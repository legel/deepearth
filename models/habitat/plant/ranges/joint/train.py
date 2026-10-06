"""Training of the joint range model and its evaluation on independent plots.

Objective. MaxEnt fits a species' relative intensity of occurrence lambda_s(x) = exp f_s(x) by maximizing the
likelihood of an inhomogeneous Poisson point process: the records are the points, the background sample stands in
for the integral of lambda over the calibration area (Renner & Warton 2013). Per species s and sampling draw

    L_s = - mean_{p in P_s} f_s(p) + log mean_{a in P_s + B_s} exp f_s(a),

with P_s a draw of the species' presences and B_s a draw of its background points. As in maxent.jar, the presences
also enter the normalizer (``addsamplestobackground``): the normalizer then always contains the points whose scores
the first term raises, which bounds the objective below, so a species can never be separated perfectly by
pushing its scores to infinity. The background is effort-weighted (see data.py), so the model learns where a
species is recorded relative to where anything is recorded, and its calibration is the species' own ecoregions.

Optimization. Each step draws ``batch_species`` species uniformly among those with records, ``n_presence``
presences and ``n_background`` background points of each (with replacement), averages L_s over the batch and takes
one AdamW step. Weight decay is the Gaussian prior of each parameter group: ``weight_decay`` on the network and the
offsets, ``weight_decay_edges`` on the branch vectors z (the Brownian-motion prior, model.py) and
``weight_decay_species`` on the species-specific vectors u. Learning rate: one-cycle schedule (5% warm-up, cosine
decay). Mixed precision: inputs stored as float16, network evaluated in bfloat16.

Evaluation. Every ``eval_every`` steps each species is scored at independent plots (VegBank, BLM AIM, FIA, ...).
A plot outside the species' calibration ecoregions ranks lowest, as it shows on the map. VegBank plots are split
into two halves by 1-degree blocks: the checkpoint with the best median AUC on the "dev" half is kept
(``model_best.pt``), the "test" half is reported. ``selection`` names the summary entry compared: by default
``dev_joint_paired``, the median over the species that also have a per-species MaxEnt map (the rule of the production
runs); without MaxEnt maps it falls back to ``dev_auc_median``, the median over all scored species.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data import JointData, PlotSet, Standardizer, TrainingPoints, block_halves, calibration_sets, plot_sets
from .model import JointRangeModel
from .tree import Tree, path_matrix


@dataclass
class TrainConfig:
    width: int = 256
    depth: int = 3
    steps: int = 21000
    batch_species: int = 256
    n_presence: int = 64
    n_background: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    weight_decay_edges: float = 1e-2
    weight_decay_species: float = 1e-4
    grad_clip: float = 5.0
    warmup: float = 0.05
    eval_every: int = 3000
    eval_chunk: int = 65536
    seed: int = 0
    selection: str = "dev_joint_paired"
    log_variables: list[str] = field(default_factory=list)
    flag_variables: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown training options: {sorted(unknown)}")
        return cls(**d)


# ------------------------------------------------------------------------------------------------- evaluation
def auc_table(present: np.ndarray, score: np.ndarray, inside: np.ndarray, sel: np.ndarray,
              min_presence: int = 5) -> np.ndarray:
    """AUC per species over the plots in ``sel``, plots not ``inside`` ranked lowest. NaN for a species with fewer
    than ``min_presence`` presence plots there, no absence plot, or no finite score (e.g. no baseline map)."""
    out = np.full(len(present), np.nan)
    for i in range(len(present)):
        y = present[i, sel]
        if y.sum() < min_presence or y.sum() == len(y) or not np.isfinite(score[i]).any():
            continue
        out[i] = roc_auc_score(y, np.where(inside[i, sel], score[i, sel], -1e9))
    return out


@torch.no_grad()
def plot_scores(model: JointRangeModel, ps: PlotSet, chunk: int = 65536) -> np.ndarray:
    """f_s at the plots of ``ps`` for its species, float32 [n_species, n_plots]."""
    dev = ps.X.device
    sid = torch.from_numpy(ps.species).to(dev).long()
    W = model.species_vectors().float()[sid]
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        H = torch.cat([model.features(ps.X[i:i + chunk].float()).float() for i in range(0, len(ps.X), chunk)])
    return (W @ H.T + model.b[sid][:, None]).cpu().numpy()


class Evaluator:
    """AUC tables of the model (and, once, of the per-species MaxEnt baseline) on every plot set. Keys: "dev" and
    "test" for the two VegBank halves, the source name (e.g. "aim", "fia") for the other sets, all plots."""

    def __init__(self, sets: Sequence[PlotSet], chunk: int = 65536):
        self.sets, self.chunk = list(sets), chunk
        vb = self.sets[0]
        dev_half = block_halves(vb.lat, vb.lon)
        self.selections = {"dev": (vb, dev_half), "test": (vb, ~dev_half)}
        for ps in self.sets[1:]:
            self.selections[ps.name] = (ps, np.ones(ps.present.shape[1], bool))
        self.baseline = {k: auc_table(ps.present, ps.baseline, ps.baseline > 0, sel)
                         for k, (ps, sel) in self.selections.items() if ps.baseline is not None}
        self.baseline = {k: v for k, v in self.baseline.items() if np.isfinite(v).any()}   # sets with MaxEnt maps

    def __call__(self, model: JointRangeModel) -> dict[str, np.ndarray]:
        was = model.training
        model.eval()
        scores = {ps.name: plot_scores(model, ps, self.chunk) for ps in self.sets}
        model.train(was)
        return {k: auc_table(ps.present, scores[ps.name], ps.inside, sel) for k, (ps, sel) in self.selections.items()}

    def summary(self, res: dict[str, np.ndarray]) -> dict:
        rec = {}
        for k, a in res.items():
            rec[f"{k}_auc_median"] = float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
            rec[f"{k}_species"] = int(np.isfinite(a).sum())
            if k in self.baseline:
                ok = np.isfinite(a) & np.isfinite(self.baseline[k])
                if ok.any():
                    rec[f"{k}_maxent_paired"] = float(np.median(self.baseline[k][ok]))
                    rec[f"{k}_joint_paired"] = float(np.median(a[ok]))
                    rec[f"{k}_win_rate"] = float((a[ok] > self.baseline[k][ok]).mean())
        return rec

    def per_species(self, res: dict[str, np.ndarray], labels: Sequence[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for ps in self.sets:
            keys = ["dev", "test"] if ps is self.sets[0] else [ps.name]
            d = {"species": [labels[i] for i in ps.species], "presence_plots": ps.present.sum(1)}
            for k in keys:
                d[f"joint_{k}" if len(keys) > 1 else "joint"] = res[k]
                if k in self.baseline:
                    d[f"maxent_{k}" if len(keys) > 1 else "maxent"] = self.baseline[k]
            out[ps.name] = pd.DataFrame(d)
        return out


# --------------------------------------------------------------------------------------------------- training
def build_model(data_dir: Path, tree_path: Path, n_inputs: int, cfg: TrainConfig) -> JointRangeModel:
    tips = list(pd.read_csv(data_dir / "species.csv").species)
    return JointRangeModel(n_inputs, path_matrix(Tree.read(tree_path), tips), cfg.width, cfg.depth)


def train(data_dir: str | Path, out: str | Path, cfg: TrainConfig, tree_path: str | Path | None = None,
          device: str | torch.device = "cuda", log=print) -> dict:
    """Train on the data directory ``data_dir`` and write ``out``: ``norm.npz`` (input standardization),
    ``model_best.pt`` (best VegBank-dev checkpoint), ``model.pt`` (final), ``run.json`` (configuration and
    evaluation log), ``per_species_<set>.csv``. Returns the contents of run.json."""
    data_dir, out = Path(data_dir), Path(out)
    tree_path = Path(tree_path) if tree_path else data_dir / "tree/natives.dated.nwk"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = JointData(data_dir)
    species = data.species()
    labels = list(species.species)
    S = len(labels)
    names = [str(n) for n in data["names"]]
    pres = data["pres"]
    st = Standardizer.fit(data.points(), np.asarray(pres) == 0, names, cfg.log_variables, cfg.flag_variables)
    st.save(out / "norm.npz")
    pts = TrainingPoints(data, st, S, dev)
    log(f"{S} species ({len(pts.active)} with records), {int(pts.n_p.sum()):,} presences, "
        f"{int(pts.n_b.sum()):,} background points, {st.n_inputs} inputs")

    model = build_model(data_dir, tree_path, st.n_inputs, cfg).to(dev)
    groups = [{"params": list(model.env.parameters()) + list(model.trunk.parameters()) + [model.b],
               "weight_decay": cfg.weight_decay},
              {"params": [model.z], "weight_decay": cfg.weight_decay_edges},
              {"params": [model.u], "weight_decay": cfg.weight_decay_species}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=cfg.warmup)
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,} (branch vectors {model.z.numel():,}, "
        f"species vectors {model.u.numel():,})")

    evaluator = Evaluator(plot_sets(data, calibration_sets(species), st, dev), cfg.eval_chunk)
    for k, v in evaluator.baseline.items():
        log(f"MaxEnt baseline {k}: median AUC {np.nanmedian(v):.4f} ({np.isfinite(v).sum()} species)")

    B, n_p, n_b = cfg.batch_species, cfg.n_presence, cfg.n_background
    record = {"config": asdict(cfg), "data_dir": str(data_dir), "tree": str(tree_path), "log": []}
    best = -math.inf
    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        s = pts.active[torch.randint(0, len(pts.active), (B,), device=dev)]
        ip = pts.p_off[s, None] + (torch.rand(B, n_p, device=dev) * pts.n_p[s, None]).long()
        ib = pts.b_off[s, None] + (torch.rand(B, n_b, device=dev) * pts.n_b[s, None]).long()
        X = pts.gather(torch.cat([ip, ib], 1).reshape(-1))
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            H = model.features(X.float()).view(B, n_p + n_b, -1)
        W = model.species_vectors()[s]
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            f = torch.einsum("bnd,bd->bn", H.float(), W.float()) + model.b[s][:, None]
        fp = f[:, :n_p]
        loss = (-fp.mean(1) + torch.logsumexp(f, 1) - math.log(f.shape[1])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % 200 == 0:
            log(f"step {step} loss {loss.item():.4f} ({(time.time() - t0) / step * 1000:.0f} ms/step)")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            res = evaluator(model)
            rec = {"step": step, "loss": float(loss), **evaluator.summary(res)}
            record["log"].append(rec)
            score = rec.get(cfg.selection, rec["dev_auc_median"])
            if score >= best:
                best = score
                torch.save(model.state_dict(), out / "model_best.pt")
                for name, df in evaluator.per_species(res, labels).items():
                    df.to_csv(out / f"per_species_{name}.csv", index=False)
            log(json.dumps(rec))
    record["seconds"] = time.time() - t0
    torch.save(model.state_dict(), out / "model.pt")
    (out / "run.json").write_text(json.dumps(record, indent=1))
    return record
