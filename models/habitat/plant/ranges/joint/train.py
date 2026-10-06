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

Background options (data.py). ``target_group``: every background point is moved to the nearest record of another
species, so presences and background share the records' fine-scale sampling. ``continental_background``: each
species also draws this many points from the background of all species across the continent (moved the same way),
so it sees where it is absent outside its calibration area, as SINR's "assume negative" loss does; with a learned
calibration penalty (model.py) this is what the penalty is learned from.

Optimization. Each step draws ``batch_species`` species uniformly among those with records, ``n_presence``
presences and ``n_background`` background points of each (with replacement), averages L_s over the batch and takes
one AdamW step. Weight decay is the Gaussian prior of each parameter group: ``weight_decay`` on the networks, the
offsets and the calibration penalty, ``weight_decay_edges`` on the branch vectors z and z_p (the Brownian-motion
prior, model.py) and ``weight_decay_species`` on the species-specific vectors u. AdamW's decay is decoupled: a
parameter shrinks by lr x decay per step, so the prior's strength is the product (lr 1e-3 and decay 1 shrink by
0.1% a step). Learning rate: one-cycle schedule (5% warm-up, cosine decay), ``place_lr`` for the place network.
Mixed precision: inputs stored as float16, networks evaluated in bfloat16.

Stages. One function trains every stage of the model (docs/joint_model.md):

* an environment model from scratch;
* the shared representation (``init``: start from a trained run's final weights; pathways it lacks start at
  zero contribution), every parameter trained;
* the species stage (``shared_from`` a run, ``freeze_shared``): the shared networks and input standardization of
  that run are kept fixed and every species parameter starts afresh and is fitted to convergence. With target-group
  background every row a step reads is a record, so the shared features of all of them are computed once
  (cache.py) and a step costs only the species' dot products.

Evaluation. Every ``eval_every`` steps each species is scored at independent plots (VegBank, BLM AIM, FIA, ...).
Two rankings: the hard calibration rule (``<set>``: a plot outside the species' calibration ecoregions ranks lowest,
as the environment model's maps show it) and the model's own scores everywhere (``<set>_noclip``: what a model with
the learned penalty maps). VegBank plots are split into two halves by 1-degree blocks: the checkpoint with the best
``selection`` value on the "dev" half is kept (``model_best.pt``), the "test" half is reported. By default
``selection`` is ``dev_joint_paired``, the median over the species that also have a per-species MaxEnt map (the rule
of the production runs); without MaxEnt maps it falls back to ``dev_auc_median``, the median over all scored species.
"""
from __future__ import annotations

import json
import math
import time
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from sklearn.metrics import roc_auc_score

from .data import JointData, PlotSet, Standardizer, TrainingPoints, block_halves, calibration_sets, plot_sets
from .field import FieldPyramid
from .model import SPECIES_TENSORS, JointRangeModel
from .tree import Tree, path_matrix

NEW_PATHWAYS = ("field.", "place.", "zp", "zc", "c0")      # tensors a warm start may lack


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
    log_variables: list[str] = dataclasses.field(default_factory=list)
    flag_variables: list[str] = dataclasses.field(default_factory=list)
    # background (data.py)
    target_group: bool = False
    continental_background: int = 0
    # data
    shoreline_records: bool = False         # add the restored shoreline presences (shore_records.npz)
    fill_plots: bool = False                # score plots with the shoreline climate fill (plot_fill_<source>.npz)
    # place pathway (place.py); 0 = none
    place_dim: int = 0
    place_hidden: int = 256
    place_blocks: int = 4
    place_lr: float | None = None
    # landscape field (field.py)
    field: bool = False
    field_dim: int = 64
    field_heads: int = 4
    field_layers: int = 2
    field_radii: list[float] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    field_angles: int = 8
    field_orders: int = 3
    field_chunk: int = 4096
    # calibration: None = the hard rule; a number = learned penalty outside the area, starting at that value
    calibration_penalty: float | None = None
    # species stage: shared networks fixed (needs ``shared_from`` and a feature cache)
    freeze_shared: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown training options: {sorted(unknown)}")
        return cls(**d)

    def model_spec(self, n_channels: int = 0) -> dict:
        """Keyword arguments of JointRangeModel beyond inputs and tree."""
        return dict(width=self.width, depth=self.depth,
                    place=dict(d_place=self.place_dim, hidden=self.place_hidden, blocks=self.place_blocks)
                    if self.place_dim else None,
                    field=dict(n_channels=n_channels, d_c=self.field_dim, heads=self.field_heads,
                               layers=self.field_layers, radii_cells=tuple(self.field_radii),
                               angles=self.field_angles, orders=self.field_orders) if self.field else None,
                    penalty_init=self.calibration_penalty)


# ------------------------------------------------------------------------------------------------- evaluation
def auc_table_reference(present: np.ndarray, score: np.ndarray, inside: np.ndarray, sel: np.ndarray,
                        min_presence: int = 5) -> np.ndarray:
    """The definition ``auc_table`` computes: per species, scikit-learn's ROC AUC over the plots in ``sel``, plots
    not ``inside`` ranked lowest."""
    out = np.full(len(present), np.nan)
    for i in range(len(present)):
        y = present[i, sel]
        if y.sum() < min_presence or y.sum() == len(y) or not np.isfinite(score[i]).any():
            continue
        out[i] = roc_auc_score(y, np.where(inside[i, sel], score[i, sel], -1e9))
    return out


def auc_table(present: np.ndarray, score: np.ndarray, inside: np.ndarray, sel: np.ndarray, min_presence: int = 5,
              device: str | torch.device | None = None) -> np.ndarray:
    """AUC per species over the plots in ``sel``, plots not ``inside`` ranked lowest. NaN for a species with fewer
    than ``min_presence`` presence plots there, no absence plot, or no finite score (e.g. no baseline map).

    Computed exactly as the Mann-Whitney statistic from average ranks (ties share their mean rank, as scikit-learn's
    roc_auc_score), in float64, for blocks of species at once (on the GPU when there is one):
    AUC = (sum of the presences' ranks - n_p (n_p + 1) / 2) / (n_p n_a)."""
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_plots = np.shape(score)[1] if np.ndim(score) == 2 else 0
    out = np.full(len(present), np.nan)
    if n_plots == 0:
        return out
    block = max(1, min(128, int(8e6 // n_plots)))                   # ~64 MB per float64 temporary
    inc_all = torch.from_numpy(np.asarray(sel, bool)).to(dev)
    for i0 in range(0, len(present), block):
        i1 = min(len(present), i0 + block)
        y = torch.from_numpy(np.array(present[i0:i1], bool)).to(dev)
        sc = torch.from_numpy(np.array(score[i0:i1], np.float64)).to(dev)
        ins = torch.from_numpy(np.array(inside[i0:i1], bool)).to(dev)
        inc = inc_all[None].expand_as(y)
        finite = torch.isfinite(sc).any(1)
        s = torch.where(ins, sc, torch.full_like(sc, -1e9))
        s = torch.where(inc, s, torch.full_like(s, float("inf")))       # not selected: after every selected plot
        srt = torch.sort(s, 1)[0].contiguous()
        s = s.contiguous()
        lo = torch.searchsorted(srt, s, right=False).double()
        hi = torch.searchsorted(srt, s, right=True).double()
        rank = (lo + 1 + hi) / 2                                         # average 1-based rank within ties
        pos = y & inc
        n_pos = pos.sum(1).double()
        n_neg = inc.sum(1).double() - n_pos
        auc = ((rank * pos).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg).clamp(min=1)
        ok = (n_pos >= min_presence) & (n_neg > 0) & finite
        out[i0:i1] = torch.where(ok, auc, torch.full_like(auc, float("nan"))).cpu().numpy()
    return out


def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16)


@torch.no_grad()
def plot_features(model: JointRangeModel, ps: PlotSet, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Shared features (F, P) of the plots of ``ps`` (float32), the networks evaluated in bfloat16 as in training."""
    dev = ps.X.device
    Fs, Ps = [], []
    for i in range(0, len(ps.X), chunk):
        with _autocast(dev):
            f, p = model.shared(ps.X[i:i + chunk].float(), ps.latlon[i:i + chunk],
                                None if ps.rc is None else ps.rc[i:i + chunk])
        Fs.append(f.float())
        if p is not None:
            Ps.append(p.float())
    return torch.cat(Fs), (torch.cat(Ps) if Ps else None)


@torch.no_grad()
def scores_from_features(model: JointRangeModel, ps: PlotSet, Fx: torch.Tensor, P: torch.Tensor | None,
                         block: int = 128) -> np.ndarray:
    """f_s at the plots of ``ps`` for its species from their shared features, float32 [n_species, n_plots], with the
    learned penalty outside each species' calibration area (if the model has one); computed in blocks of species."""
    dev = Fx.device
    sid = torch.from_numpy(ps.species).to(dev).long()
    W, b, V, pen = model.species_vectors().float(), model.b.float(), model.place_vectors(), model.outside_penalty()
    out = np.empty((len(sid), Fx.shape[0]), np.float32)
    for i in range(0, len(sid), block):
        s = sid[i:i + block]
        f = W[s] @ Fx.float().T + b[s][:, None]
        if V is not None:
            f = f + V[s].float() @ P.float().T
        if pen is not None:
            f = f - pen[s][:, None] * torch.from_numpy(ps.outside[i:i + block]).to(dev).float()
        out[i:i + block] = f.cpu().numpy()
    return out


def plot_scores(model: JointRangeModel, ps: PlotSet, chunk: int = 65536) -> np.ndarray:
    """f_s at the plots of ``ps`` for its species, float32 [n_species, n_plots]."""
    return scores_from_features(model, ps, *plot_features(model, ps, chunk))


class Evaluator:
    """AUC tables of the model (and, once, of the per-species MaxEnt baseline) on every plot set. Keys: "dev" and
    "test" for the two VegBank halves, the source name (e.g. "aim", "fia") for the other sets, all plots; each also
    as "<key>_noclip", ranked by the scores alone over the plots with predictors. ``features``: plot set -> its shared
    features (F, P); by default computed by the model (``plot_features``), in the species stage read from the cache."""

    def __init__(self, sets: Sequence[PlotSet], chunk: int = 65536,
                 features: Callable[[PlotSet], tuple] | None = None):
        self.sets, self.chunk, self.features = list(sets), chunk, features
        vb = self.sets[0]
        dev_half = block_halves(vb.lat, vb.lon)
        self.selections = {"dev": (vb, dev_half), "test": (vb, ~dev_half)}
        for ps in self.sets[1:]:
            self.selections[ps.name] = (ps, np.ones(ps.present.shape[1], bool))
        self.baseline = {k: auc_table(ps.present, ps.baseline, ps.baseline > 0, sel)
                         for k, (ps, sel) in self.selections.items() if ps.baseline is not None}
        self.baseline = {k: v for k, v in self.baseline.items() if np.isfinite(v).any()}   # sets with MaxEnt maps

    def scores(self, model: JointRangeModel) -> dict[str, np.ndarray]:
        was = model.training
        model.eval()
        out = {}
        for ps in self.sets:
            feats = self.features(ps) if self.features is not None else plot_features(model, ps, self.chunk)
            out[ps.name] = scores_from_features(model, ps, *feats)
        model.train(was)
        return out

    def __call__(self, model: JointRangeModel) -> dict[str, np.ndarray]:
        scores = self.scores(model)
        res = {}
        for k, (ps, sel) in self.selections.items():
            res[k] = auc_table(ps.present, scores[ps.name], ps.inside, sel)
            res[f"{k}_noclip"] = auc_table(ps.present, scores[ps.name], np.broadcast_to(ps.valid, ps.present.shape),
                                           sel)
        return res

    def summary(self, res: dict[str, np.ndarray]) -> dict:
        rec = {}
        for k, a in res.items():
            rec[f"{k}_auc_median"] = float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
            rec[f"{k}_species"] = int(np.isfinite(a).sum())
            base = self.baseline.get(k.removesuffix("_noclip"))
            if base is not None:
                ok = np.isfinite(a) & np.isfinite(base)
                if ok.any():
                    rec[f"{k}_maxent_paired"] = float(np.median(base[ok]))
                    rec[f"{k}_joint_paired"] = float(np.median(a[ok]))
                    rec[f"{k}_win_rate"] = float((a[ok] > base[ok]).mean())
        return rec

    def per_species(self, res: dict[str, np.ndarray], labels: Sequence[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for ps in self.sets:
            keys = ["dev", "test"] if ps is self.sets[0] else [ps.name]
            d = {"species": [labels[i] for i in ps.species], "presence_plots": ps.present.sum(1)}
            for k in keys:
                tag = f"_{k}" if len(keys) > 1 else ""
                d[f"joint{tag}"] = res[k]
                d[f"joint{tag}_noclip"] = res[f"{k}_noclip"]
                if k in self.baseline:
                    d[f"maxent{tag}"] = self.baseline[k]
            out[ps.name] = pd.DataFrame(d)
        return out


# --------------------------------------------------------------------------------------------------- training
def build_model(data_dir: Path, tree_path: Path, n_inputs: int, cfg: TrainConfig, n_channels: int = 0
                ) -> JointRangeModel:
    tips = list(pd.read_csv(data_dir / "species.csv").species)
    return JointRangeModel(n_inputs, path_matrix(Tree.read(tree_path), tips), **cfg.model_spec(n_channels))


def _load_partial(model: JointRangeModel, state: dict, take: Callable[[str], bool]) -> list[str]:
    """Copy the tensors of ``state`` that ``take`` accepts; their shapes must match. Returns the names taken."""
    own = model.state_dict()
    names = [k for k in state if take(k)]
    bad = [k for k in names if k not in own or own[k].shape != state[k].shape]
    if bad:
        raise ValueError(f"tensors that do not fit this model: {bad[:8]}")
    model.load_state_dict({k: state[k] for k in names}, strict=False)
    return names


class _CalibrationArea:
    """Membership of ecoregions in each species' calibration area, on the device: ``outside(s, eco)`` is True where
    ecoregion ``eco`` (-1 = off the grid) is not in species ``s``'s area."""

    def __init__(self, calib: list[set[int]], device):
        n_eco = max((max(c) for c in calib if c), default=0) + 2
        self.bits = torch.zeros((len(calib), n_eco), dtype=torch.bool, device=device)
        for s, c in enumerate(calib):
            if c:
                self.bits[s, torch.tensor(sorted(c), device=device)] = True

    def outside(self, s: torch.Tensor, eco: torch.Tensor) -> torch.Tensor:
        e = eco.long()
        known = (e >= 0) & (e < self.bits.shape[1])
        return ~(self.bits[s.expand_as(e), e.clamp(0, self.bits.shape[1] - 1)] & known)


def train(data_dir: str | Path, out: str | Path, cfg: TrainConfig, tree_path: str | Path | None = None,
          device: str | torch.device = "cuda", log=print, init: str | Path | None = None,
          shared_from: str | Path | None = None, cache_dir: str | Path | None = None,
          field_dir: str | Path | None = None, ecoregion_raster: str | Path | None = None) -> dict:
    """Train on the data directory ``data_dir`` and write ``out``: ``norm.npz`` (input standardization),
    ``model_best.pt`` (best VegBank-dev checkpoint), ``model.pt`` (final), ``run.json`` (configuration and
    evaluation log), ``per_species_<set>.csv``. Returns the contents of run.json.

    ``init``: a run directory whose final weights (``model.pt``) and standardization start this model (new pathways
    start fresh). ``shared_from``: a run directory whose shared networks and standardization this model takes
    (species parameters start fresh); with ``cfg.freeze_shared`` they stay fixed and the shared features are read
    from ``cache_dir`` (cache.py). ``field_dir``: the field pyramid and the grid positions of rows and plots (default
    ``<data_dir>/field``). ``ecoregion_raster``: the ecoregion ids of that grid (learned calibration)."""
    data_dir, out = Path(data_dir), Path(out)
    tree_path = Path(tree_path) if tree_path else data_dir / "tree/natives.dated.nwk"
    field_dir = Path(field_dir) if field_dir else data_dir / "field"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.freeze_shared and (shared_from is None or cache_dir is None):
        raise ValueError("freeze_shared needs shared_from and its feature cache (cache_dir)")
    if cfg.freeze_shared and not cfg.target_group:
        raise ValueError("freeze_shared reads cached features of record rows only: it needs target_group")
    if cfg.calibration_penalty is not None and ecoregion_raster is None:
        raise ValueError("a learned calibration penalty needs the grid's ecoregion raster")

    data = JointData(data_dir)
    species = data.species()
    labels = list(species.species)
    S = len(labels)
    calib = calibration_sets(species)
    names = [str(n) for n in data["names"]]
    pres = data["pres"]
    source = Path(shared_from or init) if (shared_from or init) else None
    if source is not None:                     # the networks expect inputs scaled as in their own training
        st = Standardizer.load(source / "norm.npz", cfg.flag_variables)
        if st.names != names:
            raise ValueError(f"{source}: its predictors differ from {data_dir}'s")
    else:
        st = Standardizer.fit(data.points(), np.asarray(pres) == 0, names, cfg.log_variables, cfg.flag_variables)
    st.save(out / "norm.npz")

    pyramid = FieldPyramid(field_dir, dev) if (cfg.field and not cfg.freeze_shared) else None
    n_channels = len(json.loads((field_dir / "field.json").read_text())["channels"]) if cfg.field else 0
    model = build_model(data_dir, tree_path, st.n_inputs, cfg, n_channels).to(dev)
    if pyramid is not None:
        model.attach_field(pyramid)
    if init is not None:
        state = torch.load(Path(init) / "model.pt", map_location=dev, weights_only=True)
        taken = _load_partial(model, state, lambda k: True)
        fresh = [k for k in model.state_dict() if k not in taken]
        if any(not k.startswith(NEW_PATHWAYS) for k in fresh):
            raise ValueError(f"{init}: missing tensors that are not new pathways: {fresh[:8]}")
        log(f"started from {init} ({len(taken)} tensors; {len(fresh)} new)")
    if shared_from is not None:
        state = torch.load(Path(shared_from) / "model_best.pt", map_location=dev, weights_only=True)
        taken = _load_partial(model, state, lambda k: k not in SPECIES_TENSORS)
        missing = [n for n in model.state_dict() if n not in SPECIES_TENSORS and n not in taken]
        if missing:
            raise ValueError(f"{shared_from} lacks shared tensors of this model: {missing[:8]}")
        log(f"shared networks from {shared_from} ({len(taken)} tensors); species parameters start afresh")
    if cfg.freeze_shared:
        for n, p in model.named_parameters():
            p.requires_grad_(n in SPECIES_TENSORS or n == "c0")             # c0: the penalty's common offset

    restored = data.restored() if cfg.shoreline_records else None
    if cfg.shoreline_records and restored is None:
        raise FileNotFoundError(f"{data_dir}/shore_records.npz: the shoreline records are not built")
    eco_grid = None
    if cfg.calibration_penalty is not None:
        import rasterio
        with rasterio.open(ecoregion_raster) as r:
            eco_grid = r.read(1).astype(np.int32)
    need_positions = (cfg.field and not cfg.freeze_shared) or eco_grid is not None
    pts = TrainingPoints(data, None if cfg.freeze_shared else st, S, dev, restored=restored,
                         latlon=bool(cfg.place_dim) and not cfg.freeze_shared,
                         positions=data.positions(field_dir) if need_positions else None, ecoregions=eco_grid,
                         snap=data.snap() if cfg.target_group else None)
    del eco_grid
    log(f"{S} species ({len(pts.active)} with records), {int(pts.n_p.sum()):,} presences"
        f"{f' ({len(pts) - pts.n_prepared:,} restored)' if restored is not None else ''}, "
        f"{int(pts.n_b.sum()):,} background points, {st.n_inputs} inputs")
    area = _CalibrationArea(calib, dev) if model.has_penalty else None

    cache = None
    if cfg.freeze_shared:
        from .cache import FeatureCache
        cache = FeatureCache(cache_dir, dev)
        cache.check(shared_from, len(pts))
        if cache.place is None and model.place is not None:
            raise ValueError(f"{cache_dir}: no place features cached")
        log(f"shared features of {cache.n_rows:,} record rows from {cache_dir}")

    def param_groups():
        groups = {"network": [], "place": [], "branches": [], "species": []}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            key = ("branches" if n in ("z", "zp") else "species" if n == "u"
                   else "place" if n.startswith("place.") and cfg.place_lr else "network")
            groups[key].append(p)
        decay = {"network": cfg.weight_decay, "place": cfg.weight_decay, "branches": cfg.weight_decay_edges,
                 "species": cfg.weight_decay_species}
        return [{"params": v, "weight_decay": decay[k], "lr": cfg.place_lr if k == "place" else cfg.lr}
                for k, v in groups.items() if v]
    opt = torch.optim.AdamW(param_groups(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=cfg.steps, pct_start=cfg.warmup)
    n_trained = sum(p.numel() for g in opt.param_groups for p in g["params"])
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,} (branch vectors {model.z.numel():,}, "
        f"species vectors {model.u.numel():,}; trained {n_trained:,})")

    sets = plot_sets(data, calib, st, dev, fill=cfg.fill_plots, field_dir=field_dir if cfg.field else None)
    evaluator = Evaluator(sets, cfg.eval_chunk, cache.plot_features if cache is not None else None)
    for k, v in evaluator.baseline.items():
        log(f"MaxEnt baseline {k}: median AUC {np.nanmedian(v):.4f} ({np.isfinite(v).sum()} species)")

    B, n_p, n_b, n_g = cfg.batch_species, cfg.n_presence, cfg.n_background, cfg.continental_background
    n_row = n_p + n_b + n_g
    record = {"config": asdict(cfg), "data_dir": str(data_dir), "tree": str(tree_path), "init": str(init or ""),
              "shared_from": str(shared_from or ""), "cache": str(cache_dir or ""), "log": []}
    best = -math.inf
    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        s = pts.active[torch.randint(0, len(pts.active), (B,), device=dev)]
        ip = pts.p_off[s, None] + (torch.rand(B, n_p, device=dev) * pts.n_p[s, None]).long()
        ib = pts.b_off[s, None] + (torch.rand(B, n_b, device=dev) * pts.n_b[s, None]).long()
        rows_b = pts.rows[ib]
        parts = [pts.rows[ip], pts.snap[rows_b] if cfg.target_group else rows_b]
        if n_g:
            g = pts.background[torch.randint(0, len(pts.background), (B, n_g), device=dev)]
            parts.append(pts.snap[g] if cfg.target_group else g)
        rows = torch.cat(parts, 1).reshape(-1)
        W = model.species_vectors()[s]
        V = model.place_vectors()[s] if model.place is not None else None
        if cache is not None:                          # fixed shared networks: cached features, fp32 products
            Fx, P = cache.rows(rows)
            f = torch.einsum("bnd,bd->bn", Fx.view(B, n_row, -1), W) + model.b[s][:, None]
            if V is not None:
                f = f + torch.einsum("bnd,bd->bn", P.view(B, n_row, -1), V)
        else:
            with _autocast(dev):
                H = model.features(pts.X[rows].float())
                if model.field is not None:
                    H = H + model.field(pts.positions[rows], cfg.field_chunk)
                f = torch.einsum("bnd,bd->bn", H.float().view(B, n_row, -1), W.float()) + model.b[s][:, None]
                if V is not None:
                    P = torch.cat([torch.utils.checkpoint.checkpoint(model.place, pts.latlon[rows[i:i + 32768]],
                                                                     use_reentrant=False)
                                   for i in range(0, len(rows), 32768)])       # chunked: bounded activation memory
                    f = f + torch.einsum("bnd,bd->bn", P.float().view(B, n_row, -1), V.float())
            f = f.float()
        if area is not None:                           # learned penalty outside each species' calibration area
            out_area = area.outside(s[:, None], pts.ecoregion[rows].view(B, n_row))
            f = f - model.outside_penalty()[s][:, None] * out_area.float()
        fp = f[:, :n_p]
        loss = (-fp.mean(1) + torch.logsumexp(f, 1) - math.log(f.shape[1])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
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
