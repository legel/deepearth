"""The DeepCal benchmark suite and the harmonic-mean north star.

A trained :class:`~deepearth.core.fusion.DeepEarth` is scored on a frozen set of benchmarks, each a real question of
the form "given the widely-available context, how well is a sparse target induced?" Every benchmark is normalized to
``[0, 1]`` by ``(value - baseline) / (target - baseline)`` and the single **net score** is their harmonic mean
(power mean, p = -1) -- so lifting the *weakest* benchmark helps most and no benchmark can be sacrificed for another.

Benchmarks are computed on the held-out split (spatial blocks by default) so they measure transfer, not
memorization. A benchmark whose inputs are not present in the model's configuration is reported as inactive and left
out of the net score; enabling more modalities (climate, aerial, satellite, soil, time, flowering) activates more of
them. The suite mirrors ``core/science.md`` / the 26-benchmark plan; the always-computable core is implemented here.

The scoring definitions are FIXED -- this is the ground-truth metric for autoresearch. Do not tune them to inflate a
result; improve the model instead.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

# Baseline (uninformed reference) and target (ambitious but defined) per benchmark. Net score normalizes each raw
# value into [0, 1] as (value - baseline) / (target - baseline). Baselines are deliberately non-trivial so the score
# reflects real induction beyond a naive predictor; targets are the level we are pushing toward.
BASELINE = {
    "A2_species_vision_top1": 0.02,   "A2_species_vision_top5": 0.05,
    "A1_species_geo_top10":   0.05,
    "A4_traits_vision_f1":    0.30,
    "A5_phylo_vision_cos":    0.50,
    "A6_imagine_vision_cos":  0.30,
    "Q3_geo_gain_species":    0.00,
    "Q8_phylo_to_family":     0.05,
    "loo_traits_f1":          0.30,
    "loo_vision_dino_cos":    0.30,
}
TARGET = {
    "A2_species_vision_top1": 0.90,   "A2_species_vision_top5": 0.99,
    "A1_species_geo_top10":   0.40,
    "A4_traits_vision_f1":    0.90,
    "A5_phylo_vision_cos":    0.95,
    "A6_imagine_vision_cos":  0.90,
    "Q3_geo_gain_species":    0.15,
    "Q8_phylo_to_family":     0.90,
    "loo_traits_f1":          0.90,
    "loo_vision_dino_cos":    0.90,
}


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, observed: torch.Tensor, num_classes: int) -> float:
    """Macro-F1 over the observed rows: mean per-class F1 (unweighted), so rare classes count as much as common."""
    m = observed.bool()
    if m.sum() == 0:
        return float("nan")
    p, t = pred[m], target[m]
    f1s = []
    for c in range(num_classes):
        tp = ((p == c) & (t == c)).sum().item()
        fp = ((p == c) & (t != c)).sum().item()
        fn = ((p != c) & (t == c)).sum().item()
        if tp + fn == 0:                                  # class absent from truth -> skip
            continue
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


@torch.no_grad()
def evaluate_benchmarks(model, source, device, batch: int = 1536) -> Dict[str, float]:
    """Compute every active benchmark's raw value over the held-out split. Context is built once per batch and
    reused across the different ``given`` sets, so the whole suite costs a handful of passes over the test set."""
    model.eval()
    names = [v.name for v in model.variables]
    kinds = {v.name: v.kind for v in model.variables}
    have = set(names)
    traits = [t for t in getattr(source, "trait_classes", {})]
    trait_nc = source.trait_classes if traits else {}
    fam_of_species = source.class_group if hasattr(source, "class_group") else None   # family index per species class

    vision = [v for v in ("vision_dino", "vision_bio") if v in have]
    # "U" -- universal inputs, obtainable anywhere without observing the organism (space-time is always present via
    # the position token; environmental/remote-sensing modalities join it when configured). Growing U is the principled
    # way to lift species-from-location (A1). Definition fixed; only available content changes as modalities enable.
    universal = [v for v in ("climate", "soil", "naip_rgb", "naip_ir", "clay") if v in have]
    # accumulators: [sum, count] for means; confusion pieces handled inline via stored preds
    acc = {k: [0.0, 0.0] for k in ("A2_top1", "A2_top5", "A1_top10", "A5_cos", "A6_cos", "Q3_gain", "Q8_fam")}
    # trait F1 needs full pred/target vectors; collect per trait
    tr_pred = {t: [] for t in traits}; tr_true = {t: [] for t in traits}; tr_obs = {t: [] for t in traits}
    tr_pred_loo = {t: [] for t in traits}
    dino_loo = [0.0, 0.0]

    for c0 in range(0, len(source.test), batch):
        idx = torch.tensor(source.test[c0:c0 + batch], device=device)
        values, observed, coords, nbr_coords, mani, nbrv = source.batch(idx)
        ctx = model.context(coords, nbr_coords, mani, nbrv)
        B = len(idx)

        def infer(given, targets):
            return model.infer(values, given, targets, ctx, observed)

        # --- A2: U + ground photo -> species (the flagship) ---
        if vision:
            pr = infer(universal + vision, ["identity"])["identity"]
            top5 = pr.topk(5, dim=-1).indices
            correct1 = (pr.argmax(-1) == values["identity"])
            acc["A2_top1"][0] += correct1.float().sum().item(); acc["A2_top1"][1] += B
            acc["A2_top5"][0] += (top5 == values["identity"][:, None]).any(-1).float().sum().item(); acc["A2_top5"][1] += B
            # Q8: family accuracy from the species prediction (does the guess land in the right clade?)
            if fam_of_species is not None:
                acc["Q8_fam"][0] += (fam_of_species[pr.argmax(-1)] == fam_of_species[values["identity"]]).float().sum().item()
                acc["Q8_fam"][1] += B
            # A5: U + photo -> evolutionary-position vector
            if "phylo" in have:
                pc = infer(universal + vision, ["phylo"])["phylo"]
                acc["A5_cos"][0] += F.cosine_similarity(pc, values["phylo"], dim=-1).sum().item(); acc["A5_cos"][1] += B
            # A4: U + photo -> traits (collect for macro-F1)
            if traits:
                pt = infer(universal + vision, traits)
                for t in traits:
                    tr_pred[t].append(pt[t].argmax(-1).cpu()); tr_true[t].append(values[t].cpu()); tr_obs[t].append(observed[t].cpu())

        # --- A1: U (position + geometry-only neighbors) -> species, no photo ---
        pr0 = infer(universal, ["identity"])["identity"]
        t10 = pr0.topk(10, dim=-1).indices
        acc["A1_top10"][0] += (t10 == values["identity"][:, None]).any(-1).float().sum().item(); acc["A1_top10"][1] += B
        # Q3: marginal value of location = (geo+vision top1) - (vision-only top1) is approximated by A2 - A1 at net time

        # --- A6: imagine vision -- reconstruct DINOv3 from everything but vision ---
        if "vision_dino" in have:
            non_vision = [n for n in names if n not in ("vision_dino", "vision_bio") and observed_any(observed, n)]
            pv = infer(non_vision, ["vision_dino"])["vision_dino"]
            acc["A6_cos"][0] += F.cosine_similarity(pv, values["vision_dino"], dim=-1).sum().item(); acc["A6_cos"][1] += B
            # leave-one-out DINO: predict vision_dino from ALL other observed variables
            loo_given = [n for n in names if n != "vision_dino"]
            pv2 = infer(loo_given, ["vision_dino"])["vision_dino"]
            dino_loo[0] += F.cosine_similarity(pv2, values["vision_dino"], dim=-1).sum().item(); dino_loo[1] += B

        # --- leave-one-out traits: predict each trait from ALL other observed variables ---
        if traits:
            for t in traits:
                given = [n for n in names if n != t]
                ptl = infer(given, [t])[t]
                tr_pred_loo[t].append(ptl.argmax(-1).cpu())

    def reduce(a):
        return a[0] / a[1] if a[1] > 0 else float("nan")

    out: Dict[str, float] = {}
    if vision:
        out["A2_species_vision_top1"] = reduce(acc["A2_top1"])
        out["A2_species_vision_top5"] = reduce(acc["A2_top5"])
        if fam_of_species is not None:
            out["Q8_phylo_to_family"] = reduce(acc["Q8_fam"])
        if "phylo" in have:
            out["A5_phylo_vision_cos"] = reduce(acc["A5_cos"])
        if traits:
            f1s = [_macro_f1(torch.cat(tr_pred[t]), torch.cat(tr_true[t]), torch.cat(tr_obs[t]), trait_nc[t]) for t in traits]
            out["A4_traits_vision_f1"] = float(np.nanmean(f1s))
    out["A1_species_geo_top10"] = reduce(acc["A1_top10"])
    if vision and not np.isnan(out.get("A2_species_vision_top1", np.nan)):
        out["Q3_geo_gain_species"] = max(0.0, out["A2_species_vision_top1"] - out["A1_species_geo_top10"])
    if "vision_dino" in have:
        out["A6_imagine_vision_cos"] = reduce(acc["A6_cos"])
        out["loo_vision_dino_cos"] = reduce(dino_loo)
    if traits:
        f1s_loo = [_macro_f1(torch.cat(tr_pred_loo[t]), torch.cat(tr_true[t]), torch.cat(tr_obs[t]), trait_nc[t]) for t in traits]
        out["loo_traits_f1"] = float(np.nanmean(f1s_loo))
    return out


def observed_any(observed: Dict[str, torch.Tensor], name: str) -> bool:
    """True if variable ``name`` is observed for at least one row in the batch (so it can serve as a given)."""
    return name in observed and bool(observed[name].any())


def net_score(raw: Dict[str, float]) -> float:
    """Unweighted mean of the active benchmarks, each normalized to [0,1]."""
    normed = normalized(raw)
    vals = [v for v in normed.values() if not np.isnan(v)]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def normalized(raw: Dict[str, float]) -> Dict[str, float]:
    """Each raw benchmark -> (value - baseline) / (target - baseline), clipped to [0, 1]."""
    out = {}
    for k, v in raw.items():
        if k in BASELINE and not (isinstance(v, float) and np.isnan(v)):
            b, t = BASELINE[k], TARGET[k]
            out[k] = float(np.clip((v - b) / (t - b + 1e-9), 0.0, 1.0))
    return out


def format_benchmarks(raw: Dict[str, float]) -> str:
    """Render the raw benchmark values, their normalized [0,1] scores, and the net harmonic-mean north star."""
    normed = normalized(raw)
    lines = ["benchmark                     raw     normalized"]
    for k in sorted(raw):
        n = normed.get(k, float("nan"))
        lines.append(f"  {k:<26} {raw[k]:6.3f}   {n:6.3f}")
    lines.append(f"NET SCORE (mean of {len(normed)} active): {net_score(raw):.4f}")
    return "\n".join(lines)
