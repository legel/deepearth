"""The DeepCal benchmark suite (B1-B26) and the harmonic-mean north star.

A trained :class:`~deepearth.core.fusion.DeepEarth` is scored on 26 benchmarks, each a real question of the form
"given the widely-available context U (and sometimes a ground photo), how well is a sparse target induced?" Each
benchmark's metric is ALREADY naturally in ``[0, 1]`` (top-k accuracy, family accuracy, macro-F1, cosine similarity
of unit embeddings, recall@k, or calibration MRR), so the score IS the raw value -- there is NO baseline/target
remap. A hand-set target below a metric's attainable maximum is an artificial ceiling that saturates a still-
improving benchmark at 1.0; we reject that. The single **net score** is the harmonic mean (power mean p = -1) of the
active benchmarks, so lifting the *weakest* helps most and none can be sacrificed for another.

Universal inputs ``U = {space-time position, climate, soil, clay, naip_rgb, naip_ir}`` (+ topo when wired) -- all
obtainable at a point WITHOUT observing the organism. Benchmarks are computed on the held-out split (0.5-degree
spatial blocks by default; ``holdout: temporal`` gives a strictly-future forecast split, ``holdout: phylo`` holds
out whole families) so they measure transfer, not memorization. A benchmark whose required inputs or split are not
present is reported as inactive (NaN) and left out of the net score.

The suite realizes ``science.md`` / the 26-benchmark plan (originally labelled A1-A16 + Q1-Q10; renumbered B1-B26).
Scoring is the ground-truth metric for autoresearch: never tune a definition to inflate a result -- improve the model.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

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


# The 26 benchmarks, in suite order. Each is (given-set -> target, metric). B1-B24 score on any held-out split;
# B25 needs the temporal (forecast) split; B26 needs flowering labels. Realizes the recovered A1-A16 + Q1-Q10 plan.
BENCHMARKS: List[str] = [
    "B1_species_from_env_top10",        # A1  U -> species (SDM), top-10 accuracy
    "B2_species_from_photo_top1",       # A2  U + ground photo -> species (FLAGSHIP), top-1
    "B3_species_from_photo_top5",       # A2  U + ground photo -> species, top-5
    "B4_species_from_photo_only_top1",  # Q2  photo-only -> species, top-1
    "B5_species_from_spacetime_top10",  # Q7  bare space-time -> species, top-10
    "B6_family_from_env",               # A5/Q5  U -> family (niche determinism), accuracy
    "B7_family_from_phylo",             # Q8  phylo embedding -> family, accuracy
    "B8_family_from_spacetime",         # Q7  bare space-time -> family, accuracy
    "B9_phylo_from_photo_cos",          # A5  U + photo -> phylo/evolutionary vector, cosine
    "B10_traits_from_photo_env_f1",     # A4  U + photo -> traits, macro-F1
    "B11_traits_from_photo_f1",         # Q1  photo-only -> traits, macro-F1
    "B12_traits_leave_one_out_f1",      # Q6  all-but-trait -> trait, macro-F1
    "B13_imagine_vision_cos",           # A6  non-vision -> ground-vision (DINO), cosine
    "B14_vision_leave_one_out_cos",     # Q6  all-but-vision -> ground-vision, cosine
    "B15_vision_from_aerial_cos",       # Q9  aerial (NAIP) -> ground-vision, cosine
    "B16_infer_clay_cos",               # A11 U-minus-clay -> clay (Sentinel-2), cosine
    "B17_infer_soil_cos",               # A12 U-minus-soil -> soil (SSURGO), cosine
    "B18_infer_climate_cos",            # Q6  U-minus-climate -> climate (Daymet), cosine
    "B19_infer_aerial_cos",             # Q9  U-minus-naip -> aerial (NAIP), cosine
    "B20_community_from_env_recall",    # A3  U -> local community set, recall@10
    "B21_community_from_species_recall",# Q10 focal species -> co-occurring set, recall@10
    "B22_companions_recall",            # A15 species + U -> companions, recall@10
    "B23_species_calibration_mrr",      # A14 U -> species posterior, mean reciprocal rank
    "B24_geo_information_gain",         # Q3  species gain from location = B2 - B1
    "B25_forecast_climate_cos",         # A9  future climate (temporal holdout), cosine
    "B26_flowering_auc",                # A7  U/imagined-vision -> flowering (needs labels), ROC-AUC
]


@torch.no_grad()
def evaluate_benchmarks(model, source, device, batch: int = 1536) -> Dict[str, float]:
    """Score the 26-benchmark suite over the held-out split. Context is built once per batch and the encoder is run
    once per distinct given-set (multiple targets decoded from a single encode), so the whole suite costs a bounded
    number of passes over the test set. Benchmarks whose inputs/split are unavailable are simply omitted (inactive)."""
    model.eval()
    names = [v.name for v in model.variables]
    have = set(names)
    traits = [t for t in getattr(source, "trait_classes", {})]
    trait_nc = source.trait_classes if traits else {}
    fam = source.class_group if hasattr(source, "class_group") else None    # family index per species class
    holdout = getattr(source, "holdout", "spatial")

    vision = [v for v in ("vision_dino", "vision_bio") if v in have]
    U = [v for v in ("climate", "soil", "naip_rgb", "naip_ir", "clay", "topo") if v in have]   # universal (no organism obs)
    naip = [v for v in ("naip_rgb", "naip_ir") if v in have]

    acc: Dict[str, list] = {}                              # key -> [sum, count]
    def add(key, s, n):
        a = acc.setdefault(key, [0.0, 0.0]); a[0] += float(s); a[1] += n
    # trait macro-F1 needs the full pred/target/observed vectors gathered per preset
    trc = {lab: {t: ([], [], []) for t in traits} for lab in ("photo_env", "photo", "loo")}
    RK = 10
    community_cap = 6 * batch                              # recall@k is O(K*classes); cap to the first few batches for speed

    def topk_hit(logits, target, k):
        return (logits.topk(k, -1).indices == target[:, None]).any(-1).float().sum().item()
    def fam_hit(logits, target):
        return (fam[logits.argmax(-1)] == fam[target]).float().sum().item()
    def cos_sum(pred, tgt):
        return F.cosine_similarity(pred.float(), tgt.float(), dim=-1).sum().item()
    def recall_sum(logits, target_set):                   # mean over rows of |topK ∩ set| / |set|
        kmem = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, logits.topk(RK, -1).indices, True)
        inter = (kmem & target_set).sum(1).float()
        denom = target_set.sum(1).float().clamp(min=1)
        return (inter / denom).sum().item()

    for c0 in range(0, len(source.test), batch):
        idx = torch.tensor(source.test[c0:c0 + batch], device=device)
        values, observed, coords, nbr_coords, mani, nbrv = source.batch(idx)
        ctx = model.context(coords, nbr_coords, mani, nbrv)
        B = len(idx); tid = values["identity"]
        obs = [n for n in names if observed_any(observed, n)]
        def infer(given, targets):
            return model.infer(values, given, targets, ctx, observed)

        # ---- species / family / phylo / traits from shared given-sets (one encode each) ----
        if U:
            L_U = infer(U, ["identity"])["identity"]
            add("B1_species_from_env_top10", topk_hit(L_U, tid, 10), B)
            if fam is not None:
                add("B6_family_from_env", fam_hit(L_U, tid), B)
                rank = (L_U.argsort(-1, descending=True) == tid[:, None]).float().argmax(-1)
                add("B23_species_calibration_mrr", (1.0 / (rank.float() + 1)).sum().item(), B)   # A14 calibration
        if vision:
            tg = ["identity"] + (["phylo"] if "phylo" in have else []) + traits
            P = infer(U + vision, tg)                     # U + photo, decode species/phylo/traits from one encode
            add("B2_species_from_photo_top1", topk_hit(P["identity"], tid, 1), B)
            add("B3_species_from_photo_top5", topk_hit(P["identity"], tid, 5), B)
            if "phylo" in have:
                add("B9_phylo_from_photo_cos", cos_sum(P["phylo"], values["phylo"]), B)
            for t in traits:
                a, b, o = trc["photo_env"][t]; a.append(P[t].argmax(-1).cpu()); b.append(values[t].cpu()); o.append(observed[t].cpu())
            Pv = infer(vision, ["identity"] + traits)     # photo-only
            add("B4_species_from_photo_only_top1", topk_hit(Pv["identity"], tid, 1), B)
            for t in traits:
                a, b, o = trc["photo"][t]; a.append(Pv[t].argmax(-1).cpu()); b.append(values[t].cpu()); o.append(observed[t].cpu())
        L_blank = infer([], ["identity"])["identity"]     # bare space-time
        add("B5_species_from_spacetime_top10", topk_hit(L_blank, tid, 10), B)
        if fam is not None:
            add("B8_family_from_spacetime", fam_hit(L_blank, tid), B)
            if "phylo" in have:
                add("B7_family_from_phylo", fam_hit(infer(["phylo"], ["identity"])["identity"], tid), B)

        # ---- trait leave-one-out (each trait from all other observed variables) ----
        for t in traits:
            pl = infer([n for n in names if n != t], [t])[t]
            a, b, o = trc["loo"][t]; a.append(pl.argmax(-1).cpu()); b.append(values[t].cpu()); o.append(observed[t].cpu())

        # ---- ground-vision: imagine / leave-one-out / from aerial ----
        if "vision_dino" in have:
            nonvis = [n for n in obs if n not in ("vision_dino", "vision_bio")]
            add("B13_imagine_vision_cos", cos_sum(infer(nonvis, ["vision_dino"])["vision_dino"], values["vision_dino"]), B)
            add("B14_vision_leave_one_out_cos", cos_sum(infer([n for n in names if n != "vision_dino"], ["vision_dino"])["vision_dino"], values["vision_dino"]), B)
            if naip:
                add("B15_vision_from_aerial_cos", cos_sum(infer(naip, ["vision_dino"])["vision_dino"], values["vision_dino"]), B)

        # ---- dense environmental field: reconstruct each modality from the rest of U (measure-everything) ----
        for key, tv in (("B16_infer_clay_cos", "clay"), ("B17_infer_soil_cos", "soil"), ("B18_infer_climate_cos", "climate")):
            if tv in have:
                add(key, cos_sum(infer([n for n in U if n != tv], [tv])[tv], values[tv]), B)
        if naip:
            add("B19_infer_aerial_cos", cos_sum(infer([n for n in U if n not in naip], ["naip_rgb"])["naip_rgb"], values["naip_rgb"]), B)

        # ---- community / companions (neighbor species are the eval TARGET only; never fed as input -> leak-safe) ----
        if fam is not None and c0 < community_cap and hasattr(source, "neighbors"):
            tset = torch.zeros(B, L_blank.shape[1], dtype=torch.bool, device=device)
            tset.scatter_(1, source.cls[source.neighbors[idx]], True); tset.scatter_(1, tid[:, None], True)
            if U:
                add("B20_community_from_env_recall", recall_sum(infer(U, ["identity"])["identity"], tset), B)
            add("B21_community_from_species_recall", recall_sum(infer(["identity"], ["identity"])["identity"], tset), B)
            add("B22_companions_recall", recall_sum(infer(["identity"] + U, ["identity"])["identity"], tset), B)

        # ---- forecasting (temporal holdout only): predict the held-out FUTURE environment ----
        if holdout == "temporal" and "climate" in have:
            add("B25_forecast_climate_cos", cos_sum(infer([n for n in U if n != "climate"], ["climate"])["climate"], values["climate"]), B)

        # ---- B26 flowering (A7): activates once flowering labels are wired as a variable; inactive until then ----

    # ---- reduce ----
    out: Dict[str, float] = {}
    for k, (s, n) in acc.items():
        if n > 0:
            out[k] = s / n
    if traits:
        for lab, key in (("photo_env", "B10_traits_from_photo_env_f1"),
                         ("photo", "B11_traits_from_photo_f1"),
                         ("loo", "B12_traits_leave_one_out_f1")):
            if all(trc[lab][t][0] for t in traits):
                f1s = [_macro_f1(torch.cat(trc[lab][t][0]), torch.cat(trc[lab][t][1]), torch.cat(trc[lab][t][2]), trait_nc[t]) for t in traits]
                out[key] = float(np.nanmean(f1s))
        def _tf1(t):
            c = trc["photo_env"][t]
            return float(_macro_f1(torch.cat(c[0]), torch.cat(c[1]), torch.cat(c[2]), trait_nc[t])) if c[0] else float("nan")
        if "seasonality" in traits: out["B30_seasonality_trait_f1"] = _tf1("seasonality")
        if "water" in traits and "soil_drainage" in traits:
            out["B38_water_soil_regime_f1"] = float(np.nanmean([_tf1("water"), _tf1("soil_drainage")]))
        if "form" in traits: out["B49_form_trait_f1"] = _tf1("form")
    if "B2_species_from_photo_top1" in out and "B1_species_from_env_top10" in out:
        out["B24_geo_information_gain"] = max(0.0, out["B2_species_from_photo_top1"] - out["B1_species_from_env_top10"])
    return out


def observed_any(observed: Dict[str, torch.Tensor], name: str) -> bool:
    """True if variable ``name`` is observed for at least one row in the batch (so it can serve as a given)."""
    return name in observed and bool(observed[name].any())


_SCORE_FLOOR = 1e-3   # keeps the harmonic mean finite/comparable if a benchmark reads ~0 (a zero would otherwise nuke it to 0)


def normalized(raw: Dict[str, float]) -> Dict[str, float]:
    """Each benchmark's score in [0,1]. EVERY benchmark here is defined on a metric that is ALREADY naturally in
    [0,1] -- top-k accuracy, family accuracy, macro-F1, cosine similarity of unit embeddings, recall@k, calibration
    MRR -- so the score IS the raw value (clipped for safety). No baseline/target remap: a hand-set target below the
    attainable maximum is an ARTIFICIAL ceiling that saturates a still-improving metric at 1.0, which we reject."""
    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in raw.items()
            if not (isinstance(v, float) and np.isnan(v))}


def net_score(raw: Dict[str, float]) -> float:
    """North star = HARMONIC mean (power mean p=-1) of the active benchmark scores. Chosen over the arithmetic mean
    so lifting the WEAKEST benchmark helps most and no benchmark can be sacrificed for another (matches the module
    docstring's stated design; the old code silently used an arithmetic mean -- contradiction resolved here)."""
    vals = [max(v, _SCORE_FLOOR) for v in normalized(raw).values()]
    if not vals:
        return 0.0
    return float(len(vals) / sum(1.0 / v for v in vals))


def arithmetic_net(raw: Dict[str, float]) -> float:
    """Arithmetic mean of the active scores -- reported alongside the harmonic north star for legibility (it moves
    when any benchmark improves, whereas the harmonic mean is dominated by the current weakest)."""
    vals = list(normalized(raw).values())
    return float(sum(vals) / len(vals)) if vals else 0.0


def format_benchmarks(raw: Dict[str, float]) -> str:
    """Render every benchmark's [0,1] score (weakest first, so the binding constraint is on top), plus both the
    harmonic-mean north star and the arithmetic mean over the active benchmarks."""
    normed = normalized(raw)
    order = {b: i for i, b in enumerate(BENCHMARKS)}
    lines = ["benchmark                             score"]
    for k in sorted(raw, key=lambda k: normed.get(k, 1.0)):     # weakest-first: the harmonic mean's binding benchmarks lead
        s = normed.get(k, float("nan"))
        lines.append(f"  {k:<34} {s:6.3f}")
    n_defined = len(BENCHMARKS)
    lines.append(f"NET SCORE (harmonic mean of {len(normed)}/{n_defined} active): {net_score(raw):.4f}")
    lines.append(f"  (arithmetic mean: {arithmetic_net(raw):.4f})")
    return "\n".join(lines)
