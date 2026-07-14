"""The DeepCal benchmark suite (B1-B60+, climbing) and the harmonic-mean north star.

A trained :class:`~deepearth.core.fusion.DeepEarth` is scored on the full benchmark suite, each a real question of the form
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

The suite realizes ``science.md`` (originally labelled A1-A16 + Q1-Q10; renumbered B1-B60+, still growing — three
phylogenomic-ablation families, the pollinator suite, phenology seasonality/fidelity, and forecasting).
Scoring is the ground-truth metric for autoresearch: never tune a definition to inflate a result -- improve the model.
"""
from __future__ import annotations
from typing import Dict, List
import math
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
    "B31_forecast_vision_cos",          # A10 future ground-vision / appearance (temporal holdout), cosine
    "B26_flowering_auc",                # A7  U/imagined-vision -> flowering (needs labels), ROC-AUC
    "B27_flowering_fidelity",           # phenology self-consistency: flowering agreement between imagined vision (U) and real vision (U+photo)
    "B28_flowering_peak_month_mrr",     # phenology seasonality: MRR of the true peak-flowering month from a 12-month time sweep
    "B34_lfmc_from_env",                # ecophysiology: predict a species' peak fire-season live fuel moisture
    "B42_mycorrhiza_from_env",          # biotic symbiosis: predict a plant's mycorrhizal type (FungalRoot AM/EcM/ErM/OM/NM) from env, macro-F1
    "B41_pollinator_from_species_recall",  # plant identity + env -> local pollinator set (GloBI), recall@10
    "B43_infer_hydro_cos",              # U-minus-hydro -> drainage/wind (HydroSHEDS+Winstral), cosine
    "B51_pollinator_from_env_recall",   # env only -> pollinators (interaction from habitat), recall@10
    "B52_pollinator_from_photo_recall", # env + ground photo -> pollinators, recall@10
    "B53_pollinator_calibration_mrr",   # pollinator posterior calibration, mean reciprocal rank
    "B54_pollinator_dist_kl",           # predicted vs true pollinator frequency distribution, exp(-KL)
    "B55_pollinator_phylo_transfer_recall",  # rule 27: predict a plant's pollinators from its relatives' pollinators (cross-tree induction)
    "B56_family_phylo_graph_gain",      # ablation-delta: family-from-phylo accuracy gained from the species-graph refinement
    "B57_flowering_phylo_graph_gain",   # ablation-delta (phenology family): flowering-AUC gained from the species-graph refinement
    "B58_lfmc_phylo_graph_gain",        # ablation-delta (ecophysiology family): LFMC-correlation gained from the species-graph refinement
    "B59_pollinator_phylo_graph_gain",  # ablation-delta (interactions family): pollinator-recall gained from the species-graph refinement
    "B60_community_phylo_graph_gain",   # ablation-delta (niche/community family): env->community recall gained from the species-graph refinement
    "B62_mycorrhiza_phylo_graph_gain",  # ablation-delta (symbiosis family): mycorrhiza macro-F1 gained from the species-graph refinement (symbiosis is phylo-conserved)
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
    U = [v for v in ("climate", "soil", "naip_rgb", "naip_ir", "clay", "topo", "chm", "hydro") if v in have]   # universal (no organism obs)
    naip = [v for v in ("naip_rgb", "naip_ir") if v in have]

    acc: Dict[str, list] = {}                              # key -> [sum, count]
    lfmc_p, lfmc_t = [], []                                # B34: predicted vs true live fuel moisture over the eval set
    myco_p, myco_t = [], []                                # B42: predicted vs true mycorrhizal type (class) over the eval set
    flower_p, flower_t = [], []                            # B26: predicted flowering probability vs true label over the eval set
    lfmc_p_abl, flower_p_abl, myco_p_abl = [], [], []      # B57/B58/B62: the same predictions with the species graph ABLATED (phylo-graph-gain deltas)
    flower_fid = []                                        # B27: |flowering(env-only) - flowering(env+real photo)| — imagined-vs-real fidelity
    def add(key, s, n):
        a = acc.setdefault(key, [0.0, 0.0]); a[0] += float(s); a[1] += n
    # trait macro-F1 needs the full pred/target/observed vectors gathered per preset
    trc = {lab: {t: ([], [], []) for t in traits} for lab in ("photo_env", "photo", "loo")}
    RK = 10
    community_cap = 6 * batch                              # recall@k is O(K*classes); cap to the first few batches for speed

    import os
    sdist = None                                           # local species-distribution ground truth (KL benchmarks)
    _dp = os.path.join(os.path.dirname(__file__), "..", "data", "deepcal", "gbif_species_dist.npz")
    if os.path.exists(_dp) and hasattr(source, "gbifID"):
        _z = np.load(_dp); sdist = {"m": {int(g): i for i, g in enumerate(_z["gbifID"])}, "z": _z}

    def topk_hit(logits, target, k):
        return (logits.topk(k, -1).indices == target[:, None]).any(-1).float().sum().item()
    def fam_hit(logits, target):
        return (fam[logits.argmax(-1)] == fam[target]).float().sum().item()
    def cos_sum(pred, tgt):
        return F.cosine_similarity(pred.float(), tgt.float(), dim=-1).sum().item()
    # Anomaly (mean-centered) cosine for reconstruction benchmarks. Raw cosine has a GAMEABLE floor: because a
    # modality's embeddings share a positive mean direction (measured floors: soil 0.87, clay 0.84, topo/chm 0.93-0.95,
    # hydro 0.69, vision 0.35), a constant mean prediction already scores near 1. Subtracting the modality's TRAIN mean
    # measures only the observation-specific deviation, so a mean prediction scores ~0 and the metric has honest range.
    _tr = source.train
    def _train_mean(v):
        if v in ("vision_dino", "vision_bio"):
            X = (source.dino if v == "vision_dino" else source.bio)[_tr]
        elif v == "phylo":
            X = source.phylo[source.cls[_tr]]
        elif v in getattr(source, "extra", {}):
            vals, have, _ = source.extra[v]; m = have[_tr].bool(); X = vals[_tr][m]
        else:
            return None
        if not len(X): return None
        acc = torch.zeros(X.shape[1], dtype=torch.float32, device=X.device)   # chunked sum: never materialize the full fp32 gather (OOM-safe on 24GB under temporal/phylo holdout)
        for i in range(0, len(X), 50000): acc += X[i:i + 50000].float().sum(0)
        return acc / len(X)
    CMU = {v: _train_mean(v) for v in ("vision_dino", "vision_bio", "phylo", "clay", "soil", "climate", "hydro", "naip_rgb")}
    def ccos_sum(pred, tgt, v):                            # mean-centered cosine (anomaly cosine); falls back to raw if no mean
        mu = CMU.get(v)
        if mu is None: return cos_sum(pred, tgt)
        return F.cosine_similarity(pred.float() - mu, tgt.float() - mu, dim=-1).sum().item()
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
            if hasattr(source, "flower"):                  # B26 phenology: predict flowering from env-conditioned U (per-obs)
                fv = source.flower_valid[idx].bool()
                if fv.any():
                    prb = infer(U, ["flower"])["flower"]        # env-only flowering probability (full batch, reused below)
                    pr = prb[fv]; tr = source.flower[idx][fv]
                    flower_p.append(pr.detach().cpu()); flower_t.append(tr.detach().cpu())
                    if getattr(model, "species_graph", None) is not None:   # B57: same prediction, species graph ablated
                        model._ablate_species = True
                        flower_p_abl.append(infer(U, ["flower"])["flower"][fv].detach().cpu())
                        model._ablate_species = False
                    if vision:                                  # B27 phenology fidelity: does imagined vision carry the same flowering signal as real vision?
                        vm = fv & observed.get("vision_dino", torch.zeros(B, dtype=torch.bool, device=device))
                        if vm.any():
                            pvis = infer(U + vision, ["flower"])["flower"]   # flowering from U + REAL photo
                            flower_fid.append((prb[vm] - pvis[vm]).abs().detach().cpu())
            if fam is not None:
                add("B6_family_from_env", fam_hit(L_U, tid), B)
                rank = (L_U.argsort(-1, descending=True) == tid[:, None]).float().argmax(-1)
                add("B23_species_calibration_mrr", (1.0 / (rank.float() + 1)).sum().item(), B)   # A14 calibration
                if hasattr(source, "lfmc"):                # B34 ecophysiology: predict species LFMC from environment
                    lv = source.lfmc_valid[tid]
                    if lv.any():
                        pr = infer(U, ["lfmc"])["lfmc"][lv]; tr = source.lfmc[tid][lv]
                        lfmc_p.append(pr.detach().cpu()); lfmc_t.append(tr.detach().cpu())
                        if getattr(model, "species_graph", None) is not None:   # B58: same LFMC prediction, species graph ablated
                            model._ablate_species = True
                            lfmc_p_abl.append(infer(U, ["lfmc"])["lfmc"][lv].detach().cpu())
                            model._ablate_species = False
                if hasattr(source, "myco"):                # B42 symbiosis: predict a plant's mycorrhizal type from env
                    mv = source.myco_valid[tid].bool()
                    if mv.any():
                        mp = infer(U, ["myco"])["myco"].argmax(-1)[mv]; mt = source.myco[tid][mv]
                        myco_p.append(mp.detach().cpu()); myco_t.append(mt.detach().cpu())
                        if getattr(model, "species_graph", None) is not None:   # B62: same mycorrhiza prediction, species graph ablated
                            model._ablate_species = True
                            myco_p_abl.append(infer(U, ["myco"])["myco"].argmax(-1)[mv].detach().cpu())
                            model._ablate_species = False
                if sdist is not None:                     # species-distribution KL vs local community, 3 scales
                    L_comm = infer(U, ["community"])["community"]   # dedicated community head (falls back to identity posterior if absent)
                    pm = torch.softmax(L_comm, -1).detach().cpu().numpy(); gids = source.gbifID[idx.detach().cpu().numpy()]
                    # KL SKILL SCORE vs the uniform-community reference: 1 - KL(p‖q)/KL(p‖uniform), clipped [0,1].
                    # A proper score alone (exp(-KL)) has a high, gameable floor — a no-information UNIFORM prediction
                    # scores 0.83-0.95 because local abundances are only mildly skewed. The skill score pins no-skill
                    # (uniform) to 0 and perfect to 1, so broadness can't win and the metric has full dynamic range.
                    for sc, key in (("3km", "B39_species_dist_3km_skill"), ("300m", "B40_species_dist_300m_skill"), ("30m", "B29_species_dist_30m_skill")):
                        ix, fq = sdist["z"]["idx_" + sc], sdist["z"]["frq_" + sc]; s_sk = 0.0; nk = 0
                        for b, g in enumerate(gids):
                            r = sdist["m"].get(int(g))
                            if r is None: continue
                            sp = ix[r]; msk = sp >= 0
                            if msk.sum() < 2: continue
                            p = fq[r][msk]; p = p / p.sum()
                            q = pm[b, sp[msk]] + 1e-9; q = q / q.sum()
                            kl_m = float(np.sum(p * np.log(p / q)))
                            kl_u = float(np.log(msk.sum()) + np.sum(p * np.log(p + 1e-12)))   # KL(p‖uniform) = log k - H(p)
                            if kl_u < 1e-6: continue                                          # p ~ uniform here: no discriminative signal
                            s_sk += max(0.0, 1.0 - kl_m / kl_u); nk += 1
                        if nk: add(key, s_sk, nk)
        if vision:
            tg = ["identity"] + (["phylo"] if "phylo" in have else []) + traits
            P = infer(U + vision, tg)                     # U + photo, decode species/phylo/traits from one encode
            add("B2_species_from_photo_top1", topk_hit(P["identity"], tid, 1), B)
            add("B3_species_from_photo_top5", topk_hit(P["identity"], tid, 5), B)
            if "phylo" in have:
                add("B9_phylo_from_photo_cos", ccos_sum(P["phylo"], values["phylo"], "phylo"), B)
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
                # B56 ablation-delta (rule 27 / phylo families): family-from-phylo WITH minus WITHOUT the species graph
                # refinement — isolates the phylogenomic contribution.
                if getattr(model, "species_graph", None) is not None:
                    model._ablate_species = True
                    add("_B7_ablated", fam_hit(infer(["phylo"], ["identity"])["identity"], tid), B)
                    model._ablate_species = False

        # ---- trait leave-one-out (each trait from all other observed variables) ----
        for t in traits:
            pl = infer([n for n in names if n != t], [t])[t]
            a, b, o = trc["loo"][t]; a.append(pl.argmax(-1).cpu()); b.append(values[t].cpu()); o.append(observed[t].cpu())

        # ---- ground-vision: imagine / leave-one-out / from aerial ----
        if "vision_dino" in have:
            nonvis = [n for n in obs if n not in ("vision_dino", "vision_bio")]
            add("B13_imagine_vision_cos", ccos_sum(infer(nonvis, ["vision_dino"])["vision_dino"], values["vision_dino"], "vision_dino"), B)
            add("B14_vision_leave_one_out_cos", ccos_sum(infer([n for n in names if n != "vision_dino"], ["vision_dino"])["vision_dino"], values["vision_dino"], "vision_dino"), B)
            if naip:
                add("B15_vision_from_aerial_cos", ccos_sum(infer(naip, ["vision_dino"])["vision_dino"], values["vision_dino"], "vision_dino"), B)

        # ---- dense environmental field: reconstruct each modality from the rest of U (measure-everything) ----
        for key, tv in (("B16_infer_clay_cos", "clay"), ("B17_infer_soil_cos", "soil"), ("B18_infer_climate_cos", "climate"),
                        ("B43_infer_hydro_cos", "hydro")):
            if tv in have:
                add(key, ccos_sum(infer([n for n in U if n != tv], [tv])[tv], values[tv], tv), B)
        if naip:
            add("B19_infer_aerial_cos", ccos_sum(infer([n for n in U if n not in naip], ["naip_rgb"])["naip_rgb"], values["naip_rgb"], "naip_rgb"), B)

        # ---- B28 flowering peak month (phenology seasonality): sweep the time coordinate over 12 months, condition on species, rank by predicted flowering ----
        if getattr(source, "species_peak_month", None) is not None and getattr(source, "time_axis", False) and c0 < community_cap:
            pk = source.species_peak_month[tid]; pv = pk >= 0
            if pv.any():
                mt = torch.as_tensor(source.month_tnorm, device=device)
                def _with_time(t):
                    c = coords.clone(); c[:, 3] = t; return c
                P = torch.stack([model.infer(values, ["identity"], ["flower"],
                                             model.context(_with_time(mt[mm]), nbr_coords, mani, nbrv), observed)["flower"]
                                 for mm in range(12)], 1)                          # [B,12] predicted flowering per month
                rank = (P.argsort(1, descending=True) == pk[:, None]).float().argmax(1)   # rank of the true peak month
                add("B28_flowering_peak_month_mrr", (1.0 / (rank[pv].float() + 1)).sum().item(), int(pv.sum()))

        # ---- community / companions (neighbor species are the eval TARGET only; never fed as input -> leak-safe) ----
        if fam is not None and c0 < community_cap and hasattr(source, "neighbors"):
            tset = torch.zeros(B, L_blank.shape[1], dtype=torch.bool, device=device)
            tset.scatter_(1, source.cls[source.neighbors[idx]], True); tset.scatter_(1, tid[:, None], True)
            if U:
                add("B20_community_from_env_recall", recall_sum(infer(U, ["community"])["community"], tset), B)
                if getattr(model, "species_graph", None) is not None:   # B60: env->community recall with the species graph ablated
                    model._ablate_species = True
                    add("_B20_ablated", recall_sum(infer(U, ["community"])["community"], tset), B)
                    model._ablate_species = False
            add("B21_community_from_species_recall", recall_sum(infer(["identity"], ["community"])["community"], tset), B)
            add("B22_companions_recall", recall_sum(infer(["identity"] + U, ["community"])["community"], tset), B)

        # ---- plant-pollinator interactions (GloBI): the pollinator distribution is the TARGET only (leak-safe) ----
        if hasattr(source, "poll_idx") and getattr(model, "poll_head", None) is not None and c0 < community_cap:
            c = source.cls[idx]; vi = source.poll_valid[c]; nv = int(vi.sum())
            if nv:
                pidx = source.poll_idx[c].clamp(0, source.n_pollinators - 1); pfrq = source.poll_frq[c]
                tset = torch.zeros(B, source.n_pollinators, dtype=torch.bool, device=device).scatter_(1, pidx, pfrq > 0)
                Lp = infer(["identity"] + U, ["pollinator"])["pollinator"]           # plant identity + env -> pollinators
                add("B41_pollinator_from_species_recall", recall_sum(Lp[vi], tset[vi]), nv)
                add("B51_pollinator_from_env_recall", recall_sum(infer(U, ["pollinator"])["pollinator"][vi], tset[vi]), nv)
                if getattr(model, "species_graph", None) is not None:   # B59: env->pollinator recall with the species graph ablated
                    model._ablate_species = True
                    add("_B51_ablated", recall_sum(infer(U, ["pollinator"])["pollinator"][vi], tset[vi]), nv)
                    model._ablate_species = False
                if vision:
                    add("B52_pollinator_from_photo_recall", recall_sum(infer(U + vision, ["pollinator"])["pollinator"][vi], tset[vi]), nv)
                top_true = pidx[torch.arange(B, device=device), pfrq.argmax(1)]      # most-frequent pollinator per plant
                rank = (Lp.argsort(-1, descending=True) == top_true[:, None]).float().argmax(-1)
                add("B53_pollinator_calibration_mrr", (1.0 / (rank.float() + 1))[vi].sum().item(), nv)
                q = torch.softmax(Lp, -1).gather(1, pidx); kl = torch.where(pfrq > 0, pfrq * (torch.log(pfrq + 1e-9) - torch.log(q + 1e-9)), torch.zeros_like(pfrq)).sum(1)
                add("B54_pollinator_dist_kl", torch.exp(-kl)[vi].sum().item(), nv)   # exp(-KL) of true pollinator freq vs predicted
                # B55 cross-tree phylogenomic interaction induction (rule 27): can the model predict a plant's pollinators
                # from its RELATIVES' pollinators? Target = pollinators observed for the plant's phylogenetic neighbors.
                if hasattr(source, "neighbors"):
                    nc = source.cls[source.neighbors[idx]]                       # [B,K] neighbor plant classes
                    npi = source.poll_idx[nc].reshape(B, -1).clamp(0, source.n_pollinators - 1)   # neighbors' pollinator ids
                    nfq = (source.poll_frq[nc] > 0).reshape(B, -1)               # valid (freq>0) mask
                    ntset = torch.zeros(B, source.n_pollinators, dtype=torch.bool, device=device).scatter_(1, npi, nfq)
                    rv = vi & (ntset.sum(1) > 0)
                    if int(rv.sum()):
                        add("B55_pollinator_phylo_transfer_recall", recall_sum(Lp[rv], ntset[rv]), int(rv.sum()))

        # ---- forecasting (temporal holdout only): predict the held-out FUTURE environment ----
        if holdout == "temporal" and "climate" in have:
            add("B25_forecast_climate_cos", ccos_sum(infer([n for n in U if n != "climate"], ["climate"])["climate"], values["climate"], "climate"), B)
            if "vision_dino" in have:                     # B31: forecast held-out-future ground vision (appearance/phenology) from the environment
                add("B31_forecast_vision_cos", ccos_sum(infer(U, ["vision_dino"])["vision_dino"], values["vision_dino"], "vision_dino"), B)

        # ---- B26 flowering (A7): activates once flowering labels are wired as a variable; inactive until then ----

    # ---- reduce ----
    out: Dict[str, float] = {}
    for k, (s, n) in acc.items():
        if n > 0:
            out[k] = s / n
    def _lfmc_corr(pl, tl):                                                 # log-LFMC Pearson correlation over collected preds/targets
        if not pl: return None
        p = torch.cat(pl).numpy(); t = torch.cat(tl).numpy()
        if len(p) > 2 and np.std(p) > 1e-6 and np.std(t) > 1e-6:
            return float(max(0.0, np.corrcoef(np.log(np.clip(p, 1, None)), np.log(np.clip(t, 1, None)))[0, 1]))
        return None
    def _auc(pl, tl):                                                       # rank-based ROC-AUC (Mann-Whitney) over collected preds/targets
        if not pl: return None
        p = torch.cat(pl).numpy(); t = (torch.cat(tl).numpy() > 0.5)
        npos = int(t.sum()); nneg = int((~t).sum())
        if npos == 0 or nneg == 0: return None
        r = np.argsort(np.argsort(p, kind="mergesort")).astype(np.float64) + 1.0
        return float((r[t].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    if myco_p:                                                             # B42 symbiosis: macro-F1 of predicted vs true mycorrhizal type (5 classes)
        p = torch.cat(myco_p); t = torch.cat(myco_t)
        f1 = _macro_f1(p, t, torch.ones_like(t, dtype=torch.bool), 5)
        if not (isinstance(f1, float) and np.isnan(f1)):
            out["B42_mycorrhiza_from_env"] = f1
            if myco_p_abl:                                                  # B62 = species-graph contribution to symbiosis (phylo-conserved)
                f1a = _macro_f1(torch.cat(myco_p_abl), t, torch.ones_like(t, dtype=torch.bool), 5)
                if not (isinstance(f1a, float) and np.isnan(f1a)): out["B62_mycorrhiza_phylo_graph_gain"] = max(0.0, f1 - f1a)
    _lf = _lfmc_corr(lfmc_p, lfmc_t)                                        # B34 ecophysiology; B58 = species-graph contribution to it
    if _lf is not None:
        out["B34_lfmc_from_env"] = _lf
        _lfa = _lfmc_corr(lfmc_p_abl, lfmc_t)
        if _lfa is not None: out["B58_lfmc_phylo_graph_gain"] = max(0.0, _lf - _lfa)
    _fa = _auc(flower_p, flower_t)                                          # B26 phenology; B57 = species-graph contribution to it
    if _fa is not None:
        out["B26_flowering_auc"] = _fa
        _faa = _auc(flower_p_abl, flower_t)
        if _faa is not None: out["B57_flowering_phylo_graph_gain"] = max(0.0, _fa - _faa)
    if flower_fid:                                                          # B27: 1 - mean|imagined-vision flowering - real-vision flowering|
        out["B27_flowering_fidelity"] = float(1.0 - torch.cat(flower_fid).mean())
    if "B7_family_from_phylo" in out and "_B7_ablated" in out:              # B56: phylogenomic-graph contribution (rule 27 ablation)
        out["B56_family_phylo_graph_gain"] = max(0.0, out["B7_family_from_phylo"] - out["_B7_ablated"])
    out.pop("_B7_ablated", None)
    if "B51_pollinator_from_env_recall" in out and "_B51_ablated" in out:   # B59: species-graph contribution to interaction prediction
        out["B59_pollinator_phylo_graph_gain"] = max(0.0, out["B51_pollinator_from_env_recall"] - out["_B51_ablated"])
    out.pop("_B51_ablated", None)
    if "B20_community_from_env_recall" in out and "_B20_ablated" in out:     # B60: species-graph contribution to community/niche prediction
        out["B60_community_phylo_graph_gain"] = max(0.0, out["B20_community_from_env_recall"] - out["_B20_ablated"])
    out.pop("_B20_ablated", None)
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


def is_diagnostic(k: str) -> bool:
    """A DERIVED difference benchmark (ablation-delta / information-gain): B24_geo_information_gain = B2-B1,
    B56-B60_*_phylo_graph_gain = capability WITH minus WITHOUT the species graph. These isolate a MECHANISM's
    contribution and live on a compressed 0-0.3 scale. They are reported as diagnostics but EXCLUDED from the net:
    folding a difference into the harmonic mean double-counts its own constituents and, being small (a good phylo
    gain is ~0.05, an untrained one is 0.0), it dominates/nukes the harmonic mean -> the north star would be pinned
    near 0 regardless of the model's actual capabilities."""
    return k.endswith("_gain")


_GAIN_SCALE = 0.1   # logistic scale for ablation-delta ("_gain") benchmarks in the net (see _net_value)


def _net_value(k: str, v: float) -> float:
    """The safe [0,1) contribution of benchmark ``k`` (normalized score ``v``) to the harmonic net. 100% of the
    benchmarks are included -- NOTHING is excluded (rule: every benchmark exists to be measured AND optimized). A
    plain capability metric is already in [0,1] (floored at _SCORE_FLOOR so a genuine ~0 does not nuke the mean). An
    ablation-delta ("_gain", e.g. B56 family-graph-gain, B24 geo-information-gain) is unbounded/ can sit near 0, so a
    logistic squash maps it to (0,1): it NEVER exceeds 1.0, NEVER forms a below-0 well, and is MONOTONICALLY beneficial
    to drive up -- optimizing the net therefore optimizes every benchmark, deltas included, even where signal repeats."""
    if is_diagnostic(k):
        return 1.0 / (1.0 + math.exp(-v / _GAIN_SCALE))     # v>=0 -> [0.5,1); delta 0 (no gain) = neutral 0.5
    return max(v, _SCORE_FLOOR)


def net_score(raw: Dict[str, float]) -> float:
    """North star = HARMONIC mean (power mean p=-1) of ALL active benchmarks -- capabilities AND ablation-deltas
    (deltas safely renormalized by _net_value so their inclusion can only help, never nuke). Harmonic so lifting the
    WEAKEST benchmark helps most and none can be sacrificed for another; 100% inclusion so a champion must carry the
    entire suite, not a favoured subset."""
    vals = [_net_value(k, v) for k, v in normalized(raw).items()]
    if not vals:
        return 0.0
    return float(len(vals) / sum(1.0 / v for v in vals))


def arithmetic_net(raw: Dict[str, float]) -> float:
    """Arithmetic mean of the active CAPABILITY scores -- reported alongside the harmonic north star for legibility
    (it moves when any benchmark improves, whereas the harmonic mean is dominated by the current weakest)."""
    vals = [v for k, v in normalized(raw).items() if not is_diagnostic(k)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def format_benchmarks(raw: Dict[str, float]) -> str:
    """Render every benchmark's [0,1] score (weakest first, so the binding constraint is on top), plus both the
    harmonic-mean north star and the arithmetic mean over the active benchmarks."""
    normed = normalized(raw)
    caps = {k: v for k, v in normed.items() if not is_diagnostic(k)}
    diags = {k: v for k, v in normed.items() if is_diagnostic(k)}
    lines = ["benchmark                             score"]
    for k in sorted(caps, key=lambda k: caps[k]):              # weakest-first: the harmonic mean's binding benchmarks lead
        lines.append(f"  {k:<34} {caps[k]:6.3f}")
    defined_caps = [b for b in BENCHMARKS if not is_diagnostic(b)]
    lines.append(f"NET SCORE (harmonic mean of ALL {len(normed)}/{len(BENCHMARKS)} active benchmarks, 100% incl. renormalized deltas): {net_score(raw):.4f}")
    lines.append(f"  (arithmetic mean: {arithmetic_net(raw):.4f})")
    inactive = [b for b in defined_caps if b not in caps]      # declared but not produced by THIS eval (e.g. B25/B31 need a temporal-holdout run) -- named, never silently dropped
    if inactive:
        lines.append(f"INACTIVE ({len(inactive)} declared capabilities not produced by this eval run):")
        for b in inactive:
            lines.append(f"  {b}")
    if diags:                                                  # ablation-delta / information-gain: raw here, logistic-renormalized INTO the net (_net_value)
        lines.append("ablation-delta / information-gain benchmarks (raw; logistic-renormalized into the net, never excluded):")
        for k in sorted(diags, key=lambda k: -diags[k]):
            lines.append(f"  {k:<34} {diags[k]:6.3f}  (net contrib {1.0 / (1.0 + math.exp(-diags[k] / _GAIN_SCALE)):.3f})")
    return "\n".join(lines)
