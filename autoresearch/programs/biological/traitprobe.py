"""Trait-axis comparison harness for the biological encoder (flag-gated, additive; default probe path untouched).

THE QUESTION this round answers (Ensue tag=biological, LOOP-biological-vision-seed follow-up):
the vision reseed proved the phylo species-graph *works* (bio_gain +0.3033 on family vs +0.0055 text) but on
FAMILY the text seed still wins in ABSOLUTE terms (0.90 > 0.70) -- family is exactly what the phylo-derived
text prior already saturates. Vision is FOR the axes text lacks: appearance / functional traits. This harness
scores ABSOLUTE held-out trait recovery for every seed source x graph-on/off, in ONE process, so the win
condition is directly readable:

    does  {vision or fused}+graph  BEAT  text-seed+graph  in ABSOLUTE trait score, above the +/-0.008 floor,
    across seeds?  If yes, the graph+vision path adds trait biology the champion text path cannot reach.

Seed sources (each L2-normed per-vector before the graph probes it):
    text   = E1 (2048-d BioCLIP-2.5 taxonomy-string prior; the CHAMPION seed, phylo-saturated)
    vision = mean-DINOv3 (1024-d) OR medoid-DINOv3 (--vision_medoid: denoised, robust to per-obs outliers)
    fused  = vision (+) BioCLIP-image (768-d), each L2-normed  (appearance carrying both DINO + bio-image)

For each source we report:
    seed_only    = trait-NN score of the RAW seed (no graph)              <- the seed's own ceiling
    graph        = trait-NN score of the phylo-refined REPRESENTATION      <- seed + phylo operator
    impute       = trait-NN of held-out species reconstructed from relatives (rule-25 mask)
    graph_gain   = graph - seed_only    (the phylo operator's marginal value ON THIS SEED)

The CHAMPION baseline to beat is `text graph`. A source wins iff its `graph` (or `impute`) absolute trait
score exceeds `text graph` by > floor across seeds. Reuses SpeciesGraph unchanged; edits nothing in core/.

    python -m deepearth.autoresearch.programs.biological.traitprobe \
        --cache_dir data/deepcal --trait_key multi_flower_color --steps 400 --seeds 0 1 2
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deepearth.autoresearch.programs.biological.probe import (
    load_species, load_trait, nn_trait_acc, nn_trait_ap, nn_trait_num,
)
from deepearth.encoders.biological.phylogenomic import SpeciesGraph


# --- vision seeds: mean (cached npz) and medoid (built once, cached) -----------------------------
def _mean_seeds(cache: str):
    z = np.load(Path(cache) / "derived/species_vision_seed.npz")
    return z["dino_mean"].astype(np.float32), z["bio_mean"].astype(np.float32)


def _medoid_seeds(cache: str, N: int):
    """Per-species MEDOID (the observation whose DINO is closest to all others of that species) instead of the
    MEAN. The mean of a species' DINO cloud is pulled off-manifold by outlier photos (wrong crop, non-plant,
    lighting); the medoid is a real, robust exemplar -> a cleaner appearance seed. Cached to derived/.

    We approximate the geometric medoid by the highest-mean-cosine member (argmax_i sum_j cos(x_i,x_j)),
    which for L2-normed vectors == argmax_i <x_i, mean_normed>; exact for the cosine geometry the NN metric
    uses. Both DINO and BioCLIP-image medoids are taken at the SAME chosen observation (a coherent exemplar)."""
    out = Path(cache) / "derived/species_vision_medoid.npz"
    if out.exists():
        z = np.load(out)
        return z["dino_med"].astype(np.float32), z["bio_med"].astype(np.float32)
    files = sorted(glob.glob(str(Path(cache) / "gbif_tokens/*.npz")))
    # Two streaming passes (memory-safe: never holds all 621k obs at once).
    # Pass 1: accumulate per-species SUM of L2-normed DINO -> normed centroid direction.
    z0 = np.load(files[0]); dd, bd = z0["dino"].shape[1], z0["bio"].shape[1]
    csum = np.zeros((N, dd), np.float64)
    for f in files:
        z = np.load(f)
        sl = z["species_local"].astype(np.int64)
        dn = z["dino"].astype(np.float32)
        dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-9)
        np.add.at(csum, sl, dn)
    cnorm = csum / (np.linalg.norm(csum, axis=1, keepdims=True) + 1e-9)   # [N,dd] centroid direction
    # Pass 2: for each obs, score <normed_obs, centroid>; keep the running-best obs (raw dino + bio) per species.
    best = np.full(N, -2.0, np.float64)
    Dm = np.zeros((N, dd), np.float32)
    Bm = np.zeros((N, bd), np.float32)
    for f in files:
        z = np.load(f)
        sl = z["species_local"].astype(np.int64)
        d = z["dino"].astype(np.float32)
        b = z["bio"].astype(np.float32)
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        sc = (dn * cnorm[sl]).sum(1)                      # cosine of each obs to its species centroid
        for i in range(len(sl)):
            s = sl[i]
            if sc[i] > best[s]:
                best[s] = sc[i]; Dm[s] = d[i]; Bm[s] = b[i]
    np.savez(out, dino_med=Dm, bio_med=Bm)
    return Dm, Bm


def _pooled_seeds(cache: str, N: int, mode: str, temp: float, keep: float):
    """ROUND-3 denoised per-species VISION seed. The champion vision seed is a plain MEAN of each species' DINO
    cloud -- pulled off-manifold by outlier photos (wrong crop, non-plant, bad lighting), which is what drives the
    ~1/5 vision-seed collapse. Two robust aggregators, each cached:

      mode='attn'   ATTENTION-POOLED: weight each observation by softmax(cos(obs, species-centroid)/temp), so
                    representative photos dominate and outliers are down-weighted (a soft, self-supervised pooling
                    toward the species' modal appearance; temp->0 -> medoid, temp->inf -> mean).
      mode='qfilt'  QUALITY-FILTERED trimmed mean: keep only the top `keep` fraction of observations by cosine to
                    the centroid, then mean -> hard outlier rejection.

    Two streaming passes over gbif_tokens (memory-safe, never holds all 621k obs). Both DINO and BioCLIP-image are
    pooled with the SAME per-obs weights (a coherent appearance seed)."""
    tag = f"{mode}_t{temp}_k{keep}".replace(".", "p")
    out = Path(cache) / f"derived/species_vision_pool_{tag}.npz"
    if out.exists():
        z = np.load(out); return z["dino"].astype(np.float32), z["bio"].astype(np.float32)
    import glob
    files = sorted(glob.glob(str(Path(cache) / "gbif_tokens/*.npz")))
    z0 = np.load(files[0]); dd, bd = z0["dino"].shape[1], z0["bio"].shape[1]
    # pass 1: normed centroid direction per species
    csum = np.zeros((N, dd), np.float64)
    for f in files:
        z = np.load(f); sl = z["species_local"].astype(np.int64)
        dn = z["dino"].astype(np.float32); dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-9)
        np.add.at(csum, sl, dn)
    cnorm = csum / (np.linalg.norm(csum, axis=1, keepdims=True) + 1e-9)
    if mode == "qfilt":
        # need a per-species cosine THRESHOLD (the `keep`-quantile). Pass 1b: collect cosines per species.
        from collections import defaultdict
        coslist = defaultdict(list)
        for f in files:
            z = np.load(f); sl = z["species_local"].astype(np.int64)
            d = z["dino"].astype(np.float32); dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
            sc = (dn * cnorm[sl]).sum(1)
            for i in range(len(sl)):
                coslist[int(sl[i])].append(float(sc[i]))
        thr = np.full(N, -2.0, np.float64)
        for s, cs in coslist.items():
            thr[s] = np.quantile(cs, 1.0 - keep) if len(cs) > 2 else -2.0
    # pass 2: weighted sums
    Dw = np.zeros((N, dd), np.float64); Bw = np.zeros((N, bd), np.float64); Wsum = np.zeros(N, np.float64)
    for f in files:
        z = np.load(f); sl = z["species_local"].astype(np.int64)
        d = z["dino"].astype(np.float32); b = z["bio"].astype(np.float32)
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        sc = (dn * cnorm[sl]).sum(1)                                       # cosine to species centroid
        if mode == "attn":
            w = np.exp(sc / max(temp, 1e-6))                              # softmax numerator (per-species norm below)
        elif mode == "qfilt":
            w = (sc >= thr[sl]).astype(np.float64)                        # keep top-`keep` by cosine
        else:
            raise ValueError(mode)
        np.add.at(Dw, sl, d * w[:, None]); np.add.at(Bw, sl, b * w[:, None]); np.add.at(Wsum, sl, w)
    Wsum = np.maximum(Wsum, 1e-9)
    D = (Dw / Wsum[:, None]).astype(np.float32); B = (Bw / Wsum[:, None]).astype(np.float32)
    np.savez(out, dino=D, bio=B); return D, B


def build_seed(source: str, cache: str, E1: torch.Tensor, medoid: bool, dev,
               pool: str = None, temp: float = 0.05, keep: float = 0.6):
    """Return an [N, dim] seed tensor for the requested source, each per-vector L2-normed for a fair
    cosine-NN comparison (text seed too, so no source gets a norm advantage)."""
    if source == "text":
        return F.normalize(E1.to(dev), dim=-1)
    if pool in ("attn", "qfilt"):
        D, B = _pooled_seeds(cache, E1.shape[0], pool, temp, keep)        # ROUND-3 denoised aggregation
    elif medoid:
        D, B = _medoid_seeds(cache, E1.shape[0])
    else:
        D, B = _mean_seeds(cache)
    D = torch.tensor(D).to(dev); B = torch.tensor(B).to(dev)
    if source == "vision":
        return F.normalize(D, dim=-1)
    if source == "fused":
        return torch.cat([F.normalize(D, dim=-1), F.normalize(B, dim=-1)], dim=-1)
    if source == "routed":
        # ROUND-1 AGGREGATE ROUTER (rule 26/9): a SINGLE seed carrying BOTH the phylo-saturated text prior AND
        # the appearance-carrying vision prior, each L2-normed, concatenated -> ONE SpeciesGraph. The probe
        # (Linear+LayerNorm) learns which sub-block to weight per output dimension, so the graph can route text
        # for family/plant-type axes and vision for the 9 appearance/size/vigor win-axes WITHOUT us hand-picking
        # per axis. The JOINT question: does one routed graph capture the SUM of the per-axis vision wins, or do
        # the signals interfere? Text block = E1 (2048-d), vision block = mean-DINO (1024-d). N.B. this is the PR
        # artifact -- one graph, all axes measured off it at once.
        return torch.cat([F.normalize(E1.to(dev), dim=-1), F.normalize(D, dim=-1)], dim=-1)
    raise ValueError(source)


def train_graph(seed: torch.Tensor, tree, tip_row, test, dev, d_model, steps, mask_frac, lr, impute_steps):
    """Phylo-refine one seed via the latent-clade operator (rule-25 mask-reconstruct). Identical protocol for
    every seed source. `impute_steps`>0 uses that many extra masked-imputation refinement passes at eval for
    the held-out species (push imputation-from-relatives, since pure-vision single-pass imputation was net-neg)."""
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    opt = torch.optim.Adam(graph.parameters(), lr=lr)
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & (~test)
        if not mask.any():
            continue
        refined = graph(mask=mask)
        target = graph._seed().detach()
        loss = (1.0 - F.cosine_similarity(refined[mask], target[mask], dim=-1)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):   # extra imputation refinement rounds for the held-out set
            imp = graph(mask=test)
    return s, rep, imp


def train_graph_trait_supervised(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                                 impute_steps, kind, Y, obs, nclass, dual_head=False, dual_seed=None):
    """ROUND-2 push for num_lep_support (and any axis): SEED-FROZEN, TRAIT-SUPERVISED graph. The operator is
    trained to reconstruct a MASKED species' TRAIT from its phylo relatives (rule 25+the trait-supervised recipe
    that unlocked cat_form/family): freeze the seed, mask a fraction of TRAIN species, and drive a small readout
    off the refined masked embedding toward that species' true trait (regression MSE for num_, CE for cat_).
    This aligns the objective with the held-out imputation task and pushes the axis the seed lacks.

    `dual_head` (interaction-aware): num_lep_support is lepidoptera HOST support -- a plant x insect interaction
    axis where vision(plant appearance) and phylo may COMPOUND. With dual_head we co-train a SECOND readout off a
    detached-vision projection of the SAME refined embedding, so the operator must make the refined rep explain
    the trait through BOTH a phylo-propagated channel and an appearance channel -> an interaction-aware objective
    that a single reconstruct-identity loss cannot express. `dual_seed` = the [N,vdim] vision block for that head."""
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    train_obs = obs & (~test)
    if kind == "num":
        head = torch.nn.Linear(d_model, 1).to(dev)
        yt = Y.float()
    else:
        head = torch.nn.Linear(d_model, int(nclass)).to(dev)
        yt = Y.long()
    vhead = None
    if dual_head and dual_seed is not None:
        vproj = F.normalize(dual_seed, dim=-1)
        vhead = torch.nn.Sequential(torch.nn.Linear(vproj.shape[1], d_model), torch.nn.GELU(),
                                    torch.nn.Linear(d_model, 1 if kind == "num" else int(nclass))).to(dev)
    params = list(graph.parameters()) + list(head.parameters()) + (list(vhead.parameters()) if vhead else [])
    opt = torch.optim.Adam(params, lr=lr)
    for p in graph.parameters():
        pass  # seed frozen via detached target below; operator params train
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & train_obs
        if not mask.any():
            continue
        refined = graph(mask=mask)                          # masked species reconstructed from relatives
        pred = head(refined[mask])
        if kind == "num":
            loss = F.mse_loss(pred.squeeze(-1), yt[mask])
        else:
            loss = F.cross_entropy(pred, yt[mask])
        if vhead is not None:                               # interaction-aware 2nd channel (appearance)
            vpred = vhead(vproj[mask])
            if kind == "num":
                loss = loss + F.mse_loss(vpred.squeeze(-1), yt[mask])
            else:
                loss = loss + F.cross_entropy(vpred, yt[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):
            imp = graph(mask=test)
    return s, rep, imp


# --- helpers used by both single-axis main() and the multi-axis aggregate router ------------------
def _scorer_for(cache, gidx, key, dev):
    """Return (scorer_fn, kind, metric_name) for one trait key. scorer(emb, test) -> absolute NN score."""
    kind, Y, obs, _ = load_trait(cache, gidx, key, dev)
    if kind == "cat":
        fn = lambda emb, test: nn_trait_acc(emb, Y, obs, test)
    elif kind == "num":
        fn = lambda emb, test: nn_trait_num(emb, Y, obs, test)
    else:
        fn = lambda emb, test: nn_trait_ap(emb, Y, obs, test)
    metric = {"cat": "acc", "num": "spearman"}.get(kind, "AP")
    return fn, kind, metric


# The 9 established vision-win axes (Ensue LOOP-biological-vision-*): axes where a per-species DINO seed makes
# the phylo graph ADDITIVE and beats the champion text seed. The aggregate router must recover the SUM of these.
WIN_AXES = ["num_lep_support", "cat_growth_rate", "num_width_max", "cat_sun", "num_soil_ph_max",
            "cat_form", "cat_plant_type", "cat_soil_drainage", "cat_ease_of_care"]


def run_multi_axis(a, dev):
    """ROUND-1 AGGREGATE ROUTER. Train ONE graph per (source, seed) and score EVERY win-axis off that SAME
    refined representation, then report the JOINT lift: mean over axes of (source_graph - text_graph). This
    answers whether a single routed vision+text graph captures the SUM of the per-axis vision wins measured
    individually, or whether packing all axes into one representation makes them interfere. Fair control: the
    `text` source runs the identical protocol; the reported aggregate is vs the text graph on the same axes.
    Per-process, multi-seed; changes nothing in core/. """
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    axes = a.axes if a.axes else WIN_AXES
    scorers = {k: _scorer_for(a.cache_dir, gidx, k, dev) for k in axes}
    # per (source, axis, metric) absolute scores across seeds
    agg = {s: {k: {"seed": [], "graph": [], "impute": []} for k in axes} for s in a.sources}
    t0 = time.time()
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            s, rep, imp = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                      a.steps, a.mask_frac, a.lr, a.impute_steps)
            for k in axes:
                fn = scorers[k][0]
                agg[src][k]["seed"].append(fn(s, test))
                agg[src][k]["graph"].append(fn(rep, test))
                agg[src][k]["impute"].append(fn(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    seed_word = "medoid" if a.vision_medoid else "mean"
    print(f"=== BIOLOGICAL AGGREGATE ROUTER | sources={a.sources} vision={seed_word} seeds={a.seeds} "
          f"holdout={a.holdout} N={N} axes={len(axes)} ===")
    # per-axis table: for each source, graph abs and delta vs text-graph on that axis
    txt_graph = {k: m(agg["text"][k]["graph"])[0] for k in axes} if "text" in agg else {}
    for src in a.sources:
        if src == "text":
            continue
        print(f"\n  --- source={src} vs text-graph (per axis) ---")
        print(f"  {'axis':18s} | {'text_graph':>10s} | {src+'_graph':>13s} | {'d_graph':>9s} | "
              f"{src+'_impute':>13s} | {'d_impute':>9s}")
        wins_g, wins_i = [], []
        for k in axes:
            tg = txt_graph.get(k, float("nan"))
            ti = m(agg["text"][k]["impute"])[0] if "text" in agg else float("nan")
            gm = m(agg[src][k]["graph"])[0]; im = m(agg[src][k]["impute"])[0]
            dg = gm - tg; di = im - ti
            wins_g.append(dg); wins_i.append(di)
            fg = " WIN" if dg > 0.008 else ""
            print(f"  {k:18s} | {tg:10.4f} | {gm:13.4f} | {dg:+9.4f} | {im:13.4f} | {di:+9.4f}{fg}")
        agg_g = float(np.mean(wins_g)); agg_i = float(np.mean(wins_i))
        nwin_g = sum(1 for d in wins_g if d > 0.008)
        print(f"  {'AGGREGATE(mean d)':18s} | {'':>10s} | {'':>13s} | {agg_g:+9.4f} | {'':>13s} | {agg_i:+9.4f}"
              f"   ({nwin_g}/{len(axes)} axes WIN graph @>+0.008)")

    # machine-readable summary for Ensue
    summary = {"mode": "aggregate_router", "vision": seed_word, "seeds": a.seeds, "axes": axes}
    for src in a.sources:
        if src == "text":
            continue
        pa = {}
        for k in axes:
            tg = txt_graph.get(k, float("nan"))
            ti = m(agg["text"][k]["impute"])[0] if "text" in agg else float("nan")
            gm, gs = m(agg[src][k]["graph"]); im, iss = m(agg[src][k]["impute"])
            pa[k] = {"text_graph": round(tg, 4), "graph": round(gm, 4), "graph_std": round(gs, 4),
                     "d_graph": round(gm - tg, 4), "impute": round(im, 4), "d_impute": round(im - ti, 4)}
        summary[src] = {"per_axis": pa,
                        "agg_d_graph": round(float(np.mean([pa[k]["d_graph"] for k in axes])), 4),
                        "agg_d_impute": round(float(np.mean([pa[k]["d_impute"] for k in axes])), 4),
                        "n_win_graph": sum(1 for k in axes if pa[k]["d_graph"] > 0.008)}
    import json
    print("[aggregate_router] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)} seeds x {len(a.sources)} sources x {len(axes)} axes in {time.time()-t0:.1f}s")
    return summary


def run_trait_supervised(a, dev):
    """ROUND-2: SEED-FROZEN TRAIT-SUPERVISED graph on ONE axis (default num_lep_support). Compares, per source,
    the unsupervised cosine-recon graph (the Round-1 protocol) vs the trait-supervised graph (+/- the dual-signal
    interaction head), against the text-seed champion, on held-out NN imputation. Answers: how high can the single
    best axis go when the objective is aligned with the trait and an interaction-aware appearance channel is added?"""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    key = a.trait_key
    fn, kind, metric = _scorer_for(a.cache_dir, gidx, key, dev)
    _, Y, obs, nclass = load_trait(a.cache_dir, gidx, key, dev)
    variants = a.variants  # e.g. ["unsup","sup","sup_dual"]
    agg = {src: {v: {"seed": [], "graph": [], "impute": []} for v in variants} for src in a.sources}
    t0 = time.time()
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        # vision block for the dual head (mean-DINO), aligned to N
        Dv, _ = _mean_seeds(a.cache_dir)
        Dv = torch.tensor(Dv).to(dev)
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            for v in variants:
                if v == "unsup":
                    s, rep, imp = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                              a.steps, a.mask_frac, a.lr, a.impute_steps)
                else:
                    s, rep, imp = train_graph_trait_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, kind, Y, obs, nclass,
                        dual_head=(v == "sup_dual"), dual_seed=(Dv if v == "sup_dual" else None))
                agg[src][v]["seed"].append(fn(s, test))
                agg[src][v]["graph"].append(fn(rep, test))
                agg[src][v]["impute"].append(fn(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    print(f"=== BIOLOGICAL TRAIT-SUPERVISED | trait={key} metric={metric} sources={a.sources} "
          f"variants={variants} seeds={a.seeds} holdout={a.holdout} ===")
    txt_ref = m(agg["text"]["unsup"]["graph"])[0] if "text" in agg else float("nan")
    txt_ref_imp = m(agg["text"]["unsup"]["impute"])[0] if "text" in agg else float("nan")
    print(f"  champion text-seed unsup graph={txt_ref:.4f}  impute={txt_ref_imp:.4f}")
    print(f"  {'source':7s} {'variant':9s} | {'seed':>16s} | {'GRAPH':>16s} | {'impute':>16s} | d_graph_vs_txt | d_imp_vs_txt")
    summary = {"mode": "trait_supervised", "trait": key, "metric": metric,
               "text_unsup_graph": round(txt_ref, 4), "text_unsup_impute": round(txt_ref_imp, 4), "rows": []}
    for src in a.sources:
        for v in variants:
            sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"]); im, iss = m(agg[src][v]["impute"])
            dg = gm - txt_ref; di = im - txt_ref_imp
            flag = " WIN" if dg > 0.008 else ""
            print(f"  {src:7s} {v:9s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {im:.4f}+/-{iss:.4f} | "
                  f"{dg:+.4f}       | {di:+.4f}{flag}")
            summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4),
                                    "graph": round(gm, 4), "graph_std": round(gs, 4), "impute": round(im, 4),
                                    "d_graph_vs_txt": round(dg, 4), "d_imp_vs_txt": round(di, 4)})
    import json
    print("[trait_supervised] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v in {time.time()-t0:.1f}s")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--mask_frac", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--trait_key", default="multi_flower_color")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sources", nargs="+", default=["text", "vision", "fused"])
    ap.add_argument("--vision_medoid", action="store_true", help="use per-species DINO MEDOID instead of MEAN")
    ap.add_argument("--impute_steps", type=int, default=1, help="extra masked-imputation refinement passes at eval")
    ap.add_argument("--multi_axis", action="store_true",
                    help="ROUND-1 aggregate router: one graph per source, scored on EVERY win-axis at once")
    ap.add_argument("--axes", nargs="+", default=None, help="override the win-axis list for --multi_axis")
    ap.add_argument("--trait_supervised", action="store_true",
                    help="ROUND-2: seed-frozen trait-supervised graph on --trait_key (+/- dual interaction head)")
    ap.add_argument("--variants", nargs="+", default=["unsup", "sup", "sup_dual"],
                    help="ROUND-2 variants: unsup | sup | sup_dual")
    ap.add_argument("--vision_pool", default=None, choices=[None, "attn", "qfilt"],
                    help="ROUND-3 denoised per-species DINO aggregation: attn (attention-pooled) | qfilt (quality-filtered)")
    ap.add_argument("--pool_temp", type=float, default=0.05, help="ROUND-3 attn-pool softmax temperature")
    ap.add_argument("--pool_keep", type=float, default=0.6, help="ROUND-3 qfilt keep-fraction (top cosine-to-centroid)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    if a.multi_axis:
        return run_multi_axis(a, dev)
    if a.trait_supervised:
        return run_trait_supervised(a, dev)

    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    kind, Y, obs, _ = load_trait(a.cache_dir, gidx, a.trait_key, dev)
    if kind == "cat":
        scorer = lambda emb, test: nn_trait_acc(emb, Y, obs, test)
    elif kind == "num":
        scorer = lambda emb, test: nn_trait_num(emb, Y, obs, test)
    else:
        scorer = lambda emb, test: nn_trait_ap(emb, Y, obs, test)
    N = E1.shape[0]

    # accumulate absolute scores per (source, metric) across seeds
    agg = {s: {"seed": [], "graph": [], "impute": []} for s in a.sources}
    t0 = time.time()
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            s, rep, imp = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                      a.steps, a.mask_frac, a.lr, a.impute_steps)
            agg[src]["seed"].append(scorer(s, test))
            agg[src]["graph"].append(scorer(rep, test))
            agg[src]["impute"].append(scorer(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    seed_word = "medoid" if a.vision_medoid else "mean"
    metric = {"cat": "acc", "num": "spearman"}.get(kind, "AP")
    print(f"=== BIOLOGICAL trait-compare | trait={a.trait_key} metric={metric} vision={seed_word} "
          f"seeds={a.seeds} holdout={a.holdout} N={N} ===")
    txt_graph = m(agg["text"]["graph"])[0] if "text" in agg else float("nan")
    print(f"  {'source':7s} | {'seed_only':>18s} | {'GRAPH(rep)':>18s} | {'impute':>18s} | graph_gain |  vs text-graph")
    for src in a.sources:
        sm, ss = m(agg[src]["seed"]); gm, gs = m(agg[src]["graph"]); im, iis = m(agg[src]["impute"])
        gain = gm - sm
        dvt = gm - txt_graph
        flag = "  <-- BEATS text" if (src != "text" and dvt > 0.008) else ""
        print(f"  {src:7s} | {sm:.4f} +/- {ss:.4f} | {gm:.4f} +/- {gs:.4f} | {im:.4f} +/- {iis:.4f} | "
              f"{gain:+.4f}    | {dvt:+.4f}{flag}")
    print(f"  [profile] {len(a.seeds)} seeds x {len(a.sources)} sources in {time.time()-t0:.1f}s")

    # machine-readable summary line for scoring/Ensue
    summary = {"trait": a.trait_key, "metric": metric, "vision": seed_word, "seeds": a.seeds,
               "text_graph_abs": round(txt_graph, 4)}
    for src in a.sources:
        summary[src] = {"seed": round(m(agg[src]["seed"])[0], 4),
                        "graph": round(m(agg[src]["graph"])[0], 4),
                        "graph_std": round(m(agg[src]["graph"])[1], 4),
                        "impute": round(m(agg[src]["impute"])[0], 4),
                        "graph_gain": round(m(agg[src]["graph"])[0] - m(agg[src]["seed"])[0], 4),
                        "vs_text_graph": round(m(agg[src]["graph"])[0] - txt_graph, 4)}
    import json
    print("[trait_compare] " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
