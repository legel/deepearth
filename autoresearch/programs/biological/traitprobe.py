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
from deepearth.encoders.biological.phylogenomic import SpeciesGraph, build_tree_buffers


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


def _balance_blocks(D: torch.Tensor, B: torch.Tensor, dev, energy_ratio: float = 0.95):
    """RULE-26 fused-seed stabilizer. The plain `fused` seed is cat[norm(DINO), norm(BioCLIP-img)] at EQUAL
    dimensionality/energy; under the unsupervised cosine-reconstruct objective the latent-clade operator
    diverges on it (per-axis graph std 0.22-0.27, large negative graph deltas on ~5 of 9 axes) because a
    single cosine target cannot decide which appearance block to trust and the two correlated blocks fight.

    This produces a decorrelated, variance-BALANCED single appearance seed instead of a raw concat:
      1. mean-center + PCA-whiten each block over the species population (SVD, keep components up to
         `energy_ratio` cumulative variance -> drop noise directions that destabilize reconstruction),
      2. scale each whitened block to UNIT total energy so DINO and BioCLIP-image contribute equally
         (no block dominates the cosine target purely by having larger raw norm/rank),
      3. concat -> one coherent seed. Deterministic (population stats only), so unseen species use the
         identical transform (rule-9 out-of-tree species get the same whitening)."""
    def white(X):
        Xc = X - X.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2)
        cum = torch.cumsum(var, 0) / var.sum().clamp_min(1e-12)
        k = int((cum < energy_ratio).sum().item()) + 1
        k = max(1, min(k, S.shape[0]))
        W = U[:, :k] * (Xc.shape[0] ** 0.5)          # whitened scores, unit variance per retained component
        W = W / (W.norm() + 1e-9)                     # unit TOTAL energy so both blocks weigh equally
        return W
    Dw = white(D.to(dev).float()); Bw = white(B.to(dev).float())
    return torch.cat([Dw, Bw], dim=-1)


def _balance_blocks_zca(D: torch.Tensor, B: torch.Tensor, dev, eps: float = 1e-3):
    """RULE-26 fused-seed stabilizer, FULL-RANK variant. Same goal as _balance_blocks (stop the two appearance
    blocks fighting under the unsup cosine-recon target) but WITHOUT dropping low-variance directions -- those
    dropped dirs are what mask-reconstruct/imputation needs, so PCA-whitening hurt impute. ZCA whitens
    (decorrelates + unit-variance) while keeping the FULL rank and staying in the original block basis, then
    scales each block to unit total energy so DINO and BioCLIP-image weigh equally. Deterministic population
    transform (rule-9: unseen species get the identical whitening)."""
    def zca(X):
        Xc = X - X.mean(0, keepdim=True)
        n = Xc.shape[0]
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        # symmetric inverse-sqrt of covariance via eigendecomposition (ZCA keeps original basis, full rank)
        evals, evecs = torch.linalg.eigh(cov)
        inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals.clamp_min(eps))) @ evecs.T
        W = Xc @ inv_sqrt
        W = W / (W.norm() + 1e-9)                      # unit total energy so both blocks weigh equally
        return W
    Dw = zca(D.to(dev).float()); Bw = zca(B.to(dev).float())
    return torch.cat([Dw, Bw], dim=-1)


def build_seed(source: str, cache: str, E1: torch.Tensor, medoid: bool, dev, energy_ratio: float = 0.95,
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
    if source == "fused_white":
        # RULE-26 variance-balanced, PCA-whitened fused appearance seed (stabilizes the unsup graph)
        return _balance_blocks(D, B, dev, energy_ratio=energy_ratio)
    if source == "fused_zca":
        # RULE-26 full-rank ZCA-whitened variance-balanced fused seed (keep low-var tail for imputation)
        return _balance_blocks_zca(D, B, dev)
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


def _oot_tree(cache, gidx, tip_row, keep_mask):
    """Rule-9 helper. Rebuild the latent-clade tree buffers over ONLY the species kept in (`keep_mask` True at
    their vocab row), i.e. the tree whose tips are the TRAIN species. Returns (tree_train, tip_row_train) where
    tip_row_train lists the vocab rows of the retained tips. Species NOT in keep_mask that WERE in-tree become
    genuinely OUT-OF-TREE rows of the resulting SpeciesGraph -> they carry no tree position and can only be
    imputed by the clade soft-attach cross-attention (LatentCladeAttention, the has_oot MLA-read path that the
    all-in-tree probe never exercises). This is science.md rule-9 tested for real."""
    import csv as _csv
    rows = list(_csv.DictReader(open(Path(cache) / "derived/species_index.csv")))
    labels = [rows[int(gidx[int(r)])]["tip_label"] for r in tip_row.tolist()]   # label of each currently in-tree tip
    keep = keep_mask[tip_row].tolist()                                          # keep this tip in the train tree?
    kept_pairs = [(int(r), lab) for r, lab, k in zip(tip_row.tolist(), labels, keep) if k]
    tree_tr = build_tree_buffers(str(Path(cache) / "ca_subtree.dated.nwk"), [lab for _, lab in kept_pairs])
    tip_row_tr = torch.tensor([r for r, _ in kept_pairs], dtype=torch.long)
    return tree_tr, tip_row_tr


def train_graph_oot(seed, cache, gidx, tip_row, test, dev, d_model, steps, mask_frac, lr, impute_steps,
                    sup_kind=None, sup_Y=None, sup_obs=None, sup_nclass=None,
                    resid_readout=False, clade_base=None):
    """RULE-9 OUT-OF-TREE PROJECTION (science.md rule 9, the `~` row). The standard probe leaves held-out `test`
    species AS TIPS and only masks their seed -- they still occupy a tree position and are refined by exact
    message passing. Rule-9 asks the harder question: project a species that is genuinely NOT IN THE TREE. Here we
    build the phylogeny over the TRAIN tips only; every test species becomes an out-of-tree row that must soft-
    attach to the refined clade latents via cross-attention (the LatentCladeAttention has_oot MLA read). The graph
    is trained self-supervised (rule-25 mask-reconstruct) on the in-tree TRAIN tips; test species never enter the
    tree during training or eval. Returns (seed, refined_full, oot_projection) all [N,d]; test rows of the
    projection come purely from the out-of-tree soft-attach path."""
    N = seed.shape[0]
    train = ~test
    tree_tr, tip_row_tr = _oot_tree(cache, gidx, tip_row, train)
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree_tr, tip_row=tip_row_tr,
                         species_text=seed).to(dev)
    # in-tree train tips (bool over full vocab) -- the only rows we mask/reconstruct during training
    intree = torch.zeros(N, dtype=torch.bool, device=dev); intree[tip_row_tr.to(dev)] = True
    # ITER-2: optional trait-supervised residual head. When sup_kind is set, the graph co-trains a magnitude-
    # explicit readout (predicts y - clade_mean on masked TRAIN tips) alongside the cosine-recon loss, then reads
    # the OOT-projected embedding through that head at eval -> combines the rule-9 OOT projection (which preserves
    # the seed's magnitude signal) with the residual-head clade regression (which recovers within-clade rank).
    head = None; clade_mu = None
    if sup_kind is not None:
        out_dim = 1 if sup_kind == "num" else int(sup_nclass)
        head = torch.nn.Linear(d_model, out_dim).to(dev)
        yt = sup_Y.float() if sup_kind == "num" else sup_Y.long()
        if resid_readout and sup_kind == "num" and clade_base is not None:
            clade_mu = clade_base.to(dev).float(); yt = yt - clade_mu
    params = list(graph.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    train_obs = (sup_obs & intree) if sup_kind is not None else None
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & intree            # mask only in-tree train tips
        if not mask.any():
            continue
        refined = graph(mask=mask)
        target = graph._seed().detach()
        loss = (1.0 - F.cosine_similarity(refined[mask], target[mask], dim=-1)).mean()
        if head is not None:
            hm = mask & train_obs
            if hm.any():
                pred = head(refined[hm])
                if sup_kind == "num":
                    loss = loss + F.mse_loss(pred.squeeze(-1), yt[hm])
                else:
                    loss = loss + F.cross_entropy(pred, yt[hm])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)                                             # full refine: test rows = OOT soft-attach
        oot = rep                                                          # test species already come from the OOT path
        for _ in range(max(0, impute_steps - 1)):
            oot = graph(mask=None)
        head_pred = None
        head_resid = None
        if head is not None and sup_kind == "num":
            add_mu = clade_mu if (resid_readout and clade_mu is not None) else 0.0
            head_resid = head(oot).squeeze(-1).detach()                    # WITHIN-clade residual (mean added at readout)
            head_pred = (head_resid + add_mu).detach()                     # magnitude-aware readout on OOT projection
        elif head is not None and sup_kind == "cat":
            head_pred = head(oot).argmax(-1).detach()                      # ITER-4: categorical class readout on OOT proj
    return s, rep, oot, head_pred, head_resid


def train_graph_trait_supervised(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                                 impute_steps, kind, Y, obs, nclass, dual_head=False, dual_seed=None,
                                 head_hidden=0, dual_scale=1.0, rollout_rounds=1, reveal_frac=0.25,
                                 ab_scorer=None, operator="latent-clade", phylo_dist=None,
                                 fam_id=None, resid_readout=False, clade_base=None):
    """ROUND-2 push for num_lep_support (and any axis): SEED-FROZEN, TRAIT-SUPERVISED graph. The operator is
    trained to reconstruct a MASKED species' TRAIT from its phylo relatives (rule 25+the trait-supervised recipe
    that unlocked cat_form/family): freeze the seed, mask a fraction of TRAIN species, and drive a small readout
    off the refined masked embedding toward that species' true trait (regression MSE for num_, CE for cat_).
    This aligns the objective with the held-out imputation task and pushes the axis the seed lacks.

    `dual_head` (interaction-aware): num_lep_support is lepidoptera HOST support -- a plant x insect interaction
    axis where vision(plant appearance) and phylo may COMPOUND. With dual_head we co-train a SECOND readout off a
    detached-vision projection of the SAME refined embedding, so the operator must make the refined rep explain
    the trait through BOTH a phylo-propagated channel and an appearance channel -> an interaction-aware objective
    that a single reconstruct-identity loss cannot express. `dual_seed` = the [N,vdim] vision block for that head.

    `operator` (rule 29): 'latent-clade' (default, all prior rounds) or 'ou-attention' -- the exact O(N) OU-GP
    distance-biased attention the champion uses. ou-attention needs a phylo_distance matrix (`phylo_dist`,
    built from the E1 tree-derived prior) instead of the tree buffers; this isolates the OPERATOR choice on the
    identical trait-supervised imputation task."""
    N = seed.shape[0]
    if operator == "ou-attention":
        graph = SpeciesGraph(N, d_model, operator="ou-attention", phylo_distance=phylo_dist,
                             n_heads=4, n_layers=2, species_text=seed).to(dev)
    else:
        graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                             species_text=seed).to(dev)
    train_obs = obs & (~test)
    out_dim = 1 if kind == "num" else int(nclass)
    if head_hidden > 0:   # CEILING-PUSH: deeper trait-supervised readout (2-layer GELU MLP)
        head = torch.nn.Sequential(torch.nn.Linear(d_model, head_hidden), torch.nn.GELU(),
                                   torch.nn.Linear(head_hidden, out_dim)).to(dev)
    else:
        head = torch.nn.Linear(d_model, out_dim).to(dev)
    yt = Y.float() if kind == "num" else Y.long()
    # CLADE-MEAN-RESIDUAL READOUT (resid_readout, num only): make magnitude EXPLICIT. The head predicts only the
    # WITHIN-CLADE deviation y - clade_mean; at readout we add back the held-out species' clade mean. Clade means
    # are computed from TRAIN-observed species only (per family), so no held-out leakage. This forces the graph to
    # carry the fine within-family rank (the exact signal kNN neighbor-averaging collapses) while the coarse
    # family-level magnitude comes from the phylo prior for free.
    clade_mu = None
    if resid_readout and kind == "num" and clade_base is not None:
        clade_mu = clade_base.to(dev).float()                          # caller-supplied [N] base mean (e.g. hybrid)
        yt = yt - clade_mu
    elif resid_readout and kind == "num" and fam_id is not None:
        fam = fam_id.to(dev).long()
        train_obs0 = obs & (~test)
        F_ = int(fam.max().item()) + 1
        gmean = yt[train_obs0].mean() if train_obs0.any() else yt.mean()
        ssum = torch.zeros(F_, device=dev).index_add_(0, fam[train_obs0], yt[train_obs0])
        scnt = torch.zeros(F_, device=dev).index_add_(0, fam[train_obs0], torch.ones(int(train_obs0.sum()), device=dev))
        fmean = torch.where(scnt > 0, ssum / scnt.clamp(min=1), gmean)   # per-family train mean (fallback global)
        clade_mu = fmean[fam]                                            # [N] each species' family mean
        yt = yt - clade_mu                                              # head now regresses the residual
    vhead = None
    if dual_head and dual_seed is not None:
        vproj = F.normalize(dual_seed, dim=-1)
        vh = head_hidden if head_hidden > 0 else d_model   # CEILING-PUSH: match dual-head capacity to readout
        vhead = torch.nn.Sequential(torch.nn.Linear(vproj.shape[1], vh), torch.nn.GELU(),
                                    torch.nn.Linear(vh, 1 if kind == "num" else int(nclass))).to(dev)
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
                loss = loss + dual_scale * F.mse_loss(vpred.squeeze(-1), yt[mask])
            else:
                loss = loss + dual_scale * F.cross_entropy(vpred, yt[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp_single = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):
            imp_single = graph(mask=test)
        if rollout_rounds > 1:
            imp = _rollout_impute(graph, test, rollout_rounds, reveal_frac, dev)
        else:
            imp = imp_single
        if ab_scorer is not None:                       # CLEAN same-graph A/B: isolates the readout effect (zero
            sp = ab_scorer(imp_single, test)            # training variance -- both scored on the SAME trained graph)
            ro = ab_scorer(imp, test)
            print(f"    [rollout_ab] single_pass={sp:.4f} rollout={ro:.4f} delta={ro-sp:+.4f}")
        # MAGNITUDE-AWARE READOUT (num_head_readout): the kNN impute scorer averages relatives' raw
        # values -> collapses within-clade rank (impute 0.19 << graph 0.54 on num_width_max). The trained
        # regression `head` is an MSE-fit magnitude map from the refined embedding straight to the value;
        # applying it to the imputed (mask=test) embedding is a magnitude-aware readout that keeps rank the
        # neighbor-average destroys. This isolates READOUT (same graph, same imputed embedding) vs the kNN.
        graph._head_imp = None
        graph._head_seed = None
        if kind == "num":
            add_mu = clade_mu if (resid_readout and clade_mu is not None) else 0.0
            pred_imp = head(imp).squeeze(-1) + add_mu   # [N] head prediction on IMPUTED (mask=test) embeddings
            graph._head_imp = pred_imp.detach()
            # LEAKAGE CONTROL: same head on the held-out species' OWN refined (unmasked) rep. If head-on-imputed
            # >> this, the gain genuinely rides imputation-from-relatives, not head memorization of the seed.
            pred_seed = head(rep).squeeze(-1) + add_mu
            graph._head_seed = pred_seed.detach()
    return s, rep, imp, (head, graph)


def _rollout_impute(graph, test, rounds, reveal_frac, dev):
    """READOUT LEVER (rule 10-11): autoregressive PROGRESSIVE-REVEAL imputation. Single-pass `graph(mask=test)`
    masks ALL held-out species at once, so a held-out species reconstructs only from TRAIN relatives and never
    from another (confidently imputed) held-out relative. Rollout instead reveals, each round, the most-anchored
    still-masked test species -- clear them from the mask so their seed joins the relative pool -- then re-imputes
    the rest from the enlarged neighbour set. `an observation of A updates neighbours B, C`. Confidence = a
    species' max cosine to any currently-UNMASKED species in the refined space (most tree-anchored = revealed
    first). The FINAL refined table (all test still SCORED by NN vs train, identical metric) is returned; reveals
    only change which relatives are visible during reconstruction, not the held-out train/test scoring split."""
    N = test.shape[0]
    mask = test.clone()
    n_test = int(test.sum().item())
    per_round = max(1, int(n_test * reveal_frac))
    out = graph(mask=mask).clone()                                   # frozen per-species imputed-while-masked table
    for _ in range(max(1, rounds - 1)):
        refined = graph(mask=mask)
        r = F.normalize(refined, dim=-1)
        sim = r @ r.t()
        sim.fill_diagonal_(-2.0)
        vis = ~mask                                                  # currently-visible species (train + revealed)
        anchor = torch.where(vis.unsqueeze(0), sim, torch.full_like(sim, -2.0)).max(dim=1).values  # [N]
        cand = mask.clone()                                          # only reveal currently-masked test species
        anchor = torch.where(cand, anchor, torch.full_like(anchor, -3.0))
        k = min(per_round, int(cand.sum().item()))
        if k <= 0:
            break
        top = anchor.topk(k).indices
        mask = mask.clone(); mask[top] = False                      # reveal: their seed now visible to relatives
        newimp = graph(mask=mask)                                    # re-impute the STILL-masked species anew
        out[mask] = newimp[mask]                                     # each test species keeps its imputed-while-
    #                                                                # -masked embedding (revealed ones frozen at reveal)
    return out


def train_graph_multitrait(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr, impute_steps,
                           panel, eval_key, include_eval=False):
    """NEW LEVER (rule 25/29, richer reconstruction target): SEED-FROZEN graph trained to reconstruct a PANEL of
    MANY auxiliary traits JOINTLY from relatives, then measured by held-out NN imputation on ONE frozen EVAL axis.
    Every prior trait-supervised round drove a SINGLE readout at the eval axis; here the operator must make the
    refined-from-relatives embedding simultaneously explain a whole vector of correlated biology axes (form,
    growth, size, climate, soil...).

    `include_eval=False` (variant `multi`): the eval axis is EXCLUDED from the panel -> a fair pure-TRANSFER test
    (does shared multi-axis structure carry to a held-out axis with its labels never in the loss?).
    `include_eval=True`  (variant `sup_multi`): the eval axis IS one head alongside the panel -> a MULTI-TASK test
    (do the 12 auxiliary trait heads regularize/boost the single eval-axis head above `sup` alone?).
    One head per trait; each contributes its loss only on masked species that OBSERVE that trait. Core untouched."""
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    heads, targets = [], []                                  # (head, kind, Yt, train_obs) per panel trait
    for k, (kind_k, Y_k, obs_k, nclass_k) in panel.items():
        if k == eval_key and not include_eval:
            continue                                         # transfer variant: never supervise the eval axis
        out_dim = 1 if kind_k == "num" else int(nclass_k)
        h = torch.nn.Linear(d_model, out_dim).to(dev)
        yt = Y_k.float() if kind_k == "num" else Y_k.long()
        heads.append((h, kind_k, yt, obs_k & (~test)))
        targets.append(k)
    params = list(graph.parameters())
    for h, *_ in heads:
        params += list(h.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & (~test)
        if not mask.any():
            continue
        refined = graph(mask=mask)                          # masked species reconstructed from relatives
        loss = 0.0
        n_terms = 0
        for h, kind_k, yt, tobs in heads:
            m = mask & tobs                                 # only species that OBSERVE this panel trait
            if not m.any():
                continue
            pred = h(refined[m])
            if kind_k == "num":
                loss = loss + F.mse_loss(pred.squeeze(-1), yt[m])
            else:
                loss = loss + F.cross_entropy(pred, yt[m])
            n_terms += 1
        if n_terms == 0:
            continue
        loss = loss / n_terms
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
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio, pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
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
    # rule-29 operator swap: ou-attention needs a dense phylo_distance from the E1 tree-derived prior
    phylo_dist = SpeciesGraph.distance_from_embedding(F.normalize(E1.to(dev), dim=-1)) if a.operator == "ou-attention" else None
    # clade-residual base grouping: family (fam_id) or finer genus (from tip_label 1st token), aligned to gidx
    fam_gid = torch.as_tensor(fam_id).long()
    gen_gid = None
    if a.num_resid_readout and a.resid_level in ("genus", "hybrid"):
        import csv as _csv
        _rows = list(_csv.DictReader(open(Path(a.cache_dir) / "derived/species_index.csv")))
        _gen = np.array([_rows[i]["tip_label"].split("_")[0] for i in gidx])
        gen_gid = torch.tensor(np.unique(_gen, return_inverse=True)[1], dtype=torch.long)
    resid_gid = gen_gid if a.resid_level == "genus" else fam_gid

    def _group_mean(gid, ytv, train_mask):
        """per-group mean of ytv over train_mask species (fallback global); returns ([N] mean, [N] train count)."""
        gid = gid.to(dev).long(); ytv = ytv.to(dev)
        G = int(gid.max().item()) + 1
        gm = ytv[train_mask].mean() if train_mask.any() else ytv.mean()
        ssum = torch.zeros(G, device=dev).index_add_(0, gid[train_mask], ytv[train_mask])
        scnt = torch.zeros(G, device=dev).index_add_(0, gid[train_mask], torch.ones(int(train_mask.sum()), device=dev))
        mean = torch.where(scnt > 0, ssum / scnt.clamp(min=1), gm)
        return mean[gid], scnt[gid]

    def _hybrid_base(ytv, train_mask):
        """genus mean where genus has >= resid_min_count train members, else family mean (bias/variance blend)."""
        fmean, _ = _group_mean(fam_gid, ytv, train_mask)
        gmean, gcnt = _group_mean(gen_gid, ytv, train_mask)
        return torch.where(gcnt >= a.resid_min_count, gmean, fmean)

    variants = a.variants  # e.g. ["unsup","sup","sup_dual"]
    agg = {src: {v: {"seed": [], "graph": [], "impute": [], "head_imp": [], "head_seed": []} for v in variants} for src in a.sources}
    t0 = time.time()

    def _head_spearman(pred, test_mask):
        """Spearman of trained-head predictions vs truth on held-out OBSERVED species (magnitude-aware readout)."""
        from scipy.stats import spearmanr
        if pred is None:
            return float("nan")
        tst = (obs & test_mask)
        yt = Y[tst].detach().cpu().numpy().ravel()
        pr = pred[tst].detach().cpu().numpy().ravel()
        if yt.size < 3:
            return float("nan")
        try:
            r = spearmanr(yt, pr).correlation
            return float(r) if r == r else 0.0
        except Exception:
            return 0.0
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        # vision block for the dual head (mean-DINO), aligned to N
        Dv, _ = _mean_seeds(a.cache_dir)
        Dv = torch.tensor(Dv).to(dev)
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio, pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            for v in variants:
                head_imp_pred = None
                head_seed_pred = None
                if v == "unsup":
                    s, rep, imp = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                              a.steps, a.mask_frac, a.lr, a.impute_steps)
                else:
                    s, rep, imp, (head, gr) = train_graph_trait_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, kind, Y, obs, nclass,
                        dual_head=(v == "sup_dual"), dual_seed=(Dv if v == "sup_dual" else None),
                        head_hidden=a.head_hidden, dual_scale=a.dual_scale,
                        rollout_rounds=a.rollout_rounds, reveal_frac=a.reveal_frac,
                        ab_scorer=(fn if a.rollout_rounds > 1 else None),
                        operator=a.operator, phylo_dist=phylo_dist,
                        fam_id=(resid_gid if a.num_resid_readout else None),
                        resid_readout=a.num_resid_readout,
                        clade_base=(_hybrid_base(Y.float(), obs & (~test))
                                    if (a.num_resid_readout and a.resid_level == "hybrid" and kind == "num")
                                    else None))
                    head_imp_pred = getattr(gr, "_head_imp", None)   # magnitude-aware head readout on imputed emb
                    head_seed_pred = getattr(gr, "_head_seed", None) # leakage control: head on own refined rep
                agg[src][v]["seed"].append(fn(s, test))
                agg[src][v]["graph"].append(fn(rep, test))
                agg[src][v]["impute"].append(fn(imp, test))
                agg[src][v]["head_imp"].append(_head_spearman(head_imp_pred, test))
                agg[src][v]["head_seed"].append(_head_spearman(head_seed_pred, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    print(f"=== BIOLOGICAL TRAIT-SUPERVISED | trait={key} metric={metric} sources={a.sources} "
          f"variants={variants} seeds={a.seeds} holdout={a.holdout} ===")
    ref_v = "unsup" if ("text" in agg and "unsup" in agg["text"]) else (variants[0] if "text" in agg else None)
    txt_ref = m(agg["text"][ref_v]["graph"])[0] if ref_v else float("nan")
    txt_ref_imp = m(agg["text"][ref_v]["impute"])[0] if ref_v else float("nan")
    print(f"  reference: text-seed {ref_v} graph={txt_ref:.4f}  impute={txt_ref_imp:.4f}")
    print(f"  {'source':7s} {'variant':9s} | {'seed':>16s} | {'GRAPH':>16s} | {'impute(kNN)':>16s} | {'impute(head)':>16s} | d_head_vs_kNN")
    summary = {"mode": "trait_supervised", "trait": key, "metric": metric,
               "text_unsup_graph": round(txt_ref, 4), "text_unsup_impute": round(txt_ref_imp, 4), "rows": []}
    for src in a.sources:
        for v in variants:
            sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"]); im, iss = m(agg[src][v]["impute"])
            hm, hs = m(agg[src][v]["head_imp"]); hsd, _ = m(agg[src][v]["head_seed"])
            dg = gm - txt_ref; di = im - txt_ref_imp
            dhead = (hm - im) if (hm == hm and im == im) else float("nan")   # magnitude-aware readout gain over kNN
            flag = " HEAD-WIN" if (dhead == dhead and dhead > 0.008) else ""
            print(f"  {src:7s} {v:9s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {im:.4f}+/-{iss:.4f} | "
                  f"{hm:.4f}+/-{hs:.4f} | head_seed={hsd:.4f} | {dhead:+.4f}{flag}")
            summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4),
                                    "graph": round(gm, 4), "graph_std": round(gs, 4), "impute": round(im, 4),
                                    "impute_head": round(hm, 4), "head_seed_ctrl": round(hsd, 4),
                                    "d_head_vs_knn": round(dhead, 4) if dhead == dhead else None,
                                    "d_graph_vs_txt": round(dg, 4), "d_imp_vs_txt": round(di, 4)})
    import json
    print("[trait_supervised] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v in {time.time()-t0:.1f}s")
    return summary


# Default multi-trait panel: broad, mixed cat+num biology axes spanning morphology / climate / soil / husbandry.
# The eval axis (--trait_key) is auto-excluded so the reported score is a pure TRANSFER from the panel.
PANEL_AXES = ["cat_form", "cat_growth_rate", "cat_plant_type", "cat_sun", "cat_soil_drainage",
              "cat_water", "cat_seasonality", "num_height_max", "num_width_max", "num_soil_ph_max",
              "num_rain_max", "num_elev_max", "num_lep_support"]


def run_oot(a, dev):
    """RULE-9 OUT-OF-TREE PROJECTION run. For each source, compares two held-out imputation protocols on the SAME
    split: (baseline) the in-tree masked reconstruction the probe has always used (test species stay tips, seeds
    masked) vs (oot) the test species held genuinely OUT of the tree, imputed only by clade soft-attach cross-
    attention over a TRAIN-only phylogeny. Answers: can the operator project a species that has NO tree position
    at all, and how close does that come to the (easier) in-tree masked reconstruction? Isolates rule-9."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    fn, kind, metric = _scorer_for(a.cache_dir, gidx, a.trait_key, dev)
    _, Y, obs, nclass = load_trait(a.cache_dir, gidx, a.trait_key, dev)
    do_head = a.num_resid_readout and kind == "num"          # ITER-2: num OOT projection + residual-head readout
    do_cat_head = a.num_resid_readout and kind == "cat"      # ITER-4: categorical OOT class readout (no clade mean)
    # hybrid clade base grouping for the residual head (genus mean where trusted, else family), aligned to gidx
    fam_gid = torch.as_tensor(fam_id).long()
    gen_gid = None
    if do_head and a.resid_level in ("genus", "hybrid"):
        import csv as _csv
        _rows = list(_csv.DictReader(open(Path(a.cache_dir) / "derived/species_index.csv")))
        _gen = np.array([_rows[i]["tip_label"].split("_")[0] for i in gidx])
        gen_gid = torch.tensor(np.unique(_gen, return_inverse=True)[1], dtype=torch.long)

    def _group_mean(gid, ytv, train_mask):
        gid = gid.to(dev).long(); ytv = ytv.to(dev)
        G = int(gid.max().item()) + 1
        gm = ytv[train_mask].mean() if train_mask.any() else ytv.mean()
        ssum = torch.zeros(G, device=dev).index_add_(0, gid[train_mask], ytv[train_mask])
        scnt = torch.zeros(G, device=dev).index_add_(0, gid[train_mask], torch.ones(int(train_mask.sum()), device=dev))
        return torch.where(scnt > 0, ssum / scnt.clamp(min=1), gm)[gid], scnt[gid]

    def _hybrid_base(ytv, train_mask):
        fmean, _ = _group_mean(fam_gid, ytv, train_mask)
        gmean, gcnt = _group_mean(gen_gid, ytv, train_mask)
        return torch.where(gcnt >= a.resid_min_count, gmean, fmean)

    def _head_spearman(pred, test_mask):
        from scipy.stats import spearmanr
        if pred is None:
            return float("nan")
        tst = (obs & test_mask)
        yt = Y[tst].detach().cpu().numpy().ravel(); pr = pred[tst].detach().cpu().numpy().ravel()
        if yt.size < 3:
            return float("nan")
        try:
            r = spearmanr(yt, pr).correlation
            return float(r) if r == r else 0.0
        except Exception:
            return 0.0

    def _head_acc(pred, test_mask):
        """ITER-4: accuracy of the OOT categorical class readout on held-out OBSERVED test species."""
        if pred is None:
            return float("nan")
        tst = (obs & test_mask)
        if int(tst.sum().item()) < 1:
            return float("nan")
        return (pred[tst] == Y[tst].long()).float().mean().item()

    agg = {s: {"seed": [], "intree": [], "oot": [], "oot_head": []} for s in a.sources}
    t0 = time.time()
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        # rule-9 test species must be IN the base tree (else "out of tree" is undefined) -> restrict holdout to tips
        intree_mask = torch.zeros(N, dtype=torch.bool)
        intree_mask[tip_row.cpu()] = True
        test = ((torch.rand(N, generator=g) < a.holdout) & intree_mask).to(dev)
        clade_base = _hybrid_base(Y.float(), obs & (~test)) if (do_head and a.resid_level == "hybrid") else None
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio,
                              pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            # baseline: standard in-tree masked reconstruction (test stays a tip)
            s_b, _rep_b, imp_b = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                             a.steps, a.mask_frac, a.lr, a.impute_steps)
            # rule-9: test held genuinely out of the tree, imputed by clade soft-attach (+/- residual head)
            _s_o, _rep_o, oot, head_pred, head_resid = train_graph_oot(
                seed, a.cache_dir, gidx, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr, a.impute_steps,
                sup_kind=(kind if (do_head or do_cat_head) else None), sup_Y=Y, sup_obs=obs, sup_nclass=nclass,
                resid_readout=do_head, clade_base=clade_base)
            # RULE-10/11 OOT ROLLOUT: fold confident OOT predictions into the clade base as pseudo-observed
            # relatives, then re-read the (fixed) residual head against the refined base. head_resid is the fixed
            # within-clade residual on the OOT embedding; only clade_mu (the base) updates as OOT neighbours reveal.
            if do_head and a.oot_rounds > 1 and head_resid is not None:
                cur_pred = head_pred.clone()
                revealed = torch.zeros(N, dtype=torch.bool, device=dev)
                yaug = Y.float().clone()
                test_idx = torch.nonzero(test, as_tuple=False).squeeze(1)
                n_test = int(test.sum().item())
                per = max(1, int(n_test * a.oot_reveal_frac))
                # confidence = closeness of the OOT residual to 0 is not informative; use tree-anchor = max cosine
                r = F.normalize(oot, dim=-1); sim = r @ r.t()
                anchor = torch.where((~test).unsqueeze(0), sim, torch.full_like(sim, -2.0)).max(1).values  # [N]
                order = anchor[test_idx].argsort(descending=True)          # most-anchored OOT species first
                ordered = test_idx[order]
                for ri in range(a.oot_rounds - 1):
                    take = ordered[ri * per:(ri + 1) * per]
                    if take.numel() == 0:
                        break
                    revealed[take] = True
                    yaug[take] = cur_pred[take]                            # pseudo-observed trait for revealed OOT
                    aug_train = (obs & (~test)) | revealed                 # train relatives + revealed OOT
                    new_base = _hybrid_base(yaug, aug_train)               # refined clade means include OOT neighbours
                    cur_pred = (head_resid + new_base).detach()            # re-read fixed head vs refined base
                head_pred = cur_pred
            agg[src]["seed"].append(fn(s_b, test))
            agg[src]["intree"].append(fn(imp_b, test))
            agg[src]["oot"].append(fn(oot, test))
            if do_head:
                agg[src]["oot_head"].append(_head_spearman(head_pred, test))
            elif do_cat_head:
                agg[src]["oot_head"].append(_head_acc(head_pred, test))    # ITER-4: categorical class accuracy
            else:
                agg[src]["oot_head"].append(float("nan"))
            # ITER-7/8 BLENDED READOUT: train the IN-TREE residual head too, blend with the OOT head. ITER-8 makes
            # it HONEST -- split the held-out OOT species into val/test halves, pick alpha on VAL, report on TEST
            # (alpha never sees the test scores). OOT-head preserves clean-seed signal; in-tree head tree-smooths.
            if a.oot_blend and do_head and head_pred is not None:
                _s, _rep, _imp, (ih, ig) = train_graph_trait_supervised(
                    seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr, a.impute_steps,
                    kind, Y, obs, nclass, fam_id=None, resid_readout=True,
                    clade_base=clade_base)
                intree_head = getattr(ig, "_head_imp", None)
                # deterministic val/test split of the OOT held-out species (per source-seed)
                gv = torch.Generator(device="cpu").manual_seed(1000 + sd)
                val_half = (torch.rand(N, generator=gv) < 0.5).to(dev) & test
                test_half = test & (~val_half)
                if intree_head is not None:
                    best_a, best_v = 1.0, -2.0
                    for al in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        bp = al * head_pred + (1 - al) * intree_head
                        vsc = _head_spearman(bp, val_half)                 # SELECT alpha on validation only
                        if vsc == vsc and vsc > best_v:
                            best_v, best_a = vsc, al
                    bp = best_a * head_pred + (1 - best_a) * intree_head
                    test_score = _head_spearman(bp, test_half)             # REPORT on the untouched test half
                    oot_test = _head_spearman(head_pred, test_half)        # pure OOT head on the same test half
                    agg[src].setdefault("blend", []).append(test_score)
                    agg[src].setdefault("blend_alpha", []).append(best_a)
                    agg[src].setdefault("blend_oot_ref", []).append(oot_test)

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    print(f"=== BIOLOGICAL OUT-OF-TREE (rule-9) | trait={a.trait_key} metric={metric} sources={a.sources} "
          f"seeds={a.seeds} holdout={a.holdout} N={N} resid_head={do_head or do_cat_head} ===")
    print(f"  {'source':7s} | {'seed':>16s} | {'intree_impute':>16s} | {'oot_project':>16s} | {'oot_head':>16s} | oot-intree | oot-seed")
    summary = {"mode": "out_of_tree", "trait": a.trait_key, "metric": metric,
               "resid_head": do_head, "cat_head": do_cat_head, "rows": []}
    for src in a.sources:
        sm, ss = m(agg[src]["seed"]); im, iss = m(agg[src]["intree"]); om, os = m(agg[src]["oot"])
        hm, hs = m(agg[src]["oot_head"])
        d_oi = om - im; d_os = om - sm
        flag = "  OOT-OK" if d_os > 0.008 else ""
        print(f"  {src:7s} | {sm:.4f}+/-{ss:.4f} | {im:.4f}+/-{iss:.4f} | {om:.4f}+/-{os:.4f} | "
              f"{hm:.4f}+/-{hs:.4f} | {d_oi:+.4f}   | {d_os:+.4f}{flag}")
        row = {"source": src, "seed": round(sm, 4), "intree_impute": round(im, 4),
               "oot_project": round(om, 4), "oot_std": round(os, 4),
               "oot_head": round(hm, 4) if hm == hm else None,
               "d_oot_vs_intree": round(d_oi, 4), "d_oot_vs_seed": round(d_os, 4)}
        if a.oot_blend and "blend" in agg[src]:
            bm, bs = m(agg[src]["blend"]); am, _ = m(agg[src]["blend_alpha"]); orf, _ = m(agg[src]["blend_oot_ref"])
            print(f"           blend(val-picked alpha, TEST-half) = {bm:.4f}+/-{bs:.4f}  vs pure-OOT-head {orf:.4f}  "
                  f"(delta {bm-orf:+.4f})  mean_alpha={am:.2f}")
            row["blend_test_half"] = round(bm, 4); row["blend_mean_alpha"] = round(am, 2)
            row["oot_head_test_half"] = round(orf, 4); row["blend_vs_oot_test_half"] = round(bm - orf, 4)
        summary["rows"].append(row)
    import json
    print("[out_of_tree] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src in {time.time()-t0:.1f}s")
    return summary


def run_multitrait(a, dev):
    """NEW LEVER runner (rule 25/29). Compare, per source, three graphs scored on the frozen EVAL axis
    (--trait_key) held-out NN imputation:
      unsup     = cosine-recon-identity graph (Round-1 baseline)
      sup       = SINGLE-axis trait-supervised graph on the eval axis (Round-2 recipe -- the incumbent)
      multi     = PANEL joint-reconstruct graph, eval axis EXCLUDED (this round's hypothesis; pure transfer)
    If `multi` graph/impute on the eval axis beats `sup` (and text-unsup champion) beyond the floor, a richer
    multi-axis reconstruction target transfers better than single-axis supervision. Multi-seed, one process."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    key = a.trait_key
    fn, kind, metric = _scorer_for(a.cache_dir, gidx, key, dev)
    _, Yk, obsk, nclk = load_trait(a.cache_dir, gidx, key, dev)
    panel_keys = a.axes if a.axes else PANEL_AXES
    panel = {k: load_trait(a.cache_dir, gidx, k, dev) for k in panel_keys}
    variants = a.variants  # subset of ["unsup","sup","multi"]
    agg = {src: {v: {"seed": [], "graph": [], "impute": []} for v in variants} for src in a.sources}
    t0 = time.time()
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio,
                              pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            for v in variants:
                if v == "unsup":
                    s, rep, imp = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                              a.steps, a.mask_frac, a.lr, a.impute_steps)
                elif v == "sup":
                    s, rep, imp, _hg = train_graph_trait_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, kind, Yk, obsk, nclk, dual_head=False, dual_seed=None,
                        head_hidden=a.head_hidden, dual_scale=a.dual_scale)
                else:  # multi (transfer, eval excluded) | sup_multi (multi-task, eval included)
                    s, rep, imp = train_graph_multitrait(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, panel, key, include_eval=(v == "sup_multi"))
                agg[src][v]["seed"].append(fn(s, test))
                agg[src][v]["graph"].append(fn(rep, test))
                agg[src][v]["impute"].append(fn(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    n_panel = sum(1 for k in panel_keys if k != key)
    print(f"=== BIOLOGICAL MULTI-TRAIT PANEL | eval={key} metric={metric} panel={n_panel}axes "
          f"sources={a.sources} variants={variants} seeds={a.seeds} holdout={a.holdout} ===")
    # reference = text-seed unsup if present, else text-seed of the first available variant (delta still meaningful)
    ref_v = "unsup" if ("text" in agg and "unsup" in agg["text"]) else (variants[0] if "text" in agg else None)
    txt_ref = m(agg["text"][ref_v]["graph"])[0] if ref_v else float("nan")
    txt_ref_imp = m(agg["text"][ref_v]["impute"])[0] if ref_v else float("nan")
    print(f"  reference: text-seed {ref_v} graph={txt_ref:.4f}  impute={txt_ref_imp:.4f}")
    print(f"  {'source':7s} {'variant':7s} | {'seed':>16s} | {'GRAPH':>16s} | {'impute':>16s} | d_graph_vs_txt | d_imp_vs_txt")
    summary = {"mode": "multi_trait", "eval": key, "metric": metric, "n_panel": n_panel,
               "text_unsup_graph": round(txt_ref, 4), "text_unsup_impute": round(txt_ref_imp, 4), "rows": []}
    for src in a.sources:
        for v in variants:
            sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"]); im, iss = m(agg[src][v]["impute"])
            dg = gm - txt_ref; di = im - txt_ref_imp
            flag = " WIN" if dg > 0.008 else ""
            print(f"  {src:7s} {v:7s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {im:.4f}+/-{iss:.4f} | "
                  f"{dg:+.4f}       | {di:+.4f}{flag}")
            summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4),
                                    "graph": round(gm, 4), "graph_std": round(gs, 4), "impute": round(im, 4),
                                    "d_graph_vs_txt": round(dg, 4), "d_imp_vs_txt": round(di, 4)})
    import json
    print("[multi_trait] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v panel={n_panel} in {time.time()-t0:.1f}s")
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
    ap.add_argument("--energy_ratio", type=float, default=0.95,
                    help="fused_white PCA-whiten cumulative-variance keep fraction (rule-26 seed stabilizer)")
    ap.add_argument("--head_hidden", type=int, default=0,
                    help="CEILING-PUSH: >0 -> deeper 2-layer GELU trait-supervised readout of this width (0=linear, current)")
    ap.add_argument("--dual_scale", type=float, default=1.0,
                    help="CEILING-PUSH: weight on the dual (appearance) head loss (1.0=current)")
    ap.add_argument("--operator", default="latent-clade", choices=["latent-clade", "ou-attention"],
                    help="rule-29 refinement operator for the trait-supervised graph: latent-clade (prior rounds) "
                         "or ou-attention (the champion OU-GP distance-biased attention). Only wired for --trait_supervised sup.")
    ap.add_argument("--rollout_rounds", type=int, default=1,
                    help="READOUT LEVER (rule 10-11): >1 -> autoregressive progressive-reveal imputation at eval "
                         "(reveal most-anchored held-out species each round so they anchor the rest). 1=single-pass.")
    ap.add_argument("--reveal_frac", type=float, default=0.25,
                    help="fraction of still-masked held-out species revealed per rollout round")
    ap.add_argument("--num_resid_readout", action="store_true",
                    help="MAGNITUDE-EXPLICIT: head regresses within-clade residual (y - train clade mean); "
                         "add the clade mean back at readout. num_ traits + trait_supervised only.")
    ap.add_argument("--resid_level", default="family", choices=["family", "genus", "hybrid"],
                    help="clade granularity for --num_resid_readout base mean: family (coarse), genus (finer), "
                         "or hybrid (genus mean where genus has >= --resid_min_count train members, else family).")
    ap.add_argument("--resid_min_count", type=int, default=4,
                    help="hybrid residual: min TRAIN-observed members for a genus mean to be trusted (else family).")
    ap.add_argument("--multi_trait", action="store_true",
                    help="NEW LEVER: seed-frozen graph joint-reconstructs a PANEL of aux traits; eval axis "
                         "(--trait_key) excluded -> pure transfer. Compares unsup|sup|multi on the eval axis.")
    ap.add_argument("--out_of_tree", action="store_true",
                    help="RULE-9: hold the test species OUT of the tree entirely (train-only phylogeny) and impute "
                         "them via clade soft-attach cross-attention. Compares OOT projection vs the in-tree masked "
                         "baseline on the same held-out split, per source.")
    ap.add_argument("--oot_rounds", type=int, default=1,
                    help="RULE-10/11 for OOT (--out_of_tree --num_resid_readout): >1 -> autoregressive readout. "
                         "Each round folds the most-confident OOT trait predictions back into the hybrid clade-mean "
                         "base as pseudo-observed relatives, then re-reads the residual head. 1=single-pass.")
    ap.add_argument("--oot_reveal_frac", type=float, default=0.34,
                    help="fraction of still-unrevealed OOT test species folded into the clade base per oot_round")
    ap.add_argument("--oot_blend", action="store_true",
                    help="ITER-7 (num_resid_readout num axis): also train the IN-TREE residual head and report the "
                         "best convex blend alpha*oot_head_pred + (1-alpha)*intree_head_pred over a sweep. OOT wins "
                         "on clean-seed axes, in-tree on tree-smoothable axes -> the blend should dominate both.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    if a.multi_axis:
        return run_multi_axis(a, dev)
    if a.trait_supervised:
        return run_trait_supervised(a, dev)
    if a.multi_trait:
        return run_multitrait(a, dev)
    if a.out_of_tree:
        return run_oot(a, dev)

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
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio, pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
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
