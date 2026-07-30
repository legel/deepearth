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

    python -m deepearth.autoresearch.probes.biological.editable_files.lib.traitprobe \
        --cache_dir autoresearch/data/deepcal --trait_key multi_flower_color --steps 400 --seeds 0 1 2
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deepearth.autoresearch.probes.biological.editable_files.harness.probe import (
    load_species, load_trait, nn_trait_acc, nn_trait_ap, nn_trait_num,
)
from deepearth.autoresearch.probes.biological.editable_files.phylogenomic import SpeciesGraph, build_tree_buffers


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
    if source.startswith("text_hipass"):
        # RULE-26/9 tree-ORTHOGONAL reseed (label-free). The dominant principal components of the E1 text prior
        # ARE the coarse taxonomic axis the dated tree already re-encodes (E1 is itself phylo-derived, ~0.89
        # family-NN -> redundant with the graph). Removing the top-P PCs high-passes the seed onto the finer,
        # tree-ORTHOGONAL structure the operator can be ADDITIVE on, instead of re-deriving what the seed holds.
        # Suffix P sets #PCs dropped (default 8): source="text_hipass" -> 8, "text_hipass16" -> 16.
        P = int(source[len("text_hipass"):] or 8)
        X = F.normalize(E1.to(dev), dim=-1).float()
        Xc = X - X.mean(0, keepdim=True)
        U, Sv, Vh = torch.linalg.svd(Xc, full_matrices=False)               # rows of Vh = principal axes (desc)
        keep = Vh[P:]                                                       # drop the top-P coarse axes
        Xhp = Xc @ keep.t() @ keep                                         # project onto the orthogonal complement
        return F.normalize(Xhp, dim=-1)
    if source.startswith("text_bandstop"):
        # RULE-26 MECHANISM CONTROL (additive, default-off). Drop a MIDDLE band of PCs [A:B] instead of the
        # TOP-P, keeping the top-A coarse axes. If width imputation only recovers when the TOP PCs are dropped
        # (text_hipass) and NOT when a middle band is dropped (this), the tree-redundancy thesis is confirmed:
        # the blocking signal is specifically the top principal components. Suffix 'A_B' -> band [A:B].
        spec = source[len("text_bandstop"):]
        A, B = (int(x) for x in spec.split("_"))
        X = F.normalize(E1.to(dev), dim=-1).float()
        Xc = X - X.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        idx = [i for i in range(Vh.shape[0]) if not (A <= i < B)]   # keep everything EXCEPT the middle band
        keep = Vh[idx]
        Xbs = Xc @ keep.t() @ keep
        return F.normalize(Xbs, dim=-1)
    if source.startswith("routed_hipass"):
        # RULE-26/9 combo: high-pass the text block (drop the tree-redundant coarse taxonomy) AND concat the
        # vision block to REFILL that deleted coarse-appearance capacity. Text_hipass carries fine tree-orthogonal
        # structure; vision (mean-DINO) carries appearance/size -- both are axes the phylo operator can ADD on,
        # neither re-derives the tree. One routed graph, probe learns the per-dim mix. Suffix = #PCs dropped.
        P = int(source[len("routed_hipass"):] or 16)
        X = F.normalize(E1.to(dev), dim=-1).float()
        Xc = X - X.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        keep = Vh[P:]
        Xhp = F.normalize(Xc @ keep.t() @ keep, dim=-1)
        return torch.cat([Xhp, F.normalize(D, dim=-1)], dim=-1)
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


# --- COMMUNITY / CO-OCCURRENCE axis (rule 10-12; flag-gated --cooccur; default path never calls these) ----
# HONEST JOIN provenance: gbif_tokens/*.npz["species_local"] is the row's index into the 2141-vocab
# (gbif_vocab.npz), the SAME vocab load_species/the graph use (data.py L68). obs_cache.npy is row-aligned
# to those chunks (lat/lon match 1.0) but its col2 is an ORPHANED 164-index -> IGNORED. We rebuild the
# community target directly from species_local, so every occurrence carries a real vocab row -> no fragile
# name match. Coverage: 2068/2141 species have >=1 occurrence (the 73 absent keep an all-zero community row,
# excluded from scoring via the `obs` mask). Target = per-species co-occurrence partner profile over grid
# cells (derived/cooccur_count_<res>.npy, built offline). We isolate: does the phylo graph help predict a
# species' co-occurrence community FROM ITS RELATIVES (graph-refined rep vs raw seed, held-out NN)?
def load_cooccur(cache: str, dev, res_tag: str = "005", topk: int = 64):
    """Community target aligned to the 2141-vocab. CO[i,j] = #grid-cells where species i and j co-occur.
    Returns (Y, obs, nclass): Y float32 [N, N] BINARY partner target (1 if i,j share >=1 cell, self-excluded),
    obs bool [N] species present in >=1 cell. `topk` keeps only each species' top-K co-partners in the target
    (a discriminative COMMUNITY signature, not the near-universal thresholded set) -- the held-out question is
    whether relatives recover a species' characteristic co-occurring community."""
    CO = np.load(Path(cache) / f"derived/cooccur_count_{res_tag}.npy").astype(np.float32)
    N = CO.shape[0]
    np.fill_diagonal(CO, 0.0)
    present = torch.tensor(CO.sum(1) > 0)
    Y = np.zeros_like(CO)
    if topk and topk < N:                                    # top-K strongest partners per species -> binary signature
        idx = np.argpartition(-CO, topk, axis=1)[:, :topk]
        rows = np.repeat(np.arange(N), topk)
        Y[rows, idx.ravel()] = (CO[rows, idx.ravel()] > 0).astype(np.float32)
    else:
        Y = (CO > 0).astype(np.float32)
    return torch.tensor(Y).to(dev), present.to(dev), N


def nn_cooccur_ap(emb, Y, obs, test, k=5):
    """Held-out species -> k nearest OBSERVED-TRAIN species (cosine on the representation) -> predicted community
    profile = mean of neighbours' partner rows -> micro-AP vs the true top-K partner target. Positive AP over the
    community-prior baseline = the geometry carries co-occurrence structure; graph_gain = refined - seed under an
    identical NN protocol. This is the rule 10-12 capability the phylo graph should serve (community from relatives),
    scored EXACTLY like the multi-label trait AP so it is comparable to the trait axes."""
    from sklearn.metrics import average_precision_score
    train = obs & (~test)
    tst = obs & test
    et = F.normalize(emb[train], dim=-1)
    ett = F.normalize(emb[tst], dim=-1)
    sim = ett @ et.t()
    topk = sim.topk(min(k, sim.shape[1]), dim=-1).indices
    pred = Y[train][topk].mean(1)                            # [ntest, N] predicted community profile
    # score partner columns only over species that ever appear as a partner (drop all-zero cols -> honest AP)
    col_ok = (Y[train].sum(0) > 0)
    yt = Y[tst][:, col_ok].cpu().numpy().ravel()
    pr = pred[:, col_ok].cpu().numpy().ravel()
    try:
        return float(average_precision_score(yt, pr))
    except Exception:
        return float("nan")


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
                    resid_readout=False, clade_base=None, recon_relative=False, fam_gid=None, recon_k=0,
                    oot_heads=4, oot_layers=2):
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
                         species_text=seed, n_heads=oot_heads, n_layers=oot_layers).to(dev)
    # in-tree train tips (bool over full vocab) -- the only rows we mask/reconstruct during training
    intree = torch.zeros(N, dtype=torch.bool, device=dev); intree[tip_row_tr.to(dev)] = True
    # RULE-25 (recon_relative): reconstruct a masked species toward the MEAN SEED of its same-family TRAIN
    # relatives -- the real held-out imputation target -- not the leaked identity copy of its own seed. Identity
    # recon is trivially satisfiable by the seed, so the operator learns ~identity and adds little (established
    # weak in-tree path). A relative target forces the operator to place a masked species where its relatives
    # predict, which is exactly the OOT soft-attach eval task. Default off (byte-identical when the flag is unset).
    rel_target = None
    if recon_relative and fam_gid is not None:
        fg = fam_gid.to(dev).long()
        s0 = graph._seed().detach()
        G = int(fg.max().item()) + 1
        tr = intree
        if recon_k and recon_k > 0:
            # ITER-2: rank-PRESERVING relative target. Each species' target = mean seed of its recon_k nearest
            # SAME-FAMILY TRAIN relatives (cosine in seed space), NOT the whole-family centroid (which erased
            # within-family rank in ITER-1). A local neighborhood keeps the fine structure a continuous trait
            # needs while still forcing reconstruction FROM relatives (leak-free: a species is never its own
            # neighbor, and only TRAIN tips are eligible relatives).
            sn = F.normalize(s0, dim=-1)
            eligible = tr.clone()                                        # only train tips are relatives
            same_fam = (fg.unsqueeze(1) == fg.unsqueeze(0))             # [N,N] same-family mask
            sim = sn @ sn.t()
            valid = same_fam & eligible.unsqueeze(0)                    # relative must be same-family AND a train tip
            valid.fill_diagonal_(False)                                 # never self
            sim = sim.masked_fill(~valid, -2.0)
            kk = min(int(recon_k), sim.shape[1])
            topv, topi = sim.topk(kk, dim=1)                           # [N,kk]
            has = (topv > -1.5)                                        # rows with >=1 same-family train relative
            neigh = s0[topi]                                            # [N,kk,d]
            w = (topv.clamp(min=-1.5) * has.float()).unsqueeze(-1)     # ignore pad slots
            wsum = has.float().sum(1, keepdim=True).clamp(min=1)
            knn_mean = (neigh * has.float().unsqueeze(-1)).sum(1) / wsum.unsqueeze(-1).squeeze(1)
            # fall back to family centroid for species with no same-family train relative
            cnt = torch.zeros(G, device=dev).index_add_(0, fg[tr], torch.ones(int(tr.sum()), device=dev))
            ssum = torch.zeros(G, s0.shape[1], device=dev).index_add_(0, fg[tr], s0[tr])
            fam_mean = (ssum / cnt.clamp(min=1).unsqueeze(-1))[fg]
            rel_target = torch.where(has.any(1, keepdim=True), knn_mean, fam_mean)
        else:
            cnt = torch.zeros(G, device=dev).index_add_(0, fg[tr], torch.ones(int(tr.sum()), device=dev))
            ssum = torch.zeros(G, s0.shape[1], device=dev).index_add_(0, fg[tr], s0[tr])
            fam_mean = ssum / cnt.clamp(min=1).unsqueeze(-1)
            rel_target = fam_mean[fg]
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
        if rel_target is not None:
            target = rel_target
        else:
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
                                 fam_id=None, resid_readout=False, clade_base=None,
                                 blanket_k=0, mask_curriculum=None):
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
        # blanket_k (rule-29 Markov-blanket restriction): restrict each species' reconstruction to its k NEAREST
        # phylo relatives via the operator's top_k sparsification of phylo_distance (dense = whole-clade). 0 = dense.
        graph = SpeciesGraph(N, d_model, operator="ou-attention", phylo_distance=phylo_dist,
                             n_heads=4, n_layers=2, species_text=seed,
                             top_k=(blanket_k if blanket_k and blanket_k > 0 else None)).to(dev)
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
    for _step in range(steps):
        # mask_curriculum (rule-25 mechanism knob): ('easy2hard', lo, hi) linearly RAMPS the mask fraction
        # lo->hi over training (harder recon later); ('hard2easy', lo, hi) reverses it. None = fixed mask_frac.
        mf = mask_frac
        if mask_curriculum is not None:
            _sched, _lo, _hi = mask_curriculum
            _frac = _step / max(1, steps - 1)
            mf = (_lo + (_hi - _lo) * _frac) if _sched == "easy2hard" else (_hi - (_hi - _lo) * _frac)
        mask = (torch.rand(N, device=dev) < mf) & train_obs
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
                                    else None),
                        blanket_k=a.blanket_k, mask_curriculum=a.mask_curriculum)
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
                resid_readout=do_head, clade_base=clade_base,
                recon_relative=a.recon_relative, fam_gid=fam_gid, recon_k=a.recon_k,
                oot_heads=a.oot_heads, oot_layers=a.oot_layers)
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


def run_cooccur(a, dev):
    """rule 10-12 COMMUNITY capability. Pin the community readout (top-K co-occurrence partner target,
    micro-AP over held-out species from their relatives); vary ONLY the encoder: graph-refined rep vs raw
    seed, per source, across seeds. graph_gain = graph_AP - seed_AP isolates the phylo operator's value for
    predicting a species' co-occurring community FROM RELATIVES -- a NEW axis beyond single traits. Reuses the
    identical latent-clade train_graph protocol; edits nothing in core/."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    Y, obs, N = load_cooccur(a.cache_dir, dev, res_tag=a.cooccur_res, topk=a.cooccur_topk)
    scorer = lambda emb, test: nn_cooccur_ap(emb, Y, obs, test)
    agg = {s: {"seed": [], "graph": [], "impute": []} for s in a.sources}
    t0 = time.time()
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        for src in a.sources:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio,
                              pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep)
            s, rep, imp = train_graph(seed, tree, tip_row, test, dev, a.d_model,
                                      a.steps, a.mask_frac, a.lr, a.impute_steps)
            agg[src]["seed"].append(scorer(s, test))
            agg[src]["graph"].append(scorer(rep, test))
            agg[src]["impute"].append(scorer(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    n_partner = int((Y[obs].sum(0) > 0).sum().item())
    print(f"=== BIOLOGICAL COMMUNITY (co-occurrence rule10-12) | res={a.cooccur_res} topk={a.cooccur_topk} "
          f"metric=micro-AP seeds={a.seeds} holdout={a.holdout} N={N} present={int(obs.sum())} partners={n_partner} ===")
    print(f"  {'source':7s} | {'seed_only':>18s} | {'GRAPH(rep)':>18s} | {'impute':>18s} | graph_gain")
    for src in a.sources:
        sm, ss = m(agg[src]["seed"]); gm, gs = m(agg[src]["graph"]); im, iis = m(agg[src]["impute"])
        print(f"  {src:7s} | {sm:.4f} +/- {ss:.4f} | {gm:.4f} +/- {gs:.4f} | {im:.4f} +/- {iis:.4f} | {gm - sm:+.4f}")
    print(f"  [profile] {len(a.seeds)} seeds x {len(a.sources)} sources in {time.time()-t0:.1f}s")
    import json
    summary = {"axis": "cooccur_community", "res": a.cooccur_res, "topk": a.cooccur_topk, "metric": "micro-AP",
               "seeds": a.seeds, "N": N, "present": int(obs.sum().item())}
    for src in a.sources:
        summary[src] = {"seed": round(m(agg[src]["seed"])[0], 4), "graph": round(m(agg[src]["graph"])[0], 4),
                        "graph_std": round(m(agg[src]["graph"])[1], 4),
                        "impute": round(m(agg[src]["impute"])[0], 4),
                        "graph_gain": round(m(agg[src]["graph"])[0] - m(agg[src]["seed"])[0], 4)}
    print("[cooccur_compare] " + json.dumps(summary))
    return summary


def load_myco(cache: str, gidx, dev):
    """Mycorrhizal-association trait (B63, FungalRoot -> per-GENUS majority type), aligned to the 2141-graph via
    gidx. 5 classes AM/EcM/ErM/OM/NM. Returns (Y int64 [N], obs bool [N], nclass). Phylogenetically CONSERVED
    (genus-level majority): the graph SHOULD serve it -- this is the graph-friendly counterpart to the community
    axis. Additive/flag-gated; touches nothing in core/ or probe.py."""
    z = np.load(Path(cache) / "gbif_mycorrhiza.npz", allow_pickle=True)
    myco = z["myco"]; has = z["has_myco"]
    y = torch.tensor(myco[gidx].astype(np.int64)).to(dev)
    obs = torch.tensor(has[gidx].astype(bool)).to(dev)
    y = torch.where(obs, y, torch.zeros_like(y))                 # park unobserved at 0 (excluded by obs everywhere)
    return y, obs, int(len(z["classes"]))


def run_myco_supervised(a, dev):
    """TEST 2. Myco (B63) trait-supervised imputation. Pin the myco-NN accuracy readout; vary the encoder: raw
    seed vs unsup-graph vs trait-SUPERVISED graph (frozen seed + CE on masked-species reconstruction from
    relatives -- the SAME mechanism that rescued lep/height/cat_water). Does the graph serve a phylogenetically
    conserved association trait? Reports seed / unsup-graph / sup-graph acc per source, graph_gain = sup - seed,
    and where myco lands vs the lep-like wins. Reuses train_graph_trait_supervised (kind='cat'); edits nothing
    in core/."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    Y, obs, nclass = load_myco(a.cache_dir, gidx, dev)
    scorer = lambda emb, test: nn_trait_acc(emb, Y, obs, test)
    variants = ["unsup", "sup"]
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
                else:
                    s, rep, imp, _ = train_graph_trait_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, "cat", Y, obs, nclass, operator=a.operator,
                        phylo_dist=(SpeciesGraph.distance_from_embedding(F.normalize(E1.to(dev), dim=-1))
                                    if a.operator == "ou-attention" else None),
                        blanket_k=a.blanket_k)
                agg[src][v]["seed"].append(scorer(s, test))
                agg[src][v]["graph"].append(scorer(rep, test))
                agg[src][v]["impute"].append(scorer(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    n_obs = int(obs.sum().item())
    print(f"=== BIOLOGICAL MYCO-SUPERVISED (B63) | metric=acc classes={nclass} sources={a.sources} "
          f"variants={variants} seeds={a.seeds} holdout={a.holdout} N={N} obs={n_obs} ===")
    print(f"  {'source':7s} {'variant':6s} | {'seed':>16s} | {'GRAPH':>16s} | {'impute':>16s} | graph_gain")
    summary = {"axis": "myco_B63", "metric": "acc", "nclass": nclass, "seeds": a.seeds, "N": N, "obs": n_obs,
               "rows": []}
    for src in a.sources:
        for v in variants:
            sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"]); im, iis = m(agg[src][v]["impute"])
            print(f"  {src:7s} {v:6s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {im:.4f}+/-{iis:.4f} | {gm-sm:+.4f}")
            summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4), "graph": round(gm, 4),
                                    "graph_std": round(gs, 4), "impute": round(im, 4),
                                    "graph_gain": round(gm - sm, 4)})
    import json
    print("[myco_supervised] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v in {time.time()-t0:.1f}s")
    return summary


def train_graph_cooccur_supervised(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                                   impute_steps, Y, obs):
    """COMMUNITY-SUPERVISED refinement (the untested positive branch). SAME mechanism as the trait-supervised
    recipe that rescued the trait axes, applied to the co-occurrence target: freeze the seed, mask a fraction of
    TRAIN species, and drive a small MULTI-LABEL head off each masked species' refined (reconstructed-from-
    relatives) embedding toward that species' true top-K co-occurrence partner row (BCE). If the phylo operator
    can be SUPERVISED to reconstruct a species' community from its relatives, graph_gain should flip positive as
    it did for traits; if community stays graph-resistant even under supervision, it confirms community is a
    spatial-niche (not phylo) axis. Returns (seed, rep, impute) exactly like train_graph, so the identical
    nn_cooccur_ap scorer applies. Additive; edits nothing in core/."""
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    train_obs = obs & (~test)
    head = torch.nn.Linear(d_model, Y.shape[1]).to(dev)          # multi-label partner-set readout (BCE)
    params = list(graph.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & train_obs
        if not mask.any():
            continue
        refined = graph(mask=mask)                              # masked species reconstructed from relatives
        pred = head(refined[mask])
        loss = F.binary_cross_entropy_with_logits(pred, Y[mask])
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


def run_cooccur_supervised(a, dev):
    """TEST 1. Community-SUPERVISED refinement. Pin the co-occurrence micro-AP readout (nn_cooccur_ap); vary
    ONLY the encoder: raw seed vs UNSUP graph vs community-SUPERVISED graph. graph_gain = graph_AP - seed_AP.
    The untested positive branch: does supervising the graph to reconstruct a masked species' co-occurrence
    partner-set from relatives flip graph_gain POSITIVE (as trait-supervision did for the trait axes), or does
    community stay graph-resistant even supervised (confirming spatial-niche, not phylo)? Either is clean DATA."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    Y, obs, N = load_cooccur(a.cache_dir, dev, res_tag=a.cooccur_res, topk=a.cooccur_topk)
    scorer = lambda emb, test: nn_cooccur_ap(emb, Y, obs, test)
    variants = ["unsup", "sup"]
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
                else:
                    s, rep, imp = train_graph_cooccur_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, Y, obs)
                agg[src][v]["seed"].append(scorer(s, test))
                agg[src][v]["graph"].append(scorer(rep, test))
                agg[src][v]["impute"].append(scorer(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    n_partner = int((Y[obs].sum(0) > 0).sum().item())
    print(f"=== BIOLOGICAL COMMUNITY-SUPERVISED | res={a.cooccur_res} topk={a.cooccur_topk} metric=micro-AP "
          f"sources={a.sources} variants={variants} seeds={a.seeds} holdout={a.holdout} N={N} "
          f"present={int(obs.sum())} partners={n_partner} ===")
    print(f"  {'source':7s} {'variant':6s} | {'seed':>16s} | {'GRAPH':>16s} | {'impute':>16s} | graph_gain")
    summary = {"axis": "cooccur_community_supervised", "res": a.cooccur_res, "topk": a.cooccur_topk,
               "metric": "micro-AP", "seeds": a.seeds, "N": N, "present": int(obs.sum().item()), "rows": []}
    for src in a.sources:
        for v in variants:
            sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"]); im, iis = m(agg[src][v]["impute"])
            print(f"  {src:7s} {v:6s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {im:.4f}+/-{iis:.4f} | {gm-sm:+.4f}")
            summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4), "graph": round(gm, 4),
                                    "graph_std": round(gs, 4), "impute": round(im, 4),
                                    "graph_gain": round(gm - sm, 4)})
    import json
    print("[cooccur_supervised] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v in {time.time()-t0:.1f}s")
    return summary


def run_route_contrast(a, dev):
    """TEST 3. Routing confirmation. Quantify HOW MUCH better vision/env-niche seeds are than text/phylo on the
    community axis vs on a phylo-friendly axis (lep). Reports the vision-text SEED gap on each axis. A large
    positive gap on community + small/negative on lep => community should route to the spacetime/vision path in
    the champion. Cheap: seed-only NN scores, no graph training."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    Yc, obsc, Ncc = load_cooccur(a.cache_dir, dev, res_tag=a.cooccur_res, topk=a.cooccur_topk)
    lep_fn, lep_kind, _ = _scorer_for(a.cache_dir, gidx, "num_lep_support", dev)
    axes = {"community": (lambda emb, test: nn_cooccur_ap(emb, Yc, obsc, test)),
            "lep": lep_fn}
    srcs = ["text", "vision"]
    agg = {ax: {s: [] for s in srcs} for ax in axes}
    for sd in a.seeds:
        g = torch.Generator(device="cpu").manual_seed(sd)
        test = (torch.rand(N, generator=g) < a.holdout).to(dev)
        for src in srcs:
            seed = build_seed(src, a.cache_dir, E1, a.vision_medoid, dev, energy_ratio=a.energy_ratio,
                              pool=a.vision_pool, temp=a.pool_temp, keep=a.pool_keep).to(dev)
            for ax, fn in axes.items():
                agg[ax][src].append(fn(seed, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    print(f"=== BIOLOGICAL ROUTING CONTRAST (seed-only) | axes={list(axes)} seeds={a.seeds} N={N} ===")
    print(f"  {'axis':10s} | {'text_seed':>16s} | {'vision_seed':>16s} | vision-text gap")
    summary = {"mode": "route_contrast", "seeds": a.seeds, "rows": []}
    for ax in axes:
        tm, ts = m(agg[ax]["text"]); vm, vs = m(agg[ax]["vision"])
        gap = vm - tm
        print(f"  {ax:10s} | {tm:.4f}+/-{ts:.4f} | {vm:.4f}+/-{vs:.4f} | {gap:+.4f}")
        summary["rows"].append({"axis": ax, "text_seed": round(tm, 4), "vision_seed": round(vm, 4),
                                "vision_minus_text": round(gap, 4)})
    import json
    print("[route_contrast] " + json.dumps(summary))
    return summary




# ============================================================================================
# NEW AUTHORITATIVE-DATA RETESTS (2026-07-24, additive/flag-gated; edits nothing in core/ or probe.py).
# Tests whether the REAL landed data (GloBI interactions, NatureServe G-ranks, USDA husbandry) are
# phylo-graph-served traits: for each, seed_only vs unsup-graph vs trait-SUPERVISED graph, graph_gain
# on held-out species. Targets are keyed by global_idx in derived/*, aligned to the N graph tips via gidx.
# ============================================================================================

# Curated pollinator-genus -> guild map covering the dominant GloBI genera (survey of top ~150). Higher-taxon
# tokens (Apoidea, Lepidoptera, Diptera, Coleoptera, ...) route by clade. Unknown genera -> "other".
_GUILD_GENUS = {
    # --- bees (Anthophila) ---
    "bee": ["Bombus", "Lasioglossum", "Andrena", "Hylaeus", "Halictus", "Osmia", "Megachile", "Apis",
            "Nomada", "Ceratina", "Anthophora", "Agapostemon", "Colletes", "Xylocopa", "Melissodes",
            "Hoplitis", "Eucera", "Perdita", "Habropoda", "Diadasia", "Svastra", "Peponapis", "Dufourea",
            "Sphecodes", "Augochlorella", "Augochlora", "Augochloropsis", "Dialictus", "Melitta",
            "Panurginus", "Calliopsis", "Pseudopanurgus", "Anthidium", "Coelioxys", "Triepeolus",
            "Epeolus", "Protandrena", "Macrotera", "Nomia", "Exomalopsis", "Ashmeadiella", "Chelostoma",
            "Heriades", "Stelis", "Bombus", "Apoidea", "Anthophila", "Halictidae", "Apidae", "Megachilidae",
            "Andrenidae", "Colletidae", "Melittidae", "Zadontomerus"],
    # --- butterflies + moths (Lepidoptera) ---
    "lep": ["Vanessa", "Papilio", "Callophrys", "Colias", "Danaus", "Pieris", "Strymon", "Speyeria",
            "Phyciodes", "Icaricia", "Chlosyne", "Euphydryas", "Apodemia", "Hesperia", "Hylephila",
            "Pontia", "Celastrina", "Plebejus", "Satyrium", "Erynnis", "Battus", "Lon", "Hyles",
            "Junonia", "Nymphalis", "Polygonia", "Limenitis", "Adelpha", "Coenonympha", "Cercyonis",
            "Euptoieta", "Agraulis", "Lycaena", "Glaucopsyche", "Leptotes", "Brephidium", "Atlides",
            "Ministrymon", "Ochlodes", "Poanes", "Atalopedes", "Lerema", "Pyrgus", "Thorybes",
            "Epargyreus", "Anthocharis", "Euchloe", "Nathalis", "Zerene", "Phoebis", "Manduca",
            "Autographa", "Trichoplusia", "Lepidoptera", "Nymphalidae", "Hesperiidae", "Pieridae",
            "Lycaenidae", "Papilionidae", "Riodinidae", "Sphingidae", "Noctuidae"],
    # --- flies (Diptera), syrphid + others ---
    "fly": ["Eristalis", "Toxomerus", "Eupeodes", "Sphaerophoria", "Copestylum", "Platycheirus",
            "Syrphus", "Allograpta", "Helophilus", "Palpada", "Volucella", "Bombylius", "Villa",
            "Systoechus", "Sarcophaga", "Lucilia", "Calliphora", "Musca", "Delia", "Scaeva",
            "Melanostoma", "Chrysotoxum", "Sericomyia", "Diptera", "Syrphidae", "Bombyliidae",
            "Tachinidae", "Calliphoridae", "Sarcophagidae"],
    # --- beetles (Coleoptera) ---
    "beetle": ["Acmaeodera", "Coccinella", "Hippodamia", "Chauliognathus", "Trichodes", "Diabrotica",
               "Epicauta", "Nemognatha", "Mordellistena", "Mordella", "Dasytes", "Listrus",
               "Trichochrous", "Coleoptera", "Cerambycidae", "Buprestidae", "Cantharidae",
               "Coccinellidae", "Melyridae", "Mordellidae", "Scarabaeidae", "Meloidae", "Chrysomelidae"],
    # --- wasps (non-bee Hymenoptera) ---
    "wasp": ["Polistes", "Vespula", "Dolichovespula", "Ammophila", "Sphex", "Bembix", "Philanthus",
             "Ichneumonoidea", "Ichneumonidae", "Braconidae", "Vespidae", "Sphecidae", "Crabronidae",
             "Pompilidae", "Scoliidae", "Tiphiidae", "Chrysididae", "Pemphredon", "Cerceris", "Eumenes"],
    # --- birds ---
    "bird": ["Selasphorus", "Calypte", "Archilochus", "Colibri", "Trochilidae", "Aves", "Setophaga",
             "Passerina", "Carpodacus", "Haemorhous", "Zonotrichia"],
}
def _guild_lookup():
    m = {}
    for guild, genera in _GUILD_GENUS.items():
        for g in genera:
            m[g] = guild
    return m


def _globi_guild_vectors(cache, gidx, dev, min_partners=3):
    """Build per-species pollinator-guild signatures from the REAL GloBI interaction records, aligned to the N
    graph tips via gidx. Returns (Yfrac [N, G] guild-fraction, dom [N] argmax-guild int, obs [N] bool has>=min
    recognized partners, guilds list). Catalog-noise tokens (all-caps codes / uuids / digit strings) are dropped;
    only genera in the curated guild map contribute (so 'obs' = species with >= min_partners RECOGNIZED partners)."""
    import csv as _csv
    from collections import defaultdict
    gl = _guild_lookup()
    guilds = ["bee", "lep", "fly", "beetle", "wasp", "bird"]
    gi = {g: i for i, g in enumerate(guilds)}
    counts = defaultdict(lambda: np.zeros(len(guilds), dtype=np.float32))    # global_idx -> guild counts
    with open(Path(cache) / "derived/pollinator_globi_interactions.tsv") as f:
        r = _csv.DictReader(f, delimiter="\t", quoting=_csv.QUOTE_NONE)
        for row in r:
            gi_raw = row.get("global_idx")
            if gi_raw is None or not str(gi_raw).strip().isdigit():
                continue                                # skip ~10 quote-mangled rows
            g_idx = int(gi_raw)
            parts = (row.get("partners") or "")
            for tok in parts.split("|"):
                tok = tok.strip()
                if not tok:
                    continue
                genus = tok.split()[0]
                if genus.isupper():                     # catalog codes (WSU, WSUC, WSDA_...)
                    continue
                if any(ch.isdigit() for ch in genus):
                    continue
                guild = gl.get(genus)
                if guild is not None:
                    counts[g_idx][gi[guild]] += 1.0
    N = len(gidx)
    C = np.zeros((N, len(guilds)), dtype=np.float32)
    for i, g_idx in enumerate(gidx):
        if int(g_idx) in counts:
            C[i] = counts[int(g_idx)]
    tot = C.sum(1)
    obs = tot >= float(min_partners)
    Yfrac = np.zeros_like(C)
    nz = tot > 0
    Yfrac[nz] = C[nz] / tot[nz][:, None]
    dom = C.argmax(1).astype(np.int64)
    return (torch.tensor(Yfrac).to(dev), torch.tensor(dom).to(dev),
            torch.tensor(obs).to(dev), guilds)


def run_globi_guild(a, dev):
    """RETEST (task 1+3). REAL GloBI plant->pollinator guild signature as a phylo target. Two readouts on the
    SAME held-out split: (A) DOMINANT-guild categorical accuracy (is a plant's main pollinator guild -- bee vs
    lep vs fly... -- phylo-conserved / imputable from relatives?), and (B) multi-guild fraction micro-AP (the
    full guild signature). For each: seed_only vs unsup-graph vs trait-SUPERVISED graph, graph_gain = sup-seed.
    Related plants often share pollinators -> if pollination syndrome is graph-served this flips positive where
    the old BioCLIP co-visitation proxy could not. Either way = clean DATA."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    Yfrac, dom, obs, guilds = _globi_guild_vectors(a.cache_dir, gidx, dev, min_partners=a.globi_min_partners)
    nclass = len(guilds)
    acc_scorer = lambda emb, test: nn_trait_acc(emb, dom, obs, test)
    def ap_scorer(emb, test, k=5):
        from sklearn.metrics import average_precision_score
        train = obs & (~test); tst = obs & test
        et = F.normalize(emb[train], dim=-1); ett = F.normalize(emb[tst], dim=-1)
        sim = ett @ et.t()
        topk = sim.topk(min(k, sim.shape[1]), dim=-1).indices
        pred = Yfrac[train][topk].mean(1).cpu().numpy()        # [ntest, G] predicted guild fractions
        yt = (Yfrac[tst].cpu().numpy() > 0).astype(int)        # binary: guild present for that plant
        aps = []
        for c in range(yt.shape[1]):
            if yt[:, c].sum() > 0 and yt[:, c].sum() < len(yt[:, c]):
                aps.append(average_precision_score(yt[:, c], pred[:, c]))
        return float(np.mean(aps)) if aps else float("nan")
    variants = ["unsup", "sup"]
    agg = {src: {v: {"seed_acc": [], "graph_acc": [], "seed_ap": [], "graph_ap": []} for v in variants}
           for src in a.sources}
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
                else:
                    s, rep, imp, _ = train_graph_trait_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, "cat", dom, obs, nclass, operator=a.operator,
                        phylo_dist=(SpeciesGraph.distance_from_embedding(F.normalize(E1.to(dev), dim=-1))
                                    if a.operator == "ou-attention" else None),
                        blanket_k=a.blanket_k)
                agg[src][v]["seed_acc"].append(acc_scorer(s, test))
                agg[src][v]["graph_acc"].append(acc_scorer(rep, test))
                agg[src][v]["seed_ap"].append(ap_scorer(s, test))
                agg[src][v]["graph_ap"].append(ap_scorer(rep, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    n_obs = int(obs.sum().item())
    # majority-guild floor for the dominant-guild acc
    dom_obs = dom[obs].cpu().numpy()
    import numpy as _np
    floor = float(_np.bincount(dom_obs, minlength=nclass).max() / max(1, len(dom_obs)))
    print(f"=== BIOLOGICAL GloBI-GUILD (REAL interactions) | guilds={guilds} classes={nclass} "
          f"sources={a.sources} variants={variants} seeds={a.seeds} holdout={a.holdout} N={N} "
          f"obs(>= {a.globi_min_partners} recog. partners)={n_obs} majority-floor={floor:.4f} ===")
    print(f"  {'source':7s} {'var':5s} | {'domguild_acc seed->graph':>28s} | {'guildsig_AP seed->graph':>28s} | gain(acc)")
    summary = {"axis": "globi_pollinator_guild", "guilds": guilds, "nclass": nclass, "seeds": a.seeds,
               "N": N, "obs": n_obs, "majority_floor": round(floor, 4), "rows": []}
    for src in a.sources:
        for v in variants:
            sa, _ = m(agg[src][v]["seed_acc"]); ga, gas = m(agg[src][v]["graph_acc"])
            sp, _ = m(agg[src][v]["seed_ap"]); gp, gps = m(agg[src][v]["graph_ap"])
            print(f"  {src:7s} {v:5s} | {sa:.4f} -> {ga:.4f}+/-{gas:.4f}      | "
                  f"{sp:.4f} -> {gp:.4f}+/-{gps:.4f}      | {ga-sa:+.4f}")
            summary["rows"].append({"source": src, "variant": v,
                                    "seed_acc": round(sa, 4), "graph_acc": round(ga, 4),
                                    "graph_acc_std": round(gas, 4), "gain_acc": round(ga - sa, 4),
                                    "seed_ap": round(sp, 4), "graph_ap": round(gp, 4),
                                    "gain_ap": round(gp - sp, 4)})
    import json
    print("[globi_guild] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v in {time.time()-t0:.1f}s")
    return summary


def _natureserve_grank(cache, gidx, dev, binary=True):
    """Load NatureServe G-ranks aligned to N tips via gidx. Returns (y int64 [N], obs bool [N], nclass, name).
    binary=True: at-risk (G1/G2/G3 -> 1) vs secure (G4/G5 -> 0). binary=False: ordinal G1..G5 -> 0..4."""
    import json as _json
    rank = {}                                                # global_idx -> gRank string
    with open(Path(cache) / "derived/rarity_natureserve_global.jsonl") as f:
        for line in f:
            d = _json.loads(line)
            gr = d.get("gRank")
            if gr and isinstance(gr, str) and gr.startswith("G") and len(gr) >= 2 and gr[1].isdigit():
                rank[int(d["global_idx"])] = int(gr[1])      # 1..5
    N = len(gidx)
    y = np.zeros(N, dtype=np.int64)
    obs = np.zeros(N, dtype=bool)
    for i, g_idx in enumerate(gidx):
        if int(g_idx) in rank:
            r = rank[int(g_idx)]
            if r < 1 or r > 5:
                continue
            obs[i] = True
            y[i] = (1 if r <= 3 else 0) if binary else (r - 1)
    nclass = 2 if binary else 5
    name = "natureserve_atrisk_bin" if binary else "natureserve_grank_ord"
    return (torch.tensor(y).to(dev), torch.tensor(obs).to(dev), nclass, name)


def run_natureserve(a, dev):
    """RETEST (task 2). NatureServe G-rank (REAL conservation rarity, 2063/2141) as a phylo target. Binary
    at-risk (G1-3) vs secure (G4-5) balanced-accuracy: seed_only vs unsup-graph vs trait-SUPERVISED graph. Is
    rarity a GRAPH-served trait (relatives share rarity -> graph additive over the seed) or a niche/range axis
    (like community -> flat under the graph, belongs to the env encoder)? Reports balanced-acc vs the majority
    floor, graph_gain = graph - seed. Prior E1-seed retest got bal-acc ~0.627; this isolates the GRAPH delta."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    y, obs, nclass, name = _natureserve_grank(a.cache_dir, gidx, dev, binary=not a.natureserve_ordinal)

    def bal_acc(emb, test):
        train = obs & (~test); tst = obs & test
        et = F.normalize(emb[train], dim=-1); ett = F.normalize(emb[tst], dim=-1)
        nn = (ett @ et.t()).argmax(-1)
        pred = y[train][nn]; true = y[tst]
        accs = []
        for c in range(nclass):
            m_ = (true == c)
            if m_.any():
                accs.append((pred[m_] == c).float().mean().item())
        return float(np.mean(accs)) if accs else float("nan")

    variants = ["unsup", "sup"]
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
                else:
                    s, rep, imp, _ = train_graph_trait_supervised(
                        seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                        a.impute_steps, "cat", y, obs, nclass, operator=a.operator,
                        phylo_dist=(SpeciesGraph.distance_from_embedding(F.normalize(E1.to(dev), dim=-1))
                                    if a.operator == "ou-attention" else None),
                        blanket_k=a.blanket_k)
                agg[src][v]["seed"].append(bal_acc(s, test))
                agg[src][v]["graph"].append(bal_acc(rep, test))
                agg[src][v]["impute"].append(bal_acc(imp, test))

    def m(xs):
        return float(np.mean(xs)), float(np.std(xs))
    n_obs = int(obs.sum().item())
    yo = y[obs].cpu().numpy()
    floor = float(np.bincount(yo, minlength=nclass).max() / max(1, len(yo)))
    print(f"=== BIOLOGICAL NATURESERVE {name} | metric=balanced-acc classes={nclass} sources={a.sources} "
          f"variants={variants} seeds={a.seeds} holdout={a.holdout} N={N} obs={n_obs} "
          f"majority-floor={floor:.4f} ===")
    print(f"  {'source':7s} {'var':5s} | {'seed':>16s} | {'GRAPH':>16s} | {'impute':>16s} | graph_gain")
    summary = {"axis": name, "metric": "balanced-acc", "nclass": nclass, "seeds": a.seeds, "N": N,
               "obs": n_obs, "majority_floor": round(floor, 4), "rows": []}
    for src in a.sources:
        for v in variants:
            sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"]); im, iis = m(agg[src][v]["impute"])
            print(f"  {src:7s} {v:5s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {im:.4f}+/-{iis:.4f} | {gm-sm:+.4f}")
            summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4), "graph": round(gm, 4),
                                    "graph_std": round(gs, 4), "impute": round(im, 4),
                                    "graph_gain": round(gm - sm, 4)})
    import json
    print("[natureserve] " + json.dumps(summary))
    print(f"  [profile] {len(a.seeds)}s x {len(a.sources)}src x {len(variants)}v in {time.time()-t0:.1f}s")
    return summary


_USDA_ORDINAL = {
    "Drought Tolerance": {"None": 0, "Low": 1, "Medium": 2, "High": 3},
    "Growth Rate": {"Slow": 0, "Moderate": 1, "Rapid": 2},
    "Moisture Use": {"Low": 0, "Medium": 1, "High": 2},
}
def _usda_trait(cache, gidx, dev, col):
    """Load one USDA husbandry ordinal trait (Drought Tolerance / Growth Rate / Moisture Use) aligned via gidx.
    Returns (y int64 [N], obs bool [N], nclass). Small n (~400 covered) -> report as caveat."""
    import csv as _csv
    mp = _USDA_ORDINAL[col]
    val = {}
    with open(Path(cache) / "derived/usda_plants_traits.csv") as f:
        r = _csv.DictReader(f)
        for row in r:
            v = (row.get(col) or "").strip()
            if v in mp:
                val[int(row["global_idx"])] = mp[v]
    N = len(gidx)
    y = np.zeros(N, dtype=np.int64); obs = np.zeros(N, dtype=bool)
    for i, g_idx in enumerate(gidx):
        if int(g_idx) in val:
            obs[i] = True; y[i] = val[int(g_idx)]
    return torch.tensor(y).to(dev), torch.tensor(obs).to(dev), len(mp)


def run_usda(a, dev):
    """RETEST (task 4). USDA husbandry traits (Drought Tolerance / Growth Rate / Moisture Use, ~400 covered) as
    phylo targets. Per trait: seed_only vs unsup-graph vs trait-SUPERVISED graph, categorical accuracy on
    held-out species, graph_gain = graph - seed. Does the graph serve real husbandry traits from relatives?
    SMALL-n caveat (report obs count). Reuses train_graph_trait_supervised; edits nothing in core/."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    tip_row = tip_row.to(dev)
    N = E1.shape[0]
    cols = a.usda_cols if a.usda_cols else list(_USDA_ORDINAL.keys())
    variants = ["unsup", "sup"]
    import json
    all_sum = []
    t0 = time.time()
    for col in cols:
        y, obs, nclass = _usda_trait(a.cache_dir, gidx, dev, col)
        n_obs = int(obs.sum().item())
        scorer = lambda emb, test: nn_trait_acc(emb, y, obs, test)
        agg = {src: {v: {"seed": [], "graph": []} for v in variants} for src in a.sources}
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
                    else:
                        s, rep, imp, _ = train_graph_trait_supervised(
                            seed, tree, tip_row, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
                            a.impute_steps, "cat", y, obs, nclass, operator=a.operator,
                            phylo_dist=(SpeciesGraph.distance_from_embedding(F.normalize(E1.to(dev), dim=-1))
                                        if a.operator == "ou-attention" else None),
                            blanket_k=a.blanket_k)
                    agg[src][v]["seed"].append(scorer(s, test))
                    agg[src][v]["graph"].append(scorer(rep, test))

        def m(xs):
            return float(np.mean(xs)), float(np.std(xs))
        yo = y[obs].cpu().numpy()
        floor = float(np.bincount(yo, minlength=nclass).max() / max(1, len(yo)))
        print(f"=== BIOLOGICAL USDA '{col}' | metric=acc classes={nclass} sources={a.sources} "
              f"variants={variants} seeds={a.seeds} N={N} obs={n_obs} majority-floor={floor:.4f} (SMALL-n) ===")
        print(f"  {'source':7s} {'var':5s} | {'seed':>16s} | {'GRAPH':>16s} | graph_gain")
        summary = {"axis": f"usda_{col.replace(' ', '_')}", "metric": "acc", "nclass": nclass,
                   "seeds": a.seeds, "N": N, "obs": n_obs, "majority_floor": round(floor, 4), "rows": []}
        for src in a.sources:
            for v in variants:
                sm, ss = m(agg[src][v]["seed"]); gm, gs = m(agg[src][v]["graph"])
                print(f"  {src:7s} {v:5s} | {sm:.4f}+/-{ss:.4f} | {gm:.4f}+/-{gs:.4f} | {gm-sm:+.4f}")
                summary["rows"].append({"source": src, "variant": v, "seed": round(sm, 4),
                                        "graph": round(gm, 4), "graph_std": round(gs, 4),
                                        "graph_gain": round(gm - sm, 4)})
        print(f"[usda_{col.replace(' ', '_')}] " + json.dumps(summary))
        all_sum.append(summary)
    print(f"  [profile] usda {len(cols)}cols x {len(a.seeds)}s x {len(a.sources)}src in {time.time()-t0:.1f}s")
    return all_sum


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="autoresearch/data/deepcal")
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
    ap.add_argument("--blanket_k", type=int, default=0,
                    help="rule-29 Markov-blanket restriction (ou-attention only): restrict each species' "
                         "reconstruction to its k NEAREST phylo relatives (top_k over phylo_distance). 0=whole-clade dense.")
    ap.add_argument("--mask_curriculum", default=None,
                    help="rule-25 mask curriculum: 'easy2hard:LO:HI' ramps mask_frac LO->HI over training, "
                         "'hard2easy:LO:HI' reverses. e.g. easy2hard:0.1:0.5. None=fixed --mask_frac. trait_supervised only.")
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
    ap.add_argument("--recon_relative", action="store_true",
                    help="RULE-25 objective (OOT): reconstruct masked TRAIN species toward the mean seed of their "
                         "same-family train relatives (leak-free imputation target) instead of the identity copy of "
                         "their own seed. Fixes the weak identity-recon in-tree path. Default off (byte-identical).")
    ap.add_argument("--oot_heads", type=int, default=4,
                    help="RULE-29 operator (OOT): number of clade cross-attention heads for the out-of-tree "
                         "soft-attach (default 4). More heads -> each addresses a different clade subspace when "
                         "projecting an out-of-tree species onto the shared clade latents.")
    ap.add_argument("--oot_layers", type=int, default=2,
                    help="RULE-29 operator (OOT): tree message-passing depth for the OOT graph (default 2).")
    ap.add_argument("--recon_k", type=int, default=0,
                    help="RULE-25 (with --recon_relative): >0 -> relative target = mean seed of the k NEAREST "
                         "same-family TRAIN relatives (rank-preserving local target) instead of the family centroid "
                         "(0, ITER-1, which erased within-family rank). Leak-free: never self, train tips only.")
    ap.add_argument("--oot_blend", action="store_true",
                    help="ITER-7 (num_resid_readout num axis): also train the IN-TREE residual head and report the "
                         "best convex blend alpha*oot_head_pred + (1-alpha)*intree_head_pred over a sweep. OOT wins "
                         "on clean-seed axes, in-tree on tree-smoothable axes -> the blend should dominate both.")
    ap.add_argument("--cooccur", action="store_true",
                    help="rule 10-12 COMMUNITY axis: predict a species' co-occurrence partner set from relatives (graph-on vs off)")
    ap.add_argument("--cooccur_res", default="005", help="grid-res tag of derived/cooccur_count_<tag>.npy (005=0.05deg)")
    ap.add_argument("--cooccur_topk", type=int, default=64, help="keep each species' top-K strongest co-partners as the target signature")
    ap.add_argument("--cooccur_supervised", action="store_true",
                    help="TEST1: community-SUPERVISED graph (BCE on masked-species partner-set reconstruction) vs unsup vs seed")
    ap.add_argument("--myco_supervised", action="store_true",
                    help="TEST2: myco (B63) trait-supervised imputation -- seed vs unsup-graph vs CE-supervised graph")
    ap.add_argument("--route_contrast", action="store_true",
                    help="TEST3: seed-only vision-text gap on community vs lep (routing contrast)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--globi_guild", action="store_true",
                    help="RETEST: REAL GloBI pollinator-guild signature (dominant-guild acc + guild-frac AP) as a phylo target")
    ap.add_argument("--globi_min_partners", type=int, default=3,
                    help="GloBI: min RECOGNIZED-guild partner records for a species to count as observed")
    ap.add_argument("--natureserve", action="store_true",
                    help="RETEST: NatureServe G-rank rarity (binary at-risk G1-3 vs secure) balanced-acc; graph vs seed")
    ap.add_argument("--natureserve_ordinal", action="store_true",
                    help="NatureServe: ordinal G1..G5 (5-class) instead of the binary at-risk target")
    ap.add_argument("--usda", action="store_true",
                    help="RETEST: USDA husbandry traits (Drought Tolerance/Growth Rate/Moisture Use, ~400 sp) graph vs seed")
    ap.add_argument("--usda_cols", nargs="+", default=None,
                    help="USDA: subset of ordinal columns to test (default all three)")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"
    if isinstance(a.mask_curriculum, str) and a.mask_curriculum:   # 'easy2hard:LO:HI' -> (sched, lo, hi) tuple
        _p = a.mask_curriculum.split(":")
        a.mask_curriculum = (_p[0], float(_p[1]), float(_p[2]))

    if a.cooccur_supervised:
        return run_cooccur_supervised(a, dev)
    if a.globi_guild:
        return run_globi_guild(a, dev)
    if a.natureserve:
        return run_natureserve(a, dev)
    if a.usda:
        return run_usda(a, dev)
    if a.myco_supervised:
        return run_myco_supervised(a, dev)
    if a.route_contrast:
        return run_route_contrast(a, dev)
    if a.cooccur:
        return run_cooccur(a, dev)
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
