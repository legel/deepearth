"""Species SEED construction — the rule-26 lever.

What a species embedding starts as, BEFORE the phylogenetic operator touches it. This is the science
`program.md` names as the loop's root cause: the champion E1 text prior is itself phylogeny-derived
(~0.89 family-NN on its own), so the tree can only re-derive what the seed already carries, and
`bio_gain` sits at ~0 by redundancy rather than by the operator failing. Every source here is an
attempt to hand the operator a seed it can actually ADD to.

Sources (`build_seed`): text · vision · fused · fused_white · fused_zca · routed · text_hipass{P} ·
text_bandstop{A}_{B} · routed_hipass{P}. The hipass variants SVD out the top-P principal components of
E1 on the thesis that those PCs are exactly what the tree would otherwise re-encode.

EDITABLE. Nothing in here decides what a number means -- it decides what the encoder is given.
"""
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
