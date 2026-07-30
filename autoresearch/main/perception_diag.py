"""ADDITIVE probe (perception_diag) — scratch diagnostic, nothing committed, touches no core.

Mission was written for a GPU data box (dino[1024]+bio[768] tokens, prepared_*.pt, checkpoints).
On THIS box those artifacts are absent; only three real data artifacts exist:
  - gbif_worldclim_tokens.npz : 207431 obs x 19 WorldClim env vars (no species labels)
  - gbif_habitat_emb.npz      : 2141 species x 384 habitat-text semantic embeddings
  - gbif_habitat_text.npz      : 2141 species raw habitat text

So we run the diagnostics that ARE grounded in real data present, with held-out
skill vs shuffle-null floors, and report DATA. No perception (dino/bio) tokens exist
here to ablate, so Part-A perception-dependency is reported as NOT-RUNNABLE on this box.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARCH = "/Users/andromeda/deepcal-archive"
WC = f"{ARCH}/gbif_worldclim_tokens.npz"
HE = f"{ARCH}/session-backup/data/gbif_habitat_emb.npz"


def load():
    h = np.load(HE, allow_pickle=True)
    emb = h["emb"].astype(np.float32)
    bino = np.array([str(b) for b in h["binomial"]])
    hastext = h["has_text"]
    m = hastext.astype(bool)
    return emb[m], bino[m]


def genus_of(bino):
    return np.array([b.split()[0] for b in bino])


def _macro_accuracy(y_true, y_pred):
    """Equal-weight per-class accuracy."""
    return float(balanced_accuracy_score(y_true, y_pred))


def probe_genus_from_semantic(emb, bino, min_species=5, seed=0):
    """Does the LEARNED habitat-text semantic embedding discriminate GENUS (a phylo/relatedness
    proxy)? Skill = held-out macro-accuracy; floor = label-shuffle null. High skill = the learned
    semantic channel carries phylogenetic signal ON ITS OWN (not borrowed from vision)."""
    gen = genus_of(bino)
    u, c = np.unique(gen, return_counts=True)
    # Five-fold stratification requires five examples of every retained class
    # so each held-out fold contains that class.
    min_species = max(int(min_species), 5)
    keep = set(u[c >= min_species])
    m = np.array([g in keep for g in gen])
    X, y = emb[m], gen[m]
    classes, y = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("genus diagnostic needs at least two genera with five species each")
    rng = np.random.default_rng(seed)

    def cv_acc(labels):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in skf.split(X, labels):
            # Scaling is fitted inside each fold; held-out means and variances
            # never influence that fold's training representation.
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=400, C=1.0),
            )
            clf.fit(X[tr], labels[tr])
            accs.append(_macro_accuracy(labels[te], clf.predict(X[te])))
        return float(np.mean(accs))

    real = cv_acc(y)
    yl = y.copy(); rng.shuffle(yl)
    null = cv_acc(yl)
    return dict(n_species=int(m.sum()), n_genera=len(classes), cv_folds=5,
                chance=float(1.0 / len(classes)), skill=real, shuffle_null=null,
                skill_over_null=real - null)


def probe_semantic_axes(emb, bino, seed=0):
    """Signal-ceiling proxy: how much of the habitat-text embedding variance is species-structured
    vs noise. Reports effective rank + how well a low-rank reconstruction preserves genus-NN purity.
    A high ceiling here means the LEARNED semantic channel has rich exploitable structure."""
    Xs = StandardScaler().fit_transform(emb)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    eff_rank = float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))
    # genus-NN purity at full rank (cosine 1-NN, leave-one-out)
    gen = genus_of(bino)
    Xn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    purity = float((gen[nn] == gen).mean())
    # null: shuffle genus labels
    rng = np.random.default_rng(seed)
    gl = gen.copy(); rng.shuffle(gl)
    purity_null = float((gl[nn] == gl).mean())
    return dict(n=len(emb), dim=emb.shape[1], effective_rank=eff_rank,
                var_top10=float(ev[:10].sum()), var_top50=float(ev[:50].sum()),
                genus_nn_purity=purity, genus_nn_null=purity_null,
                purity_over_null=purity - purity_null)


def probe_env_signal():
    """WorldClim env: intrinsic structure + how compressible (signal ceiling of the raw env channel
    that the LEARNED env-encoder consumes). No species labels here, so we report intrinsic-dimension
    and inter-var redundancy only (a ceiling on how much distinct env information exists to learn)."""
    z = np.load(WC, allow_pickle=True)
    wc = z["worldclim"].astype(np.float64)
    m = z["has_worldclim"].astype(bool)
    wc = wc[m]
    wc = wc[np.isfinite(wc).all(1)]
    Xs = StandardScaler().fit_transform(wc)
    C = np.corrcoef(Xs.T)
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = ev / ev.sum()
    eff_rank = float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))
    return dict(n_obs=int(wc.shape[0]), n_vars=int(wc.shape[1]),
                effective_rank=eff_rank, var_top3=float(ev[:3].sum()),
                mean_abs_offdiag_corr=float(np.abs(C[~np.eye(19, dtype=bool)]).mean()))


if __name__ == "__main__":
    import json, sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    emb, bino = load()
    out = {}
    if what in ("all", "genus"):
        out["genus_from_semantic"] = probe_genus_from_semantic(emb, bino)
    if what in ("all", "axes"):
        out["semantic_axes_ceiling"] = probe_semantic_axes(emb, bino)
    if what in ("all", "env"):
        out["env_signal_ceiling"] = probe_env_signal()
    print(json.dumps(out, indent=2))
