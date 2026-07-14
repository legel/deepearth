"""Regenerate derived/patristic_ref.npy — the reference cophenetic (branch-length path) distance matrix over the
dated tree's tips, indexed by global_idx (species_index row). It is the self-check reference for
phylogenomic._test_real_tree (which confirms the pruned/compact tree preserves these distances).

Reuses the model's OWN tree parser (build_tree_buffers) + the same rootdist/LCA math, so ordering and the unit-mean
branch rescaling match exactly. Validated against the shipped matrix (asserts max abs err < 1e-3) before writing.

    python -m deepearth.data.deepcal.build_patristic_ref            # cache = data/deepcal (DEEPCAL_DATA_DIR override)
"""
import os, csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))       # parent of the `deepearth` package -> importable
from deepearth.encoders.biological.phylogenomic import build_tree_buffers


def cophenetic(cparent, cblen, nsp, n):
    """All-pairs cophenetic over the nsp tips via one post-order sweep (each pair filled once, at its LCA)."""
    depth = np.zeros(n, int); changed = True
    while changed:                                                 # depth by relaxation (parents shallower than children)
        changed = False
        for c in range(n):
            if cparent[c] >= 0 and depth[c] != depth[cparent[c]] + 1:
                depth[c] = depth[cparent[c]] + 1; changed = True
    rootdist = np.zeros(n)
    for c in np.argsort(depth):
        if cparent[c] >= 0: rootdist[c] = rootdist[cparent[c]] + cblen[c]
    kids = defaultdict(list)
    for c in range(n):
        if cparent[c] >= 0: kids[cparent[c]].append(c)
    tips = [[c] if c < nsp else [] for c in range(n)]             # species indices < nsp are tips
    M = np.zeros((nsp, nsp), np.float32)
    for u in np.argsort(-depth):                                  # deepest first -> children ready before their parent
        ch = kids[u]
        for a in range(len(ch)):
            ia = np.array(tips[ch[a]], int)
            for b in range(a):                                    # pairs across two different children first meet at u
                jb = np.array(tips[ch[b]], int)
                blk = rootdist[ia][:, None] + rootdist[jb][None, :] - 2 * rootdist[u]
                M[np.ix_(ia, jb)] = blk; M[np.ix_(jb, ia)] = blk.T
        if u >= nsp:
            tips[u] = [t for c in ch for t in tips[c]]
    np.fill_diagonal(M, 0.0)
    return M


def main():
    cache = Path(os.environ.get("DEEPCAL_DATA_DIR", Path(__file__).resolve().parent))
    rows = list(csv.DictReader(open(cache / "derived/species_index.csv")))
    import re
    tree_toks = set(re.findall(r"[^(),:;\s]+", open(cache / "ca_subtree.dated.nwk").read()))
    tree_rows = [r for r in rows if r["tip_label"] in tree_toks]           # tree tips, in species_index (idx) order
    tips = [r["tip_label"] for r in tree_rows]
    gidx = np.array([int(r["idx"]) for r in tree_rows])                    # each tree tip's global_idx (species_index row)
    print(f"{len(rows)} species_index rows -> {len(tips)} tree tips (global_idx 0..{int(gidx.max())})", flush=True)
    b = build_tree_buffers(str(cache / "ca_subtree.dated.nwk"), tips)      # species position k <-> tips[k] <-> gidx[k]
    n, nsp = b["n_nodes"], b["n_species"]
    cparent = np.full(n, -1, int); cblen = np.zeros(n)
    for c, p, bl in zip(b["down_child"], b["down_parent"], b["down_blen"]):
        cparent[c] = p; cblen[c] = bl * b["branch_scale"]                 # undo the unit-mean scaling -> Myr
    Mpos = cophenetic(cparent, cblen, nsp, n)                             # [nsp,nsp] indexed by tip position
    M = np.zeros((int(gidx.max()) + 1,) * 2, np.float32)                  # reindex to global_idx (like the shipped matrix)
    M[np.ix_(gidx, gidx)] = Mpos
    print(f"built cophenetic {M.shape}", flush=True)

    ref = cache / "derived/patristic_ref.npy"
    if ref.exists():                                                      # audit vs the shipped matrix before overwriting
        D = np.load(ref)
        err = float(np.abs(M[:D.shape[0], :D.shape[1]] - D).max()) if D.shape == M.shape else float("nan")
        print(f"audit vs shipped patristic_ref {D.shape}: max abs err {err:.3g}", flush=True)
        assert err < 1e-3, f"builder drifted from the shipped reference (max err {err:.3g})"
    np.save(ref, M)
    print(f"wrote {ref}", flush=True)


if __name__ == "__main__":
    main()
