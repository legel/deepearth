"""Assemble the pollinator distance from DATED clade patristics + the topological/text backbone. For each clade with
a published chronogram (bees, ants, butterflies, hawkmoths — see pollinator_dated_patristic.R), replace its
within-clade block of pollinator_distance.npy with the clade's dated patristic (cophenetic in Myr), RESCALED so the
block's mean matches the existing (topological) block — this injects real dated relative-structure (congeners weighted
by divergence *time*, not just edge count) while preserving the global scale that the cross-clade/uncovered entries
(OpenTree topology + BioCLIP text shadow) already set. The topological approximation is thus replaced where we have
gold-standard dated trees. Backs up the topological matrix first.
"""
import numpy as np, csv, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
COPHEN = Path("/home/photon/4tb/deepcal_data/trees/dated_cophen")
DIST = HERE / "pollinator_distance.npy"
CLADES = ["bees", "ants", "butterflies", "moths", "birds"]


def binom_to_index():
    m = {}
    for r in csv.reader(open(HERE / "pollinator" / "pollinator_vocab.csv")):
        if r and r[0].isdigit():
            m[r[1].strip().lower().replace(" ", "_")] = int(r[0])
    return m


def main():
    b2i = binom_to_index()
    topo = HERE / "pollinator_distance_topo.npy"
    if not topo.exists():                                             # preserve the ORIGINAL topological+text matrix once
        np.save(topo, np.load(DIST).astype(np.float32))
    D = np.load(topo).astype(np.float64)                              # always rebuild from the topological original (idempotent)
    print(f"distance {D.shape}, vocab-mapped binomials {len(b2i)}")
    total_overwritten = 0
    for clade in CLADES:
        f = COPHEN / f"{clade}_cophen.csv"
        if not f.exists():
            print(f"  {clade}: no cophen, skip"); continue
        rows = list(csv.reader(open(f)))
        labels = [c.strip().lower() for c in rows[0][1:]]              # header: "",tip1,tip2,...
        M = np.array([[float(x) for x in r[1:]] for r in rows[1:]], np.float64)   # dated patristic (Myr)
        idx = np.array([b2i.get(l, -1) for l in labels])
        keep = idx >= 0
        if keep.sum() < 3:
            print(f"  {clade}: <3 mapped, skip"); continue
        sub = np.where(keep)[0]; gi = idx[keep]                        # rows/cols to use; their global indices
        Msub = M[np.ix_(sub, sub)]
        cur = D[np.ix_(gi, gi)]                                        # current (topological) block
        cm, dm = cur[~np.eye(len(gi), dtype=bool)].mean(), Msub[~np.eye(len(sub), dtype=bool)].mean()
        Msub = Msub * (cm / dm) if dm > 0 else Msub                   # rescale dated block -> match current block mean (scale-consistent)
        np.fill_diagonal(Msub, 0.0)
        D[np.ix_(gi, gi)] = Msub                                       # overwrite within-clade with dated structure
        total_overwritten += len(gi)
        print(f"  {clade}: {len(gi)} taxa dated | block mean {cm:.3f} kept | congener range [{Msub[Msub>0].min():.4f}, {Msub.max():.3f}]")
    D = 0.5 * (D + D.T)                                                # enforce symmetry
    D = np.clip(D, 0.0, None)                                          # clip tiny pre-existing float noise (~-1e-7)
    np.save(DIST, D.astype(np.float32))
    print(f"wrote {DIST} ({total_overwritten} taxa now on dated patristics; topological backup -> pollinator_distance_topo.npy)")


if __name__ == "__main__":
    main()
