"""Build the PLANT species-graph DATED distance from ca_subtree.dated.nwk (rules 7-12), replacing the crude
distance_from_embedding shadow the ou-attention graph used for ALL species (fusion.py). ~65% of model species are
tree tips -> real dated cophenetic (Myr); the ~35% inductively-placed species (rules 25/26) keep the embedding shadow,
filled in fusion.py at model-build time. Output gbif_plant_dist.npz {dated[m,m] float32, model_idx[m] int64} where
model_idx are the vocab-order indices of the tree-covered species and dated is their cophenetic. Mirrors the pollinator
dated pipeline (pollinator_dated_distance.py). Reproducible: cophenetic is deterministic given the versioned tree.

    Rscript deepearth/data/deepcal/plant_dated_patristic.R   # is invoked by this script
    python -m deepearth.data.deepcal.plant_dated_distance
"""
import numpy as np, csv, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE
DERIVED = CACHE / "derived"
RSCRIPT = "/home/photon/miniconda3/envs/rphylo/bin/Rscript"


def main():
    vocab = np.load(CACHE / "gbif_vocab.npz", allow_pickle=True)
    gi = vocab["global_idx"]
    rows = list(csv.DictReader(open(DERIVED / "species_index.csv")))
    tips = [rows[g]["tip_label"] for g in gi]                          # model species tip_labels in vocab order
    # clean reproducible intermediate the R builder consumes
    with open(DERIVED / "model_species.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model_idx", "tip_label"])
        for i, t in enumerate(tips): w.writerow([i, t])
    # cophenetic in R (ape) over the tree-covered model species
    r = subprocess.run([RSCRIPT, "deepearth/data/deepcal/plant_dated_patristic.R"],
                       cwd=str(HERE.parents[2]), capture_output=True, text=True)
    sys.stdout.write(r.stdout);  sys.stderr.write(r.stderr)
    if "PLANT_DATED_PATRISTIC_DONE" not in r.stdout:
        raise SystemExit("R cophenetic failed")
    # read the dated cophenetic (tip_label-labelled) and map back to model indices
    rr = list(csv.reader(open(DERIVED / "plant_cophen.csv")))
    labels = [c.strip() for c in rr[0][1:]]
    M = np.array([[float(x) for x in row[1:]] for row in rr[1:]], np.float64)
    t2i = {t: i for i, t in enumerate(tips)}
    midx = np.array([t2i[l] for l in labels], np.int64)               # vocab-order index of each covered species
    assert M.shape[0] == M.shape[1] == len(midx)
    assert np.allclose(M, M.T, atol=1e-4) and np.allclose(np.diag(M), 0)
    np.savez(CACHE / "gbif_plant_dist.npz", dated=M.astype(np.float32), model_idx=midx)
    print(f"wrote {CACHE/'gbif_plant_dist.npz'}: {len(midx)}/{len(tips)} model species on real dated cophenetic "
          f"(mean {M[~np.eye(len(M),dtype=bool)].mean():.1f} Myr); {len(tips)-len(midx)} keep the embedding shadow")


if __name__ == "__main__":
    main()
