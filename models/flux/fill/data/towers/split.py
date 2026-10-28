"""The frozen 80/20 site split, spatially blocked and stratified, with a validation subset inside the training sites.

1. Blocks. Included sites are linked when they lie within BLOCK_KM of each other; each connected group (single
   linkage) is one block. Whole blocks go to one side, so no test site lies within BLOCK_KM of a training site.
2. Strata. Each site's stratum is (IGBP group, Koppen main class, measures FCH4). Blocks are visited in a seeded
   random order; a block joins the test set when that brings the test share of its sites' strata closer to
   TEST_FRACTION and the test set is not yet full. Methane sites are a stratum dimension, so the FCH4 subset splits
   in the same proportion and by the same blocks.
3. Validation. The same procedure, seeded separately, takes VAL_FRACTION of the training sites (by block) as the
   validation set used for all model selection. Test sites are never used for tuning.

    python split.py --out /home/photon/4tb/fluxfill/towers

Writes <out>/split.csv (site_id, split, block, stratum) and split.json (parameters, counts, checksum). An existing
split.json is never overwritten: the split is frozen (pass --force only to rebuild a dataset release).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from tables import haversine_km

SEED = 20261026
BLOCK_KM = 50.0
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
"""Of the training sites (8 % of all sites)."""
IGBP_GROUP = {"ENF": "forest_needle", "DNF": "forest_needle", "EBF": "forest_broad", "DBF": "forest_broad",
              "MF": "forest_mixed", "CSH": "shrub_savanna", "OSH": "shrub_savanna", "WSA": "shrub_savanna",
              "SAV": "shrub_savanna", "GRA": "grassland", "CRO": "cropland", "CVM": "cropland", "WET": "wetland"}
"""IGBP classes grouped so that each stratum holds enough sites to split; the rest (URB, WAT, BSV, SNO) are
"other"."""


def blocks(lat: np.ndarray, lon: np.ndarray, km: float = BLOCK_KM) -> np.ndarray:
    d = haversine_km(lat[:, None], lon[:, None], lat[None], lon[None])
    return connected_components(csr_matrix(d < km), directed=False)[1]


def assign(strata: pd.Series, block: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean selection of whole blocks, about `fraction` of the sites, balanced over strata."""
    rng = np.random.default_rng(seed)
    target = strata.value_counts() * fraction
    have = pd.Series(0.0, index=target.index)
    chosen = np.zeros(len(strata), bool)
    cap = fraction * len(strata)
    ids = np.unique(block)
    for b in rng.permutation(ids):
        members = np.flatnonzero(block == b)
        if chosen.sum() + members.size > cap + 0.5:
            continue
        add = strata.iloc[members].value_counts()
        before = ((have[add.index] - target[add.index]) ** 2).sum()
        after = ((have[add.index] + add - target[add.index]) ** 2).sum()
        if after < before:
            chosen[members] = True
            have[add.index] += add
    # Strata of one or two sites never gain from a test member above; fill the remaining test share with whole
    # blocks in a second seeded pass, smallest blocks first so the share lands close to `fraction`.
    rest = [b for b in rng.permutation(ids) if not chosen[block == b].any()]
    rest.sort(key=lambda b: int((block == b).sum()))
    for b in rest:
        members = np.flatnonzero(block == b)
        if chosen.sum() + members.size > cap + 0.5:
            break
        chosen[members] = True
    return chosen


def make_split(sites: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    s = sites[sites.included].reset_index(drop=True).copy()
    s["igbp_group"] = s.igbp.map(IGBP_GROUP).fillna("other")
    s["stratum"] = (s.igbp_group + "|" + s.koppen_main.fillna("?") + "|"
                    + np.where(s.has_fch4, "ch4", "noch4"))
    s["block"] = blocks(s.lat.values.astype(float), s.lon.values.astype(float))
    test = assign(s.stratum, s.block.values, TEST_FRACTION, seed)
    s["split"] = np.where(test, "test", "train")
    tr = np.flatnonzero(~test)
    val = assign(s.stratum.iloc[tr].reset_index(drop=True), s.block.values[tr], VAL_FRACTION, seed + 1)
    s.loc[tr[val], "split"] = "val"
    return s[["site_id", "split", "block", "stratum", "igbp", "koppen_main", "has_fch4", "lat", "lon"]]


def check(split: pd.DataFrame, km: float = BLOCK_KM) -> float:
    """The smallest distance (km) from any test site to any training or validation site."""
    te = split[split.split == "test"]
    tr = split[split.split != "test"]
    if te.empty or tr.empty:
        return float("inf")
    d = haversine_km(te.lat.values[:, None], te.lon.values[:, None], tr.lat.values[None], tr.lon.values[None])
    return float(d.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if (a.out / "split.json").exists() and not a.force:
        raise SystemExit("split.json exists: the split is frozen")
    sites = pd.read_parquet(a.out / "sites.parquet")
    sp = make_split(sites)
    gap = check(sp)
    assert gap >= BLOCK_KM, gap
    sp.to_csv(a.out / "split.csv", index=False)
    digest = hashlib.sha256((a.out / "split.csv").read_bytes()).hexdigest()
    info = {"seed": SEED, "block_km": BLOCK_KM, "test_fraction": TEST_FRACTION, "val_fraction_of_train": VAL_FRACTION,
            "counts": sp.split.value_counts().to_dict(),
            "fch4_counts": sp[sp.has_fch4].split.value_counts().to_dict(),
            "min_test_to_train_km": round(gap, 2), "blocks": int(sp.block.nunique()), "sha256_split_csv": digest}
    (a.out / "split.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
