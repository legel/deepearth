"""Split v2: more methane validation towers, test towers unchanged.

The frozen split (split.csv) has 5 validation towers that measure FCH4, against 21 test and 95 training towers, and
its validation methane towers are 4 wetlands and 1 cropland while the test methane towers span wetlands, needleleaf
forests, grasslands, a shrubland and a lake. Split v2 moves whole training blocks (the spatial blocks of split.py, so
no validation tower shares a block with a training tower) to validation until the validation methane towers follow
the test methane mix.

Rule, fixed before any model is scored on the moved towers:
1. Goal: about 20 validation methane towers in the IGBP proportions of the test methane towers.
2. Training blocks with at least one methane tower are visited in a seeded random order (numpy default_rng(20261030)).
3. A block is moved when it has at most 6 towers (large blocks stay in training) and adding its methane towers keeps
   every IGBP count at or below the goal.
4. Test towers and the frozen split.csv are unchanged (the test release manifest records split.csv's checksum).

Usage: python split_v2.py <towers_dir>   (reads split.csv; writes split_v2.csv and split_v2.json)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED, GOAL_N, MAX_BLOCK = 20261030, 20, 6


def main():
    d = Path(sys.argv[1])
    s = pd.read_csv(d / "split.csv")
    c = s[s.has_fch4]
    target = c[c.split == "test"].igbp.value_counts()
    goal = (target / target.sum() * GOAL_N).round().astype(int)
    cur = c[c.split == "val"].igbp.value_counts().to_dict()
    info = {b: (len(g), g[g.has_fch4].igbp.value_counts().to_dict())
            for b, g in s[s.split == "train"].groupby("block") if g.has_fch4.any()}
    order = list(info)
    np.random.default_rng(SEED).shuffle(order)
    moved = []
    for b in order:
        n, mix = info[b]
        if n > MAX_BLOCK or any(cur.get(k, 0) + v > goal.get(k, 0) for k, v in mix.items()):
            continue
        for k, v in mix.items():
            cur[k] = cur.get(k, 0) + v
        moved.append(b)
    s2 = s.copy()
    s2.loc[s2.block.isin(moved) & (s2.split == "train"), "split"] = "val"
    assert s2[s2.split == "test"].site_id.tolist() == s[s.split == "test"].site_id.tolist()
    s2.to_csv(d / "split_v2.csv", index=False)
    counts = {k: {str(b): int(v) for b, v in g.items()} for k, g in s2.groupby("split").has_fch4.value_counts().unstack().iterrows()}
    doc = {"base": "split.csv", "base_sha256": hashlib.sha256((d / "split.csv").read_bytes()).hexdigest(),
           "rule": __doc__.split("\n\n")[2], "seed": SEED, "goal": {k: int(v) for k, v in goal.items()},
           "moved_blocks": sorted(int(b) for b in moved), "moved_sites": sorted(s[s.block.isin(moved)].site_id),
           "counts": counts}
    (d / "split_v2.json").write_text(json.dumps(doc, indent=1))
    print(f"moved {len(moved)} blocks ({len(doc['moved_sites'])} towers); methane towers {counts}")


if __name__ == "__main__":
    main()
