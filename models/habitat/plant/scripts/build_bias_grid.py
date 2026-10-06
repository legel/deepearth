#!/usr/bin/env python3
"""Sampling-bias grid (Daru 2024 step 5; ranges/background.py): the kernel density of all vascular-plant records
(GBIF map counts per ~9 km cell, fetch_gbif_effort.py), spatialEco::sp.kde's Gaussian kernel with Silverman's
bandwidth, on a 10 km Behrmann equal-area grid, standardized to 0..1. Background points are drawn in proportion to it.

usage: build_bias_grid.py [--config configs/conus.json] [--counts CSV] [--out NPZ]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import background, config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--counts")
    ap.add_argument("--out")
    ap.add_argument("--cell-km", type=float, default=10.0)
    a = ap.parse_args()
    cfg = config.load(a.config)
    ps = cfg["per_species"]
    g = background.bias_grid(pd.read_csv(a.counts or cfg.path(ps["effort_counts"])), a.cell_km)
    g["grid"] = g["grid"].astype(np.float32)                   # stored in single precision
    out = Path(a.out) if a.out else cfg.path(ps["bias"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **g)
    print(f"{g['grid'].shape} cells of {a.cell_km:g} km; bandwidth {g['bandwidth_m'][0] / 1e3:.1f} x "
          f"{g['bandwidth_m'][1] / 1e3:.1f} km; {g['n_records']:,} records")


if __name__ == "__main__":
    main()
