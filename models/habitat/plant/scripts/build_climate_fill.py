#!/usr/bin/env python3
"""The climate fill of a 240 m region grid (ranges/joint/climate_fill.py): every land cell without climate takes
all 20 WorldClim bands of its nearest cell with climate within shoreline.max_km (exact distance on the equal-area
grid). The map store's GridInputs applies it, so shoreline cells are mapped like every other land cell.

Land = inside the region's mask (shoreline.land_mask) and under shoreline.max_water water by NALCMS 2020 at 30 m
(shoreline.nalcms_30m, warped to the 30 m grid aligned with the 240 m grid): the administrative mask follows state
boundaries ~3 nautical miles offshore, so the water test keeps coastal sea out of the fill.

usage: build_climate_fill.py [--config configs/conus.json] [--region conus]
Writes the configured shoreline.grid_fill[<region>] (e.g. <grid>/climate_fill_conus240.npz).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint import climate_fill  # noqa: E402
from ranges.predictors import ConusStack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--region", default="conus")
    a = ap.parse_args()
    cfg = config.load(a.config)
    P, sh = cfg.path, cfg["shoreline"]
    grid = P(cfg["store"]["grids"][a.region])
    shape = ConusStack(grid).shape
    water = climate_fill.water_fraction(P(sh["nalcms_30m"]), shape)
    land = np.load(P(sh["land_mask"][a.region])).astype(bool) & np.isfinite(water) & \
        (np.nan_to_num(water, nan=1.0) < sh["max_water"])
    del water
    climate_fill.build_grid_fill(grid, land, P(sh["grid_fill"][a.region]), max_km=sh["max_km"],
                                 log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
