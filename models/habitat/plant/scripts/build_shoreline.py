#!/usr/bin/env python3
"""Shoreline locations take the nearest climate (ranges/joint/climate_fill.py, ranges/joint/shoreline.py).

WorldClim has no value where a 30" pixel centre lies in the sea or a large lake, so shoreline plots could not be
scored and shoreline presences were dropped by the per-species preparation. This script, for the configured joint
data directory:
  1. fills every plot set (VegBank, AIM, FIA): <data dir>/plot_fill_<source>.npz (filled predictors, fill distance);
  2. restores the dropped presences: <data dir>/shore_records.npz (replayed seeded draw, nearest climate within
     shoreline.max_km, SoilGrids at the point, position on the 240 m grid).

usage: build_shoreline.py [--config configs/conus.json] [--data-dir D] [--plots-only | --records-only] [--workers 12]
"""
import argparse
import sys
from pathlib import Path

import pyproj  # noqa: F401,I001  (before rasterio: its bundled PROJ can break pyproj's transforms)
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint import climate_fill, shoreline  # noqa: E402
from ranges.joint.reader import ECOREGION_RASTER  # noqa: E402
from ranges.predictors import GlobalStack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    g_ = ap.add_mutually_exclusive_group()
    g_.add_argument("--plots-only", action="store_true")
    g_.add_argument("--records-only", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = config.load(a.config)
    P, g, sh = cfg.path, cfg["geo"], cfg["shoreline"]
    data_dir = Path(a.data_dir) if a.data_dir else P(cfg["joint"]["data_dir"])
    log = lambda m: print(m, flush=True)  # noqa: E731
    if not a.records_only:
        climate_fill.fill_plot_sets(data_dir, GlobalStack(P(g["worldclim_stack"])).data, max_km=sh["max_km"], log=log)
    if not a.plots_only:
        grid = P(cfg["store"]["grids"][cfg["scope"]["region"].lower()])
        with rasterio.open(grid / ECOREGION_RASTER) as r:
            transform = tuple(r.transform)[:6]
        shoreline.restore_records(data_dir, cfg.paths(cfg["per_species"]["products"]), P(g["worldclim_stack"]),
                                  P(g["soil_dir"]), transform, P(g["proj_grids"]), max_km=sh["max_km"],
                                  workers=a.workers, log=log)


if __name__ == "__main__":
    main()
