#!/usr/bin/env python3
"""Independent plot sets of the joint model (ranges/joint/prepare.py ``plot_sets``): VegBank, BLM AIM and FIA plots
in the contiguous US, never used in training, with their predictors, their ecoregion on the 240 m grid and, where a
per-species MaxEnt range card exists, its suitability at the plots (the paired baseline of the evaluation).

usage: national_plots.py [--config configs/conus.json] [--data-dir D] [--sources vegbank,aim,fia]
Writes <data dir>/joint_data.npz (VegBank, with the predictor names) and plots_<source>.npz (default data dir: the
configured national directory).
"""
import argparse
import sys
from pathlib import Path

import pyproj  # noqa: F401,I001  (before rasterio: its bundled PROJ can break pyproj's transforms)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config, validate  # noqa: E402
from ranges.joint.prepare import Predictors, plot_sets  # noqa: E402
from ranges.maxent import default_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    ap.add_argument("--sources", default="vegbank,aim,fia")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ev, g = cfg["evaluation"], cfg["geo"]
    P = cfg.path
    truth = validate.PlotTruth(P(ev["aim_csv"]), P(ev["fia_dir"]), vegbank_dir=P(ev["vegbank_dir"]),
                               wcvp_dir=P(ev["wcvp_dir"]))
    cards = {mode: cfg.paths(dirs) for mode, dirs in cfg["per_species"]["range_cards"].items()}
    plot_sets(a.data_dir or P(cfg["joint"]["national_dir"]), truth, Predictors(P(g["worldclim_stack"]), P(g["soil_dir"])),
              P(cfg["store"]["grids"]["conus"]), cards, a.sources.split(","), cfg["scope"]["region"],
              ev["min_presence"], str(default_device()), log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
