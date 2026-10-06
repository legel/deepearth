#!/usr/bin/env python3
"""Training points of the joint model for every US native species with per-species products
(ranges/joint/prepare.py): presences and effort-weighted background exactly as each species' MaxEnt saw them, with
the 24 shared predictors (WorldClim 2.1 at 30", SoilGrids 2.0 at 7.5").

usage: national_prepare.py [--config configs/conus.json] [--out DIR] [--workers 12]
Reads the configured tree (build_tree.py), species table and product directories (run_fit.py, searched in order);
writes the configured national data directory (joint.national_dir).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint.prepare import Predictors, training_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cfg = config.load(a.config)
    g = cfg["geo"]
    training_data(a.out or cfg.path(cfg["joint"]["national_dir"]), cfg.path(cfg["joint"]["tree_dir"]),
                  cfg.path(cfg["species"]["table"]), cfg.paths(cfg["per_species"]["products"]),
                  Predictors(cfg.path(g["worldclim_stack"]), cfg.path(g["soil_dir"])), a.workers,
                  log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
