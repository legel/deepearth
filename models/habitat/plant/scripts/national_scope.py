#!/usr/bin/env python3
"""Restrict the national training data to the mapped region (ranges/joint/scope.py ``restrict``): the species native
to it, without the training points inside the excluded areas (for CONUS: Alaska with the Aleutians, and Hawaii),
keeping each species' points in its native range abroad.

usage: national_scope.py [--config configs/conus.json] [--src DIR] [--dst DIR]
Default: the configured national directory (joint.national_dir) -> the configured training data (joint.data_dir).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint.scope import restrict  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    a = ap.parse_args()
    cfg = config.load(a.config)
    s, j = cfg["scope"], cfg["joint"]
    restrict(a.src or cfg.path(j["national_dir"]), a.dst or cfg.path(j["data_dir"]), s["region"],
             cfg.path(cfg["geo"]["wgsrpd_l3"]), s["exclude_l3"], s["exclude_buffer_deg"], s.get("exclude_boxes", ()),
             s.get("drop_plot_sets", ()), log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
