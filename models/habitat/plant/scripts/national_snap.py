#!/usr/bin/env python3
"""Target-group background of the joint model (ranges/joint/prepare.py ``snap_to_records``, data.py): for every
training row of a joint data directory, the row of the nearest record of another species, by great-circle distance.
Training moves each background point there, so presences and background share the fine-scale sampling of records.

usage: national_snap.py [--config configs/conus.json] [--data-dir D]
Writes <data dir>/community/snap.npy (default data dir: joint.data_dir).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint.prepare import snap_to_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    a = ap.parse_args()
    cfg = config.load(a.config)
    snap_to_records(Path(a.data_dir) if a.data_dir else cfg.path(cfg["joint"]["data_dir"]),
                    log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
