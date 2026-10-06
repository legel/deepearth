#!/usr/bin/env python3
"""Cache the shared features of the representation for the species stage (ranges/joint/cache.py).

The species stage keeps the representation's networks fixed, so the features of every row a training step can read
(presences, restored shoreline presences, and the records background points move to) and of every plot are computed
once, with the representation's own inputs and arithmetic.

usage: national_cache.py [--config configs/conus.json] [--run DIR] [--data-dir D] [--out DIR] [--device cuda]
Defaults: the representation stage's run, the species stage's data directory and cache directory (joint.stages).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint.cache import build_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--run", help="run whose shared networks are cached (default: the species stage's shared_from)")
    ap.add_argument("--data-dir")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = config.load(a.config)
    st = cfg["joint"]["stages"]
    spc = st["species"]
    src = st[spc["shared_from"]]
    run = Path(a.run) if a.run else cfg.path(src["runs_dir"]) / src["run"]
    data_dir = Path(a.data_dir) if a.data_dir else cfg.path(spc["data_dir"])
    tc = spc["train"]
    build_cache(run, data_dir, Path(a.out) if a.out else cfg.path(spc["cache"]), tc["flag_variables"],
                data_dir / "field", tc["shoreline_records"], tc["fill_plots"], a.device,
                log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
