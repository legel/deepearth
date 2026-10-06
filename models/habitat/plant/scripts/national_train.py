#!/usr/bin/env python3
"""Train the joint range model (docs/joint_model.md) on the configured training data.

usage: national_train.py [name] [--config configs/conus.json] [--data-dir D] [--out DIR] [--steps N]
                         [--eval-every N] [--seed S] [--device cuda]
Writes <runs_dir>/<name>/ (default name: the configured run): norm.npz, model_best.pt, model.pt, run.json,
per_species_<plots>.csv.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ranges import config  # noqa: E402
from ranges.joint.train import TrainConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    ap.add_argument("--tree", help="dated Newick tree (default <data-dir>/tree/natives.dated.nwk)")
    ap.add_argument("--out", help="run directory (default <runs_dir>/<name>)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = config.load(a.config)
    j = cfg["joint"]
    tc = TrainConfig.from_dict(j["train"])
    for k in ("steps", "eval_every", "seed"):
        if getattr(a, k) is not None:
            setattr(tc, k, getattr(a, k))
    data_dir = Path(a.data_dir) if a.data_dir else cfg.path(j["data_dir"])
    out = Path(a.out) if a.out else cfg.path(j["runs_dir"]) / (a.name or j["run"])
    train(data_dir, out, tc, a.tree, a.device, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
