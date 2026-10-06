#!/usr/bin/env python3
"""Train the joint range model (docs/joint_model.md) on the configured training data.

usage: national_train.py [name] [--stage base|representation|species] [--config configs/conus.json] [--data-dir D]
                         [--out DIR] [--steps N] [--eval-every N] [--seed S] [--device cuda]

Without --stage: the environment model (joint.train; run joint.run in joint.runs_dir). With --stage, one stage of the
model with the landscape field, place pathway and learned calibration (joint.stages):
  base            the environment model the representation starts from;
  representation  every pathway trained jointly, started from the base run's final weights;
  species         the representation's networks fixed, every species parameter fitted afresh on its cached features
                  (build the cache first: scripts/national_cache.py).
Writes <runs_dir>/<name>/ (default name: the configured run): norm.npz, model_best.pt, model.pt, run.json,
per_species_<plots>.csv.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ranges import config  # noqa: E402
from ranges.joint.reader import ECOREGION_RASTER  # noqa: E402
from ranges.joint.train import TrainConfig, train  # noqa: E402


def stage_run(cfg, name: str) -> Path:
    """The run directory of a configured stage."""
    s = cfg["joint"]["stages"][name]
    return cfg.path(s["runs_dir"]) / s["run"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--stage", choices=["base", "representation", "species"])
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
    spec = (j["stages"][a.stage] if a.stage else
            {"data_dir": j["data_dir"], "runs_dir": j["runs_dir"], "run": j["run"], "train": j["train"]})
    tc = TrainConfig.from_dict(spec["train"])
    for k in ("steps", "eval_every", "seed"):
        if getattr(a, k) is not None:
            setattr(tc, k, getattr(a, k))
    data_dir = Path(a.data_dir) if a.data_dir else cfg.path(spec["data_dir"])
    out = Path(a.out) if a.out else cfg.path(spec["runs_dir"]) / (a.name or spec["run"])
    grid = cfg.path(cfg["store"]["grids"][cfg["scope"]["region"].lower()])
    train(data_dir, out, tc, a.tree, a.device, log=lambda m: print(m, flush=True),
          init=stage_run(cfg, spec["init"]) if spec.get("init") else None,
          shared_from=stage_run(cfg, spec["shared_from"]) if spec.get("shared_from") else None,
          cache_dir=cfg.path(spec.get("cache")), field_dir=data_dir / "field",
          ecoregion_raster=grid / ECOREGION_RASTER if tc.calibration_penalty is not None else None)


if __name__ == "__main__":
    main()
