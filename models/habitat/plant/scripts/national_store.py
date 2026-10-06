#!/usr/bin/env python3
"""Build, update or recode the map store of the joint model (ranges/joint/store.py, docs/joint_model.md).

usage:
  national_store.py build [--config configs/conus.json] [--model M] [--run RUN] [--out DIR] [--no-zero-shot]
      every species of the run, plus the species without records (zero-shot, from the configured inventory, tree and
      region), with calibration areas extended so that every species native to the region has one on its grid; the
      run is that of --model (default store.model): "species" (the species stage of joint.stages, whose grids take the
      shoreline climate fill of shoreline.grid_fill) or "environment" (joint.run)
  national_store.py update-zero-shot <store>... [--all] [--config ...] [--model M] [--run RUN] [--device cuda]
      recompute the species without records of existing stores (current zero-shot rule) without rebuilding fields
  national_store.py recode <src> <dst>
      rewrite a store's tiles in the current layout (lossless)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj  # noqa: F401,I001  (before rasterio: its bundled PROJ can break pyproj's transforms)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ranges import config  # noqa: E402
from ranges.joint.climate_fill import GridFill  # noqa: E402
from ranges.joint.data import JointData, Standardizer  # noqa: E402
from ranges.joint.model import JointRangeModel  # noqa: E402
from ranges.joint.reader import ECOREGION_RASTER  # noqa: E402
from ranges.joint.scope import extend_calibration  # noqa: E402
from ranges.joint.store import GridInputs, build_store, recode_store, update_zero_shot  # noqa: E402
from ranges.joint.tree import Tree  # noqa: E402
from ranges.joint.zero_shot import infer_species, l3_ecoregions  # noqa: E402


def load_run(run: Path, flag_variables, device, field_dir: Path | None = None):
    """The checkpoint chosen on VegBank-dev (model_best.pt, else the final model.pt) and its standardization; a model
    with a landscape field gets the pyramid of ``field_dir``."""
    f = run / "model_best.pt" if (run / "model_best.pt").exists() else run / "model.pt"
    return JointRangeModel.load(f, device, field_dir), Standardizer.load(run / "norm.npz", flag_variables)


def stored_model(cfg, name: str | None = None) -> dict:
    """The run a store maps (``name``, default store.model: "environment" or a stage of joint.stages): its directory,
    data directory and training settings."""
    j = cfg["joint"]
    name = name or cfg["store"].get("model", "environment")
    spec = (j["stages"][name] if name != "environment" else
            {"data_dir": j["data_dir"], "runs_dir": j["runs_dir"], "run": j["run"], "train": j["train"]})
    return {"run": cfg.path(spec["runs_dir"]) / spec["run"], "data_dir": cfg.path(spec["data_dir"]),
            "train": spec["train"]}


def region_inventory(cfg) -> pd.DataFrame:
    """The species inventory (wcvp_accepted_name, native_l3, us_regions), restricted to the species native to the
    configured region."""
    inv = pd.read_csv(cfg.path(cfg["species"]["inventory"]))
    region = cfg["scope"]["region"]
    return inv[inv.us_regions.fillna("").str.split(",").apply(lambda r: region in r)]


def l3eco(cfg) -> dict:
    z, g = cfg["store"]["zero_shot"], cfg["geo"]
    return l3_ecoregions(cfg.path(z["l3_ecoregions"]), cfg.path(g["wgsrpd_l3"]), cfg.path(g["ecoregions"]),
                         z.get("min_share", 0.10))


def zero_shot(cfg, model, data):
    return infer_species(model, list(data.species().species), Tree.read(cfg.path(cfg["store"]["zero_shot"]["full_tree"])),
                         region_inventory(cfg), l3eco(cfg))


def calibration_rule(cfg, log):
    """scope.extend_calibration over the region's grid (its ecoregion ids) and level-3 areas."""
    import rasterio
    grid = cfg.path(cfg["store"]["grids"][cfg["scope"]["region"].lower()])
    with rasterio.open(grid / ECOREGION_RASTER) as r:
        on_grid = set(np.unique(r.read(1)).tolist()) - {0}
    inv, table, g = region_inventory(cfg), l3eco(cfg), cfg["geo"]
    return lambda labels, calib: extend_calibration(labels, calib, inv, on_grid, table, cfg["scope"]["region_l3"],
                                                    cfg.path(g["wgsrpd_l3"]), cfg.path(g["ecoregions"]), log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config")
    b.add_argument("--run", help="run directory (default <runs_dir>/<run>)")
    b.add_argument("--data-dir")
    b.add_argument("--out")
    b.add_argument("--no-zero-shot", action="store_true", help="only species with records")
    b.add_argument("--model", help="environment, or a stage of joint.stages (default store.model)")
    b.add_argument("--device", default="cuda")
    u = sub.add_parser("update-zero-shot")
    u.add_argument("stores", nargs="+")
    u.add_argument("--config")
    u.add_argument("--run")
    u.add_argument("--data-dir")
    u.add_argument("--device", default="cuda")
    u.add_argument("--model", help="environment, or a stage of joint.stages (default store.model)")
    u.add_argument("--all", action="store_true",
                   help="recompute every species without records (per-row seeds), not only those whose vector changed")
    r = sub.add_parser("recode")
    r.add_argument("src")
    r.add_argument("dst")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.cmd == "recode":
        recode_store(a.src, a.dst, log)
        return
    cfg = config.load(a.config)
    sc, m = cfg["store"], stored_model(cfg, a.model)
    run = Path(a.run) if a.run else m["run"]
    data_dir = Path(a.data_dir) if a.data_dir else m["data_dir"]
    model, st = load_run(run, m["train"]["flag_variables"], a.device, data_dir / "field")
    data = JointData(data_dir)
    positions = data.positions(data_dir / "field") if model.field is not None else None
    if a.cmd == "update-zero-shot":
        zs = zero_shot(cfg, model, data)
        for store in a.stores:
            update_zero_shot(store, model, st, data, zs, a.device, recompute_all=a.all, positions=positions, log=log)
        return
    fills = cfg.get("shoreline", {}).get("grid_fill", {}) if m["train"].get("fill_plots") else {}
    grids = {rg: GridInputs(cfg.path(d), cfg.path(sc["soil_dir"]), st.names,
                            GridFill.load(cfg.path(fills[rg])) if rg in fills else None)
             for rg, d in sc["grids"].items()}
    zs = None if a.no_zero_shot else zero_shot(cfg, model, data)
    build_store(model, st, data, a.out or cfg.path(sc["out"]), grids, sc["delta"], zs, sc["sample"], sc["tile"],
                a.device, calibration_rule(cfg, log), positions, log)


if __name__ == "__main__":
    main()
