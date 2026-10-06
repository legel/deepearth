#!/usr/bin/env python3
"""The landscape field of the joint model (ranges/joint/field.py) and where every training row and plot lies on it.

usage: national_field.py [pyramid] [positions] [--config configs/conus.json] [--data-dir D] [--device cpu]
  pyramid    build the field pyramid of joint.field (channels of the 240 m stack, standardized, int8, 2x levels)
             in joint.field.pyramid;
  positions  for a joint data directory (default joint.data_dir), the fractional (row, column) of every training
             row (field/rc_train.npy) and of every plot set (field/rc_plots_<source>.npy) on the field's 240 m grid,
             by the WGS 84 -> EPSG:5070 transformation PROJ applied when the rasters were warped (geodesy.py; the
             PROJ grids of geo.proj_grids); the pyramid files are linked into <data dir>/field.
Without a subcommand both run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint.data import JointData  # noqa: E402
from ranges.joint.field import build_pyramid  # noqa: E402
from ranges.joint.geodesy import WGS84ToConusAlbers, grid_positions  # noqa: E402


def positions(data_dir: Path, pyramid: Path, proj_grids: Path, device: str, log) -> None:
    out = data_dir / "field"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(pyramid.glob("field_L*.npy")) + [pyramid / "field.json"]:
        link = out / f.name
        if pyramid.resolve() != out.resolve() and not link.exists():
            link.symlink_to(f.resolve())
    transform = json.loads((pyramid / "field.json").read_text())["transform"]
    to5070 = WGS84ToConusAlbers(proj_grids, device)
    t0 = time.time()
    D = JointData(data_dir)
    lat, lon = D["lat"], D["lon"]
    rc = np.lib.format.open_memmap(out / "rc_train.npy", mode="w+", dtype=np.float32, shape=(len(lat), 2))
    step = 8_000_000
    for i in range(0, len(lat), step):
        rc[i:i + step] = grid_positions(np.asarray(lat[i:i + step]), np.asarray(lon[i:i + step]), transform, to5070,
                                        device=device)
        log(f"training rows {min(i + step, len(lat)):,}/{len(lat):,} ({time.time() - t0:.0f} s)")
    rc.flush()
    sets = {"vegbank": D.npz} | {f.stem.split("_", 1)[1]: np.load(f) for f in sorted(data_dir.glob("plots_*.npz"))}
    for name, z in sets.items():
        np.save(out / f"rc_plots_{name}.npy", grid_positions(z["plot_lat"], z["plot_lon"], transform, to5070,
                                                             device=device))
        log(f"{name}: {len(z['plot_lat']):,} plots")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", choices=["pyramid", "positions"], default=[])
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (same arithmetic)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    f, log = cfg["joint"]["field"], (lambda m: print(m, flush=True))
    pyramid = cfg.path(f["pyramid"])
    steps = a.steps or ["pyramid", "positions"]
    if "pyramid" in steps:
        build_pyramid(cfg.path(f["stack"]), f["channels"], pyramid, f["levels"], f["log_channels"], a.device, log)
    if "positions" in steps:
        positions(Path(a.data_dir) if a.data_dir else cfg.path(cfg["joint"]["data_dir"]), pyramid,
                  cfg.path(cfg["geo"]["proj_grids"]), a.device, log)


if __name__ == "__main__":
    main()
