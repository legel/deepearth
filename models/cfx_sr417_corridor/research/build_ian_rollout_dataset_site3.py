#!/usr/bin/env python3
"""The real-Hurricane-Ian rollout test both surrogates (mesh-GNN and grid-transformer) have been
missing since the 2026-08-24 ablation started — flagged repeatedly (HYDROLINK_PAPER_PLAN.md §5c,
CLAUDE.md's 2026-08-25 entries) and finally built here. Every rollout result so far, for either
surrogate architecture, has been synthetic-Atlas-14-design-storm-only; this produces the first
REAL, gauge-relevant event at the grid-transformer's own training resolution.

Reuses `run_site3_ian.py`'s already-calibrated site3-repointed `flood_sim_ian` module (same real
KSFB ASOS hyetograph, same DEM/soil/roads/buildings) — but at the grid-transformer's 25m TRAINING
resolution (not the 5m production resolution `run_site3_ian.py` normally uses), and saves RAW
native-resolution frames + per-frame forcing directly (not the 256x256-downsampled SIML format
`run_site3_ian.py --save-frames` writes for the viewer, which is the wrong resolution for this
purpose). Exact per-frame forcing reconstruction follows the same method
`build_grid_surrogate_dataset_site3.py` already established: interpolate the CANONICAL
(t_sim, rain_sim) forcing timeline — here `fsi.load_ian_hyetograph()`'s own output, not a
resampled design hyetograph — at each frame's own recorded elapsed time.

Output: `simulation/data/grid_surrogate_site3/storm_ian.npz`, same schema as every synthetic
storm's own .npz, so it's directly loadable by evaluate_grid_transformer_checkpoints.py's
load_storms()/rollout_eval() with a one-line change (add "ian" to the storms dict, split="ian").
"""
import os
import sys
import json
import time
import argparse

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "data", "grid_surrogate_site3")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

import contextlib                     # noqa: E402
with contextlib.redirect_stdout(sys.stderr):
    import flood_sim_ian as fsi       # noqa: E402
    from test_sites import get_site   # noqa: E402

SITE = "site3"
site = get_site(SITE)
SITE3_DIR = os.path.join(PROJ_DIR, "site3_gee_creek")

# Same monkey-patch pattern run_site3_ian.py already uses — repoint at site3's own real,
# calibrated data (not a copy of run_site3_ian.py's own module-level code, since that script has
# no importable function form; this mirrors it directly, non-invasively, same as every other
# site3 script in this project).
fsi.DEM_COND     = site["dem_cond_path"]
fsi.SOIL_JSON    = site["soil_json_path"]
fsi.MUKEY_MAP    = site["mukey_map_path"]
fsi.MUKEY_LEGEND = site["mukey_legend_path"]
fsi.ROADS_PATH   = site["roads_path"]
fsi.BUILDINGS_PATH = site["buildings_path"]
fsi.NLCD_IMPERVIOUS_PATH = site["nlcd_path"]
fsi.ASOS_CSV     = os.path.join(SITE3_DIR, "precipitation", "data", "asos_hourly_SFB.csv")
fsi.HORTON = fsi._load_horton_params()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-size", type=float, default=25.0,
                    help="MUST match the grid-transformer's training resolution.")
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--frame-interval-min", type=float, default=20.0,
                    help="MUST match the grid-transformer's training frame interval.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Real Hurricane Ian — site3, grid-transformer training resolution "
          f"({args.cell_size:.0f}m)")
    print("=" * 70)

    print(f"\n[1/3] DEM (cell size = {args.cell_size:.0f}m) …")
    z, profile, dx = fsi.load_dem_for_sim(args.cell_size)
    print(f"  grid: {z.shape}")

    print(f"\n[2/3] Spatial Horton infiltration …")
    horton = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton is not None:
        horton = fsi.apply_impervious_mask(horton, z.shape, profile["transform"], profile["crs"])
        horton = fsi.apply_nlcd_graded_impervious(horton, z.shape, profile["transform"], profile["crs"])

    print(f"\n[3/3] Real Ian hyetograph (ASOS KSFB, {fsi.IAN_START} to {fsi.IAN_END}) + solver …")
    rain_sim, hours, rain_mm = fsi.load_ian_hyetograph(args.dt)
    t_sim = np.arange(0, len(rain_sim) * args.dt, args.dt)[:len(rain_sim)]
    print(f"  total rain: {rain_mm.sum():.0f}mm, peak {rain_mm.max():.1f}mm/hr, "
          f"{len(rain_sim)} steps = {len(rain_sim)*args.dt/3600:.1f}hr")

    t0 = time.time()
    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = fsi.run_sim(
        z, dx, rain_sim, args.dt, frame_interval_min=args.frame_interval_min,
        verbose=False, use_infiltration=True, horton_arrays=horton)
    el = time.time() - t0

    frames = np.stack(frame_data["frames"]).astype(np.float32)
    times_min = np.array(frame_data["times_min"], dtype=np.float32)
    frame_rain_mm_hr = np.interp(times_min * 60.0, t_sim, rain_sim) * 3600.0 * 1000.0

    print(f"  DONE: {el:.1f}s  n_frames={len(frames)}  peak_depth={h_max.max():.3f}m  "
          f"peak_flooded={np.max(flooded_ha_ts):.1f}ha")

    out_path = os.path.join(OUT_DIR, "storm_ian.npz")
    np.savez_compressed(out_path, frames=frames.astype(np.float32), times_min=times_min,
                        rain_mm_hr=frame_rain_mm_hr.astype(np.float32),
                        return_period_yr=-1, aep=0.0,   # not a design storm -- real event
                        cell_size_m=args.cell_size, dx=dx, split="real_event")
    print(f"\nSaved {os.path.relpath(out_path, PROJ_DIR)}  "
          f"({len(frames)} frames, {z.shape[0]}x{z.shape[1]} cells)")


if __name__ == "__main__":
    main()
