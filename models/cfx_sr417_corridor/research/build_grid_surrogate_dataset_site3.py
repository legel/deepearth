#!/usr/bin/env python3
"""Multi-storm grid-solver training data for a FloodSformer-style learned surrogate — site3.

Why this exists
----------------
Every learned-surrogate experiment this project has run so far (`train_mesh_gnn_site3.py`,
`validate_gnn_rollout.py`, `benchmark_gnn_forward.py`) operated on the from-scratch UNSTRUCTURED
MESH solver (`mesh_shallow_water.py`) and found a real negative result at full site3 scale: a
per-edge message-passing GNN is ~56x SLOWER than the physics solver once the graph reaches
site3's true resolution (8.67M edges) — see CLAUDE.md's "GNN INFERENCE benchmarked at full site3
scale" entry. A 2026-08-24 literature review (HYDROLINK_PAPER_PLAN.md §4) found that every
published "AI surrogate beats the physics solver" result reviewed operates on a COARSER or
fundamentally different representation than a per-edge mesh graph — most directly, FloodSformer
(Pianforini et al. 2025, Environmental Modelling & Software) trains a CNN-autoencoder +
Transformer directly on GRID-based shallow-water solver output (images, not a mesh graph) and
reports ~90x speedup.

This script builds the training corpus for a FloodSformer-style ablation on OUR OWN grid solver
(`flood_sim_ian.py`, the same solver family already calibrated and gauge-validated for site3 —
see CLAUDE.md's "real Hurricane Ian event run for site3" entry), so the eventual comparison is:
same project, same physical system, two different learned-surrogate architectures, apples to
apples. Mirrors the GNN study's own experimental design deliberately: TRAIN at a small/coarse
scale (here: a coarse cell size, not a mesh crop), then separately BENCHMARK inference cost at
site3's true full resolution (`benchmark_grid_transformer_forward.py`) — the same split between
"does it work" and "is it fast enough at real scale" the GNN study already used.

Method
------
Reuses `analysis/flood_probability.py`'s own machinery almost verbatim (NOAA Atlas 14 IDF ->
SCS Type II design hyetograph -> `flood_sim_ian.run_sim()`), the ONLY change being real
`frame_interval_min` (that script deliberately passes `frame_interval_min=10**9` — "no frames
needed" — since it only wants peak depth). Six design storms (return periods 1, 2, 10, 25, 100,
500 yr) give real intensity diversity from a single physical domain, split train/held-out the
same way `validate_gnn_rollout.py` held out 2 scenarios: two storms (1yr = most frequent, 500yr
= most extreme) are reserved for rollout validation; the middle four (2/10/25/100yr) are for
training. This is NOT independent replication of a real historical event (site3 only has one:
Hurricane Ian) — it is intensity diversity from the same physically-calibrated solver, which is
what a temporal surrogate actually needs to learn (how depth evolves given different forcing),
not a claim of meteorological diversity.

Per-frame forcing (the surrogate's cross-attention input, FloodSformer's analog of "inflow
discharge") is recovered by re-evaluating the EXACT rain-rate timeline used to force the solver
(`rain_sim`/`t_sim`, built the same way `flood_probability.run_ensemble()` builds it) at each
frame's own recorded time — not by re-deriving it from the solver's per-STEP arrays, which are
indexed by step count under an adaptive CFL timestep and therefore do not correspond 1:1 to a
fixed wall-clock spacing. This sidesteps that mismatch entirely and is exact by construction.

Output: one .npz per storm in `simulation/data/grid_surrogate_site3/`:
    frames        float32 [n_frames, ny, nx]   depth [m]
    times_min     float32 [n_frames]
    rain_mm_hr    float32 [n_frames]           forcing at each frame's own time
    return_period_yr, aep, cell_size_m, dx, split ('train' | 'held_out')
"""
import os
import sys
import time
import argparse

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "data", "grid_surrogate_site3")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PROJ_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
sys.path.insert(0, os.path.join(PROJ_DIR, "precipitation"))
sys.path.insert(0, os.path.join(PROJ_DIR, "analysis"))

import contextlib                     # noqa: E402
with contextlib.redirect_stdout(sys.stderr):
    import flood_sim_ian as fsi       # noqa: E402  the calibrated solver, reused unchanged
    import noaa_atlas14 as a14        # noqa: E402  IDF curves + SCS Type II hyetograph
    import flood_probability as fp    # noqa: E402  repoint_solver_to_site(), load_idf()

# 1yr and 500yr held out for rollout validation (frequent + extreme, same "don't just memorize
# the middle of the range" spirit as validate_gnn_rollout.py's own 2-scenario holdout).
RETURN_PERIODS_YR = [1, 2, 10, 25, 100, 500]
HELD_OUT_YR = {1, 500}

SITE_LAT, SITE_LON = 28.690514, -81.287539   # site3 registry center (lidar/test_sites.py)


def build_storm(T, depth_mm, duration_hr, cell_size_m, dt_s, frame_interval_min):
    """One design-storm run with real frames. Mirrors flood_probability.run_ensemble()'s inner
    loop exactly, except frame_interval_min is real and we keep (t_sim, rain_sim) for the exact
    per-frame forcing reconstruction described in the module docstring."""
    z, profile, dx = fsi.load_dem_for_sim(cell_size_m)
    horton = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton is not None:
        horton = fsi.apply_impervious_mask(horton, z.shape, profile["transform"], profile["crs"])
        horton = fsi.apply_nlcd_graded_impervious(horton, z.shape,
                                                  profile["transform"], profile["crs"])

    HY_DT_MIN = 5
    hy = a14.make_design_hyetograph(depth_mm, duration_hr, dt_min=HY_DT_MIN)
    step_s = HY_DT_MIN * 60.0
    rate_ms = np.asarray(hy["incremental_depth_mm"], dtype=float) / 1000.0 / step_s
    t_hy = np.asarray(hy["time_min"], dtype=float) * 60.0
    tail_s = duration_hr * 3600.0
    t_sim = np.arange(0.0, t_hy[-1] + tail_s, dt_s)
    rain_sim = np.interp(t_sim, t_hy, rate_ms, left=0.0, right=0.0)
    applied_mm = float(rain_sim.sum() * dt_s * 1000.0)
    if applied_mm > 1e-9:
        rain_sim *= depth_mm / applied_mm

    t0 = time.time()
    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = fsi.run_sim(
        z, dx, rain_sim, dt_s, frame_interval_min=frame_interval_min,
        verbose=False, use_infiltration=True, horton_arrays=horton)
    el = time.time() - t0

    frames = np.stack(frame_data["frames"]).astype(np.float32)          # [n_frames, ny, nx]
    times_min = np.array(frame_data["times_min"], dtype=np.float32)
    # Exact per-frame forcing: interpolate the CANONICAL (t_sim, rain_sim) forcing timeline at
    # each frame's own recorded elapsed time, in mm/hr for readability.
    frame_rain_mm_hr = np.interp(times_min * 60.0, t_sim, rain_sim) * 3600.0 * 1000.0

    print(f"  T={T:>4}yr  grid={z.shape}  n_frames={len(frames)}  "
          f"peak_depth={h_max.max():.3f}m  peak_flooded={np.max(flooded_ha_ts):.1f}ha  [{el:.1f}s]")
    return frames.astype(np.float32), times_min, frame_rain_mm_hr.astype(np.float32), dx, z.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-size", type=float, default=25.0,
                    help="Coarse TRAINING cell size [m] — deliberately coarse, mirrors the GNN "
                         "study's own small-crop training scale. Full-resolution benchmarking "
                         "happens separately, in benchmark_grid_transformer_forward.py.")
    ap.add_argument("--duration-hr", type=float, default=24.0)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--frame-interval-min", type=float, default=20.0)
    args = ap.parse_args()

    print("═" * 74)
    print("  Grid-surrogate training corpus — site3, multi-storm design ensemble")
    print("═" * 74)
    fp.repoint_solver_to_site("site3")

    print(f"\n[1/2] NOAA Atlas 14 IDF for site3 ({SITE_LAT}, {SITE_LON}), "
          f"{args.duration_hr:.0f}hr duration …")
    depths = fp.load_idf(SITE_LAT, SITE_LON, args.duration_hr)

    print(f"\n[2/2] Running {len(RETURN_PERIODS_YR)} design storms at {args.cell_size:.0f}m "
          f"cell size (train: {[t for t in RETURN_PERIODS_YR if t not in HELD_OUT_YR]}, "
          f"held-out: {sorted(HELD_OUT_YR)}) …")
    manifest = []
    for T in RETURN_PERIODS_YR:
        frames, times_min, rain_mm_hr, dx, shape = build_storm(
            T, depths[T], args.duration_hr, args.cell_size, args.dt, args.frame_interval_min)
        split = "held_out" if T in HELD_OUT_YR else "train"
        out_path = os.path.join(OUT_DIR, f"storm_T{T:04d}yr.npz")
        np.savez_compressed(out_path, frames=frames, times_min=times_min,
                            rain_mm_hr=rain_mm_hr, return_period_yr=T, aep=1.0 / T,
                            cell_size_m=args.cell_size, dx=dx, split=split)
        manifest.append(dict(return_period_yr=T, split=split, n_frames=int(len(frames)),
                             grid_shape=list(shape), peak_depth_m=float(frames.max()),
                             file=os.path.basename(out_path)))

    import json
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(dict(site="site3", cell_size_m=args.cell_size, duration_hr=args.duration_hr,
                      dt_s=args.dt, frame_interval_min=args.frame_interval_min,
                      storms=manifest), f, indent=2)
    print(f"\nSaved {len(manifest)} storms to {os.path.relpath(OUT_DIR, PROJ_DIR)}/")


if __name__ == "__main__":
    main()
