"""
Real Hurricane Milton event, run on site3's grid-based fast solver (Gee Creek gauge-matched site)
===================================================================================================
Trimmed sibling of run_site3_ian.py, pointed at the second real storm this project's magnitude-
gap investigation already validated against (see ../../NEXT_STEPS.md's "Second real-storm
validation: Hurricane Milton" entry, 2026-08-31) -- reuses flood_sim_ian.py's own functions
directly via the same monkey-patched-constants pattern (ASOS_CSV, IAN_START/IAN_END; the shared
module's own naming still says "Ian" everywhere, this script just points the same machinery at a
different real storm and observed record). flood_sim_ian.py itself is never edited.

Only the subset of run_site3_ian.py's features this project's Milton work has actually used are
kept: no --baseflow (Milton's own pre-storm baseflow of 7.87 cfs was used only as evidence the
ground was drier, never as an initial condition), no --save-frames (Milton has no viewer page).
--storage-scale, --no-storage-cap, --no-ponded-infiltration, --tag, and the new --extend-hours
(see run_site3_ian.py's own docstring for --extend-hours' purpose) are all kept, since the
2026-08-31/2026-09-03 sweeps used them for both storms.

Storm window: 2024-10-06 00:00 - 2024-10-10 23:00 UTC (120h), matching the already-fetched
site3_gee_creek/precipitation/data/asos_hourly_SFB_milton.csv total of 288.0mm exactly (see
../../NEXT_STEPS.md's MRMS cross-check entry, 2026-09-02).

Usage:
    python3 simulation/run_site3_milton.py --cell-size 25 --dt 20
    python3 simulation/run_site3_milton.py --cell-size 25 --dt 20 --extend-hours 96 --tag _168h
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

import flood_sim_ian as fsi  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE = "site3"
site = get_site(SITE)

SITE3_DIR = os.path.join(PROJ_DIR, "site3_gee_creek")
OUT_DIR = fsi.OUT_DIR

# Monkey-patch flood_sim_ian's module-level path constants -- same registry-key pattern
# run_site3_ian.py uses, plus the Milton-window ASOS record and storm dates in place of Ian's.
fsi.DEM_COND     = site["dem_cond_path"]
fsi.SOIL_JSON    = site["soil_json_path"]
fsi.MUKEY_MAP    = site["mukey_map_path"]
fsi.MUKEY_LEGEND = site["mukey_legend_path"]
fsi.ROADS_PATH   = site["roads_path"]
fsi.BUILDINGS_PATH = site["buildings_path"]
fsi.NLCD_IMPERVIOUS_PATH = site["nlcd_path"]
fsi.SOIL_STORAGE_CSV = os.path.join(os.path.dirname(site["soil_json_path"]), "soil_storage.csv")
fsi.ASOS_CSV = os.path.join(SITE3_DIR, "precipitation", "data", "asos_hourly_SFB_milton.csv")
fsi.IAN_START = "2024-10-06 00:00"
fsi.IAN_END = "2024-10-10 23:00"

# fsi.HORTON was computed at import time from whatever SOIL_JSON was in effect then -- re-derive
# now that SOIL_JSON points at site3's own soil_parameters.json (same trick run_site3_ian.py
# uses for the identical reason).
fsi.HORTON = fsi._load_horton_params()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-size", type=float, default=25.0)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--storage-scale", type=float, default=1.0)
    ap.add_argument("--no-storage-cap", action="store_true")
    ap.add_argument("--no-ponded-infiltration", action="store_true")
    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--extend-hours", type=float, default=0.0,
                     help="See run_site3_ian.py's --extend-hours docstring -- same test, "
                          "same reasoning, applied to the second storm.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Real Hurricane Milton event — site3: {site['label']}")
    print(f"Precipitation: ASOS KSFB (Orlando Sanford Intl, 10.8km from site3)")
    print("=" * 70)

    print(f"\n[1/4] DEM (target cell size = {args.cell_size}m) …")
    z, profile, dx = fsi.load_dem_for_sim(args.cell_size)

    print("\n[2/4] Spatial Horton infiltration (SSURGO + impervious mask + NLCD graded) …")
    horton_arrays = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton_arrays is not None:
        horton_arrays = fsi.apply_impervious_mask(horton_arrays, z.shape, profile["transform"], profile["crs"])
        horton_arrays = fsi.apply_nlcd_graded_impervious(horton_arrays, z.shape, profile["transform"], profile["crs"])

    print(f"\n[3/4] Milton hyetograph (ASOS KSFB hourly, {fsi.IAN_START} – {fsi.IAN_END} UTC) …")
    rain_sim, hours, rain_mm = fsi.load_ian_hyetograph(args.dt)
    if args.extend_hours > 0:
        n_extra_steps = int(round(args.extend_hours * 3600 / args.dt))
        n_extra_hours = int(round(args.extend_hours))
        rain_sim = np.concatenate([rain_sim, np.zeros(n_extra_steps)])
        rain_mm = np.concatenate([rain_mm, np.zeros(n_extra_hours)])
        hours = np.arange(len(rain_mm))
        print(f"  Extended with {args.extend_hours:.0f}h of zero rain "
              f"({n_extra_steps} extra steps) -> total window "
              f"{len(rain_sim) * args.dt / 3600:.1f}h")
    total_rain = rain_mm.sum()
    n_steps = len(rain_sim)
    print(f"  Simulation: {n_steps} steps x {args.dt:.0f}s = {n_steps*args.dt/3600:.1f} hrs, "
          f"total rain {total_rain:.0f}mm")

    if args.dry_run:
        print("\nDry run -- exiting.")
        return

    print(f"\n[4/4] Running solver ({z.shape[0]}x{z.shape[1]} cells @ {dx:.1f}m) …")
    t0 = time.time()
    max_deficit_m = fsi.load_soil_storage_capacity(z.shape, profile["transform"], profile["crs"])
    if args.no_storage_cap:
        max_deficit_m = None
    elif max_deficit_m is not None and args.storage_scale != 1.0:
        max_deficit_m = max_deficit_m * args.storage_scale

    from pyproj import Transformer as _Tf
    _tr = _Tf.from_crs("epsg:4326", profile["crs"], always_xy=True)
    _gx, _gy = _tr.transform(site["gauge_lon"], site["gauge_lat"])
    _gc, _gr = ~profile["transform"] * (_gx, _gy)
    _gr, _gc = int(round(_gr)), int(round(_gc))
    _rad = max(1, int(round(25.0 / dx)))
    _r0, _r1 = max(0, _gr - _rad), min(z.shape[0], _gr + _rad + 1)
    _c0, _c1 = max(0, _gc - _rad), min(z.shape[1], _gc + _rad + 1)
    _sub = z[_r0:_r1, _c0:_c1]
    _fl = int(np.nanargmin(np.where(np.isfinite(_sub), _sub, np.inf)))
    gauge_rc = (_r0 + _fl // _sub.shape[1], _c0 + _fl % _sub.shape[1])
    print(f"  Gauge {site['gauge_site_no']} at grid {gauge_rc}, bed {z[gauge_rc]:.2f} m")

    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = fsi.run_sim(
        z, dx, rain_sim, args.dt, frame_interval_min=60.0, use_infiltration=True,
        horton_arrays=horton_arrays, max_deficit_m=max_deficit_m, gauge_rc=gauge_rc,
        initial_h=None, ponded_infiltration=not args.no_ponded_infiltration,
    )
    elapsed = time.time() - t0

    peak_ha = float(flooded_ha_ts.max())
    peak_h = float(h_max.max())
    print(f"\n  Solver: {elapsed:.0f}s  |  peak depth={peak_h:.3f}m  |  peak flooded={peak_ha:.1f}ha")

    CMS_TO_CFS = 35.3147
    outflow_total_cms = frame_data["outflow_total_cms"]
    outflow_south_cms = frame_data["outflow_south_cms"]
    step_hrs = np.arange(n_steps) * args.dt / 3600.0
    hydro_df = pd.DataFrame({
        "time_min": step_hrs * 60,
        "rain_mm_hr": rain_ts,
        "Pe_mm_hr": Pe_ts,
        "flooded_ha": flooded_ha_ts,
        "mean_depth_m": mean_depth_ts,
        "outflow_total_cms": outflow_total_cms,
        "outflow_total_cfs": outflow_total_cms * CMS_TO_CFS,
        "outflow_south_cms": outflow_south_cms,
        "outflow_south_cfs": outflow_south_cms * CMS_TO_CFS,
        "gauge_cms": frame_data["gauge_cms"],
        "gauge_cfs": frame_data["gauge_cms"] * CMS_TO_CFS,
    })
    hydro_path = os.path.join(OUT_DIR, f"hydrograph_milton_site3{args.tag}.csv")
    hydro_df.to_csv(hydro_path, index=False)
    print(f"  {os.path.basename(hydro_path)} ({n_steps} rows)")

    summary = {
        "site": SITE, "storm": "milton",
        "cell_size_m": dx, "grid_shape": list(z.shape), "dt_s": args.dt,
        "n_steps": n_steps, "wall_s": elapsed,
        "total_rain_mm": float(total_rain), "peak_rain_mm_hr": float(rain_mm.max()),
        "peak_depth_m": peak_h, "peak_flooded_ha": peak_ha,
    }
    with open(os.path.join(OUT_DIR, f"milton_sim_summary_site3{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  milton_sim_summary_site3{args.tag}.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()
