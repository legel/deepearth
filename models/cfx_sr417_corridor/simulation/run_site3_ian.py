"""
Real Hurricane Ian event, run on site3's grid-based fast solver (Gee Creek gauge-matched site)
================================================================================================
The mesh solver (mesh_shallow_water.py / run_site3_swe_checkpointed.py) is a demo-scale
prototype for fine mesh-resolution water behavior — its own docstring says so, and the first
full run confirmed it in practice: an 8-minute synthetic rain burst took 2.5 hours of wall time,
and even then only 0.02% of the rain reached the domain boundary (site3's 6x6km box is far too
large for water to route to the edge in 8 minutes). Reproducing the REAL, ~72-hour Ian event
through that solver is not tractable — this project's OTHER solver, flood_sim_ian.py (grid-
based, LISFLOOD-FP), already handles a real 72-hour event for the original AOI in ~30-45s at
5m/160,000 cells, and already tracks per-step domain-boundary outflow (the exact "civil
engineering hydrograph" signal a gauge comparison needs). This script reuses that solver's own
functions directly (monkey-patched paths, same pattern as every other site3 script this project
uses — flood_sim_ian.py itself is never edited) pointed at site3's own already-built DEM/soil/
roads/buildings, plus a newly-fetched real ASOS hourly record for KSFB (Orlando Sanford Intl,
10.8km from site3 — much closer than KMCO's 29.1km, which is what the original AOI uses).

Real, honest caveats carried from this project's own established convention (not new):
  - Only outflow_total_cms (all 4 domain edges combined) is used, not outflow_south_cms — the
    real Gee Creek gauge sits NORTH of site3's own box center (see lidar/test_sites.py's site3
    entry), not south the way both Shingle Creek gauges sit relative to the original AOI. Using
    the total across all edges is a defensible, honest simplification for a first exploratory
    run rather than picking one edge without checking which one actually faces the gauge.
  - This is a SHAPE/TIMING comparison, not magnitude — this project's own
    entry established that principle for the original AOI's ~44x watershed-area mismatch; site3
    has its own, smaller (but still real) mismatch: 11.65 km^2 delineated vs 33.15 km^2
    documented gauge drainage area (35% capture — see test_sites.py's site3 comment).

Usage:
    python3 simulation/run_site3_ian.py --cell-size 5 --dt 20
"""
import os, sys, json, time, shutil, argparse
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

import flood_sim_ian as fsi  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE = "site3"
site = get_site(SITE)

SITE3_DIR = os.path.join(PROJ_DIR, "site3_gee_creek")

# Monkey-patch flood_sim_ian's module-level path constants to point at site3's own already-
# built data (dem/soil/roads/buildings, all fetched during the 2026-07-27 site3 pipeline build)
# plus the newly-fetched real KSFB Ian-window ASOS record — same non-invasive pattern every
# other site3 script in this project uses (never edit the production script itself).
fsi.DEM_COND     = site["dem_cond_path"]
fsi.SOIL_JSON    = site["soil_json_path"]
fsi.MUKEY_MAP    = site["mukey_map_path"]
fsi.MUKEY_LEGEND = site["mukey_legend_path"]
fsi.ROADS_PATH   = site["roads_path"]
fsi.BUILDINGS_PATH = site["buildings_path"]
fsi.NLCD_IMPERVIOUS_PATH = site["nlcd_path"]
# Storage table for the finite-infiltration cap. Not in site["..."] because it postdates the
# registry entry; derived from the same soil/data directory as the other soil inputs.
fsi.SOIL_STORAGE_CSV = os.path.join(os.path.dirname(site["soil_json_path"]), "soil_storage.csv")
fsi.ASOS_CSV     = os.path.join(SITE3_DIR, "precipitation", "data", "asos_hourly_SFB.csv")

# fsi.HORTON was computed at import time from the ORIGINAL AOI's SOIL_JSON — re-derive it now
# that SOIL_JSON points at site3's own soil_parameters.json, and reassign the module global
# (load_spatial_horton/run_sim look this up via the module namespace at call time, so
# reassigning it here is picked up correctly, same trick used throughout this project).
fsi.HORTON = fsi._load_horton_params()

OUT_DIR = fsi.OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-size", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--frame-interval", type=float, default=60.0,
                     help="Minutes between saved animation frames (default 60)")
    ap.add_argument("--save-frames", action="store_true",
                     help="Save SIML animation frames for the viewer (site3.html)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Real Hurricane Ian event — site3: {site['label']}")
    print(f"Precipitation: ASOS KSFB (Orlando Sanford Intl, 10.8km from site3)")
    print("=" * 70)

    print(f"\n[1/4] DEM (target cell size = {args.cell_size}m) …")
    z, profile, dx = fsi.load_dem_for_sim(args.cell_size)

    print("\n[2/4] Spatial Horton infiltration (SSURGO + impervious mask + NLCD graded) …")
    horton_arrays = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton_arrays is None:
        print("  mukey_map.tif not found — falling back to uniform mean "
              f"f0={fsi.HORTON['f0']} fc={fsi.HORTON['fc']} k={fsi.HORTON['k']}")
    else:
        horton_arrays = fsi.apply_impervious_mask(horton_arrays, z.shape, profile["transform"], profile["crs"])
        horton_arrays = fsi.apply_nlcd_graded_impervious(horton_arrays, z.shape, profile["transform"], profile["crs"])

    print(f"\n[3/4] Ian hyetograph (ASOS KSFB hourly, {fsi.IAN_START} – {fsi.IAN_END} UTC) …")
    rain_sim, hours, rain_mm = fsi.load_ian_hyetograph(args.dt)
    total_rain = rain_mm.sum()
    n_steps = len(rain_sim)
    print(f"  Sep 28: {rain_mm[:24].sum():.0f} mm  Sep 29: {rain_mm[24:48].sum():.0f} mm  "
          f"Sep 30: {rain_mm[48:].sum():.0f} mm  total={total_rain:.0f} mm")
    print(f"  Simulation: {n_steps} steps x {args.dt:.0f}s = {n_steps*args.dt/3600:.1f} hrs")

    if args.dry_run:
        print("\nDry run -- exiting.")
        return

    print(f"\n[4/4] Running solver ({z.shape[0]}x{z.shape[1]} cells @ {dx:.1f}m) …")
    t0 = time.time()
    # Finite soil storage (saturation excess). This driver calls run_sim directly rather than
    # going through fsi.main(), so it has to load the cap itself.
    max_deficit_m = fsi.load_soil_storage_capacity(z.shape, profile["transform"], profile["crs"])
    if max_deficit_m is not None:
        print(f"  Soil storage cap: mean {1000*float(max_deficit_m.mean()):.0f} mm, "
              f"range {1000*float(max_deficit_m.min()):.0f}-{1000*float(max_deficit_m.max()):.0f} mm, "
              f"{100*float((max_deficit_m == 0).mean()):.0f}% of cells depressional (zero storage)")
    else:
        print(f"  WARNING: no soil storage table at {fsi.SOIL_STORAGE_CSV} — "
              f"infiltration is UNBOUNDED (run soil/fetch_soil_storage.py --site site3)")

    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = fsi.run_sim(
        z, dx, rain_sim, args.dt, frame_interval_min=args.frame_interval, use_infiltration=True,
        horton_arrays=horton_arrays, max_deficit_m=max_deficit_m,
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
    })
    hydro_path = os.path.join(OUT_DIR, "hydrograph_ian_site3.csv")
    hydro_df.to_csv(hydro_path, index=False)
    print(f"  {os.path.basename(hydro_path)} ({n_steps} rows)")

    peak_idx = int(hydro_df["outflow_total_cfs"].idxmax())
    peak_outflow_cfs = float(hydro_df["outflow_total_cfs"].iloc[peak_idx])
    peak_outflow_time_hr = float(hydro_df["time_min"].iloc[peak_idx]) / 60.0
    rain_peak_idx = int(hydro_df["rain_mm_hr"].idxmax())
    rain_peak_time_hr = float(hydro_df["time_min"].iloc[rain_peak_idx]) / 60.0
    print(f"\n  Simulated total-boundary outflow peak: {peak_outflow_cfs:.2f} cfs "
          f"at t={peak_outflow_time_hr:.1f}h into the sim window")
    print(f"  Rain peak at t={rain_peak_time_hr:.1f}h  -->  lag = "
          f"{peak_outflow_time_hr - rain_peak_time_hr:.1f}h")
    # Real Gee Creek gauge (02234400) Ian response, corrected 2026-07-27 — the original
    # "baseflow ~4 cfs -> peak 35.4 cfs" figure recorded during site selection was WRONG
    # (see test_sites.py's site3 comment); the real USGS NWIS instantaneous-values record
    # (site3_gee_creek/infrastructure/data/gee_creek_ian_discharge.csv) shows baseline
    # ~45.2 cfs (2022-09-27 19:30 UTC) -> peak 1,190 cfs (2022-09-29 13:31 UTC).
    print(f"  Real Gee Creek gauge (02234400) Ian response: baseflow ~45.2 cfs -> peak 1,190 cfs "
          f"on 2022-09-29 (corrected via direct NWIS IV-service fetch, 2026-07-27)")

    summary = {
        "site": SITE, "site_label": site["label"],
        "precip_station": "KSFB (Orlando Sanford Intl, 10.8km)",
        "cell_size_m": dx, "grid_shape": list(z.shape), "dt_s": args.dt,
        "n_steps": n_steps, "wall_s": elapsed,
        "total_rain_mm": float(total_rain), "peak_rain_mm_hr": float(rain_mm.max()),
        "peak_depth_m": peak_h, "peak_flooded_ha": peak_ha,
        "peak_outflow_total_cfs": peak_outflow_cfs,
        "peak_outflow_time_hr": peak_outflow_time_hr,
        "rain_peak_time_hr": rain_peak_time_hr,
        "outflow_lag_hr": peak_outflow_time_hr - rain_peak_time_hr,
        "gauge": {
            "site_no": site["gauge_site_no"],
            "documented_drainage_area_km2": site["documented_drainage_area_km2"],
            "delineated_drainage_area_km2": site["delineated_drainage_area_km2"],
            "real_ian_baseflow_cfs": 45.2, "real_ian_peak_cfs": 1190.0,
            "real_ian_peak_time_utc": "2022-09-29T13:31:00Z",
        },
    }
    with open(os.path.join(OUT_DIR, "ian_sim_summary_site3.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  ian_sim_summary_site3.json")

    if args.save_frames and frame_data["frames"]:
        print(f"\n  Saving {len(frame_data['frames'])} animation frames for the viewer …")
        with rasterio.open(fsi.DEM_COND) as src:
            b = src.bounds
        true_left, true_right = sorted([b.left, b.right])
        true_bottom, true_top = sorted([b.bottom, b.top])

        class _Bounds:
            left, bottom, right, top = true_left, true_bottom, true_right, true_top
        fsi.write_siml_bin(
            os.path.join(OUT_DIR, "depth_frames_ian_site3.bin"),
            frame_data["frames"], profile, _Bounds(), frame_data["times_min"],
        )
        print(f"  depth_frames_ian_site3.bin  ({len(frame_data['frames'])} frames)")

        hydro_json = {
            "scenario_id": "ian_site3",
            "times_min": frame_data["times_min"],
            "rain_mm_hr": [float(rain_ts[min(int(tm*60/args.dt), n_steps-1)]) for tm in frame_data["times_min"]],
            "flooded_ha": [float(flooded_ha_ts[min(int(tm*60/args.dt), n_steps-1)]) for tm in frame_data["times_min"]],
            "total_rain_mm": float(total_rain),
            "peak_flooded_ha": peak_ha,
        }
        with open(os.path.join(OUT_DIR, "simulation_ian_site3_hydrograph.json"), "w") as fh:
            json.dump(hydro_json, fh)
        print(f"  simulation_ian_site3_hydrograph.json")

        # Copy into viewer/data/ (added 2026-08-04). Previously this never happened
        # automatically -- confirmed via MD5 during the post-friction-fix regeneration that
        # depth_frames_ian_site3.bin had regenerated correctly here in simulation/outputs/
        # while viewer/data/simulation_ian_site3_frames.bin silently stayed 8 days stale,
        # because nothing ever copied it there except a one-off manual `cp`. Every other
        # AOI-scale Ian export (export_ian_simulation.py) already does this step; this script
        # never had the equivalent.
        viewer_data_dir = os.path.join(PROJ_DIR, "viewer", "data")
        for src_name, dst_name in [
            ("depth_frames_ian_site3.bin", "simulation_ian_site3_frames.bin"),
            ("simulation_ian_site3_hydrograph.json", "simulation_ian_site3_hydrograph.json"),
        ]:
            shutil.copy2(os.path.join(OUT_DIR, src_name), os.path.join(viewer_data_dir, dst_name))
        print(f"  -> copied to viewer/data/ "
              f"(simulation_ian_site3_frames.bin, simulation_ian_site3_hydrograph.json)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
