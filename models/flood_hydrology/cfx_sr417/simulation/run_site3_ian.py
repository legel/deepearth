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

BASEFLOW_CFS = 45.2   # gauge 02234400, 2022-09-27 19:30 UTC — pre-Ian steady flow

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
    ap.add_argument("--storage-scale", type=float, default=1.0,
                     help="Multiply the SSURGO-derived soil-storage cap by this factor. 1.0 is "
                          "the derived value (mean 206 mm at site3); 0 forces immediate "
                          "saturation excess; use --no-storage-cap for unbounded infiltration. "
                          "For the sensitivity sweep — the cap has never had one, and site3's "
                          "206 mm against the main AOI's 36 mm is the leading unexplained "
                          "difference between a site that reproduces plausible runoff and one "
                          "that does not.")
    ap.add_argument("--no-storage-cap", action="store_true",
                     help="Disable the cap entirely (unbounded Horton infiltration).")
    ap.add_argument("--tag", default="", type=str,
                     help="Suffix for output filenames, so sweep arms don't overwrite each "
                          "other or the canonical hydrograph_ian_site3.csv.")
    ap.add_argument("--baseflow", action="store_true",
                     help="Pre-fill the channel to the depth gauge 02234400's measured 45.2 cfs "
                          "baseflow implies, instead of starting it dry. OFF by default: the "
                          "initial condition validates itself well (simulated 47.9 cfs at the "
                          "gauge cell at t=0 vs 45.2 observed) but MEASURABLY HURTS the run — "
                          "total infiltration rose 2.56 -> 5.00 million m³ and storm runoff "
                          "reaching the gauge fell 0.189 -> 0.080 million m³. Zeroing the "
                          "channel's soil storage (a perennial channel sits at the water table) "
                          "was tried and changed nothing: only 3,504 of 8,846 channel cells had "
                          "any storage to zero. The extra infiltration is on the floodplain — "
                          "a conveying channel moves water across more distinct cells, and each "
                          "one it wets unlocks its own soil storage. Kept for that experiment, "
                          "not because it improves the result.")
    ap.add_argument("--no-ponded-infiltration", action="store_true",
                     help="Restore the previous rainfall-only-limited infiltration "
                          "(Pe = max(P - inf, 0)): already-ponded water cannot infiltrate "
                          "once rain stops. Default (ponded infiltration on) lets standing "
                          "depth keep draining into any remaining soil storage after rain ends.")
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
    if args.no_storage_cap:
        max_deficit_m = None
        print("  Soil storage cap DISABLED — infiltration is unbounded")
    elif max_deficit_m is not None and args.storage_scale != 1.0:
        max_deficit_m = max_deficit_m * args.storage_scale
        print(f"  Soil storage cap scaled x{args.storage_scale:g} -> mean "
              f"{1000 * float(max_deficit_m.mean()):.0f} mm")
    if max_deficit_m is not None:
        print(f"  Soil storage cap: mean {1000*float(max_deficit_m.mean()):.0f} mm, "
              f"range {1000*float(max_deficit_m.min()):.0f}-{1000*float(max_deficit_m.max()):.0f} mm, "
              f"{100*float((max_deficit_m == 0).mean()):.0f}% of cells depressional (zero storage)")
    else:
        print(f"  WARNING: no soil storage table at {fsi.SOIL_STORAGE_CSV} — "
              f"infiltration is UNBOUNDED (run soil/fetch_soil_storage.py --site site3)")

    # Locate USGS 02234400 on the solver grid, and snap it onto the channel: the gauge
    # coordinate can land a cell or two off the burned centreline, and reading a dry
    # floodplain cell next to the creek would report ~no discharge. Snap to the deepest
    # (lowest-bed) cell within 25 m, which on a burned DEM is the channel itself.
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
    print(f"  Gauge {site['gauge_site_no']} at grid {gauge_rc}, bed {z[gauge_rc]:.2f} m "
          f"(snapped {np.hypot(gauge_rc[0]-_gr, gauge_rc[1]-_gc)*dx:.0f} m to the channel)")

    # ── Baseflow initial condition ───────────────────────────────────────────────────────
    # Pre-fill the burned channel to the depth its own measured baseflow implies. Gauge 02234400
    # read 45.2 cfs immediately before Ian; a perennial creek is not empty when a storm starts.
    # Depth from Manning at normal flow, using the along-channel bed slope measured on THIS grid
    # rather than an assumed value:  h = ( Q n / (w sqrt(S)) )^(3/5).
    if args.baseflow:
        import geopandas as _gpd
        from rasterio.features import rasterize as _rasterize
        _fl = os.path.join(SITE3_DIR, "hydrography", "data", "3dhp_flowlines.geojson")
        _g = _gpd.read_file(_fl).to_crs(profile["crs"])
        _ch = _rasterize([(x, 1) for x in _g.geometry.buffer(dx) if x is not None],
                         out_shape=z.shape, transform=profile["transform"],
                         fill=0, dtype=np.uint8).astype(bool)
        # along-channel bed slope on the solver grid
        _inv = ~profile["transform"]; _D = _L = 0.0
        for _ln in _g.geometry:
            if _ln is None or _ln.length < 20:
                continue
            _zs = []
            for _d in np.arange(0, _ln.length, dx):
                _p = _ln.interpolate(float(_d)); _c, _r = _inv * (_p.x, _p.y)
                _r, _c = int(_r), int(_c)
                if 0 <= _r < z.shape[0] and 0 <= _c < z.shape[1] and np.isfinite(z[_r, _c]):
                    _zs.append(z[_r, _c])
            if len(_zs) > 3:
                _D += _zs[0] - _zs[-1]; _L += _ln.length
        _S = max(_D / _L, 1e-4) if _L > 0 else 1e-3
        _Q = BASEFLOW_CFS / 35.3147                     # m³/s
        _h0 = ((_Q * fsi.MANNING_N) / (dx * np.sqrt(_S))) ** 0.6
        initial_h = np.zeros_like(z); initial_h[_ch & np.isfinite(z)] = _h0

        # A channel carrying perennial baseflow is, by definition, at the water table: its
        # available soil storage is ZERO. Leaving channel cells with the SSURGO column storage
        # (206 mm mean here) let the pre-filled channel soak away for 72 h — measured, that
        # DOUBLED total infiltration (2.56 -> 5.00 million m³) and cut storm runoff reaching
        # the gauge 13x (0.189 -> 0.015 million m³), while inflating the runoff coefficient to
        # a flattering 25 % that was almost entirely baseflow passing through. The solver
        # already zeroes storage for the 26 % of cells SSURGO flags depressional; a perennial
        # channel belongs in exactly that category.
        if max_deficit_m is not None:
            _nz = int((max_deficit_m[_ch] > 0).sum())
            max_deficit_m[_ch] = 0.0
            print(f"  Channel soil storage zeroed in {_nz:,} cells "
                  f"(perennial channel sits at the water table)")
        print(f"  Baseflow IC: {BASEFLOW_CFS} cfs, along-channel S={_S:.2e} "
              f"-> channel depth {_h0:.3f} m over {int(_ch.sum()):,} cells")
    else:
        initial_h = None
        print("  Channel starts dry (pass --baseflow to pre-fill it; see the flag's help)")

    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = fsi.run_sim(
        z, dx, rain_sim, args.dt, frame_interval_min=args.frame_interval, use_infiltration=True,
        horton_arrays=horton_arrays, max_deficit_m=max_deficit_m, gauge_rc=gauge_rc,
        initial_h=initial_h, ponded_infiltration=not args.no_ponded_infiltration,
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
    hydro_path = os.path.join(OUT_DIR, f"hydrograph_ian_site3{args.tag}.csv")
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
    with open(os.path.join(OUT_DIR, f"ian_sim_summary_site3{args.tag}.json"), "w") as fh:
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
