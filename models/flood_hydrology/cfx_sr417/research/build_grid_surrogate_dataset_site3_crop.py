#!/usr/bin/env python3
"""The apples-to-apples half of the mesh-GNN vs. grid-transformer comparison.

The 2026-08-24/25 comparison so far has a real asymmetry, flagged honestly but not fixed until
now: the mesh-GNN was trained on `site3_crop_coarse` (lidar/test_sites.py — a small, ~500m×500m
SPATIAL CROP around Gee Creek's pour point, chosen to make mesh training tractable), while the
grid-transformer was trained on site3's FULL spatial domain at a COARSENED (25m) resolution.
Different axes were shrunk (space vs. resolution) — not a controlled comparison.

This script trains the grid-transformer's own solver-driven data pipeline on the EXACT SAME
region `site3_crop_coarse` registers (same lat/lon/radius_km, same real pour-point location the
GNN study picked), at 5m — this project's PRODUCTION cell size, not an arbitrary coarse one —
since the crop is small enough that production resolution is cheap here (~100×100 cells,
comparable in count to the GNN's own ~6,500-7,000 training nodes). This directly tests: does the
grid-transformer's own accuracy profile (rollout collapse, near-zero spatial IoU — see
research/README.md) hold at a small-area, full-production-resolution scale too, or
was the full-domain-coarsened setup itself contributing to the failure?

DEM windowing note: `flood_sim_ian.load_dem_for_sim()` has no bounding-box-clipping capability
(always reads site3's FULL registered DEM) — this script reads/windows/resamples the DEM
independently rather than editing that shared function (same non-invasive pattern every other
site3 script in this project uses), then calls `fsi.run_sim()`/`fsi.load_spatial_horton()`
directly with the resulting clipped z/profile, which is all those functions actually need. Uses
4-CORNER bbox reprojection into the DEM's native EPSG:5070 CRS, not 2 corners — this project has
already hit real meridian-convergence-skew bugs from the 2-corner shortcut at this exact
longitude — not repeating that mistake.
"""
import os
import sys
import time
import argparse

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "data", "grid_surrogate_site3_crop")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
sys.path.insert(0, PROJ_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
sys.path.insert(0, os.path.join(PROJ_DIR, "precipitation"))
sys.path.insert(0, os.path.join(PROJ_DIR, "analysis"))

import contextlib                     # noqa: E402
with contextlib.redirect_stdout(sys.stderr):
    import flood_sim_ian as fsi       # noqa: E402
    import noaa_atlas14 as a14        # noqa: E402
    import flood_probability as fp    # noqa: E402
    from test_sites import get_site   # noqa: E402

SITE = "site3_crop_coarse"
site = get_site(SITE)   # lat=28.703898850286457, lon=-81.29064288498651, radius_km=0.25

fsi.DEM_COND     = site["dem_cond_path"]
fsi.SOIL_JSON    = site["soil_json_path"]
fsi.MUKEY_MAP    = site["mukey_map_path"]
fsi.MUKEY_LEGEND = site["mukey_legend_path"]
fsi.ROADS_PATH   = site["roads_path"]
fsi.BUILDINGS_PATH = site["buildings_path"]
fsi.NLCD_IMPERVIOUS_PATH = site["nlcd_path"]
fsi.HORTON = fsi._load_horton_params()

RETURN_PERIODS_YR = [1, 2, 10, 25, 100, 500]
HELD_OUT_YR = {1, 500}


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def load_cropped_dem(cell_size_m):
    """Window + resample the DEM to the site3_crop_coarse bbox, 4-corner reprojection.

    Uses the raster's own INVERSE transform to convert world coords -> pixel row/col directly,
    not rasterio.windows.from_bounds (which assumes standard north-up orientation) — site3's DEM
    has a positive y-resolution (transform.e > 0, bounds.bottom > bounds.top), the same real
    orientation quirk this project has hit and fixed the same way multiple times before."""
    west, south, east, north = bbox_from_center(site["lat"], site["lon"], site["radius_km"])
    with rasterio.open(fsi.DEM_COND) as src:
        native_res = abs(src.transform.a)
        tx = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        corners = [tx.transform(lon, lat) for lon, lat in
                   [(west, south), (west, north), (east, south), (east, north)]]
        xs, ys = zip(*corners)
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)

        inv = ~src.transform
        cols, rows = [], []
        for x, y in [(left, bottom), (left, top), (right, bottom), (right, top)]:
            c, r = inv * (x, y)
            cols.append(c); rows.append(r)
        col0, col1 = int(np.floor(min(cols))), int(np.ceil(max(cols)))
        row0, row1 = int(np.floor(min(rows))), int(np.ceil(max(rows)))
        col0, row0 = max(col0, 0), max(row0, 0)
        col1, row1 = min(col1, src.width), min(row1, src.height)

        win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        dem_crop = src.read(1, window=win).astype(np.float32)
        crop_tf = src.window_transform(win)
        nodata = src.nodata if src.nodata is not None else -9999.0  # 0.0 is falsy
        crs = src.crs

    print(f"  DEM native res {native_res:.2f}m, cropped window {dem_crop.shape} "
          f"({(right-left):.0f}m x {(top-bottom):.0f}m)")

    if cell_size_m <= native_res * 1.1:
        z = dem_crop.copy()
        z[z == nodata] = np.nan
        return z, {"transform": crop_tf, "crs": crs}, native_res

    new_h = int((top - bottom) / cell_size_m)
    new_w = int((right - left) / cell_size_m)
    dst_tf = transform_from_bounds(left, bottom, right, top, new_w, new_h)
    z_c = np.zeros((new_h, new_w), dtype=np.float32)
    # Was Resampling.bilinear — a hand-rolled copy of flood_sim_ian.load_dem_for_sim that
    # carried the same defect: averaging kernels blend richdem's one-cell-wide breach channels
    # back into the surrounding ground, reintroducing depression storage the conditioning had
    # removed (measured on site3 at 5 m: 3.710e6 m³, 20.3 % of the Ian storm). Reference the
    # solver's own constant so the two cannot drift apart again.
    reproject(dem_crop, z_c, src_transform=crop_tf, src_crs=crs,
              dst_transform=dst_tf, dst_crs=crs,
              src_nodata=nodata, dst_nodata=nodata,
              resampling=fsi.DEM_RESAMPLING)
    z_c[z_c == nodata] = np.nan
    return z_c, {"transform": dst_tf, "crs": crs}, cell_size_m


def build_storm(T, depth_mm, duration_hr, cell_size_m, dt_s, frame_interval_min):
    z, profile, dx = load_cropped_dem(cell_size_m)
    horton = fsi.load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton is not None:
        horton = fsi.apply_impervious_mask(horton, z.shape, profile["transform"], profile["crs"])
        horton = fsi.apply_nlcd_graded_impervious(horton, z.shape, profile["transform"], profile["crs"])

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

    frames = np.stack(frame_data["frames"]).astype(np.float32)
    times_min = np.array(frame_data["times_min"], dtype=np.float32)
    frame_rain_mm_hr = np.interp(times_min * 60.0, t_sim, rain_sim) * 3600.0 * 1000.0

    print(f"  T={T:>4}yr  grid={z.shape}  n_frames={len(frames)}  "
          f"peak_depth={h_max.max():.3f}m  peak_flooded={np.max(flooded_ha_ts):.2f}ha  [{el:.1f}s]")
    return frames.astype(np.float32), times_min, frame_rain_mm_hr.astype(np.float32), dx, z.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-size", type=float, default=5.0,
                    help="Production resolution — the crop is small enough this is affordable.")
    ap.add_argument("--duration-hr", type=float, default=24.0)
    ap.add_argument("--dt", type=float, default=10.0,
                    help="Finer than the full-domain run's 20s — smaller cells need a shorter "
                         "CFL-safe step; matches this project's own resolution-vs-dt convention.")
    ap.add_argument("--frame-interval-min", type=float, default=10.0,
                    help="Finer than the full-domain run's 20min — a small, fast-responding crop "
                         "needs finer temporal sampling to resolve its own dynamics.")
    args = ap.parse_args()

    print("=" * 74)
    print(f"  Grid-surrogate training corpus — site3_crop_coarse (GNN's own region), "
          f"{args.cell_size:.0f}m production resolution")
    print("=" * 74)
    print(f"  region: lat={site['lat']}, lon={site['lon']}, radius_km={site['radius_km']} "
          f"(same as the mesh-GNN's own site3_crop_coarse)")

    print(f"\n[1/2] NOAA Atlas 14 IDF …")
    depths = fp.load_idf(site["lat"], site["lon"], args.duration_hr)

    print(f"\n[2/2] Running {len(RETURN_PERIODS_YR)} design storms …")
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
        json.dump(dict(site=SITE, cell_size_m=args.cell_size, duration_hr=args.duration_hr,
                      dt_s=args.dt, frame_interval_min=args.frame_interval_min,
                      storms=manifest), f, indent=2)
    print(f"\nSaved {len(manifest)} storms to {os.path.relpath(OUT_DIR, PROJ_DIR)}/")


if __name__ == "__main__":
    main()
