"""
DEM Download — CFX SR417 Corridor Test-Landscape AOI
=======================================================
Downloads the highest-available lidar DEM from USGS 3DEP for a 2x2 km
study area centered on the candidate SR417 corridor test-landscape site
near Lake Nona / south Orlando, FL (28.36687N, -81.43299W).

Uses py3dep (USGS TNM / 3DEP API, free, no account required).
Available resolutions: 1m (where lidar exists), 3m (1/9 arc-sec), 10m (1/3 arc-sec).
Florida has statewide QL1 lidar coverage; 1m is preferred.

Vertical datum: NAVD88 (hydro-flattened — water surfaces appear flat at
water-surface elevation, not the channel/pond bed).

Outputs (saved under dem/data/):
    sr417_corridor_dem.tif       — raw downloaded DEM, best available resolution
    sr417_corridor_dem_meta.json — bounding box, CRS, resolution, source info

Usage:
    python3 dem/dem_download.py
    python3 dem/dem_download.py --lat 28.36687 --lon -81.43299 --radius_km 1.0 --resolution 3
"""

import os
import sys
import json
import argparse
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Shared site registry (added 2026-08-04) ──────────────────────────────────
# Makes `--site <name>` resolve lat/lon/radius AND the output directory from the ONE registry
# (site_registry.py -> lidar/test_sites.py) instead of hand-typed coordinates. Purely additive:
# with no --site flag this script behaves exactly as it always has. See site_registry.py's
# docstring for why: hand-typed per-invocation coordinates leave fetched data on disk with no
# script that can reproduce it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import site_registry  # noqa: E402

os.makedirs(DATA_DIR, exist_ok=True)

# AOI center — candidate SR417 corridor test-landscape site (Lake Nona / south Orlando, FL)
DEFAULT_LAT    = 28.36687
DEFAULT_LON    = -81.43299
DEFAULT_RADIUS = 1.0       # km -> 2x2 km study box
DEFAULT_RES    = 3         # meters; try 1 first, fall back to 3 then 10


def bbox_from_center(lat, lon, radius_km):
    """Return (west, south, east, north) bounding box in EPSG:4326."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def download_dem(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=DEFAULT_RADIUS,
                 resolution=DEFAULT_RES, out_path=None):
    import py3dep
    import rioxarray  # noqa: F401 — needed for .rio accessor on returned DataArray

    if out_path is None:
        out_path = os.path.join(DATA_DIR, "sr417_corridor_dem.tif")

    west, south, east, north = bbox_from_center(lat, lon, radius_km)
    bbox = (west, south, east, north)
    print(f"Study area  : {south:.5f}°N - {north:.5f}°N, {west:.5f}°E - {east:.5f}°E")
    print(f"Box size    : {2*radius_km:.1f} x {2*radius_km:.1f} km")

    resolutions_to_try = sorted(set([resolution, 3, 10]))  # fallback ladder
    dem = None
    used_res = None

    for res in resolutions_to_try:
        print(f"Requesting DEM at {res}m resolution from USGS 3DEP …")
        try:
            dem = py3dep.get_dem(bbox, crs="epsg:4326", resolution=res)
            used_res = res
            print(f"  ✓ Received {dem.shape} grid at {res}m")
            break
        except Exception as exc:
            print(f"  ✗ {res}m failed: {exc}")

    if dem is None:
        sys.exit("Could not download DEM at any resolution. Check internet connection or USGS TNM service.")

    dem.rio.to_raster(out_path)
    print(f"Saved DEM   : {out_path}")

    meta = {
        "source": "USGS 3DEP via py3dep",
        "property": "CFX SR417 corridor test-landscape AOI, near Lake Nona / south Orlando, FL",
        "center_lat": lat,
        "center_lon": lon,
        "radius_km": radius_km,
        "bbox_wsen": list(bbox),
        "resolution_m": used_res,
        "crs": str(dem.rio.crs),
        "shape_yx": list(dem.shape),
        "nodata": float(dem.rio.nodata) if dem.rio.nodata is not None else None,
        "z_min_m": float(dem.min().values),
        "z_max_m": float(dem.max().values),
        "vertical_datum": "NAVD88 (hydro-flattened)",
    }
    meta_path = out_path.replace(".tif", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta  : {meta_path}")
    print(f"Elevation   : {meta['z_min_m']:.1f} – {meta['z_max_m']:.1f} m NAVD88")
    return out_path, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download USGS 3DEP DEM for the SR417 corridor AOI")
    parser.add_argument("--lat",        type=float, default=DEFAULT_LAT,    help="Center latitude")
    parser.add_argument("--lon",        type=float, default=DEFAULT_LON,    help="Center longitude")
    parser.add_argument("--radius_km",  type=float, default=DEFAULT_RADIUS, help="Half-width of study box in km")
    parser.add_argument("--resolution", type=int,   default=DEFAULT_RES,    help="Target resolution in meters (1, 3, or 10)")
    parser.add_argument("--out",        type=str,   default=None,           help="Output GeoTIFF path")
    site_registry.add_site_arg(parser)
    args = site_registry.resolve(parser.parse_args(), category="dem")
    if args.site_data_root:
        # Rebind the module-level DATA_DIR so every function writing output lands in the
        # selected site's own tree instead of the main AOI's (the exact clobbering
        # fetch_naip_site3.py's docstring warns about — both share e.g. naip_2021_RGB.tif).
        globals()["DATA_DIR"] = args.site_data_dir

    download_dem(args.lat, args.lon, args.radius_km, args.resolution, args.out)
