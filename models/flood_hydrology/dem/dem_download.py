"""
DEM Download — Winter Garden FL Property
==========================================
Downloads the highest-available lidar DEM from USGS 3DEP for a 2×2 km
study area centered on 17801 Champagne Dr, Winter Garden, FL 34787.

Uses py3dep (USGS TNM / 3DEP API, free, no account required).
Available resolutions: 1m (where lidar exists), 3m (1/9 arc-sec), 10m (1/3 arc-sec).
Florida has statewide QL1 lidar coverage; 1m is preferred.

Vertical datum: NAVD88 (hydro-flattened — lake surfaces appear flat at
water-surface elevation, not the lake bed).

Outputs (saved under dem/data/):
    winter_garden_dem_1m.tif     — raw downloaded DEM, best available resolution
    winter_garden_dem_meta.json  — bounding box, CRS, resolution, source info

Usage:
    python3 dem/dem_download.py
    python3 dem/dem_download.py --lat 28.5652 --lon -81.5868 --radius_km 1.0 --resolution 3
"""

import os
import sys
import json
import argparse
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Property center
DEFAULT_LAT    = 28.521592   # 17801 Champagne Dr, Winter Garden FL (28°31'17.73"N 81°39'25.13"W)
DEFAULT_LON    = -81.656981
DEFAULT_RADIUS = 1.0       # km → 2×2 km study box
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
        out_path = os.path.join(DATA_DIR, "winter_garden_dem.tif")

    west, south, east, north = bbox_from_center(lat, lon, radius_km)
    bbox = (west, south, east, north)
    print(f"Study area  : {south:.5f}°N – {north:.5f}°N, {west:.5f}°E – {east:.5f}°E")
    print(f"Box size    : {2*radius_km:.1f} × {2*radius_km:.1f} km")

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
        "property": "17801 Champagne Dr, Winter Garden, FL 34787",
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
    parser = argparse.ArgumentParser(description="Download USGS 3DEP DEM for property AOI")
    parser.add_argument("--lat",        type=float, default=DEFAULT_LAT,    help="Center latitude")
    parser.add_argument("--lon",        type=float, default=DEFAULT_LON,    help="Center longitude")
    parser.add_argument("--radius_km",  type=float, default=DEFAULT_RADIUS, help="Half-width of study box in km")
    parser.add_argument("--resolution", type=int,   default=DEFAULT_RES,    help="Target resolution in meters (1, 3, or 10)")
    parser.add_argument("--out",        type=str,   default=None,           help="Output GeoTIFF path")
    args = parser.parse_args()

    download_dem(args.lat, args.lon, args.radius_km, args.resolution, args.out)
