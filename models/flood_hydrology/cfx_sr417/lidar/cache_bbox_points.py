"""
Per-tile, checkpointed LiDAR bbox-filtering cache
====================================================
Added 2026-07-27 after load_points_in_bbox() (build_lidar_pointcloud.py) repeatedly got killed
mid-run for site3 — it reads all matching .laz files in one uninterrupted call, and site3 has
25 large tiles (up to 655MB each) vs. the original AOI's 6 smaller ones, making one call take
long enough to hit whatever is killing it (no OOM entries in the system log, no traceback —
looks like an external/harness-level interruption, not a code bug). Same fix philosophy as
lidar/download_laz_tiles.py's resumable download: process ONE tile at a time, cache each tile's
already-filtered (much smaller) result to disk immediately, so a kill mid-run only costs the
one tile in progress, not the whole batch — and a re-run skips every tile already cached.

Usage:
    python3 lidar/cache_bbox_points.py --lat 28.690514 --lon -81.287539 --radius_km 2.99 \
        --cache-dir site3_gee_creek/lidar/data/bbox_cache
"""
import os, sys, glob, argparse
import numpy as np
import laspy
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from build_lidar_pointcloud import LAS_CRS, DEM_CRS, FT_TO_M, RAW_DIR  # noqa: E402


def bbox_from_center(lat, lon, radius_km):
    import math
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def cache_one_tile(laz_path, xmin_ft, xmax_ft, ymin_ft, ymax_ft, cache_path):
    if os.path.exists(cache_path):
        print(f"    already cached: {os.path.basename(cache_path)}")
        return
    print(f"    reading {os.path.basename(laz_path)} …")
    las = laspy.read(laz_path)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    mask = (x >= xmin_ft) & (x <= xmax_ft) & (y >= ymin_ft) & (y <= ymax_ft)
    n = int(mask.sum())
    print(f"      {len(x):,} points in tile -> {n:,} in bbox")
    np.savez_compressed(
        cache_path,
        x=x[mask], y=y[mask], z=z[mask],
        classification=np.asarray(las.classification)[mask],
        return_number=np.asarray(las.return_number)[mask],
        num_returns=np.asarray(las.number_of_returns)[mask],
    )


def load_cached_points(lon_min, lat_min, lon_max, lat_max, cache_dir, classification_filter=6):
    """Read every cached .npz in cache_dir and assemble the same dict load_points_in_bbox()
    returns (already in EPSG:5070 meters).

    classification_filter: added 2026-07-27 after this function itself got killed for site3 —
    concatenating ALL bbox-filtered points (1.16 BILLION for site3's 46.78km2 area) needs 25+ GB
    just for x/y/z as float64, well past this machine's 17GB RAM. The only caller of pts in
    this project's mesh-building pipeline (build_building_surfaces in droplet_flow_test.py) only
    ever uses classification==6 (building) points — filtering PER-FILE, before concatenating,
    keeps peak memory bounded by the much-smaller building-only subset instead of the full raw
    cloud. Pass None to disable (get everything, the old behavior) if a future caller needs it.

    Note: the lon_min/lat_min/lon_max/lat_max parameters were once accepted but never used to filter
    anything — every call loaded and returned ALL of cache_dir's points regardless of the
    requested box. Not a correctness bug (build_building_surfaces() still spatially filters
    correctly downstream via its own cKDTree query), but a real efficiency one: a small-area
    query (site3_crop's ~500m box) was loading and returning the SAME 52.19M full-site3-scale
    point set as a full-site3 query, for no benefit. Fixed by converting the requested lon/lat
    box to EPSG:2881 feet (same CRS the cached x/y are stored in, same conversion
    cache_one_tile() itself already uses) and filtering each tile's points against it, same
    per-tile-before-concatenating pattern the classification filter above already uses.
    """
    to_las = Transformer.from_crs("EPSG:4326", LAS_CRS, always_xy=True)
    bx0_ft, by0_ft = to_las.transform(lon_min, lat_min)
    bx1_ft, by1_ft = to_las.transform(lon_max, lat_max)
    bxmin_ft, bxmax_ft = sorted([bx0_ft, bx1_ft])
    bymin_ft, bymax_ft = sorted([by0_ft, by1_ft])

    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    xs, ys, zs, cls, rn, nr = [], [], [], [], [], []
    n_raw_total = 0
    for f in files:
        d = np.load(f)
        n_raw_total += len(d["x"])
        if len(d["x"]) == 0:
            continue
        fx, fy, fz = d["x"], d["y"], d["z"]
        fcls, frn, fnr = d["classification"], d["return_number"], d["num_returns"]
        in_box = (fx >= bxmin_ft) & (fx <= bxmax_ft) & (fy >= bymin_ft) & (fy <= bymax_ft)
        fx, fy, fz, fcls, frn, fnr = fx[in_box], fy[in_box], fz[in_box], fcls[in_box], frn[in_box], fnr[in_box]
        if classification_filter is not None:
            m = fcls == classification_filter
            fx, fy, fz, fcls, frn, fnr = fx[m], fy[m], fz[m], fcls[m], frn[m], fnr[m]
        if len(fx) == 0:
            continue
        xs.append(fx); ys.append(fy); zs.append(fz)
        cls.append(fcls); rn.append(frn); nr.append(fnr)
    n_kept = sum(len(a) for a in xs)
    print(f"    load_cached_points: {n_raw_total:,} raw bbox points -> {n_kept:,} kept "
          f"(classification_filter={classification_filter})")
    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    classification = np.concatenate(cls)
    return_number = np.concatenate(rn)
    num_returns = np.concatenate(nr)

    z_m = z * FT_TO_M
    to_dem = Transformer.from_crs(LAS_CRS, DEM_CRS, always_xy=True)
    X, Y = to_dem.transform(x, y)
    return {"x": X, "y": Y, "z": z_m, "classification": classification,
            "return_number": return_number, "num_returns": num_returns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius_km", type=float, required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox_from_center(args.lat, args.lon, args.radius_km)

    to_las = Transformer.from_crs("EPSG:4326", LAS_CRS, always_xy=True)
    xmin_ft, ymin_ft = to_las.transform(lon_min, lat_min)
    xmax_ft, ymax_ft = to_las.transform(lon_max, lat_max)
    xmin_ft, xmax_ft = sorted([xmin_ft, xmax_ft])
    ymin_ft, ymax_ft = sorted([ymin_ft, ymax_ft])

    laz_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.laz")))
    print(f"{len(laz_files)} raw tiles in {RAW_DIR}, caching bbox-filtered points to "
          f"{args.cache_dir} one tile at a time …")
    for i, laz_path in enumerate(laz_files, 1):
        cache_path = os.path.join(args.cache_dir, os.path.basename(laz_path) + ".npz")
        print(f"  [{i}/{len(laz_files)}] {os.path.basename(laz_path)}")
        cache_one_tile(laz_path, xmin_ft, xmax_ft, ymin_ft, ymax_ft, cache_path)

    print("\nAll tiles cached.")


if __name__ == "__main__":
    main()
