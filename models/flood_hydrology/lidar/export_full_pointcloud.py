"""
Export a LiDAR point cloud → viewer/data/ — Johns Lake AOI
===============================================================
Ported from cfx_sr417_corridor's export_full_pointcloud.py 2026-07-28 (same binary format,
same NAIP-coloring approach), reading from this project's own build_lidar_pointcloud.py
(different CRS-handling story — see that module's docstring — since this AOI spans two
different LiDAR acquisitions, unlike CFX's single-acquisition AOI).

Binary format (unchanged): b'PCLD' + uint32 n_points + float32[n*3] positions (scene-space,
VERT_EXAG/z_min/origin baked in, same convention as terrain.js) + uint8[n*3] colors.

Usage:
    python3 lidar/export_full_pointcloud.py                      # decimated, NAIP colors
    python3 lidar/export_full_pointcloud.py --full                # every point, no decimation
    python3 lidar/export_full_pointcloud.py --color-by classification
"""
import os, sys, json, struct, argparse
import numpy as np
import rasterio
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lidar_pointcloud import (
    bbox_from_center, load_points_in_bbox, DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM,
    GEO_META, VERT_EXAG, CLASS_NAMES, DEM_CRS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
NAIP_PATH = os.path.join(PROJ_DIR, "soil", "data", "naip_2021_RGB.tif")

NOISE_CLASSES = {7, 18}   # low_point_noise, high_noise — excluded entirely

CLASS_COLORS = {
    1:  (110, 110, 110),   # unclassified
    2:  (150, 120, 80),    # ground — tan/brown
    3:  (120, 180, 90),    # low vegetation
    4:  (70, 150, 60),     # medium vegetation
    5:  (30, 110, 40),     # high vegetation
    6:  (175, 175, 180),   # building — gray
    9:  (60, 120, 210),    # water — blue
    17: (255, 85, 51),     # bridge deck
}
DEFAULT_COLOR = (100, 100, 100)


def color_by_classification(cls):
    colors = np.full((len(cls), 3), DEFAULT_COLOR, dtype=np.uint8)
    for code, rgb in CLASS_COLORS.items():
        colors[cls == code] = rgb
    return colors


def color_by_naip(x, y):
    with rasterio.open(NAIP_PATH) as src:
        img = src.read()  # (3, H, W) uint8
        transform, crs = src.transform, src.crs
        H, W = src.height, src.width

    to_naip = Transformer.from_crs(DEM_CRS, crs, always_xy=True)
    nx, ny = to_naip.transform(x, y)
    rows, cols = rasterio.transform.rowcol(transform, nx, ny)
    rows = np.clip(np.asarray(rows), 0, H - 1)
    cols = np.clip(np.asarray(cols), 0, W - 1)
    return img[:, rows, cols].T.astype(np.uint8)   # (n, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS_KM)
    ap.add_argument("--target-points", type=int, default=4_000_000)
    ap.add_argument("--full", action="store_true", help="Export every point, no decimation")
    ap.add_argument("--color-by", choices=["naip", "classification"], default="naip")
    ap.add_argument("--out-name", default="lidar_pointcloud.bin")
    args = ap.parse_args()

    lon_min, lat_min, lon_max, lat_max = bbox_from_center(args.lat, args.lon, args.radius_km)

    print("=" * 62)
    print(f"LiDAR point cloud export (Johns Lake) → viewer/data/{args.out_name}")
    print(f"  color-by={args.color_by}  full={args.full}  radius_km={args.radius_km}")
    print("=" * 62)

    print("\n[1/3] Loading + filtering point cloud …")
    pts = load_points_in_bbox(lon_min, lat_min, lon_max, lat_max)
    n_total = len(pts["x"])
    print(f"  Total points in query area: {n_total:,}")

    keep = ~np.isin(pts["classification"], list(NOISE_CLASSES))
    n_kept = int(keep.sum())
    print(f"  After dropping noise classes: {n_kept:,} ({100*n_kept/n_total:.2f}%)")

    if args.full:
        idx = np.where(keep)[0]
        step = 1
    else:
        step = max(1, n_kept // args.target_points)
        idx = np.where(keep)[0][::step]
    n_out = len(idx)
    print(f"  {'No decimation (--full)' if args.full else f'Decimating every {step}th point'} "
          f"→ {n_out:,} points")

    x, y, z = pts["x"][idx], pts["y"][idx], pts["z"][idx]
    cls = pts["classification"][idx]

    print(f"\n[2/3] Coloring by {args.color_by} + transforming to scene-space …")
    with open(GEO_META) as fh:
        geo_meta = json.load(fh)
    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]

    sx = (x - ox - w / 2).astype(np.float32)
    sy = ((z - z_min) * VERT_EXAG).astype(np.float32)
    sz = (oy + h / 2 - y).astype(np.float32)

    if args.color_by == "naip":
        colors = color_by_naip(x, y)
    else:
        colors = color_by_classification(cls)

    hist = {CLASS_NAMES.get(int(c), f"class_{int(c)}"): int((cls == c).sum())
            for c in np.unique(cls)}
    print("  Points by class in the exported cloud:")
    for name, count in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"    {name:28s} {count:>10,}")

    print("\n[3/3] Writing binary …")
    positions = np.empty(n_out * 3, dtype=np.float32)
    positions[0::3] = sx
    positions[1::3] = sy
    positions[2::3] = sz

    color_flat = colors.reshape(-1)

    out_path = os.path.join(DATA_DIR, args.out_name)
    with open(out_path, "wb") as fh:
        fh.write(b"PCLD")
        fh.write(struct.pack("<I", n_out))
        fh.write(positions.tobytes())
        fh.write(color_flat.tobytes())

    kb = os.path.getsize(out_path) / 1024
    print(f"  {args.out_name}: {n_out:,} points  ({kb/1024:.1f} MB)")

    summary_name = os.path.splitext(args.out_name)[0] + "_summary.json"
    with open(os.path.join(DATA_DIR, summary_name), "w") as fh:
        json.dump({"n_points": n_out, "n_total_query_area": n_total, "decimation_step": step,
                    "color_by": args.color_by, "full": args.full,
                    "classification_histogram": hist}, fh, indent=2)
    print(f"  {summary_name}")


if __name__ == "__main__":
    main()
