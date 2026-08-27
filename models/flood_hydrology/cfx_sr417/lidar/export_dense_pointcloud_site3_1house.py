"""
Export a real, raw dense LiDAR point cloud for JUST the site3_1house crop (~120m box)
-> viewer/data/lidar_pointcloud_site3_1house.bin
==============================================================================================
Direct response to feedback while looking at the 1-house 3D shallow-water demo: the flat
placeholder-tinted mesh doesn't read as a recognizable house up close, and the existing full-
site3 point cloud (30.5M points decimated across the ENTIRE 6x6km box) thins out to almost
nothing at any single ~120m spot. This exports a dedicated, un-decimated cloud scoped to just
this house's own small bbox -- reuses load_cached_points() (already correctly bbox-filters
per-tile as of the 2026-07-27 fix) with classification_filter=None (every class: ground,
vegetation, building -- not just the class-6 building points the mesh-building step uses), then
the same NAIP-coloring + PCLD-binary-export logic export_dense_pointcloud_site3.py already
uses. No decimation needed here (unlike the full-site3 export) -- at this crop's scale the real
point count is small enough (tens of thousands, not billions) to keep every single point.

Usage:
    python3 lidar/export_dense_pointcloud_site3_1house.py
"""
import os, sys, json, struct
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from build_lidar_pointcloud import VERT_EXAG, bbox_from_center  # noqa: E402
from cache_bbox_points import load_cached_points  # noqa: E402
from export_full_pointcloud import CLASS_NAMES  # noqa: E402
import export_full_pointcloud as efp  # noqa: E402
from test_sites import get_site  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
GEO_META_SITE3 = os.path.join(PROJ_DIR, "viewer", "data", "geo_meta_site3.json")
SITE3_NAIP_PATH = os.path.join(PROJ_DIR, "site3_gee_creek", "imagery", "data", "naip_2021_RGB.tif")


def main():
    site = get_site("site3_1house")
    lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])

    print("[1/3] Loading ALL LiDAR classes for the 1-house crop (no decimation) …")
    pts = load_cached_points(lon_min, lat_min, lon_max, lat_max, site["bbox_cache_dir"],
                              classification_filter=None)
    x, y, z, cls = pts["x"], pts["y"], pts["z"], pts["classification"]
    n_out = len(x)
    print(f"  {n_out:,} points kept")

    print(f"\n[2/3] Coloring {n_out:,} points by NAIP + transforming to site3 scene-space …")
    efp.NAIP_PATH = SITE3_NAIP_PATH
    colors = efp.color_by_naip(x, y)

    with open(GEO_META_SITE3) as fh:
        geo_meta = json.load(fh)
    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]

    sx = (x - ox - w / 2).astype(np.float32)
    sy = ((z - z_min) * VERT_EXAG).astype(np.float32)
    sz = (oy + h / 2 - y).astype(np.float32)

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

    out_path = os.path.join(DATA_DIR, "lidar_pointcloud_site3_1house.bin")
    with open(out_path, "wb") as fh:
        fh.write(b"PCLD")
        fh.write(struct.pack("<I", n_out))
        fh.write(positions.tobytes())
        fh.write(color_flat.tobytes())

    kb = os.path.getsize(out_path) / 1024
    print(f"  lidar_pointcloud_site3_1house.bin: {n_out:,} points ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
