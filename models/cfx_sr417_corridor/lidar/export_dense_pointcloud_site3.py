"""
Export a real, raw dense LiDAR point cloud for site3 -> viewer/data/lidar_pointcloud_site3.bin
==================================================================================================
Direct response to feedback: the earlier "dense LiDAR" layer for site3 was actually the
triangulated ground+roof MESH (dense_test_area_mesh_site3.obj, decimate=8 on ground + capped
roof points) — a solved surface, not raw points, and reads visually as sparse/blocky next to
the real thing. This script instead replicates export_full_pointcloud.py's OWN approach (the
one behind the main page's "Full-res cloud" / site1/site2's "Dense point cloud, NAIP colors"
layers, http://localhost:5051/) for site3: real, individual LiDAR returns (ground AND
vegetation AND buildings, not just class-6 buildings), colored by sampling NAIP, exported as a
THREE.Points cloud in the exact same PCLD binary format lidarPointCloud.js already knows how to
load -- so the JS side needs zero new rendering code, just a URL pointing at this file.

Real, memory-bounded difference from export_full_pointcloud.py needed here: that script's
load_points_in_bbox() reads all matching LAZ tiles in ONE call and decimates AFTER
concatenating everything into one array. For site3's bbox that's ~1.16 BILLION raw points
(all classes, unfiltered) across 25 tiles -- concatenating first would need on the order of
25-30GB just for x/y/z as float64, already the exact problem cache_bbox_points.py's own
docstring documents hitting for site3 at this scale (that fix was per-tile classification
filtering; this export needs a different per-tile fix since it can't drop non-building classes
the way the mesh-building step did). Fixed here by decimating EACH cached tile file
independently, right after loading it and before concatenating -- peak memory is bounded by one
tile's own point count, never the full 1.16B-point union.

Usage:
    python3 lidar/export_dense_pointcloud_site3.py --target-points 8000000
"""
import os, sys, glob, json, struct, argparse
import numpy as np
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from build_lidar_pointcloud import LAS_CRS, DEM_CRS, FT_TO_M, VERT_EXAG  # noqa: E402
from export_full_pointcloud import NOISE_CLASSES, CLASS_NAMES  # noqa: E402
import export_full_pointcloud as efp  # noqa: E402
from test_sites import get_site  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
GEO_META_SITE3 = os.path.join(PROJ_DIR, "viewer", "data", "geo_meta_site3.json")
SITE3_NAIP_PATH = os.path.join(PROJ_DIR, "site3_gee_creek", "imagery", "data",
                                "naip_2021_RGB.tif")


def load_decimated(cache_dir, overall_step):
    """Same per-tile logic as cache_bbox_points.load_cached_points(), but decimates EACH
    tile's own points (after dropping noise classes) by `overall_step` BEFORE concatenating,
    instead of loading everything then decimating once at the end -- see module docstring for
    why that ordering matters at site3's scale."""
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    xs, ys, zs, cls_list = [], [], [], []
    n_raw_total = 0
    n_after_noise_drop = 0
    for f in files:
        d = np.load(f)
        n_raw_total += len(d["x"])
        if len(d["x"]) == 0:
            continue
        fx, fy, fz, fcls = d["x"], d["y"], d["z"], d["classification"]
        keep = ~np.isin(fcls, list(NOISE_CLASSES))
        fx, fy, fz, fcls = fx[keep], fy[keep], fz[keep], fcls[keep]
        n_after_noise_drop += len(fx)
        if len(fx) == 0:
            continue
        xs.append(fx[::overall_step]); ys.append(fy[::overall_step])
        zs.append(fz[::overall_step]); cls_list.append(fcls[::overall_step])
    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    z = np.concatenate(zs) if zs else np.array([])
    cls = np.concatenate(cls_list) if cls_list else np.array([])
    print(f"  {n_raw_total:,} raw points -> {n_after_noise_drop:,} after noise-class drop "
          f"-> {len(x):,} after per-tile 1/{overall_step} decimation")

    z_m = z * FT_TO_M
    to_dem = Transformer.from_crs(LAS_CRS, DEM_CRS, always_xy=True)
    X, Y = to_dem.transform(x, y)
    return np.asarray(X), np.asarray(Y), z_m, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-points", type=int, default=8_000_000,
                     help="approx. final point count (site3's bbox has ~1.16B raw points "
                          "across all classes, so this is a real decimation, not --full)")
    args = ap.parse_args()

    site = get_site("site3")
    bbox_cache_dir = site["bbox_cache_dir"]

    # Estimate the overall decimation step from the already-known total (this project's own
    # earlier bbox-caching run printed 1,162,301,150 raw points for this exact bbox) -- avoids
    # a full first pass just to count. If that count is ever wrong, the true kept-count is
    # still printed below so the mismatch would be immediately visible, not silent.
    KNOWN_RAW_TOTAL = 1_162_301_150
    step = max(1, KNOWN_RAW_TOTAL // args.target_points)
    print(f"Target ~{args.target_points:,} points from ~{KNOWN_RAW_TOTAL:,} raw -> "
          f"per-tile decimation step {step}")

    print("\n[1/3] Loading + decimating cached tiles …")
    x, y, z, cls = load_decimated(bbox_cache_dir, step)
    n_out = len(x)

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

    out_path = os.path.join(DATA_DIR, "lidar_pointcloud_site3.bin")
    with open(out_path, "wb") as fh:
        fh.write(b"PCLD")
        fh.write(struct.pack("<I", n_out))
        fh.write(positions.tobytes())
        fh.write(color_flat.tobytes())

    kb = os.path.getsize(out_path) / 1024
    print(f"  lidar_pointcloud_site3.bin: {n_out:,} points ({kb/1024:.1f} MB)")


if __name__ == "__main__":
    main()
