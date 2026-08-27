"""
Checkpointed site3 mesh build — same output as droplet_flow_test.py --site site3
==================================================================================
Added 2026-07-27 after droplet_flow_test.py's main() repeatedly got killed running for site3
(background execution in this environment seems to have a reliable window well under the
~350-450s the full pipeline needs at site3's scale — confirmed by timing each stage in
isolation: ground surface ~30-60s, point loading ~53-100s, build_building_surfaces ~268s even
after the cKDTree fix — each stage alone completes fine, but the CUMULATIVE total across all of
them in one uninterrupted run does not). Same resumability principle as
lidar/download_laz_tiles.py and lidar/cache_bbox_points.py: cache each expensive stage's result
to disk immediately, so a kill at any point only costs whatever stage was in progress, not
everything before it — re-running skips every already-cached stage.

Usage:
    python3 lidar/build_site3_mesh_checkpointed.py
    # re-run as many times as needed; already-completed stages are skipped instantly
"""
import os, sys, json, pickle, time


def atomic_pickle_dump(obj, path):
    """Write to a temp file then rename atomically — added 2026-07-27 after a real bug: a kill
    mid-write of the 5.98GB buildings.pkl left a truncated-but-existing file at the final path,
    which the NEXT run's os.path.exists() check treated as valid, then failed with EOFError
    trying to unpickle it. POSIX rename is atomic on the same filesystem, so the final path only
    ever contains a complete file — a kill during the write leaves an orphaned .tmp file instead
    of a corrupt "real" checkpoint."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp_path, path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from droplet_flow_test import (  # noqa: E402
    build_ground_surface, build_building_surfaces, run_droplets_fused, export_mesh_obj,
    DATA_DIR,
)
from cache_bbox_points import load_cached_points  # noqa: E402
from build_lidar_pointcloud import bbox_from_center, GEO_META, VERT_EXAG  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE = "site3"
CKPT_DIR = os.path.join(PROJ_DIR, "site3_gee_creek", "lidar", "data", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
GROUND_CKPT = os.path.join(CKPT_DIR, "ground.pkl")
BUILDINGS_CKPT = os.path.join(CKPT_DIR, "buildings.pkl")


def load_pickle_checkpoint(path):
    """Returns None (treat as not-cached) if the file is missing OR corrupt/truncated — the
    latter can genuinely happen if a previous run was killed mid-write before the atomic-rename
    fix above existed. Safe to call unconditionally."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"  WARNING: {path} exists but is corrupt/truncated ({e}) — rebuilding this stage")
        return None


def stage_ground(site, lon_min, lat_min, lon_max, lat_max):
    cached = load_pickle_checkpoint(GROUND_CKPT)
    if cached is not None:
        print("[stage: ground] already cached, loading …")
        return cached
    print("[stage: ground] building from DEM …")
    t0 = time.time()
    ground = build_ground_surface(lon_min, lat_min, lon_max, lat_max,
                                   dem_cond_path=site.get("dem_cond_path"),
                                   decimate=site.get("ground_decimate", 1))
    print(f"  {len(ground.simplices):,} ground triangles ({time.time()-t0:.1f}s)")
    atomic_pickle_dump(ground, GROUND_CKPT)
    print("  cached to", GROUND_CKPT)
    return ground


def stage_buildings(site, ground):
    cached = load_pickle_checkpoint(BUILDINGS_CKPT)
    if cached is not None:
        print("[stage: buildings] already cached, loading …")
        return cached
    print("[stage: buildings] loading LiDAR points + meshing roofs …")
    t0 = time.time()
    bbox_cache_dir = site.get("bbox_cache_dir")
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)
    lon_min = lat_min = lon_max = lat_max = None  # only needed if no bbox_cache_dir
    if bbox_cache_dir:
        # lon/lat bbox is only used by load_cached_points' internal signature; ground's own
        # extent (gxmin..gymax, already in EPSG:5070) is what actually matters here, but
        # load_cached_points is (lon,lat)-based, so pass through the site's own bbox instead.
        lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])
        pts = load_cached_points(lon_min, lat_min, lon_max, lat_max, bbox_cache_dir)
    else:
        from build_lidar_pointcloud import load_points_in_bbox
        lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])
        pts = load_points_in_bbox(lon_min, lat_min, lon_max, lat_max)
    print(f"  points loaded ({time.time()-t0:.1f}s so far)")
    buildings, building_polys = build_building_surfaces(
        pts, (gxmin, gymin, gxmax, gymax), buildings_path=site.get("buildings_path"),
        max_points_per_building=site.get("roof_max_points"))
    n_roof_tri = sum(len(b.simplices) for b in buildings)
    print(f"  {n_roof_tri:,} roof triangles across {len(buildings)} buildings "
          f"(total stage: {time.time()-t0:.1f}s)")
    atomic_pickle_dump((buildings, building_polys), BUILDINGS_CKPT)
    print("  cached to", BUILDINGS_CKPT)
    return buildings, building_polys


def main():
    site = get_site(SITE)
    suffix = f"_{SITE}"
    lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])

    print("=" * 70)
    print(f"Checkpointed mesh build — {SITE}: {site['label']}")
    print("=" * 70)

    ground = stage_ground(site, lon_min, lat_min, lon_max, lat_max)
    buildings, building_polys = stage_buildings(site, ground)

    print("\n[final] Exporting mesh + tracing droplets …")
    with open(GEO_META) as fh:
        geo_meta = json.load(fh)
    verts_list = [ground.verts] + [b.verts for b in buildings]
    simplices_list = [ground.simplices] + [b.simplices for b in buildings]
    export_mesh_obj(verts_list, simplices_list,
                     os.path.join(DATA_DIR, f"dense_test_area_mesh{suffix}.obj"), geo_meta)

    paths, settle_reason = run_droplets_fused(ground, buildings, building_polys,
                                               n_droplets=500, max_steps=50, step_m=0.4)

    import struct
    import numpy as np
    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]
    reason_code = {"local_min": 0, "left_mesh": 1, "max_steps": 2}
    out_path = os.path.join(DATA_DIR, f"droplet_paths{suffix}.bin")
    n_written = 0
    with open(out_path, "wb") as fh:
        fh.write(b"DROP")
        fh.write(struct.pack("<I", len(paths)))
        for path, reason in zip(paths, settle_reason):
            pts_arr = np.array(path, dtype=np.float32)
            if len(pts_arr) < 2:
                fh.write(struct.pack("<I", 0))
                fh.write(struct.pack("<B", reason_code.get(reason, 2)))
                continue
            sx = pts_arr[:, 0] - ox - w / 2
            sy = (pts_arr[:, 2] - z_min) * VERT_EXAG
            sz = oy + h / 2 - pts_arr[:, 1]
            scene = np.column_stack([sx, sy, sz]).astype(np.float32).reshape(-1)
            fh.write(struct.pack("<I", len(pts_arr)))
            fh.write(scene.tobytes())
            fh.write(struct.pack("<B", reason_code.get(reason, 2)))
            n_written += 1
    print(f"  {os.path.basename(out_path)}: {n_written} droplet paths written")

    n_roof_tri = sum(len(b.simplices) for b in buildings)
    summary = {
        "site": SITE, "site_label": site["label"],
        "n_droplets_seeded": len(paths), "n_paths_written": n_written,
        "n_ground_triangles": len(ground.simplices), "n_roof_triangles": n_roof_tri,
        "n_buildings_meshed": len(buildings),
    }
    with open(os.path.join(DATA_DIR, f"droplet_paths_summary{suffix}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nDONE.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
