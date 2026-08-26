"""
Build (and cache) the mesh+masks bundle for the FULL site3 registry entry (ground_decimate=8,
the same production mesh flood_sim/mesh_shallow_water already use for real Ian runs) -- WITHOUT
running any GPU physics scenarios. Purpose: get the real full-scale edge count and mesh
structure so GNN INFERENCE (not training) tractability can be benchmarked before attempting a
full-scale surrogate run, per CLAUDE.md's 2026-07-27 GNN-rollout-validation entry's "not yet
done" list. Mirrors run_gnn_training_sweep.py's build_mesh_and_masks() exactly, just pointed at
the real "site3" site (not the training-only site3_crop/site3_crop_coarse crops) and with the
scenario-sweep step skipped entirely.

Usage:
    .venv/bin/python3 simulation/build_full_site3_mesh_masks.py
"""
import os, sys, pickle, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

from mesh_shallow_water import (  # noqa: E402
    build_combined_mesh, compute_ground_impervious_mask, compute_lake_mask,
    load_spatial_horton_points, load_nlcd_impervious_fraction_points,
    build_ground_surface, build_building_surfaces,
)
from cache_bbox_points import load_cached_points  # noqa: E402
from build_lidar_pointcloud import bbox_from_center  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE = "site3"
CKPT_DIR = os.path.join(PROJ_DIR, "site3_gee_creek", "gnn_training", SITE, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
MESH_CKPT = os.path.join(CKPT_DIR, "mesh_and_masks.pkl")


def main():
    if os.path.exists(MESH_CKPT):
        with open(MESH_CKPT, "rb") as fh:
            mesh, gim, lm, gh, gng = pickle.load(fh)
        print(f"[mesh+masks] already cached: {mesh['T']:,} triangles, {len(mesh['edges']['i']):,} edges")
        return

    print(f"[mesh+masks] building from scratch for full '{SITE}' (ground_decimate=8, the real "
          f"production-scale mesh) …")
    t0 = time.time()
    site = get_site(SITE)
    lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])
    ground = build_ground_surface(lon_min, lat_min, lon_max, lat_max,
                                   dem_cond_path=site.get("dem_cond_path"),
                                   decimate=site.get("ground_decimate", 1))
    pts = load_cached_points(lon_min, lat_min, lon_max, lat_max, site["bbox_cache_dir"])
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)
    buildings, building_polys = build_building_surfaces(
        pts, (gxmin, gymin, gxmax, gymax), buildings_path=site.get("buildings_path"),
        max_points_per_building=site.get("roof_max_points"))
    mesh = build_combined_mesh(ground, buildings)

    ground_xy = mesh["xy"][:mesh["Tg"]]
    ground_impervious_mask = compute_ground_impervious_mask(ground_xy, roads_path=site.get("roads_path"))
    lake_mask = compute_lake_mask(ground_xy, site.get("pond_id3dhp"))
    ground_horton = load_spatial_horton_points(
        ground_xy, mukey_map_path=site.get("mukey_map_path"),
        mukey_legend_path=site.get("mukey_legend_path"), soil_json_path=site.get("soil_json_path"))
    already_hard = ground_impervious_mask.copy()
    if lake_mask is not None:
        already_hard |= lake_mask
    ground_nlcd_grade = load_nlcd_impervious_fraction_points(
        ground_xy, already_hard, nlcd_path=site.get("nlcd_path"))

    elapsed = time.time() - t0
    print(f"  {mesh['T']:,} triangles, {len(mesh['edges']['i']):,} edges  ({elapsed:.1f}s)")
    bundle = (mesh, ground_impervious_mask, lake_mask, ground_horton, ground_nlcd_grade)
    tmp = MESH_CKPT + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh)
    os.replace(tmp, MESH_CKPT)
    print(f"Saved: {MESH_CKPT}")


if __name__ == "__main__":
    main()
