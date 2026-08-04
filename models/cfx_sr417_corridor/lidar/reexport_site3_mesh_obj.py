"""
Re-export dense_test_area_mesh_site3.obj with site3's OWN geo_meta (not the main AOI's),
and real per-vertex NAIP color (added 2026-07-27, same session, per direct request: "why is
the dense LiDAR green, can we not use NAIP's color for ground and roof like site2?")
==========================================================================================
Real bug found 2026-07-27 while wiring up the site3 viewer page: build_site3_mesh_checkpointed
.py's export_mesh_obj() call used the imported GEO_META constant from build_lidar_pointcloud.py,
which points at viewer/data/geo_meta.json -- the MAIN AOI's own scene metadata (origin_x,
origin_y, width_m, height_m, z_min). geo_meta_site3.json didn't exist yet at the time that mesh
was built (it's a product of this same session's viewer work). Baking site3's real-world
EPSG:5070 vertex coordinates against the main AOI's origin/dimensions would place the mesh
wildly out of alignment if loaded into the site3 scene -- verts[:,0] (site3's real X, ~37km from
the main AOI) minus the main AOI's own origin_x is a huge, meaningless offset.

Coloring: site1/site2's "dense point cloud" layers color each LiDAR point by sampling the NAIP
orthophoto at that point's own (x,y) -- export_full_pointcloud.py's color_by_naip(). This mesh
is a triangulated SURFACE, not a raw point cloud, but the same idea applies at the vertex level:
reuses color_by_naip() directly (monkey-patched NAIP_PATH -> site3's own naip_2021_RGB.tif,
same non-invasive pattern as every other site3 script) and writes the result as real per-vertex
color via export_mesh_obj()'s new colors_list parameter, instead of the flat placeholder tint
site3_main.js used originally.

Fast to fix/re-run: reuses the already-cached ground.pkl/buildings.pkl checkpoints (no rebuild),
just redoes the lightweight OBJ-writing step.

Usage:
    python3 lidar/reexport_site3_mesh_obj.py
"""
import os, sys, json, pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from droplet_flow_test import export_mesh_obj, DATA_DIR  # noqa: E402
import export_full_pointcloud as efp  # noqa: E402

CKPT_DIR = os.path.join(PROJ_DIR, "site3_gee_creek", "lidar", "data", "checkpoints")
GROUND_CKPT = os.path.join(CKPT_DIR, "ground.pkl")
BUILDINGS_CKPT = os.path.join(CKPT_DIR, "buildings.pkl")
GEO_META_SITE3 = os.path.join(PROJ_DIR, "viewer", "data", "geo_meta_site3.json")
SITE3_NAIP_PATH = os.path.join(PROJ_DIR, "site3_gee_creek", "imagery", "data",
                                "naip_2021_RGB.tif")


def main():
    with open(GROUND_CKPT, "rb") as fh:
        ground = pickle.load(fh)
    with open(BUILDINGS_CKPT, "rb") as fh:
        buildings, building_polys = pickle.load(fh)
    with open(GEO_META_SITE3) as fh:
        geo_meta = json.load(fh)

    efp.NAIP_PATH = SITE3_NAIP_PATH

    verts_list = [ground.verts] + [b.verts for b in buildings]
    simplices_list = [ground.simplices] + [b.simplices for b in buildings]

    # color_by_naip() opens + reads the ENTIRE NAIP GeoTIFF from disk on every call — fine for
    # site1/site2's single whole-point-cloud call, but calling it once per surface here (1
    # ground + 10,739 buildings) would re-read a ~317MB file 10,740 times. Caught before it ran
    # away (killed a first attempt after seeing it start). Fixed by concatenating every
    # surface's vertices into one array, sampling ONCE, then splitting the result back apart —
    # same total answer, one disk read instead of 10,740.
    print(f"Sampling NAIP color for {sum(len(v) for v in verts_list):,} vertices "
          f"(one batched read, not one per surface) …")
    all_x = np.concatenate([v[:, 0] for v in verts_list])
    all_y = np.concatenate([v[:, 1] for v in verts_list])
    all_colors = efp.color_by_naip(all_x, all_y)
    colors_list = []
    offset = 0
    for v in verts_list:
        colors_list.append(all_colors[offset:offset + len(v)])
        offset += len(v)

    out_path = os.path.join(DATA_DIR, "dense_test_area_mesh_site3.obj")
    export_mesh_obj(verts_list, simplices_list, out_path, geo_meta, colors_list=colors_list)


if __name__ == "__main__":
    main()
