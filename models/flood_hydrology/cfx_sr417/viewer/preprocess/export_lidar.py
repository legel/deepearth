"""
Export raw-LiDAR viewer assets → viewer/data/

Copies two kinds of LiDAR-derived assets into viewer/data/:
  1. The two SR417 bridge-crossing TIN meshes (lidar/build_lidar_pointcloud.py) — see
     lidar/data/BRIDGE_VALIDATION.md for why these exist (bare-earth DEM drops the highway
     ~7.5-8.4m to grade level at both crossings; these meshes are the real point-cloud surface).
  2. The full-AOI decimated, classification-colored point cloud
     (lidar/export_full_pointcloud.py) — ~4M points, all of the downloaded LiDAR data, not just
     the two bridges.
Both are pre-baked into the viewer's scene-space coordinate convention (see terrain.js's
VERT_EXAG/z_min/origin transform) — no positioning math needed on the JS side.

Run both lidar scripts before this one:
    python3 lidar/build_lidar_pointcloud.py
    python3 lidar/export_full_pointcloud.py

Then run this script:
    python3 viewer/preprocess/export_lidar.py
"""
import os, shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
PROJ_DIR = os.path.dirname(BASE_DIR)
LIDAR_DIR = os.path.join(PROJ_DIR, "lidar", "data")
SIM_DIR   = os.path.join(PROJ_DIR, "simulation", "outputs")
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = [
    "bridge_mesh_town_loop_blvd.obj",
    "bridge_mesh_john_young_pkwy.obj",
    "lidar_pointcloud.bin",
    "lidar_pointcloud_full.bin",
    "lidar_pointcloud_5houses.bin",          # site1 dense point cloud (NAIP colors)
    "lidar_pointcloud_5houses_class.bin",    # kept on disk; no longer a separate panel toggle
    "dense_test_area_mesh.obj",              # site1 fused mesh (meshShallowWater.js's own mesh)
]
# droplet_paths*.bin (droplet_flow_test.py's output) intentionally NOT copied here anymore —
# the standalone droplet layer was removed from the panel 2026-07-21 (superseded by the
# physics-driven flow tracers, see flow_tracer_paths* in OPTIONAL_FILES below); nothing in the
# viewer reads droplet_paths.bin any more, so it shouldn't be able to block the whole
# preprocessing pipeline the way a strictly-required file would.

# site2 (retention pond + 3 houses, added 2026-07-20 — see lidar/test_sites.py) mirrors every
# site1 file above with a "_site2" suffix. Treated as OPTIONAL alongside the mesh-shallow-water
# outputs below (warn, don't block) for the same reason: these are slow (~minutes) runs someone
# may not have done yet, and failing the whole preprocessing pipeline over one missing layer
# would break every OTHER viewer feature too.
SITE2_FILES = [
    "lidar_pointcloud_site2.bin",
    "dense_test_area_mesh_site2.obj",
]

# simulation/mesh_shallow_water.py's outputs — swe_surface_heightmap*.bin lands in
# lidar/data/, swe_mesh_frames*.bin/swe_mesh_summary*.json in simulation/outputs/ (matching
# flood_sim_ian.py's convention). Treated as OPTIONAL (warn, don't block) for the same reason
# as SITE2_FILES above.
#
# 2026-07-21: the viewer's rain-intensity control moved from a live on-demand re-simulation
# (rejected — it blocks the UI for 4-13min on a solver run) to 3 pre-computed
# presets per site (Low=40mm/hr, Medium=100mm/hr, High=180mm/hr — see
# simulation/build_rain_presets.sh or the individual mesh_shallow_water.py --peak-rain-mm-hr
# invocations). main.js only ever fetches the *_low/_medium/_high-suffixed filenames now — the
# bare (unsuffixed) names below are legacy/unused by the panel but still copied for reference.
RAIN_LEVELS = ["low", "medium", "high"]
OPTIONAL_FILES = [
    (LIDAR_DIR, "swe_surface_heightmap.bin"),
    (SIM_DIR, "swe_mesh_frames.bin"),
    (SIM_DIR, "swe_mesh_summary.json"),
    (SIM_DIR, "flow_tracer_paths.bin"),          # site1 physics-driven tracers (2026-07-21)
    (LIDAR_DIR, "swe_surface_heightmap_site2.bin"),
    (SIM_DIR, "swe_mesh_frames_site2.bin"),
    (SIM_DIR, "swe_mesh_summary_site2.json"),
    (SIM_DIR, "lake_hydrograph_site2.csv"),
    (SIM_DIR, "flow_tracer_paths_site2.bin"),    # site2 physics-driven tracers
] + [(LIDAR_DIR, f) for f in SITE2_FILES]
for _level in RAIN_LEVELS:
    OPTIONAL_FILES += [
        (LIDAR_DIR, f"swe_surface_heightmap_{_level}.bin"),
        (SIM_DIR, f"swe_mesh_frames_{_level}.bin"),
        (SIM_DIR, f"swe_mesh_summary_{_level}.json"),
        (SIM_DIR, f"flow_tracer_paths_{_level}.bin"),
        (LIDAR_DIR, f"swe_surface_heightmap_site2_{_level}.bin"),
        (SIM_DIR, f"swe_mesh_frames_site2_{_level}.bin"),
        (SIM_DIR, f"swe_mesh_summary_site2_{_level}.json"),
        (SIM_DIR, f"lake_hydrograph_site2_{_level}.csv"),
        (SIM_DIR, f"flow_tracer_paths_site2_{_level}.bin"),
    ]

# site3_1house was never wired into this file's copy list at all (found 2026-08-04, during the
# post-friction-fix mesh regeneration): its 4 output files were regenerating correctly on disk
# every time `run_site3_1house_demo.py` ran, but this script had no entry for them, so
# viewer/data/ silently kept whatever copy happened to be there from the file's first-ever
# manual `cp` -- on this occasion 8 days stale relative to the corrected solver. Added here so a
# future regeneration can't repeat that silent miss.
OPTIONAL_FILES += [
    (LIDAR_DIR, "swe_surface_heightmap_site3_1house.bin"),
    (SIM_DIR, "swe_mesh_frames_site3_1house.bin"),
    (SIM_DIR, "swe_mesh_summary_site3_1house.json"),
    (SIM_DIR, "flow_tracer_paths_site3_1house.bin"),
]


def main():
    print("Exporting LiDAR + mesh shallow-water viewer assets → viewer/data/ …")
    any_missing = False
    for fname in FILES:
        src = os.path.join(LIDAR_DIR, fname)
        dst = os.path.join(DATA_DIR, fname)
        if not os.path.exists(src):
            print(f"  MISSING: {fname} — run the lidar/ build scripts first")
            any_missing = True
            continue
        shutil.copy2(src, dst)
        print(f"  {fname}  ({os.path.getsize(dst)//1024} KB)")
    for src_dir, fname in OPTIONAL_FILES:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(DATA_DIR, fname)
        if not os.path.exists(src):
            print(f"  optional, not yet built: {fname} (run simulation/mesh_shallow_water.py)")
            continue
        shutil.copy2(src, dst)
        print(f"  {fname}  ({os.path.getsize(dst)//1024} KB)")
    if any_missing:
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
