"""
Build the site3_1house demo mesh + run the 3D shallow-water solver (falling rain + physics-
driven flow), reusing droplet_flow_test.py + mesh_shallow_water.py directly, exactly the same
scripts/physics site1/site2's own demo layers use.
=======================================================================================================
Real reason this needs its own wrapper instead of just `python3 lidar/droplet_flow_test.py
--site site3_1house` directly: both droplet_flow_test.py and mesh_shallow_water.py import a
module-level GEO_META constant from build_lidar_pointcloud.py that points at the MAIN AOI's own
viewer/data/geo_meta.json -- correct for site1/site2 (which live INSIDE the main AOI's own scene
space) but wrong for anything in the site3_gee_creek/ family (37km away, its own separate scene
space defined by viewer/data/geo_meta_site3.json) -- this is the exact same bug already found and
fixed once for the full site3 mesh (see lidar/reexport_site3_mesh_obj.py). Monkey-patches
GEO_META on both imported modules before calling their own main() functions (with sys.argv
patched to pass --site site3_1house), same non-invasive pattern as every other site3 script in
this project -- neither shared script is edited.

Usage:
    .venv/bin/python3 site3_gee_creek/run_site3_1house_demo.py
    # single-droplet, forced onto the roof (demo variant, added per direct request):
    .venv/bin/python3 site3_gee_creek/run_site3_1house_demo.py --n-tracers 1 --force-roof-seed
"""
import os, sys, argparse

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
sys.path.insert(0, os.path.join(PROJ_DIR, "simulation"))

import droplet_flow_test as dft  # noqa: E402
import mesh_shallow_water as msw  # noqa: E402

SITE3_GEO_META = os.path.join(PROJ_DIR, "viewer", "data", "geo_meta_site3.json")
SITE = "site3_1house"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tracers", type=int, default=2500)
    ap.add_argument("--force-roof-seed", action="store_true")
    args = ap.parse_args()

    dft.GEO_META = SITE3_GEO_META
    msw.GEO_META = SITE3_GEO_META

    print("### [1/2] Building dense mesh (ground + roof) for site3_1house ###")
    sys.argv = ["droplet_flow_test.py", "--site", SITE]
    dft.main()

    print("\n### [2/2] Running 3D shallow-water solver (falling rain + physics-driven flow) ###")
    sys.argv = ["mesh_shallow_water.py", "--site", SITE,
                "--peak-rain-mm-hr", "100", "--rain-duration-min", "4", "--total-min", "8",
                "--n-tracers", str(args.n_tracers)]
    if args.force_roof_seed:
        sys.argv.append("--force-roof-seed")
    msw.main()


if __name__ == "__main__":
    main()
