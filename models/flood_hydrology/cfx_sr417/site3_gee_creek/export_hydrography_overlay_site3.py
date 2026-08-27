"""
Export a 3DHP hydrography PNG for site3's viewer page -> viewer/data/hydrography_site3.png
===============================================================================================
Reuses export_overlays.py's own export_hydrography() (waterbody polygon fill + buffered
flowline rasterization) directly rather than duplicating it. That function reads module-level
HYDRO_DIR/OUT_DIR constants and writes a HARDCODED "hydrography.png" filename under its own
OUT_DIR -- both monkey-patched here (HYDRO_DIR -> site3's own hydrography dir, populated by
fetch_3dhp_site3.py; OUT_DIR -> a temp dir) so this never touches or overwrites the main AOI's
own hydrography.png, then the result is moved to its real site3-suffixed name. Same
non-invasive pattern as export_naip_overlay_site3.py.

Usage:
    python3 site3_gee_creek/export_hydrography_overlay_site3.py
    (requires site3_gee_creek/fetch_3dhp_site3.py to have been run first)
"""
import os, sys, shutil, tempfile
import rasterio

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "viewer", "preprocess"))
import export_overlays as eo  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

VIEWER_DATA_DIR = os.path.join(PROJ_DIR, "viewer", "data")
SITE3_HYDRO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hydrography", "data")
DEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dem", "data", "site3_dem.tif")


def main():
    site = get_site("site3")

    with rasterio.open(DEM_PATH) as src:
        b = src.bounds
        dem_crs = src.crs
    # Same inverted-bounds normalization as export_naip_overlay_site3.py / export_dem_site3.py --
    # site3's DEM has bounds.bottom > bounds.top.
    true_left, true_right = sorted([b.left, b.right])
    true_bottom, true_top = sorted([b.bottom, b.top])

    class _Bounds:
        left, right, bottom, top = true_left, true_right, true_bottom, true_top

    with tempfile.TemporaryDirectory() as tmp_dir:
        eo.HYDRO_DIR = SITE3_HYDRO_DIR
        eo.OUT_DIR = tmp_dir
        # Same reasoning as export_naip_overlay_site3.py's SIZE bump: export_overlays.py's own
        # SIZE=512 was tuned for the main AOI's ~2.3km box. Site3's box is ~6.9km wide, and its
        # flowlines are thin buffered lines (12m buffer) -- at 512px/6.9km (~13.5m/px) a 12m-wide
        # buffered creek would be sub-pixel in places. Bumped to 2048, matching the NAIP overlay's
        # own site3-only override, for a legible ~3.35m/px line width.
        eo.SIZE = 2048
        eo.export_hydrography(_Bounds(), dem_crs)

        tmp_png = os.path.join(tmp_dir, "hydrography.png")
        if not os.path.exists(tmp_png):
            print("Hydrography export produced no output -- check that "
                  f"{SITE3_HYDRO_DIR}/3dhp_flowlines.geojson and 3dhp_waterbodies.geojson "
                  "exist and are non-empty (run fetch_3dhp_site3.py first).")
            return
        dst = os.path.join(VIEWER_DATA_DIR, "hydrography_site3.png")
        shutil.copy2(tmp_png, dst)
        print(f"hydrography_site3.png written to {dst}")


if __name__ == "__main__":
    main()
