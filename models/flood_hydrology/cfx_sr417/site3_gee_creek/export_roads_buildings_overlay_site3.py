"""
Export a roads+buildings PNG for site3's viewer page -> viewer/data/roads_buildings_site3.png
==================================================================================================
Reuses export_overlays.py's own export_roads_buildings() directly (OSM roads buffered by
highway type, dark gray, + building footprints, warm tan), monkey-patching its module-level
INFRA_DIR/OUT_DIR to site3's own paths -- same non-invasive pattern as
export_hydrography_overlay_site3.py. Site3 already has its own real roads.geojson/
buildings.geojson (fetched as part of the original site3 data pipeline for the solver's own
impervious-surface masking) -- this just gives that same real data a viewer overlay.

Usage:
    python3 site3_gee_creek/export_roads_buildings_overlay_site3.py
"""
import os, sys, shutil, tempfile
import rasterio

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "viewer", "preprocess"))
import export_overlays as eo  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

VIEWER_DATA_DIR = os.path.join(PROJ_DIR, "viewer", "data")
SITE3_INFRA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infrastructure", "data")
DEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dem", "data", "site3_dem.tif")


def main():
    with rasterio.open(DEM_PATH) as src:
        b = src.bounds
        dem_crs = src.crs
    true_left, true_right = sorted([b.left, b.right])
    true_bottom, true_top = sorted([b.bottom, b.top])

    class _Bounds:
        left, right, bottom, top = true_left, true_right, true_bottom, true_top

    with tempfile.TemporaryDirectory() as tmp_dir:
        eo.INFRA_DIR = SITE3_INFRA_DIR
        eo.OUT_DIR = tmp_dir
        eo.SIZE = 2048
        eo.export_roads_buildings(_Bounds(), dem_crs)

        tmp_png = os.path.join(tmp_dir, "roads_buildings.png")
        if not os.path.exists(tmp_png):
            print(f"Roads/buildings export produced no output -- check {SITE3_INFRA_DIR}/"
                  "roads.geojson and buildings.geojson exist.")
            return
        shutil.copy2(tmp_png, os.path.join(VIEWER_DATA_DIR, "roads_buildings_site3.png"))
        print(f"roads_buildings_site3.png written to {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()
