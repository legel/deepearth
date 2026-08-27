"""
Export a FEMA flood-zone PNG for site3's viewer page -> viewer/data/floodplain_site3.png
============================================================================================
Reuses export_overlays.py's own export_floodplain() directly (SFHA/AE blue under regulatory-
floodway red), monkey-patching its module-level FLOOD_DIR/OUT_DIR to site3's own paths -- same
non-invasive pattern as export_hydrography_overlay_site3.py. Requires
site3_gee_creek/fetch_fema_site3.py to have been run first (site3 had no floodplain/ directory
at all before that).

Usage:
    python3 site3_gee_creek/export_floodplain_overlay_site3.py
"""
import os, sys, shutil, tempfile
import rasterio

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "viewer", "preprocess"))
import export_overlays as eo  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

VIEWER_DATA_DIR = os.path.join(PROJ_DIR, "viewer", "data")
SITE3_FLOOD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floodplain", "data")
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
        eo.FLOOD_DIR = SITE3_FLOOD_DIR
        eo.OUT_DIR = tmp_dir
        eo.SIZE = 2048
        eo.export_floodplain(_Bounds(), dem_crs)

        tmp_png = os.path.join(tmp_dir, "floodplain.png")
        if not os.path.exists(tmp_png):
            print(f"Floodplain export produced no output -- check "
                  f"{SITE3_FLOOD_DIR}/fema_flood_zones.geojson exists and is non-empty "
                  "(run fetch_fema_site3.py first).")
            return
        shutil.copy2(tmp_png, os.path.join(VIEWER_DATA_DIR, "floodplain_site3.png"))
        print(f"floodplain_site3.png written to {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()
