"""
Export a SSURGO soils PNG for site3's viewer page -> viewer/data/ssurgo_site3.png
====================================================================================
Reuses export_overlays.py's own export_ssurgo() directly (mukey raster -> per-series color
fill), monkey-patching its module-level SOIL_DIR/OUT_DIR to site3's own paths -- same
non-invasive pattern as export_hydrography_overlay_site3.py. Site3 already has its own real
SSURGO fetch (mukey_map.tif + mukey_map_legend.csv, built as part of the original site3 data
pipeline for the solver's own per-cell Horton infiltration) -- this just gives that same real
data a viewer overlay, which it never had before.

Usage:
    python3 site3_gee_creek/export_ssurgo_overlay_site3.py
"""
import os, sys, shutil, tempfile
import rasterio

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "viewer", "preprocess"))
import export_overlays as eo  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

VIEWER_DATA_DIR = os.path.join(PROJ_DIR, "viewer", "data")
SITE3_SOIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soil", "data")
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
        eo.SOIL_DIR = SITE3_SOIL_DIR
        eo.OUT_DIR = tmp_dir
        eo.SIZE = 2048   # same site3-wide-box override every other site3 overlay export uses
        eo.export_ssurgo(_Bounds(), dem_crs)

        tmp_png = os.path.join(tmp_dir, "ssurgo.png")
        if not os.path.exists(tmp_png):
            print(f"SSURGO export produced no output -- check {SITE3_SOIL_DIR}/mukey_map.tif exists.")
            return
        shutil.copy2(tmp_png, os.path.join(VIEWER_DATA_DIR, "ssurgo_site3.png"))
        shutil.copy2(os.path.join(tmp_dir, "ssurgo_legend.json"),
                     os.path.join(VIEWER_DATA_DIR, "ssurgo_legend_site3.json"))
        print(f"ssurgo_site3.png + ssurgo_legend_site3.json written to {VIEWER_DATA_DIR}")


if __name__ == "__main__":
    main()
