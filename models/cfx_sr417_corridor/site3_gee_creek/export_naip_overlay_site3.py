"""
Export a NAIP true-color PNG for site3's viewer page -> viewer/data/naip_site3.png
=====================================================================================
Reuses export_overlays.py's own export_naip() (reproject-from-georeferenced-GeoTIFF logic)
directly rather than duplicating it. That function reads a module-level IMAGERY_DIR constant
and writes a HARDCODED "naip_rgb.png" filename under its own OUT_DIR -- both monkey-patched
here (IMAGERY_DIR -> site3's own imagery dir, OUT_DIR -> a temp dir) so this never touches or
overwrites the main AOI's own naip_rgb.png, then the result is moved to its real site3-suffixed
name. Same non-invasive pattern every other site3 script in this project uses.

Usage:
    python3 site3_gee_creek/export_naip_overlay_site3.py
"""
import os, sys, shutil, tempfile
import rasterio

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "viewer", "preprocess"))
import export_overlays as eo  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

VIEWER_DATA_DIR = os.path.join(PROJ_DIR, "viewer", "data")
SITE3_IMAGERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagery", "data")
DEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dem", "data", "site3_dem.tif")


def main():
    site = get_site("site3")

    with rasterio.open(DEM_PATH) as src:
        b = src.bounds
        dem_crs = src.crs
    # Same inverted-bounds normalization as export_dem_site3.py / flood_sim_ian.py's
    # load_dem_for_sim() -- site3's DEM has bounds.bottom > bounds.top.
    true_left, true_right = sorted([b.left, b.right])
    true_bottom, true_top = sorted([b.bottom, b.top])

    class _Bounds:
        left, right, bottom, top = true_left, true_right, true_bottom, true_top

    with tempfile.TemporaryDirectory() as tmp_dir:
        eo.IMAGERY_DIR = SITE3_IMAGERY_DIR
        eo.OUT_DIR = tmp_dir
        # export_overlays.py's own SIZE=512 was tuned for the main AOI's ~2.3km box (~4.5m/px
        # against NAIP's 0.6m native res -- already soft, but tolerable). Site3's box is ~6.9km
        # wide -- the same 512px texture works out to ~13.4m/px, visibly blurry when draped.
        # Bumped to 2048 here (site3-only, via monkey-patch, same as IMAGERY_DIR/OUT_DIR above)
        # for ~3.35m/px, actually crisper than the main AOI's own texture.
        eo.SIZE = 2048
        eo.export_naip(_Bounds(), dem_crs)

        tmp_png = os.path.join(tmp_dir, "naip_rgb.png")
        if not os.path.exists(tmp_png):
            print("NAIP export failed -- naip_rgb.png not produced (see export_naip's own "
                  "message above for why, e.g. no NAIP GeoTIFF found in "
                  f"{SITE3_IMAGERY_DIR})")
            return
        dst = os.path.join(VIEWER_DATA_DIR, "naip_site3.png")
        shutil.copy2(tmp_png, dst)
        print(f"naip_site3.png written to {dst}")


if __name__ == "__main__":
    main()
