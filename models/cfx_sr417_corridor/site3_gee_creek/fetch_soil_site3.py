"""
Fetch SSURGO soils + NLCD impervious for site3 (Gee Creek) -> site3_gee_creek/soil/data/
==========================================================================================
Real gap this fills: site3's `soil/data/` already has real outputs on disk (mukey_map.tif +
legend, soil_parameters.json, cn_by_hsg.csv, nlcd_impervious.tif) but no committed script ever
produced them -- flagged in the 2026-08-03 project audit (INTERNSHIP_AUDIT_2026-08-03.md).
`export_ssurgo_overlay_site3.py`'s own docstring assumes this fetch already happened; this
script is the fetch that docstring was assuming existed.

Two real subtleties this script deliberately does NOT paper over by just calling each
upstream script's own main() unmodified:

1. `ssurgo_download.py`'s main() rasterizes the mukey map against a HARDCODED main-AOI DEM
   path (`dem/data/sr417_corridor_dem.tif`) -- monkey-patching DATA_DIR alone would silently
   rasterize site3's soil polygons onto the WRONG (main AOI's) grid. Fixed at the source
   (ssurgo_download.py now takes an optional `dem_path` param, default unchanged for existing
   callers) so this script can pass site3's own DEM explicitly.
2. `fetch_nlcd.py`'s `load_dem_profile()` reads a hardcoded `DEM_DIR`/`DEM_FILENAME` module
   global -- both are monkey-patched here (same non-invasive pattern used everywhere else in
   site3_gee_creek/), pointed at site3's own `site3_dem.tif` instead of the main AOI's
   `sr417_corridor_dem_1m.tif`.

Requires site3's own DEM to exist first: python3 site3_gee_creek/fetch_dem_site3.py

Usage:
    python3 site3_gee_creek/fetch_soil_site3.py
"""
import os, sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "soil"))
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
import ssurgo_download  # noqa: E402
import fetch_nlcd  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE3_DIR = os.path.dirname(os.path.abspath(__file__))
SITE3_SOIL_DIR = os.path.join(SITE3_DIR, "soil", "data")
SITE3_DEM_PATH = os.path.join(SITE3_DIR, "dem", "data", "site3_dem.tif")


def main():
    os.makedirs(SITE3_SOIL_DIR, exist_ok=True)
    site = get_site("site3")

    if not os.path.exists(SITE3_DEM_PATH):
        sys.exit(f"Site3 DEM not found: {SITE3_DEM_PATH}. Run fetch_dem_site3.py first.")

    print("── SSURGO (Horton/CN params + mukey_map.tif rasterized on site3's own DEM) ──")
    ssurgo_download.DATA_DIR = SITE3_SOIL_DIR
    ssurgo_download.main(
        lat=site["lat"], lon=site["lon"], radius_km=site["radius_km"],
        dem_path=SITE3_DEM_PATH,
    )

    print("\n── NLCD 2021 impervious surface (resampled onto site3's own DEM grid) ──")
    fetch_nlcd.DATA_DIR = SITE3_SOIL_DIR
    fetch_nlcd.DEM_DIR = os.path.dirname(SITE3_DEM_PATH)
    fetch_nlcd.DEM_FILENAME = os.path.basename(SITE3_DEM_PATH)
    fetch_nlcd.OUTPUT_TIF = os.path.join(SITE3_SOIL_DIR, "nlcd_impervious.tif")
    fetch_nlcd.OUTPUT_PNG = os.path.join(SITE3_SOIL_DIR, "nlcd_impervious.png")
    fetch_nlcd.main(lat=site["lat"], lon=site["lon"], radius_km=site["radius_km"])


if __name__ == "__main__":
    main()
