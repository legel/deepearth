"""
Fetch USGS 3DEP DEM for site3 (Gee Creek near Longwood) -> site3_gee_creek/dem/data/
=====================================================================================
Real gap this fills: `dem/dem_download.py` was almost certainly invoked by hand with site3's
coordinates to produce the DEM already on disk (`site3_gee_creek/dem/data/site3_dem.tif`,
confirmed via its own `site3_dem_meta.json`: lat=28.690514, lon=-81.287539, radius_km=2.99,
resolution=1, EPSG:5070, 7810x7819) -- but that invocation was never saved as a script, so
site3's DEM was not reproducible from scratch (flagged in the 2026-08-03 project audit,
INTERNSHIP_AUDIT_2026-08-03.md). This script closes that gap the same non-invasive way every
other site3 fetch script does: reuse the main-AOI function directly, monkey-patch nothing (the
function already takes an explicit out_path), and read coordinates from the single source of
truth (`lidar/test_sites.py`) instead of retyping them a second time.

Usage:
    python3 site3_gee_creek/fetch_dem_site3.py
"""
import os, sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "dem"))
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
import dem_download  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE3_DEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dem", "data")


def main():
    os.makedirs(SITE3_DEM_DIR, exist_ok=True)
    site = get_site("site3")
    out_path = os.path.join(SITE3_DEM_DIR, "site3_dem.tif")
    dem_download.download_dem(
        lat=site["lat"], lon=site["lon"], radius_km=site["radius_km"],
        resolution=1,  # matches the DEM already on disk (site3_dem_meta.json: resolution_m=1)
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
