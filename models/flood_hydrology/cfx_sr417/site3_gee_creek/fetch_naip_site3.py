"""
NAIP aerial imagery for site3 (Gee Creek gauge-matched validation site)
=========================================================================
imagery/fetch_naip.py's own main() hardcodes its output to imagery/data/ (the main AOI's own
directory) — calling it as-is for site3's coordinates would silently overwrite the main AOI's
existing NAIP files (both would be named naip_{year}_RGB.tif in the same directory). Its own
download_naip_mosaic() function takes the output directory as a parameter though, so this calls
search_naip()/download_naip_mosaic() directly with a site3-specific directory instead of running
main() — same non-invasive pattern every other site3 script in this project uses.

Usage:
    python3 site3_gee_creek/fetch_naip_site3.py
"""
import os, sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "imagery"))
from fetch_naip import bbox_from_center, search_naip, download_naip_mosaic  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagery", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    site = get_site("site3")
    lat, lon, radius_km = site["lat"], site["lon"], site["radius_km"]
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"NAIP download for site3 ({site['label']})")
    print(f"bbox [W,S,E,N]: {[round(x,5) for x in bbox]}")

    items = search_naip(bbox, years=None)
    if not items:
        print("No NAIP scenes found — trying explicit recent years …")
        items = search_naip(bbox, years=[2022, 2021, 2020, 2019])
    if not items:
        print("Still no NAIP scenes found for site3.")
        return

    meta = download_naip_mosaic(items, bbox, OUT_DIR)
    if meta:
        print(f"\nNAIP downloaded: year={meta['year']} tiles={meta['n_tiles']} "
              f"resolution={meta['resolution_m']:.2f}m")


if __name__ == "__main__":
    main()
