"""
Fetch FEMA NFHL flood hazard zones for site3 -> site3_gee_creek/floodplain/data/
====================================================================================
Reuses floodplain/fetch_fema_nfhl.py's own query/summarize/save functions directly (its
main() already accepts lat/lon/radius_km as plain arguments -- no internal path patching
needed for the QUERY itself) but monkey-patches its module-level DATA_DIR first, same
non-invasive pattern as every other site3 script, so this never overwrites the main AOI's own
floodplain/data/fema_flood_zones.geojson.

Real gap this fills: site3_gee_creek/ has never had a floodplain/ directory at all (flagged in
the dataset-selection-logic audit) -- site3 is the site actually being
compared against a real USGS gauge, so a mapped FEMA layer is directly relevant there too.

Usage:
    python3 site3_gee_creek/fetch_fema_site3.py
"""
import os, sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "floodplain"))
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
import fetch_fema_nfhl as fema  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE3_FLOOD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floodplain", "data")


def main():
    os.makedirs(SITE3_FLOOD_DIR, exist_ok=True)
    fema.DATA_DIR = SITE3_FLOOD_DIR
    site = get_site("site3")
    fema.main(site["lat"], site["lon"], site["radius_km"])


if __name__ == "__main__":
    main()
