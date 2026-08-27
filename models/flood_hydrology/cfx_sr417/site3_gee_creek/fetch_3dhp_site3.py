"""
USGS 3DHP flowlines + waterbodies for site3 (Gee Creek gauge-matched validation site)
=========================================================================================
`hydrography/fetch_3dhp.py` hardcodes its output to hydrography/data/ (the main AOI's own
directory) — calling it as-is for site3's coordinates would silently overwrite the main AOI's
existing 3dhp_flowlines.geojson/3dhp_waterbodies.geojson (both would be named identically in
the same directory). Its own query_layer()/summarize_flowlines()/summarize_waterbodies()/
save_outputs() are reused directly here with a site3-specific DATA_DIR monkey-patched in —
same non-invasive pattern every other site3 script in this project uses.

This was never fetched for site3 before now — `site3_gee_creek/hydrography/` was a genuinely
empty (0B) directory prior to this script (confirmed 2026-07-27; `fetch_3dhp.py` was never one
of the scripts ported into site3's build chain, unlike DEM/soil/imagery/roads). Also uses the
newer, richer `3dhp.nationalmap.gov` FeatureServer (already switched over in fetch_3dhp.py
itself as of 2026-07-27) rather than the older MapServer that returns null Flow-Network-
Derivative attributes.

Purpose of this layer for site3 specifically: 3DHP is what identified site3's own gauge-bearing
creek in the first place indirectly (via NWIS, not 3DHP) — but the flowline/waterbody geometry
itself is directly relevant here as a real, independently-mapped drainage network to visually
and analytically cross-check against this project's own D8-delineated stream network
(dem/data/hydro/stream_network.geojson-equivalent for site3) and the real Gee Creek gauge pour
point. Unlike the main AOI, site3 had NO mapped hydrography layer at all until this script.

Usage:
    python3 site3_gee_creek/fetch_3dhp_site3.py
"""
import os, sys

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "hydrography"))
from fetch_3dhp import (  # noqa: E402
    bbox_from_center, query_layer, summarize_flowlines, summarize_waterbodies,
    save_outputs, LAYER_FLOWLINE, LAYER_WATERBODY,
)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

import fetch_3dhp as f3  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hydrography", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    site = get_site("site3")
    lat, lon, radius_km = site["lat"], site["lon"], site["radius_km"]
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"Querying 3DHP for site3 ({site['label']})")
    print(f"  Bounding box: {[round(x, 5) for x in bbox]}")

    print("  Querying 3DHP flowlines (layer 50) …")
    flowlines = query_layer(LAYER_FLOWLINE, bbox)
    print("  Querying 3DHP waterbodies (layer 60) …")
    waterbodies = query_layer(LAYER_WATERBODY, bbox)

    flow_summary = summarize_flowlines(flowlines)
    water_summary = summarize_waterbodies(waterbodies)

    # save_outputs() writes to fetch_3dhp's own module-level DATA_DIR — monkey-patch it to
    # site3's own directory for the duration of this call so the main AOI's files are untouched.
    f3.DATA_DIR = DATA_DIR
    save_outputs(flowlines, waterbodies, bbox, flow_summary, water_summary)


if __name__ == "__main__":
    main()
