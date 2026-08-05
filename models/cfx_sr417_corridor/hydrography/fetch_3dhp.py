"""
USGS 3D Hydrography Program (3DHP) — Flowlines & Waterbodies for the SR417 Corridor AOI
===========================================================================================
Pulls USGS 3DHP flowlines and waterbodies intersecting the 2x2 km AOI around
28.36687N, -81.43299W, via the 3DHP_all ArcGIS REST MapServer. 3DHP is the
USGS's next-generation hydrography product (3D-enabled, elevation-integrated),
replacing legacy NHD over time — queried here as a second, more current source
alongside `fetch_nhd.py`'s NHDPlus HR pull, per Lance Legel's explicit guidance
to study 3DHP for this project.

Service: https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer
  Layer 50 = Flowline
  Layer 60 = Waterbody
  (Layer 80 = Catchment — confirmed EMPTY for this whole region as of 2026-07-27,
  see TASK_3DHP_FLOW_NETWORK_DERIVATIVES.md; not queried here.)

Switched 2026-07-27 from the older `hydro.nationalmap.gov/.../3DHP_all/MapServer` endpoint
(same layer IDs, confirmed via `?f=pjson`) — the older MapServer returns real geometry but
every Flow-Network-Derivative attribute (arbolatesum, streamorder, pathlength, mainstemid,
hydrosequence, etc.) as null; the newer FeatureServer returns real populated values for the
same fields on the same features (confirmed via a direct side-by-side query for this AOI's own
bbox, same feature count/geometry, richer attributes only). Output schema/field names are
identical, so nothing downstream (export_overlays.py's export_hydrography, or any other
consumer of 3dhp_flowlines.geojson/3dhp_waterbodies.geojson) needed to change.

Outputs (saved under hydrography/data/):
    3dhp_flowlines.geojson    — flowline features intersecting the AOI (may be empty)
    3dhp_waterbodies.geojson  — waterbody features intersecting the AOI (may be empty)
    3dhp_summary.json         — counts, total length/area, bbox used

Usage:
    python3 hydrography/fetch_3dhp.py
    python3 hydrography/fetch_3dhp.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Shared site registry (added 2026-08-04) ──────────────────────────────────
# Makes `--site <name>` resolve lat/lon/radius AND the output directory from the ONE registry
# (site_registry.py -> lidar/test_sites.py) instead of hand-typed coordinates. Purely additive:
# with no --site flag this script behaves exactly as it always has. See site_registry.py's
# docstring for why (INTERNSHIP_AUDIT_2026-08-03.md §4: site3's data existed on disk with no
# script that could reproduce it, because coordinates were typed by hand per-invocation).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import site_registry  # noqa: E402

os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT    = 28.36687   # CFX SR417 corridor test-landscape AOI (Lake Nona / south Orlando)
DEFAULT_LON    = -81.43299
DEFAULT_RADIUS = 1.0        # km -> 2x2 km study box

MAPSERVER_URL = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer"
LAYER_FLOWLINE  = 50
LAYER_WATERBODY = 60


def bbox_from_center(lat, lon, radius_km):
    """Return (west, south, east, north) bounding box in EPSG:4326."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def query_layer(layer_id, bbox_wsen, max_retries=4):
    """Query an ArcGIS REST MapServer layer by bbox; return a GeoJSON FeatureCollection dict."""
    west, south, east, north = bbox_wsen
    url = f"{MAPSERVER_URL}/{layer_id}/query"
    params = {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                print(f"    ⚠ layer {layer_id} query failed after {max_retries} attempts: {exc}")
                return {"type": "FeatureCollection", "features": []}
            backoff = 2 ** attempt
            print(f"    ⚠ transient error querying layer {layer_id}, retrying in {backoff}s …")
            time.sleep(backoff)


def summarize_flowlines(fc):
    feats = fc.get("features", [])
    if not feats:
        print("\n── 3DHP Flowlines ───────────────────────────────────────────")
        print("  None found in AOI.")
        return {"count": 0, "total_length_km": 0.0}

    total_length_km = sum(f["properties"].get("lengthkm", 0) or 0 for f in feats)
    named = sorted({f["properties"].get("gnisidlabel") for f in feats
                    if f["properties"].get("gnisidlabel")})

    print("\n── 3DHP Flowlines ───────────────────────────────────────────")
    print(f"  Count: {len(feats)}  |  Total length: {total_length_km:.2f} km")
    print(f"  Named: {named if named else '(none — unnamed/unclassified segments)'}")
    return {"count": len(feats), "total_length_km": round(total_length_km, 3), "named": named}


def summarize_waterbodies(fc):
    feats = fc.get("features", [])
    if not feats:
        print("\n── 3DHP Waterbodies ─────────────────────────────────────────")
        print("  None found in AOI.")
        return {"count": 0}

    print("\n── 3DHP Waterbodies ─────────────────────────────────────────")
    print(f"  Count: {len(feats)}")
    ftypes = sorted({f["properties"].get("featuretypelabel") for f in feats
                      if f["properties"].get("featuretypelabel")})
    print(f"  Feature types present: {ftypes}")
    return {"count": len(feats), "feature_types": ftypes}


def save_outputs(flowlines_fc, waterbodies_fc, bbox, flow_summary, water_summary):
    flow_path  = os.path.join(DATA_DIR, "3dhp_flowlines.geojson")
    water_path = os.path.join(DATA_DIR, "3dhp_waterbodies.geojson")

    with open(flow_path, "w") as f:
        json.dump(flowlines_fc, f)
    print(f"\nSaved flowlines   → {flow_path}")

    with open(water_path, "w") as f:
        json.dump(waterbodies_fc, f)
    print(f"Saved waterbodies → {water_path}")

    summary = {
        "bbox_wsen": list(bbox),
        "service": MAPSERVER_URL,
        "layers_queried": {"flowline": LAYER_FLOWLINE, "waterbody": LAYER_WATERBODY},
        "flowlines": flow_summary,
        "waterbodies": water_summary,
    }
    summary_path = os.path.join(DATA_DIR, "3dhp_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary     → {summary_path}")


def main(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=DEFAULT_RADIUS):
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"Querying 3DHP for ({lat}, {lon}), radius {radius_km} km")
    print(f"  Bounding box: {[round(x, 5) for x in bbox]}")

    print("  Querying 3DHP flowlines (layer 50) …")
    flowlines = query_layer(LAYER_FLOWLINE, bbox)
    print("  Querying 3DHP waterbodies (layer 60) …")
    waterbodies = query_layer(LAYER_WATERBODY, bbox)

    flow_summary  = summarize_flowlines(flowlines)
    water_summary = summarize_waterbodies(waterbodies)

    save_outputs(flowlines, waterbodies, bbox, flow_summary, water_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch USGS 3DHP flowlines + waterbodies for the SR417 corridor AOI")
    parser.add_argument("--lat",       type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",       type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS)
    site_registry.add_site_arg(parser)
    args = site_registry.resolve(parser.parse_args(), category="hydrography")
    if args.site_data_root:
        # Rebind the module-level DATA_DIR so every function writing output lands in the
        # selected site's own tree instead of the main AOI's (the exact clobbering
        # fetch_naip_site3.py's docstring warns about — both share e.g. naip_2021_RGB.tif).
        globals()["DATA_DIR"] = args.site_data_dir
    main(args.lat, args.lon, args.radius_km)
