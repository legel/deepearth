"""
USGS 3D Hydrography Program (3DHP) — Flowlines & Waterbodies for the Johns Lake AOI
===========================================================================================
Pulls USGS 3DHP flowlines and waterbodies intersecting the 2x2 km Johns Lake AOI around
28.521592N, -81.656981W (Winter Garden, FL), via the 3DHP_all ArcGIS REST FeatureServer.
3DHP is the USGS's next-generation hydrography product (3D-enabled, elevation-integrated),
replacing legacy NHD over time. Ported from the sibling cfx_sr417_corridor project 2026-07-28
(same script/endpoint, different AOI) — Johns Lake had no 3DHP layer at all before this.

Service: https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer
  Layer 50 = Flowline
  Layer 60 = Waterbody
  (Layer 80 = Catchment — confirmed EMPTY for this whole region as of 2026-07-27,
  not queried here.)

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
    python3 hydrography/fetch_3dhp.py --lat 28.521592 --lon -81.656981 --radius_km 1.0
"""

import os
import json
import argparse
import time
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT    = 28.521592   # Johns Lake AOI (Winter Garden, FL)
DEFAULT_LON    = -81.656981
DEFAULT_RADIUS = 1.0         # km -> 2x2 km study box

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
    parser = argparse.ArgumentParser(description="Fetch USGS 3DHP flowlines + waterbodies for the Johns Lake AOI")
    parser.add_argument("--lat",       type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",       type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS)
    args = parser.parse_args()
    main(args.lat, args.lon, args.radius_km)
