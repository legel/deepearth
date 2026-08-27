"""
FEMA National Flood Hazard Layer (NFHL) — Flood Zones & Floodway for the SR417 Corridor AOI
===============================================================================================
Pulls FEMA NFHL flood hazard zone polygons intersecting the 2x2 km AOI around
28.36687N, -81.43299W, via the public NFHL ArcGIS REST MapServer, and flags
which (if any) are the regulatory floodway — the as-engineered hydrological
constraint, studied alongside 3DHP ahead of any erosion/slope-stabilization
design work.

Service: https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer
  Layer 28 = Flood Hazard Zones (polygons; FLD_ZONE, ZONE_SUBTY, SFHA_TF, STATIC_BFE)
(confirmed via MapServer metadata. There is no separate "Floodway" layer — the
regulatory floodway is a ZONE_SUBTY value, e.g. "FLOODWAY", within layer 28's
polygons, not a distinct layer ID.)

STATIC_BFE is in feet, vertical datum given by the V_DATUM field (commonly
NAVD88) — compare directly against dem/data/sr417_corridor_dem_meta.json's
z_min/z_max (also NAVD88 meters) after unit conversion, not at face value.

Outputs (saved under floodplain/data/):
    fema_flood_zones.geojson  — all flood hazard zone polygons intersecting the AOI
    fema_summary.json         — zone counts by FLD_ZONE/ZONE_SUBTY, floodway flag, bbox used

Usage:
    python3 floodplain/fetch_fema_nfhl.py
    python3 floodplain/fetch_fema_nfhl.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
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
# docstring for why: hand-typed per-invocation coordinates leave fetched data on disk with no
# script that can reproduce it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import site_registry  # noqa: E402

os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT    = 28.36687   # CFX SR417 corridor test-landscape AOI (Lake Nona / south Orlando)
DEFAULT_LON    = -81.43299
DEFAULT_RADIUS = 1.0        # km -> 2x2 km study box

MAPSERVER_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
LAYER_FLOOD_HAZARD_ZONES = 28

# ZONE_SUBTY values that denote a regulatory floodway (FEMA NFHL data dictionary)
FLOODWAY_SUBTYPES = {
    "FLOODWAY", "ADMINISTRATIVE FLOODWAY", "COMMUNITY ENCROACHMENT AREA",
    "FLOWAGE EASEMENT AREA", "NARROW FLOODWAY",
}


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


def summarize_zones(fc):
    feats = fc.get("features", [])
    if not feats:
        print("\n── FEMA Flood Hazard Zones ──────────────────────────────────")
        print("  None found in AOI.")
        return {"count": 0, "zones": [], "has_floodway": False, "has_sfha": False}

    zones = []
    has_floodway = False
    has_sfha = False
    print("\n── FEMA Flood Hazard Zones ──────────────────────────────────")
    print(f"  Count: {len(feats)}")
    print(f"  {'FLD_ZONE':10} {'ZONE_SUBTY':28} {'SFHA':5} {'BFE_ft':8}")
    for f in feats:
        p = f["properties"]
        fld_zone = p.get("FLD_ZONE", "")
        subty = p.get("ZONE_SUBTY") or ""
        sfha = p.get("SFHA_TF", "")
        bfe = p.get("STATIC_BFE", -9999)
        is_floodway = subty.upper() in FLOODWAY_SUBTYPES
        has_floodway = has_floodway or is_floodway
        has_sfha = has_sfha or (sfha == "T")
        bfe_str = str(bfe) if bfe != -9999 else "n/a"
        print(f"  {fld_zone:10} {subty:28} {sfha:5} {bfe_str:8}" + ("  <-- FLOODWAY" if is_floodway else ""))
        zones.append({"fld_zone": fld_zone, "zone_subty": subty, "sfha_tf": sfha,
                       "static_bfe_ft": bfe if bfe != -9999 else None,
                       "is_floodway": is_floodway})

    print()
    print(f"  Regulatory floodway present in AOI: {has_floodway}")
    print(f"  Special Flood Hazard Area (SFHA) present in AOI: {has_sfha}")
    return {"count": len(feats), "zones": zones, "has_floodway": has_floodway, "has_sfha": has_sfha}


def save_outputs(zones_fc, bbox, summary):
    zones_path = os.path.join(DATA_DIR, "fema_flood_zones.geojson")
    with open(zones_path, "w") as f:
        json.dump(zones_fc, f)
    print(f"\nSaved flood zones → {zones_path}")

    full_summary = {
        "bbox_wsen": list(bbox),
        "service": MAPSERVER_URL,
        "layer_queried": {"flood_hazard_zones": LAYER_FLOOD_HAZARD_ZONES},
        **summary,
    }
    summary_path = os.path.join(DATA_DIR, "fema_summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved summary     → {summary_path}")


def main(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=DEFAULT_RADIUS):
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"Querying FEMA NFHL for ({lat}, {lon}), radius {radius_km} km")
    print(f"  Bounding box: {[round(x, 5) for x in bbox]}")

    print("  Querying Flood Hazard Zones (layer 28) …")
    zones_fc = query_layer(LAYER_FLOOD_HAZARD_ZONES, bbox)

    summary = summarize_zones(zones_fc)
    save_outputs(zones_fc, bbox, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch FEMA NFHL flood hazard zones for the SR417 corridor AOI")
    parser.add_argument("--lat",       type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",       type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS)
    site_registry.add_site_arg(parser)
    args = site_registry.resolve(parser.parse_args(), category="floodplain")
    if args.site_data_root:
        # Rebind the module-level DATA_DIR so every function writing output lands in the
        # selected site's own tree instead of the main AOI's (the exact clobbering
        # fetch_naip_site3.py's docstring warns about — both share e.g. naip_2021_RGB.tif).
        globals()["DATA_DIR"] = args.site_data_dir
    main(args.lat, args.lon, args.radius_km)
