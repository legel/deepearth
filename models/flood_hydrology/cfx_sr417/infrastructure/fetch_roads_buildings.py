#!/usr/bin/env python3
"""
Fetch roads and building footprints for the CFX SR417 corridor AOI.

Roads/buildings become the geometric mask for
separating built surfaces from natural ground in later analysis
(soil/landcover work, and eventually a generalized flood solver that
treats roads/roofs differently from bare ground).

Approach (generalized from boundary/generate_cfx_corridor_kml.py, which
queries Overpass for SR417 only — this script pulls every road and every
building footprint in the AOI instead):
  1. Query Overpass API for:
       (a) way["highway"]      — every highway type (motorway down to path)
       (b) way["building"]     — simple building footprints (closed ways)
       (c) relation["building"] — multipolygon building relations
     within the AOI bbox.
  2. Convert highway ways → LineString features, keeping the `highway`
     tag value (and `name` / `ref` if present) as properties so road
     types remain distinguishable later.
  3. Convert building ways → Polygon features from the closed way ring.
     Convert building relations → Polygon (or MultiPolygon) features from
     the union of their "outer" member ways only — inner rings/holes are
     not extracted at this pass (per project scope for this task).
  4. Write both as separate GeoJSON FeatureCollections in EPSG:4326.

Usage:
    python3 infrastructure/fetch_roads_buildings.py
    python3 infrastructure/fetch_roads_buildings.py --lat 28.36687 --lon -81.43299 --radius_km 1.0

Output:
    infrastructure/data/roads.geojson
    infrastructure/data/buildings.geojson
"""

import argparse
import json
import math
import os
import sys
import time

import requests
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import linemerge, unary_union, polygonize
from pyproj import Transformer

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

DEFAULT_LAT = 28.36687
DEFAULT_LON = -81.43299
DEFAULT_RADIUS_KM = 1.0

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 90


def bbox_from_center(lat, lon, radius_km):
    """Return (south, west, north, east) in EPSG:4326."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def fetch_roads_buildings(south, west, north, east, retries=3):
    """Query Overpass for all highway ways + all building ways/relations in the bbox."""
    query = (
        f'[out:json][timeout:{OVERPASS_TIMEOUT}];'
        f'('
        f'way["highway"]({south},{west},{north},{east});'
        f'way["building"]({south},{west},{north},{east});'
        f'relation["building"]({south},{west},{north},{east});'
        f');'
        f'out body;>;out skel qt;'
    )
    headers = {
        "User-Agent": "DeepEarth/1.0 (CFX SR417 corridor research; contact qhuang62@asu.edu)",
        "Accept": "application/json",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=headers,
                timeout=OVERPASS_TIMEOUT + 10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Overpass attempt {attempt} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Overpass fetch failed after {retries} attempts: {e}") from e


def build_roads(osm_data):
    """Convert highway ways → (LineString, properties) tuples, lon/lat order."""
    nodes = {
        e["id"]: (e["lon"], e["lat"])
        for e in osm_data["elements"]
        if e["type"] == "node"
    }
    roads = []
    for e in osm_data["elements"]:
        if e["type"] != "way":
            continue
        tags = e.get("tags", {})
        if "highway" not in tags:
            continue
        coords = [nodes[n] for n in e["nodes"] if n in nodes]
        if len(coords) < 2:
            continue
        roads.append((
            LineString(coords),
            {
                "osm_id": e["id"],
                "highway": tags.get("highway"),
                "name": tags.get("name"),
                "ref": tags.get("ref"),
            },
        ))
    return roads


def _way_ring(way_elem, nodes):
    """Return a closed-ring coordinate list for a way, or None if not usable."""
    coords = [nodes[n] for n in way_elem["nodes"] if n in nodes]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def build_buildings(osm_data):
    """Convert building ways + building relations → (Polygon(s), properties) tuples."""
    nodes = {
        e["id"]: (e["lon"], e["lat"])
        for e in osm_data["elements"]
        if e["type"] == "node"
    }
    ways_by_id = {e["id"]: e for e in osm_data["elements"] if e["type"] == "way"}

    buildings = []

    # --- Simple building ways (closed ways with a building tag) ---
    relation_member_way_ids = set()
    for e in osm_data["elements"]:
        if e["type"] == "relation" and "building" in e.get("tags", {}):
            for m in e.get("members", []):
                if m.get("type") == "way":
                    relation_member_way_ids.add(m.get("ref"))

    for e in osm_data["elements"]:
        if e["type"] != "way":
            continue
        tags = e.get("tags", {})
        if "building" not in tags:
            continue
        if e["id"] in relation_member_way_ids:
            # Already represented (or will be) via its parent relation — skip to
            # avoid double-counting the same footprint.
            continue
        ring = _way_ring(e, nodes)
        if ring is None:
            continue
        try:
            poly = Polygon(ring)
            if not poly.is_valid or poly.area == 0:
                continue
        except Exception:
            continue
        buildings.append((
            poly,
            {
                "osm_id": e["id"],
                "osm_type": "way",
                "building": tags.get("building"),
                "name": tags.get("name"),
            },
        ))

    # --- Building relations (multipolygons) — outer rings only ---
    for e in osm_data["elements"]:
        if e["type"] != "relation":
            continue
        tags = e.get("tags", {})
        if "building" not in tags:
            continue
        outer_lines = []
        for m in e.get("members", []):
            if m.get("type") != "way" or m.get("role") != "outer":
                continue
            way = ways_by_id.get(m.get("ref"))
            if way is None:
                continue
            coords = [nodes[n] for n in way["nodes"] if n in nodes]
            if len(coords) >= 2:
                outer_lines.append(LineString(coords))
        if not outer_lines:
            continue
        try:
            merged = linemerge(outer_lines)
            polys = list(polygonize([merged]) if merged.geom_type == "LineString"
                         else polygonize(merged))
            if not polys:
                # Fall back: try polygonizing the raw (unmerged) lines directly
                polys = list(polygonize(outer_lines))
            if not polys:
                continue
        except Exception:
            continue
        for poly in polys:
            if not poly.is_valid or poly.area == 0:
                continue
            buildings.append((
                poly,
                {
                    "osm_id": e["id"],
                    "osm_type": "relation",
                    "building": tags.get("building"),
                    "name": tags.get("name"),
                },
            ))

    return buildings


def write_geojson(output_path, geom_props_list):
    features = [
        {"type": "Feature", "geometry": mapping(geom), "properties": props}
        for geom, props in geom_props_list
    ]
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(fc, f)


def main():
    parser = argparse.ArgumentParser(description="Fetch OSM roads + building footprints for the AOI")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS_KM)
    site_registry.add_site_arg(parser)
    args = site_registry.resolve(parser.parse_args(), category="infrastructure")
    if args.site_data_root:
        # Rebind the module-level DATA_DIR so every function writing output lands in the
        # selected site's own tree instead of the main AOI's (the exact clobbering
        # fetch_naip_site3.py's docstring warns about — both share e.g. naip_2021_RGB.tif).
        globals()["DATA_DIR"] = args.site_data_dir

    south, west, north, east = bbox_from_center(args.lat, args.lon, args.radius_km)
    print(f"AOI bbox (EPSG:4326): S={south:.5f} W={west:.5f} N={north:.5f} E={east:.5f}")

    print("Fetching roads + buildings from Overpass API...")
    osm_data = fetch_roads_buildings(south, west, north, east)
    n_nodes = sum(1 for e in osm_data["elements"] if e["type"] == "node")
    n_ways = sum(1 for e in osm_data["elements"] if e["type"] == "way")
    n_rels = sum(1 for e in osm_data["elements"] if e["type"] == "relation")
    print(f"  Raw elements: {n_nodes} nodes, {n_ways} ways, {n_rels} relations")

    roads = build_roads(osm_data)
    buildings = build_buildings(osm_data)

    print(f"  Roads found: {len(roads)}")
    print(f"  Buildings found: {len(buildings)}")

    if not roads:
        print("ERROR: No roads found in AOI. This is almost certainly a transient Overpass "
              "error (see fetch_nhd.py precedent in this repo where a CloudFront hiccup "
              "silently returned zero features) — retry before treating this as a real "
              "'no roads here' finding.")
        sys.exit(1)
    if not buildings:
        print("WARNING: No buildings found in AOI — same caveat as above; retry before trusting this.")

    # --- Project to UTM17N to compute total length/area sanity metrics ---
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)

    def project_geom(geom):
        if geom.geom_type == "LineString":
            return LineString([tf.transform(x, y) for x, y in geom.coords])
        if geom.geom_type == "Polygon":
            ext = [tf.transform(x, y) for x, y in geom.exterior.coords]
            ints = [[tf.transform(x, y) for x, y in ring.coords] for ring in geom.interiors]
            return Polygon(ext, ints)
        return geom

    total_road_length_m = sum(project_geom(geom).length for geom, _ in roads)
    total_building_area_m2 = sum(project_geom(geom).area for geom, _ in buildings)

    roads_path = os.path.join(DATA_DIR, "roads.geojson")
    write_geojson(roads_path, roads)
    print(f"\nGeoJSON written: {roads_path}  ({len(roads)} features)")

    buildings_path = os.path.join(DATA_DIR, "buildings.geojson")
    write_geojson(buildings_path, buildings)
    print(f"GeoJSON written: {buildings_path}  ({len(buildings)} features)")

    summary = {
        "aoi": {"lat": args.lat, "lon": args.lon, "radius_km": args.radius_km},
        "roads": {
            "count": len(roads),
            "total_length_m": round(total_road_length_m, 1),
            "total_length_km": round(total_road_length_m / 1000, 3),
        },
        "buildings": {
            "count": len(buildings),
            "total_footprint_area_m2": round(total_building_area_m2, 1),
            "total_footprint_area_ha": round(total_building_area_m2 / 1e4, 3),
        },
        "source": "OpenStreetMap via Overpass API",
        "note": "Buildings from multipolygon relations use outer rings only (no holes extracted).",
    }
    summary_path = os.path.join(DATA_DIR, "roads_buildings_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {summary_path}")

    print("\nSummary:")
    print(f"  Roads: {len(roads)} segments, {total_road_length_m/1000:.2f} km total")
    print(f"  Buildings: {len(buildings)} footprints, {total_building_area_m2/1e4:.2f} ha total")


if __name__ == "__main__":
    main()
