#!/usr/bin/env python3
"""
Generate an estimated KML boundary for the CFX SR417 corridor parcel
within the 2x2km AOI around (28.36687, -81.43299).

Approach:
  1. Fetch SR417 centerline ways from OpenStreetMap via Overpass API.
  2. Project to UTM 17N (EPSG:32617) for metric buffering.
  3. Buffer each carriageway by ROW_HALF_WIDTH_M (default 100 m, giving a
     ~200 m total corridor — a reasonable estimate for a CFX toll expressway).
  4. Union all buffered carriageway polygons.
  5. Clip to the 2x2 km AOI bounding box.
  6. Re-project to WGS84 and write as KML.

The resulting boundary is *estimated* from OSM centerlines + a standard ROW
buffer. It should be replaced by a surveyed parcel boundary once CFX provides
one, but serves as a valid planning polygon in the interim.

Usage:
    python3 boundary/generate_cfx_corridor_kml.py
    python3 boundary/generate_cfx_corridor_kml.py --lat 28.36687 --lon -81.43299 --radius_km 1.0 --row_half_m 100

Output:
    boundary/data/cfx_sr417_corridor_estimated.kml
    boundary/data/cfx_sr417_corridor_estimated.geojson
"""

import argparse
import json
import math
import os
import sys
import time

import requests
from shapely.geometry import (
    LineString, MultiLineString, MultiPolygon, Polygon, box, mapping, shape
)
from shapely.ops import unary_union
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT = 28.36687
DEFAULT_LON = -81.43299
DEFAULT_RADIUS_KM = 1.0
DEFAULT_ROW_HALF_M = 100  # half-width of estimated ROW in meters

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 60


def bbox_from_center(lat, lon, radius_km):
    """Return (south, west, north, east) in EPSG:4326."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def fetch_sr417_ways(south, west, north, east, retries=3):
    """Query Overpass for SR 417 motorway ways within the bbox."""
    query = (
        f'[out:json][timeout:{OVERPASS_TIMEOUT}];'
        f'(way["highway"]["ref"~"417"]({south},{west},{north},{east}););'
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


def build_linestrings(osm_data):
    """Convert Overpass JSON elements to Shapely LineStrings (lon, lat order)."""
    nodes = {
        e["id"]: (e["lon"], e["lat"])
        for e in osm_data["elements"]
        if e["type"] == "node"
    }
    lines = []
    for e in osm_data["elements"]:
        if e["type"] != "way":
            continue
        coords = [nodes[n] for n in e["nodes"] if n in nodes]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def project_lines(lines, from_crs="EPSG:4326", to_crs="EPSG:32617"):
    """Project a list of Shapely LineStrings to a target CRS."""
    tf = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    projected = []
    for line in lines:
        pts = [tf.transform(x, y) for x, y in line.coords]
        projected.append(LineString(pts))
    return projected


def project_polygon(poly, from_crs="EPSG:32617", to_crs="EPSG:4326"):
    """Project a Shapely Polygon (or MultiPolygon) back to WGS84."""
    tf = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    def transform_ring(ring):
        return [tf.transform(x, y) for x, y in ring.coords]

    def transform_poly(p):
        exterior = transform_ring(p.exterior)
        interiors = [transform_ring(i) for i in p.interiors]
        return Polygon(exterior, interiors)

    if isinstance(poly, MultiPolygon):
        return MultiPolygon([transform_poly(p) for p in poly.geoms])
    return transform_poly(poly)


def to_kml_coords(coords):
    """Format a coordinate list as KML coordinate string (lon,lat,0)."""
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y in coords)


def polygon_to_kml_placemark(poly, name, description, style_id):
    """Render a single Shapely Polygon as a KML Placemark."""
    outer = to_kml_coords(poly.exterior.coords)
    inner_rings = ""
    for interior in poly.interiors:
        inner_rings += f"""
            <innerBoundaryIs>
              <LinearRing><coordinates>{to_kml_coords(interior.coords)}</coordinates></LinearRing>
            </innerBoundaryIs>"""
    return f"""  <Placemark>
    <name>{name}</name>
    <description>{description}</description>
    <styleUrl>#{style_id}</styleUrl>
    <Polygon>
      <extrude>0</extrude>
      <altitudeMode>clampToGround</altitudeMode>
      <outerBoundaryIs>
        <LinearRing><coordinates>{outer}</coordinates></LinearRing>
      </outerBoundaryIs>{inner_rings}
    </Polygon>
  </Placemark>"""


def write_kml(output_path, placemarks, doc_name="CFX SR417 Corridor"):
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{doc_name}</name>
  <Style id="cfx_corridor">
    <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
    <PolyStyle><color>660000ff</color></PolyStyle>
  </Style>
  <Style id="aoi_box">
    <LineStyle><color>ff00ffff</color><width>1.5</width></LineStyle>
    <PolyStyle><fill>0</fill></PolyStyle>
  </Style>
{"".join(placemarks)}
</Document>
</kml>"""
    with open(output_path, "w") as f:
        f.write(kml)


def write_geojson(output_path, geom, properties):
    feature = {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": properties,
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    with open(output_path, "w") as f:
        json.dump(fc, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate estimated CFX SR417 corridor KML")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument("--row_half_m", type=float, default=DEFAULT_ROW_HALF_M,
                        help="Half-width of estimated ROW buffer in meters (default: 100)")
    args = parser.parse_args()

    south, west, north, east = bbox_from_center(args.lat, args.lon, args.radius_km)
    print(f"AOI bbox (EPSG:4326): S={south:.5f} W={west:.5f} N={north:.5f} E={east:.5f}")

    # --- 1. Fetch SR417 from Overpass ---
    print("Fetching SR417 centerlines from Overpass API...")
    osm_data = fetch_sr417_ways(south, west, north, east)
    ways = [e for e in osm_data["elements"] if e["type"] == "way"]
    nodes = {e["id"]: e for e in osm_data["elements"] if e["type"] == "node"}
    print(f"  {len(ways)} way segments, {len(nodes)} nodes")
    if not ways:
        print("ERROR: No SR417 ways found in AOI. Check coordinates or Overpass connectivity.")
        sys.exit(1)

    lines_wgs84 = build_linestrings(osm_data)
    print(f"  Built {len(lines_wgs84)} LineStrings")

    # --- 2. Project to UTM 17N for buffering ---
    lines_utm = project_lines(lines_wgs84, from_crs="EPSG:4326", to_crs="EPSG:32617")

    # --- 3. Buffer each carriageway ---
    buffered = [line.buffer(args.row_half_m, cap_style=2) for line in lines_utm]
    corridor_utm = unary_union(buffered)
    print(f"  Buffered by {args.row_half_m} m → corridor area: {corridor_utm.area/1e6:.4f} km²")

    # --- 4. Clip to AOI bbox ---
    tf_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    aoi_corners_utm = [
        tf_to_utm.transform(west, south),
        tf_to_utm.transform(east, south),
        tf_to_utm.transform(east, north),
        tf_to_utm.transform(west, north),
    ]
    aoi_box_utm = box(
        min(c[0] for c in aoi_corners_utm),
        min(c[1] for c in aoi_corners_utm),
        max(c[0] for c in aoi_corners_utm),
        max(c[1] for c in aoi_corners_utm),
    )
    corridor_clipped_utm = corridor_utm.intersection(aoi_box_utm)
    print(f"  Clipped corridor area: {corridor_clipped_utm.area/1e6:.4f} km²")

    # --- 5. Project back to WGS84 ---
    corridor_wgs84 = project_polygon(corridor_clipped_utm, from_crs="EPSG:32617", to_crs="EPSG:4326")
    aoi_box_wgs84 = box(west, south, east, north)

    # --- 6. Write KML ---
    placemarks = []
    desc = (
        f"Estimated CFX SR417 right-of-way corridor. "
        f"Source: OpenStreetMap centerlines buffered {args.row_half_m} m each side "
        f"(~{int(args.row_half_m * 2 / 0.3048)} ft total estimated ROW width). "
        f"Clipped to {args.radius_km * 2:.1f} km × {args.radius_km * 2:.1f} km AOI "
        f"centered on ({args.lat}, {args.lon}). "
        f"ESTIMATED — replace with CFX surveyed boundary when available."
    )

    geoms = (
        list(corridor_wgs84.geoms)
        if isinstance(corridor_wgs84, MultiPolygon)
        else [corridor_wgs84]
    )
    for i, poly in enumerate(geoms):
        label = "CFX SR417 Estimated ROW Corridor" + (f" (part {i+1})" if len(geoms) > 1 else "")
        placemarks.append(polygon_to_kml_placemark(poly, label, desc, "cfx_corridor"))

    # AOI reference box
    aoi_coords = to_kml_coords(list(aoi_box_wgs84.exterior.coords))
    placemarks.append(f"""  <Placemark>
    <name>2×2 km AOI Reference Box</name>
    <styleUrl>#aoi_box</styleUrl>
    <Polygon>
      <extrude>0</extrude><altitudeMode>clampToGround</altitudeMode>
      <outerBoundaryIs>
        <LinearRing><coordinates>{aoi_coords}</coordinates></LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>""")

    kml_path = os.path.join(DATA_DIR, "cfx_sr417_corridor_estimated.kml")
    write_kml(kml_path, placemarks)
    print(f"\nKML written: {kml_path}")

    # --- 7. Write GeoJSON ---
    geojson_path = os.path.join(DATA_DIR, "cfx_sr417_corridor_estimated.geojson")
    write_geojson(geojson_path, corridor_wgs84, {
        "name": "CFX SR417 Estimated ROW Corridor",
        "source": "OpenStreetMap via Overpass API",
        "method": f"SR417 centerlines buffered {args.row_half_m} m each side",
        "row_half_width_m": args.row_half_m,
        "row_total_width_m": args.row_half_m * 2,
        "row_total_width_ft": round(args.row_half_m * 2 / 0.3048),
        "aoi_center_lat": args.lat,
        "aoi_center_lon": args.lon,
        "aoi_radius_km": args.radius_km,
        "status": "ESTIMATED — replace with CFX surveyed boundary when available",
    })
    print(f"GeoJSON written: {geojson_path}")

    area_m2 = corridor_clipped_utm.area
    print(f"\nSummary:")
    print(f"  OSM ways used: {len(ways)}")
    print(f"  ROW half-width: {args.row_half_m} m ({args.row_half_m / 0.3048:.0f} ft)")
    print(f"  Estimated total ROW width: {args.row_half_m * 2} m ({args.row_half_m * 2 / 0.3048:.0f} ft)")
    print(f"  Corridor area within AOI: {area_m2:.0f} m² = {area_m2/4e6*100:.1f}% of AOI")


if __name__ == "__main__":
    main()
