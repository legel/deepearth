"""
FDEP Lake Bathymetry Search — Winter Garden FL
================================================
Searches the Florida Department of Environmental Protection (FDEP) GIS data
portal for real lake bathymetric survey data for the Winter Garden chain-of-lakes
near the study site (28.521592°N, 81.656981°W).

Priority lakes searched (Orange County, FL):
  - Lake Apopka (largest nearby lake)
  - Lake Butler (directly adjacent to study area)
  - Lake Down, Lake Bessie, Lake Frozen (Winter Garden chain-of-lakes)

Data sources tried (in priority order):
  1. FDEP STORET / WIN portal (REST API)
  2. FDEP GIS Open Data portal (ArcGIS REST services)
  3. USGS NHD Plus lake bathymetry layer
  4. Florida Lakes Information System (FLIS)

If found: saves bathymetric contours as GeoJSON and rasterizes to match
          the DEM grid → lake_bed_dem_fdep.tif  (replaces estimated bed)

If not found: prints clear message; existing estimated bed (lake_bed_dem.tif)
              remains the authoritative source for volume calculations.

Usage:
    python3 dem/fetch_fdep_bathymetry.py
    python3 dem/fetch_fdep_bathymetry.py --lake "Lake Butler"
"""

import os
import sys
import json
import time
import argparse
import warnings
import urllib.request
import urllib.parse

import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_TIF  = os.path.join(DATA_DIR, "lake_bed_dem_fdep.tif")
OUTPUT_JSON = os.path.join(DATA_DIR, "fdep_bathymetry.geojson")

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981
SEARCH_RADIUS_KM = 5.0

# Winter Garden / Orange County lakes of interest
TARGET_LAKES = [
    "Lake Butler",
    "Lake Down",
    "Lake Bessie",
    "Lake Frozen",
    "Lake Apopka",
    "Johns Lake",
]

# ── FDEP / GIS endpoints ──────────────────────────────────────────────────────

FDEP_ARCGIS_BASE = "https://geodata.dep.state.fl.us/arcgis/rest/services"
FDEP_LAKES_URL   = (
    "https://geodata.dep.state.fl.us/arcgis/rest/services/"
    "OpenData/DEP_Surface_Waters/MapServer/0/query"
)
# FDEP BMAP (Basin Management Action Plans) lake monitoring
FDEP_WIN_STATIONS = "https://floridadep.gov/dear/watershed-monitoring-section"

# USGS NHD WFS endpoint
USGS_NHD_WFS = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/"
    "NHDPlus/MapServer/9/query"
)

# FWRI (FWC) lake data
FWRI_LAKES_URL = "https://hub.arcgis.com/api/v3/datasets/a2c5da6c39524e5c80a4e88e11a0a47e_0/downloads/data"


def _get_json(url, params=None, timeout=30, verify_ssl=True):
    """Fetch URL, return parsed JSON or None on error."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FloodDigitalTwin/1.0"})
        import ssl
        ctx = ssl.create_default_context()
        if not verify_ssl:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return None


def search_fdep_arcgis_lakes():
    """
    Query FDEP ArcGIS REST service for lake polygons near the study site.
    Returns list of dicts with lake name and geometry info, or empty list.
    """
    print("\n[1/4] Querying FDEP ArcGIS surface water layer …")

    bbox = (
        PROPERTY_LON - 0.05, PROPERTY_LAT - 0.05,
        PROPERTY_LON + 0.05, PROPERTY_LAT + 0.05,
    )
    params = {
        "where": "1=1",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "GNIS_NAME,WATERBODY_TYPE,AREASQKM",
        "returnGeometry": "true",
        "f": "geojson",
    }
    data = _get_json(FDEP_LAKES_URL, params, verify_ssl=False)
    if data and "features" in data:
        features = data["features"]
        lakes = []
        for feat in features:
            name = feat.get("properties", {}).get("GNIS_NAME", "")
            if name:
                lakes.append(feat)
        print(f"  Found {len(features)} water body features near study site")
        for feat in features[:10]:
            props = feat.get("properties", {})
            print(f"    {props.get('GNIS_NAME','(unnamed)')} — {props.get('WATERBODY_TYPE','?')} "
                  f"({props.get('AREASQKM','?')} km²)")
        return features
    else:
        print("  FDEP ArcGIS layer not reachable or no features returned")
        return []


def search_usgs_nhd_lakes():
    """
    Query USGS NHD Plus lake polygons near the study site.
    Returns GeoJSON features or empty list.
    """
    print("\n[2/4] Querying USGS NHD lake layer …")
    bbox = (
        PROPERTY_LON - 0.05, PROPERTY_LAT - 0.05,
        PROPERTY_LON + 0.05, PROPERTY_LAT + 0.05,
    )
    params = {
        "where": "FTYPE=390 OR FTYPE=436",   # 390=LakePond, 436=Reservoir
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "GNIS_NAME,AREASQKM,FTYPE",
        "returnGeometry": "false",
        "f": "json",
    }
    data = _get_json(USGS_NHD_WFS, params, verify_ssl=False)
    if data and "features" in data:
        features = data["features"]
        print(f"  Found {len(features)} NHD lake/reservoir features")
        for feat in features[:10]:
            attrs = feat.get("attributes", {})
            print(f"    {attrs.get('GNIS_NAME','(unnamed)')} — area {attrs.get('AREASQKM','?')} km²")
        return features
    else:
        print("  USGS NHD layer not reachable or no features returned")
        return []


def search_fdep_bathymetric_surveys():
    """
    Search FDEP portal for bathymetric survey datasets.
    FDEP conducts periodic lake bathymetry surveys; results posted on the
    FDEP open data portal or ftp.dep.state.fl.us.

    Returns: list of found survey records (dicts), or empty list.
    """
    print("\n[3/4] Searching FDEP open data for bathymetric surveys …")

    # FDEP open data CKAN/ArcGIS Hub — search for lake bathymetry
    search_urls = [
        (
            "https://geodata.dep.state.fl.us/arcgis/rest/services/OpenData/DEP_Lake_Bathymetry/MapServer",
            "FDEP Lake Bathymetry MapServer"
        ),
        (
            "https://geodata.dep.state.fl.us/arcgis/rest/services/OpenData/DEP_Lake_Monitoring/MapServer",
            "FDEP Lake Monitoring MapServer"
        ),
        (
            "https://geodata.dep.state.fl.us/arcgis/rest/services",
            "FDEP ArcGIS Services root"
        ),
    ]

    found_services = []
    for url, label in search_urls:
        params = {"f": "json"}
        data = _get_json(url, params, verify_ssl=False)
        if data:
            print(f"  ✓ Reachable: {label}")
            if "services" in data:
                for svc in data["services"]:
                    name = svc.get("name", "")
                    if any(kw in name.lower() for kw in ["bath", "lake", "depth", "contour"]):
                        print(f"    → Relevant service: {name}")
                        found_services.append(svc)
            elif "layers" in data:
                for lyr in data["layers"]:
                    print(f"    Layer: {lyr.get('name','?')}")
        else:
            print(f"  ✗ Not reachable: {label}")

    # Also try the FDEP FTP / direct download for Orange County
    fdep_orange_url = (
        "https://geodata.dep.state.fl.us/datasets/dep-lake-bathymetry-contours"
    )
    print(f"\n  Checking FDEP lake bathymetry contours dataset …")
    data = _get_json(fdep_orange_url + "/FeatureServer/0/query",
                     {"where": "1=1", "resultRecordCount": 1, "f": "json"},
                     verify_ssl=False)
    if data and "features" in data:
        print(f"  ✓ FDEP bathymetry contours dataset is accessible")
        found_services.append({"name": "DEP Lake Bathymetry Contours",
                                "url": fdep_orange_url})
    else:
        print("  ✗ FDEP bathymetry contours dataset not found at that URL")

    return found_services


def fetch_bathymetry_for_lake(lake_name):
    """
    Try to fetch actual bathymetric depth contours for a named lake.
    Returns GeoJSON FeatureCollection or None.
    """
    print(f"\n[4/4] Fetching bathymetric data for '{lake_name}' …")

    # Try ArcGIS Hub / FDEP with lake name filter
    base_urls = [
        "https://geodata.dep.state.fl.us/arcgis/rest/services/OpenData/DEP_Lake_Bathymetry/MapServer/0/query",
        "https://services.arcgis.com/2Lt3er5cNy9PWHZ9/arcgis/rest/services/FL_Lakes_Bathymetry/FeatureServer/0/query",
    ]

    for url in base_urls:
        params = {
            "where": f"LAKE_NAME LIKE '%{lake_name.upper()}%' OR GNIS_NAME LIKE '%{lake_name}%'",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        }
        data = _get_json(url, params, verify_ssl=False)
        if data and data.get("features"):
            print(f"  ✓ Found {len(data['features'])} bathymetric features for {lake_name}")
            return data

    print(f"  ✗ No bathymetric data found for {lake_name} via API")
    return None


def rasterize_contours_to_dem(geojson_fc, dem_path, out_tif):
    """
    Interpolate bathymetric depth contours onto the DEM grid.
    Uses linear interpolation (scipy griddata) from contour vertices.
    """
    import rasterio
    from scipy.interpolate import griddata

    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        rows, cols = dem.shape

    # Extract (x, y, depth) points from GeoJSON contours
    points = []
    for feat in geojson_fc.get("features", []):
        depth = feat.get("properties", {}).get("DEPTH_FT") or feat.get("properties", {}).get("depth_m")
        if depth is None:
            continue
        depth_m = float(depth) * 0.3048 if "FT" in str(feat.get("properties",{}).keys()) else float(depth)
        geom = feat.get("geometry", {})
        coords = []
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
        elif geom.get("type") == "MultiLineString":
            for part in geom.get("coordinates", []):
                coords.extend(part)
        for x, y, *_ in coords:
            points.append((x, y, depth_m))

    if len(points) < 3:
        print(f"  Only {len(points)} depth points found — cannot interpolate")
        return False

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    ds = np.array([p[2] for p in points])

    # Build grid of DEM pixel centers in geographic coords
    col_idx = np.arange(cols)
    row_idx = np.arange(rows)
    cc, rr = np.meshgrid(col_idx, row_idx)
    xs_grid = transform.c + (cc + 0.5) * transform.a
    ys_grid = transform.f + (rr + 0.5) * transform.e

    # Interpolate
    bed_interp = griddata(
        np.column_stack([xs, ys]), ds,
        (xs_grid, ys_grid),
        method="linear", fill_value=np.nan
    ).astype(np.float32)

    profile.update(dtype="float32", count=1, compress="lzw", nodata=-9999.0)
    bed_interp = np.where(np.isnan(bed_interp), -9999.0, bed_interp)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(bed_interp, 1)
    print(f"  Saved FDEP bathymetry raster → {out_tif}")
    return True


def summarize_findings(lake_features, nhd_features, survey_services, target_lake):
    """Print a clear summary of what was found and what to use for volume calc."""
    dem_path = os.path.join(DATA_DIR, "winter_garden_dem.tif")

    print("\n" + "=" * 70)
    print("FDEP BATHYMETRY SEARCH SUMMARY")
    print("=" * 70)
    print(f"Study site : {PROPERTY_LAT}°N, {PROPERTY_LON}°W (Winter Garden FL)")
    print(f"Search radius : {SEARCH_RADIUS_KM} km")
    print()

    print("Lakes found in AOI (FDEP/NHD):")
    found_names = set()
    for feat in lake_features:
        name = feat.get("properties", {}).get("GNIS_NAME", "(unnamed)")
        area = feat.get("properties", {}).get("AREASQKM", "?")
        if name not in found_names:
            found_names.add(name)
            marker = "★" if any(t.lower() in name.lower() for t in TARGET_LAKES) else " "
            print(f"  {marker} {name} ({area} km²)")

    print()
    print("FDEP Bathymetric Survey Services:")
    if survey_services:
        for svc in survey_services:
            print(f"  ✓ {svc.get('name','?')}")
    else:
        print("  ✗ No accessible bathymetric survey services found")

    # Check if we managed to write a bathymetry raster
    if os.path.exists(OUTPUT_TIF):
        print()
        print(f"  ✓ FDEP raster written → {OUTPUT_TIF}")
        print("    Use this in lake_volume.py instead of estimated lake_bed_dem.tif")
        print("    Label all volume visualizations: 'Lake bed: FDEP survey data'")
    else:
        print()
        print("  ✗ No FDEP bathymetric raster could be generated")
        print()
        print("  RECOMMENDATION: Continue with estimated lake_bed_dem.tif")
        print("  Label all volume visualizations:")
        print("    'Lake bed: estimated from shoreline slope (not surveyed)'")
        print()
        print("  MANUAL ALTERNATIVE:")
        print("  If you need real bathymetry data, request it directly from FDEP:")
        print("    → Email: GIS.Libraries@dep.state.fl.us")
        print("    → Subject: 'Lake bathymetry data request — Winter Garden chain-of-lakes'")
        print("    → Lakes: Lake Butler, Lake Down, Lake Bessie, Lake Frozen")
        print("    → Orange County, FL — for academic/research use")
        print()
        print("  Also check:")
        print("    → USGS Science Base: sciencebase.gov")
        print("      Search: 'Lake Butler Orange County Florida bathymetry'")
        print("    → SJRWMD (St. Johns River Water Management District):")
        print("      www.sjrwmd.com — may have lake surveys for Orange County")

    print("=" * 70)


def main(target_lake=None):
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("FDEP Lake Bathymetry Search")
    print(f"Study site: {PROPERTY_LAT}°N, {PROPERTY_LON}°W — Winter Garden FL")
    print("=" * 70)

    lake_features  = search_fdep_arcgis_lakes()
    nhd_features   = search_usgs_nhd_lakes()
    survey_services = search_fdep_bathymetric_surveys()

    # Try target lake if specified, otherwise try all priority lakes
    bathymetry_fc = None
    lakes_to_try = [target_lake] if target_lake else TARGET_LAKES
    for lake_name in lakes_to_try:
        fc = fetch_bathymetry_for_lake(lake_name)
        if fc:
            bathymetry_fc = fc
            break

    # If we got bathymetric contours, rasterize them
    dem_path = os.path.join(DATA_DIR, "winter_garden_dem.tif")
    if bathymetry_fc and os.path.exists(dem_path):
        try:
            rasterize_contours_to_dem(bathymetry_fc, dem_path, OUTPUT_TIF)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(bathymetry_fc, f)
            print(f"Saved GeoJSON → {OUTPUT_JSON}")
        except Exception as exc:
            print(f"  Could not rasterize contours: {exc}")

    summarize_findings(lake_features, nhd_features, survey_services, target_lake)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search FDEP for lake bathymetry data")
    parser.add_argument("--lake", type=str, default=None,
                        help="Specific lake name to search (default: try all priority lakes)")
    args = parser.parse_args()
    main(args.lake)
