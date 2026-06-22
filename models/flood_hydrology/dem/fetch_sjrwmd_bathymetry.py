"""
SJRWMD / FWC / FDEP Bathymetric Survey Fetch — Winter Garden FL
================================================================
Searches multiple Florida water management and environmental agency databases
for real lake bathymetric survey data for the study lake near 17801 Champagne
Dr, Winter Garden FL (Johns Lake or nearby unnamed lake body).

Data sources queried (in priority order):
    1. FWC (Florida Fish & Wildlife Conservation Commission)
       — Bathymetric Survey Program, depth contour polygons
       — ArcGIS MapServer: https://geodata.myfwc.com/...
    2. SJRWMD (St. Johns River Water Management District)
       — ArcGIS Open Data Hub (Hub REST API)
       — Covers Orange County; manages most central FL lakes
    3. FDEP (Florida Department of Environmental Protection)
       — Open Data CKAN hub + ArcGIS services
    4. USGS NHD Plus High-Resolution lake depth polygons
    5. USACE (Army Corps) hydrographic surveys (nationwide)

When depth contour data is found:
    — Extract (lon, lat, depth_m) from LineString/Polygon contours
    — Interpolate to DEM grid using scipy.griddata (linear)
    — Save: dem/data/lake_bed_dem_survey.tif (authoritative bathymetry)
    — Save: dem/data/bathymetry_source.txt (metadata: agency, survey date)

If no survey data is found for this specific lake, prints manual data request
instructions (email templates for FWC and SJRWMD).

Usage:
    python3 dem/fetch_sjrwmd_bathymetry.py
    python3 dem/fetch_sjrwmd_bathymetry.py --radius_km 3.0
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981
DEFAULT_RADIUS_KM = 2.0  # search radius around property

OUT_TIF  = os.path.join(DATA_DIR, "lake_bed_dem_survey.tif")
OUT_META = os.path.join(DATA_DIR, "bathymetry_source.txt")


def bbox_from_center(lat, lon, radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat  # W, S, E, N


def _get_with_retry(url, params=None, timeout=30, retries=3):
    """HTTP GET with retry logic."""
    import requests
    import time
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return None


# ── Source 1: FWC Bathymetric Survey Program ──────────────────────────────────

FWC_BATHYMETRY_SERVICES = [
    # ArcGIS MapServer for FWC lake bathymetry surveys (depth contour lines)
    "https://geodata.myfwc.com/arcgis/rest/services/fishing/Florida_Lakes_Bathymetry/MapServer/0",
    "https://geodata.myfwc.com/arcgis/rest/services/fishing/Florida_Lakes_Bathymetry/MapServer/1",
    "https://geodata.myfwc.com/arcgis/rest/services/fishing/Florida_Lakes_Bathymetry/MapServer/2",
    "https://geodata.myfwc.com/arcgis/rest/services/fishing/Florida_Lakes_Bathymetry/MapServer/3",
]


def fetch_fwc_bathymetry(bbox):
    """Query FWC ArcGIS MapServer for bathymetric depth contours."""
    west, south, east, north = bbox
    envelope = f"{west},{south},{east},{north}"

    depth_points = []
    for service_url in FWC_BATHYMETRY_SERVICES:
        try:
            params = {
                "geometry": envelope,
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "inSR": "4326",
                "outSR": "4326",
            }
            resp = _get_with_retry(f"{service_url}/query", params=params, timeout=30)
            if resp is None:
                continue
            data = resp.json()
            features = data.get("features", [])
            if not features:
                continue
            print(f"  FWC: {len(features)} features from {service_url.split('/')[-1]}")
            pts = _extract_depth_points(features)
            depth_points.extend(pts)
            if depth_points:
                return depth_points, "FWC Bathymetric Survey Program"
        except Exception as e:
            print(f"  FWC service error: {e}")
    return [], None


# ── Source 2: SJRWMD Open Data Hub ───────────────────────────────────────────

SJRWMD_SERVICES = [
    # SJRWMD ArcGIS Hub REST API — lake bathymetry surveys
    "https://opendata.sjrwmd.com/datasets/sjrwmd::lake-bathymetric-surveys/FeatureServer/0",
    "https://maps.sjrwmd.com/arcgis/rest/services/Public/SJRWMD_Open_Data/MapServer/0",
    "https://maps.sjrwmd.com/arcgis/rest/services/PublicServices/LakeBathymetry/MapServer/0",
    # ArcGIS Online hosted layer
    "https://services.arcgis.com/1KSVSmnHT2Lw9ea6/arcgis/rest/services/LakeBathymetricSurveys/FeatureServer/0",
]


def fetch_sjrwmd_bathymetry(bbox):
    """Query SJRWMD ArcGIS services for lake depth contour data."""
    west, south, east, north = bbox
    envelope = f"{west},{south},{east},{north}"

    depth_points = []
    for service_url in SJRWMD_SERVICES:
        try:
            params = {
                "geometry": envelope,
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "inSR": "4326",
                "outSR": "4326",
            }
            base = service_url.replace("/FeatureServer/0", "/query").replace("/MapServer/0", "/query")
            if "/query" not in base:
                base = service_url + "/query"
            resp = _get_with_retry(base, params=params, timeout=30)
            if resp is None:
                continue
            data = resp.json()
            features = data.get("features", [])
            if not features:
                continue
            print(f"  SJRWMD: {len(features)} features from {service_url}")
            pts = _extract_depth_points(features)
            depth_points.extend(pts)
            if depth_points:
                return depth_points, "SJRWMD Lake Bathymetric Survey"
        except Exception as e:
            print(f"  SJRWMD service error ({service_url}): {e}")

    # Also try SJRWMD CKAN open data
    try:
        ckan_url = "https://opendata.sjrwmd.com/api/3/action/datastore_search"
        params = {"resource_id": "lake_bathymetry", "limit": 500}
        resp = _get_with_retry(ckan_url, params=params, timeout=30)
        if resp and resp.status_code == 200:
            result = resp.json().get("result", {})
            records = result.get("records", [])
            if records:
                print(f"  SJRWMD CKAN: {len(records)} records")
    except Exception:
        pass

    return [], None


# ── Source 3: FDEP Surface Water ─────────────────────────────────────────────

FDEP_SERVICES = [
    "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/WATER_MAP/MapServer/0",
    "https://geodata.dep.state.fl.us/datasets/fdep::lake-bathymetric-survey-depth-contours/FeatureServer/0",
]


def fetch_fdep_bathymetry(bbox):
    """Query FDEP ArcGIS services for lake depth data."""
    west, south, east, north = bbox
    envelope = f"{west},{south},{east},{north}"
    depth_points = []
    for service_url in FDEP_SERVICES:
        try:
            params = {
                "geometry": envelope,
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "inSR": "4326",
                "outSR": "4326",
            }
            query_url = service_url + "/query" if "/query" not in service_url else service_url
            resp = _get_with_retry(query_url, params=params, timeout=30)
            if resp is None:
                continue
            data = resp.json()
            features = data.get("features", [])
            if not features:
                continue
            print(f"  FDEP: {len(features)} features")
            pts = _extract_depth_points(features)
            depth_points.extend(pts)
            if depth_points:
                return depth_points, "FDEP Lake Bathymetric Survey"
        except Exception as e:
            print(f"  FDEP error: {e}")
    return [], None


# ── Source 4: USGS NHD Plus ───────────────────────────────────────────────────

def fetch_nhd_bathymetry(bbox):
    """Query USGS NHD Plus for lake depth features (where available)."""
    try:
        west, south, east, north = bbox
        url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/10/query"
        params = {
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FTYPE,FCODE,GNIS_NAME,Shape_Area",
            "returnGeometry": "true",
            "f": "geojson",
            "inSR": "4326", "outSR": "4326",
        }
        resp = _get_with_retry(url, params=params, timeout=30)
        if resp is None:
            return [], None
        features = resp.json().get("features", [])
        if features:
            print(f"  NHD: {len(features)} lake features")
            # NHD doesn't provide depth contours directly, but lists lake names
            for feat in features:
                name = feat.get("properties", {}).get("GNIS_NAME", "")
                if name:
                    print(f"    Lake name: {name}")
    except Exception as e:
        print(f"  NHD query error: {e}")
    return [], None


# ── Depth point extraction from GeoJSON features ─────────────────────────────

def _extract_depth_points(features):
    """
    Extract (lon, lat, depth_m) tuples from GeoJSON features.

    Looks for depth values in: CONTOUR, CONTOUR_FT, DEPTH, DEPTH_FT,
    DEPTH_M, value, elev, Z attributes. Converts feet → metres if needed.
    Handles LineString, MultiLineString, Point, Polygon geometries.
    """
    points = []
    depth_fields = ["CONTOUR", "CONTOUR_FT", "DEPTH", "DEPTH_FT", "DEPTH_M",
                    "value", "elev", "Z", "depth", "contour", "DEPTH_FEET",
                    "BATHYMETRY", "FEET"]

    for feat in features:
        props = feat.get("properties", {}) or {}
        geom  = feat.get("geometry", {}) or {}

        # Find depth value from properties
        depth_raw = None
        is_feet   = False
        for field in depth_fields:
            if field in props and props[field] is not None:
                depth_raw = props[field]
                is_feet = "FT" in field.upper() or "FEET" in field.upper()
                break

        if depth_raw is None:
            continue

        try:
            depth_val = float(depth_raw)
        except (TypeError, ValueError):
            continue

        if depth_val <= 0:
            continue  # depth must be positive (below surface)

        depth_m = depth_val * 0.3048 if is_feet else depth_val

        # Extract coordinates from geometry
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Point":
            points.append((coords[0], coords[1], depth_m))
        elif gtype == "LineString":
            for c in coords:
                points.append((c[0], c[1], depth_m))
        elif gtype == "MultiLineString":
            for line in coords:
                for c in line:
                    points.append((c[0], c[1], depth_m))
        elif gtype in ("Polygon", "MultiPolygon"):
            rings = coords if gtype == "MultiPolygon" else [coords]
            for poly in rings:
                for ring in poly:
                    for c in ring:
                        points.append((c[0], c[1], depth_m))

    return points


# ── Interpolate depth points to DEM grid ─────────────────────────────────────

def interpolate_to_dem(depth_points, dem_path, lake_mask_path, water_surface_elev=None):
    """
    Interpolate scattered (lon, lat, depth_m) points to DEM grid.

    The output raster gives lake bed elevation in metres NAVD88:
        bed_elev(x,y) = water_surface_elev − depth_m(x,y)

    Returns (lake_bed_arr, dem_profile) or (None, None) if too few points.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from scipy.interpolate import griddata

    if len(depth_points) < 4:
        print(f"  Only {len(depth_points)} depth points — too few to interpolate reliably")
        return None, None

    with rasterio.open(dem_path) as src:
        dem_arr  = src.read(1).astype(np.float32)
        dem_prof = src.profile.copy()
        dem_crs  = src.crs
        dem_transform = src.transform
        dem_shape = dem_arr.shape

    with rasterio.open(lake_mask_path) as src:
        lake_mask = src.read(1).astype(np.uint8)

    lake_bool = (lake_mask == 1) & np.isfinite(dem_arr)
    if lake_bool.sum() == 0:
        print("  No lake cells found in lake mask")
        return None, None

    if water_surface_elev is None:
        water_surface_elev = float(np.nanmean(dem_arr[lake_bool]))
    print(f"  Water surface elevation: {water_surface_elev:.2f} m NAVD88")

    # Convert lon/lat depth points to raster CRS
    lons = np.array([p[0] for p in depth_points])
    lats = np.array([p[1] for p in depth_points])
    depths = np.array([p[2] for p in depth_points])

    # Transform to raster CRS
    import pyproj
    if "4326" in str(dem_crs) or "WGS" in str(dem_crs).upper():
        xs, ys = lons, lats
    else:
        transformer = pyproj.Transformer.from_crs("epsg:4326", dem_crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)

    # Convert to pixel coordinates
    T = dem_transform
    cols_pts = (xs - T.c) / T.a
    rows_pts = (ys - T.f) / T.e

    # Grid coordinates for interpolation target (lake cells only)
    all_rows, all_cols = np.where(lake_bool)

    # Interpolate depth using griddata
    print(f"  Interpolating {len(depth_points)} depth points to {len(all_rows):,} lake cells …")
    interp_depths = griddata(
        np.column_stack([cols_pts, rows_pts]),
        depths,
        np.column_stack([all_cols, all_rows]),
        method="linear",
        fill_value=np.nan,
    )

    # Fill any NaN holes (outside convex hull of points) with nearest-neighbor
    nan_pts = np.isnan(interp_depths)
    if nan_pts.sum() > 0:
        interp_depths_nn = griddata(
            np.column_stack([cols_pts, rows_pts]),
            depths,
            np.column_stack([all_cols[nan_pts], all_rows[nan_pts]]),
            method="nearest",
        )
        interp_depths[nan_pts] = interp_depths_nn

    # Build lake bed elevation array
    lake_bed = dem_arr.copy()
    # Convert depth → bed elevation: bed = surface − depth (positive depth = below surface)
    bed_elevs = water_surface_elev - np.maximum(interp_depths, 0.1)  # min 0.1m depth
    # Cap at surface and at reasonable max depth (15m for FL karst lakes)
    bed_elevs = np.clip(bed_elevs, water_surface_elev - 15.0, water_surface_elev - 0.1)
    lake_bed[lake_bool] = bed_elevs

    mean_depth = float(np.mean(water_surface_elev - bed_elevs))
    max_depth  = float(np.max(water_surface_elev - bed_elevs))
    print(f"  Interpolated: mean depth {mean_depth:.2f} m, max depth {max_depth:.2f} m")

    return lake_bed, dem_prof


# ── Main ──────────────────────────────────────────────────────────────────────

def main(radius_km=DEFAULT_RADIUS_KM):
    import rasterio

    dem_path = os.path.join(DATA_DIR, "winter_garden_dem.tif")
    if not os.path.exists(dem_path):
        sys.exit("DEM not found. Run dem_download.py first.")

    # Prefer S2 lake mask for authoritative lake boundary
    mask_path = (os.path.join(DATA_DIR, "lake_mask_s2.tif")
                 if os.path.exists(os.path.join(DATA_DIR, "lake_mask_s2.tif"))
                 else os.path.join(DATA_DIR, "lake_mask.tif"))
    if not os.path.exists(mask_path):
        sys.exit("Lake mask not found. Run dem_process.py first.")

    bbox = bbox_from_center(PROPERTY_LAT, PROPERTY_LON, radius_km)
    print(f"Searching for bathymetric survey data …")
    print(f"  Study location: {PROPERTY_LAT:.4f}°N, {PROPERTY_LON:.4f}°W")
    print(f"  Search radius: {radius_km} km")
    print(f"  Bbox [W,S,E,N]: {[round(x,5) for x in bbox]}")
    print()

    depth_points = []
    source_name  = None

    # Source 1: FWC
    print("── Source 1: FWC Bathymetric Survey Program ─────────────────────")
    pts, src = fetch_fwc_bathymetry(bbox)
    if pts:
        depth_points, source_name = pts, src
        print(f"  ✓ FWC: {len(pts)} depth points")

    # Source 2: SJRWMD
    if not depth_points:
        print("\n── Source 2: SJRWMD Open Data Hub ───────────────────────────────")
        pts, src = fetch_sjrwmd_bathymetry(bbox)
        if pts:
            depth_points, source_name = pts, src
            print(f"  ✓ SJRWMD: {len(pts)} depth points")

    # Source 3: FDEP
    if not depth_points:
        print("\n── Source 3: FDEP Open Data ──────────────────────────────────────")
        pts, src = fetch_fdep_bathymetry(bbox)
        if pts:
            depth_points, source_name = pts, src
            print(f"  ✓ FDEP: {len(pts)} depth points")

    # Source 4: NHD (informational only — provides lake names, not depth contours)
    print("\n── Source 4: USGS NHD Plus (lake identification) ─────────────────")
    fetch_nhd_bathymetry(bbox)

    # ── Process depth points if found ────────────────────────────────────────
    if depth_points:
        print(f"\nInterpolating {len(depth_points)} depth points to DEM grid …")
        lake_bed, dem_prof = interpolate_to_dem(depth_points, dem_path, mask_path)

        if lake_bed is not None:
            dem_prof.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
            with rasterio.open(OUT_TIF, "w", **dem_prof) as dst:
                dst.write(lake_bed.astype(np.float32), 1)
            print(f"\n✓ Saved survey bathymetry → {OUT_TIF}")

            with open(OUT_META, "w") as f:
                f.write(f"Source: {source_name}\n")
                f.write(f"Points: {len(depth_points)}\n")
                f.write(f"Search radius: {radius_km} km\n")
                f.write(f"Location: {PROPERTY_LAT}, {PROPERTY_LON}\n")
            print(f"✓ Saved metadata     → {OUT_META}")
            print(f"\nTo use survey bathymetry in lake_volume.py and flood_sim.py,")
            print(f"rename this file to lake_bed_dem_fwc.tif or update _bed_src logic.")
        else:
            print("  ✗ Interpolation failed (too few points or no lake mask)")

    else:
        print("\n── No survey data found ──────────────────────────────────────────")
        print("No bathymetric survey data returned from any source.")
        print("This lake may not have a public survey on record.")
        print()
        _print_manual_request_instructions()


def _print_manual_request_instructions():
    print("═" * 68)
    print("MANUAL DATA REQUEST INSTRUCTIONS")
    print("═" * 68)
    print()
    print("1. FWC Bathymetric Survey Program:")
    print("   https://myfwc.com/research/freshwater/programs/bathymetric-surveys/")
    print("   Email: FreshwaterFisheries@MyFWC.com")
    print("   Subject: Lake Bathymetry Data Request — [Lake Name], Orange County FL")
    print()
    print("2. SJRWMD Water Resources:")
    print("   https://www.sjrwmd.com/data/lake-water-levels/")
    print("   Phone: (386) 329-4500")
    print("   Request: Lake bathymetric survey data and water level records")
    print("   Orange County lakes contact: St. Johns River WMD, Palatka FL 32177")
    print()
    print("3. FDEP OCULUS Environmental Data Portal:")
    print("   https://oculus.dep.state.fl.us/oculus/default.screen")
    print("   Search: Lake bathymetric survey, Orange County")
    print()
    print("4. UF LAKEWATCH Program (lake morphometry data):")
    print("   https://lakewatch.ifas.ufl.edu/resources/lake-data/")
    print("   Orange County lake list → search for lake name")
    print()
    print("5. Upload manual survey CSV to: dem/data/lake_bathymetry_manual.csv")
    print("   Required columns: lon, lat, depth_m")
    print("   Then re-run: python3 dem/fetch_sjrwmd_bathymetry.py")
    print("═" * 68)

    # Check for manually uploaded CSV
    manual_csv = os.path.join(DATA_DIR, "lake_bathymetry_manual.csv")
    if os.path.exists(manual_csv):
        import pandas as pd
        import rasterio
        print(f"\nManual CSV found: {manual_csv}")
        try:
            df = pd.read_csv(manual_csv)
            pts = [(row["lon"], row["lat"], row["depth_m"])
                   for _, row in df.iterrows()
                   if all(c in df.columns for c in ["lon", "lat", "depth_m"])]
            if pts:
                print(f"  Loaded {len(pts)} manual depth points")
                dem_path  = os.path.join(DATA_DIR, "winter_garden_dem.tif")
                mask_path = (os.path.join(DATA_DIR, "lake_mask_s2.tif")
                             if os.path.exists(os.path.join(DATA_DIR, "lake_mask_s2.tif"))
                             else os.path.join(DATA_DIR, "lake_mask.tif"))
                lake_bed, dem_prof = interpolate_to_dem(pts, dem_path, mask_path)
                if lake_bed is not None:
                    dem_prof.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
                    with rasterio.open(OUT_TIF, "w", **dem_prof) as dst:
                        dst.write(lake_bed.astype(np.float32), 1)
                    print(f"  ✓ Saved manual bathymetry → {OUT_TIF}")
        except Exception as e:
            print(f"  Error loading manual CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch real lake bathymetric survey data from FWC, SJRWMD, FDEP")
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS_KM,
                        help="Search radius in km (default: 2.0)")
    args = parser.parse_args()
    main(args.radius_km)
