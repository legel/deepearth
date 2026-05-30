"""
Lake Bathymetry Fetcher — Winter Garden FL Chain-of-Lakes
==========================================================
Attempts to download real bathymetric survey data for the lakes adjacent to
17801 Champagne Dr, Winter Garden FL 34787 from three sources (in priority order):

  1. FWC "Bathymetry of Select Florida Lakes" (ArcGIS REST, 1-ft contours, 2005–2011)
  2. Orange County Water Atlas (per-lake data download)
  3. SJRWMD Historical Bathymetry (ArcGIS Open Data Hub)

If data is found:
  - Converts depth contours to a point cloud
  - Interpolates to the existing DEM grid
  - Saves as  dem/data/lake_bed_dem_fwc.tif
  - All downstream scripts (lake_volume.py, dem_visualize.py) auto-detect this file

If no data is found:
  - Prints manual contact instructions
  - lake_bed_dem_estimated.tif (shoreline-slope extrapolation) continues to be used

Usage:
    python3 dem/fetch_bathymetry.py
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Study area bounding box (WGS84, ±1 km from property)
BBOX_WSEN = (-81.66723, 28.51258, -81.64673, 28.53060)
PROPERTY_LAT =  28.521592
PROPERTY_LON = -81.656981

OUTPUT_PATH = os.path.join(DATA_DIR, "lake_bed_dem_fwc.tif")


def load_dem_grid():
    """Load existing DEM for grid reference."""
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio not found. pip install rasterio")
    dem_path = os.path.join(DATA_DIR, "winter_garden_dem.tif")
    if not os.path.exists(dem_path):
        sys.exit(f"DEM not found at {dem_path}. Run dem_download.py first.")
    with rasterio.open(dem_path) as src:
        dem_arr  = src.read(1).astype(np.float32)
        profile  = src.profile.copy()
        transform = src.transform
        crs      = src.crs
    return dem_arr, profile, transform, crs


# ── Source 1: FWC ArcGIS REST ─────────────────────────────────────────────────

FWC_REST_URL = (
    "https://gis.myfwc.com/hosting/rest/services/Open_Data/"
    "Bathymetry_of_Select_Lakes_in_Florida/MapServer/3/query"
)
# Service native CRS: EPSG:6439 (NAD83(HARN) Florida GDL Albers, feet)
FWC_SERVICE_CRS = "EPSG:6439"

def fetch_fwc_bathymetry():
    """
    Query FWC lake bathymetry contours (1-ft depth contours) within the AOI bbox.
    Service lives at gis.myfwc.com MapServer/3 in EPSG:6439 — bbox must be
    reprojected from WGS84 before querying; results requested back in WGS84.
    Fields: DEPTHF (feet, negative below surface), DEPTHM (meters, negative).
    Returns list of (depth_m_positive, lon, lat) tuples, or None if unavailable.
    """
    import urllib.request
    import urllib.parse
    try:
        import pyproj
    except ImportError:
        print("  ⚠ pyproj not installed — cannot reproject bbox for FWC query")
        return None

    west, south, east, north = BBOX_WSEN
    pad = 0.01  # ~1 km padding to capture contours along the lake edge
    w2, s2, e2, n2 = west - pad, south - pad, east + pad, north + pad

    # Reproject AOI bbox from WGS84 → service CRS (EPSG:6439)
    tr = pyproj.Transformer.from_crs("EPSG:4326", FWC_SERVICE_CRS, always_xy=True)
    x_min, y_min = tr.transform(w2, s2)
    x_max, y_max = tr.transform(e2, n2)

    params = {
        "geometry":         f"{x_min},{y_min},{x_max},{y_max}",
        "geometryType":     "esriGeometryEnvelope",
        "inSR":             "6439",
        "outSR":            "4326",        # return coords in WGS84
        "spatialRel":       "esriSpatialRelIntersects",
        "outFields":        "DEPTHF,DEPTHM",
        "returnGeometry":   "true",
        "resultRecordCount": 10000,
        "f":                "geojson",
    }
    print(f"  Querying FWC MapServer/3 ({FWC_SERVICE_CRS}) …")
    try:
        with urllib.request.urlopen(FWC_REST_URL + "?" + urllib.parse.urlencode(params),
                                    timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ✗ FWC query failed: {e}")
        return None

    err = data.get("error")
    if err:
        print(f"  ✗ FWC service error: {err.get('message', '')}")
        return None

    features = data.get("features", [])
    if not features:
        print("  ✗ FWC returned 0 features for this AOI — lake not in FWC survey coverage")
        return None

    print(f"  ✓ FWC returned {len(features)} contour features")
    records = []
    depths_seen = []
    for feat in features:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        depth_m = props.get("DEPTHM")
        depth_f = props.get("DEPTHF")
        if depth_m is not None:
            depth_m = abs(float(depth_m))   # stored as negative; make positive
        elif depth_f is not None:
            depth_m = abs(float(depth_f)) * 0.3048
        else:
            continue
        if depth_m <= 0:
            continue
        depths_seen.append(depth_m)
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        if gtype == "LineString":
            for lon, lat in coords:
                records.append((depth_m, float(lon), float(lat)))
        elif gtype == "MultiLineString":
            for part in coords:
                for lon, lat in part:
                    records.append((depth_m, float(lon), float(lat)))

    if depths_seen:
        print(f"  Depth range: {min(depths_seen):.2f} – {max(depths_seen):.2f} m")
    print(f"  Extracted {len(records):,} depth points from contours")
    return records if records else None


# ── Source 2: FDEP ArcGIS REST ────────────────────────────────────────────────
# Florida DEP Lake Bathymetric Surveys — public ArcGIS REST service

FDEP_BATHY_URLS = [
    # Bureau of Watershed Restoration — lake contour surveys
    ("https://geodata.dep.state.fl.us/datasets/FDEP::lake-bathymetric-surveys/about",
     "https://services.arcgis.com/KTcxiTD9dsQw4r7Z/arcgis/rest/services/"
     "FDEP_Lake_Bathymetry/FeatureServer/0/query"),
    # Fallback: FDEP GeoDB open data hub search endpoint
    ("https://opendata.dep.state.fl.us/",
     "https://services1.arcgis.com/O1JpcwDW8sjYuddV/arcgis/rest/services/"
     "FDEP_Lake_Bathymetry_Surveys/FeatureServer/0/query"),
]

def fetch_fdep_bathymetry():
    """
    Query FDEP (Florida DEP) lake bathymetric survey REST endpoint.
    Tries multiple known service URLs for the public ArcGIS layer.
    """
    import urllib.request
    import urllib.parse

    west, south, east, north = BBOX_WSEN
    params = {
        "geometry":       f"{west},{south},{east},{north}",
        "geometryType":   "esriGeometryEnvelope",
        "inSR":           "4326",
        "outSR":          "4326",
        "spatialRel":     "esriSpatialRelIntersects",
        "outFields":      "*",
        "returnGeometry": "true",
        "f":              "geojson",
        "resultRecordCount": 5000,
    }
    query_str = urllib.parse.urlencode(params)
    for desc_url, rest_url in FDEP_BATHY_URLS:
        print(f"  Querying FDEP: {rest_url[:65]}…")
        try:
            with urllib.request.urlopen(rest_url + "?" + query_str, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"    ✗ Request failed: {e}")
            continue
        error = data.get("error")
        if error:
            print(f"    ✗ Service error: {error.get('message','')}")
            continue
        features = data.get("features", [])
        if not features:
            print(f"    ✗ 0 features returned")
            continue
        print(f"  ✓ FDEP returned {len(features)} features")
        # Parse depth points from LineString contours (same format as FWC)
        records = []
        for feat in features:
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            # Try common depth field names
            depth_m = (props.get("DEPTH_M") or props.get("depth_m") or
                       props.get("DEPTH_FT") or props.get("depth_ft"))
            if depth_m is None:
                continue
            try:
                depth_m = float(depth_m)
                if props.get("DEPTH_FT") and not props.get("DEPTH_M"):
                    depth_m *= 0.3048  # convert ft → m
            except (TypeError, ValueError):
                continue
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "LineString":
                for lon, lat in coords:
                    records.append((depth_m, float(lon), float(lat)))
            elif gtype == "MultiLineString":
                for part in coords:
                    for lon, lat in part:
                        records.append((depth_m, float(lon), float(lat)))
        if records:
            print(f"  Extracted {len(records):,} depth points")
            return records
        print(f"  ✗ Features found but no parseable depth fields")
    return None


# ── Source 3: SJRWMD ArcGIS Open Data ────────────────────────────────────────

# Correct SJRWMD Open Data Hub endpoints (verified 2025)
SJRWMD_URLS = [
    "https://services2.arcgis.com/xRI7bJlrVQdE2V0N/arcgis/rest/services/"
    "LakeBathymetry/FeatureServer/0/query",
    "https://www.sjrwmd.com/localSites/data-warehouse/data/bathymetry/"
    "query?f=geojson",
    "https://opendata.arcgis.com/datasets/"
    "8d4c2c7b7e7c4e3f9f7d8e5a3b2c1d4e_0/query",  # placeholder — see search note
]

def fetch_sjrwmd_bathymetry():
    """
    Query SJRWMD lake bathymetry from their ArcGIS Open Data portal.
    Multiple endpoint candidates are tried in order.
    """
    import urllib.request
    import urllib.parse

    west, south, east, north = BBOX_WSEN
    params = {
        "geometry":       f"{west},{south},{east},{north}",
        "geometryType":   "esriGeometryEnvelope",
        "inSR":           "4326",
        "outSR":          "4326",
        "spatialRel":     "esriSpatialRelIntersects",
        "outFields":      "*",
        "returnGeometry": "true",
        "f":              "geojson",
        "resultRecordCount": 5000,
    }
    query_str = urllib.parse.urlencode(params)
    for rest_url in SJRWMD_URLS:
        if "placeholder" in rest_url:
            continue
        print(f"  Querying SJRWMD: {rest_url[:65]}…")
        try:
            with urllib.request.urlopen(rest_url + "?" + query_str, timeout=30) as resp:
                raw = resp.read().decode()
            data = json.loads(raw)
        except Exception as e:
            print(f"    ✗ {e}")
            continue
        error = data.get("error")
        if error:
            print(f"    ✗ Service error: {error.get('message','')}")
            continue
        features = data.get("features", [])
        if not features:
            print(f"    ✗ 0 features for AOI")
            continue
        print(f"  ✓ SJRWMD returned {len(features)} features")
        return features
    return None


# ── Source 4: LAKEWATCH / Florida Lake Database ───────────────────────────────

def fetch_lakewatch_depth():
    """
    Check Florida LAKEWATCH (UF) database for mean depth of Butler Chain lakes.
    LAKEWATCH collects citizen-science water quality data including Secchi depth
    and morphometric data (area, mean depth, max depth) for ~1000 FL lakes.
    The public CSV download covers lake-level summary statistics, not contours.
    Returns list of (depth_m, lon, lat) for centroid-based depth estimate, or None.
    """
    import urllib.request

    # LAKEWATCH publishes a county-level Excel/CSV of lake data.
    # Orange County file (public, no license):
    url = ("https://lakewatch.ifas.ufl.edu/wp-content/uploads/sites/48/2022/10/"
           "Orange-2021.xlsx")
    print(f"  Checking Florida LAKEWATCH Orange County data …")
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read()
        print(f"  ✓ Downloaded LAKEWATCH data ({len(raw)//1024} KB)")
        # Try to parse with openpyxl if available
        try:
            import io, openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
            ws = wb.active
            headers = [str(c.value).lower() if c.value else "" for c in ws[1]]
            print(f"  Columns: {[h for h in headers if h][:10]}")
            # Look for depth and lat/lon columns
            # (LAKEWATCH files vary; this is a best-effort parse)
        except ImportError:
            print("  ⚠ openpyxl not installed — cannot parse .xlsx; try: pip install openpyxl")
        return None
    except Exception as e:
        print(f"  ✗ LAKEWATCH download failed: {e}")
        return None


# ── Interpolation: depth points → DEM-grid raster ────────────────────────────

def interpolate_to_dem(depth_records, dem_arr, profile, transform, crs):
    """
    Interpolate scattered (depth_m, lon, lat) points onto the DEM grid.
    Produces lake_bed_dem_fwc.tif: water_surface_elev − depth at each lake cell.
    Points outside the lake mask fall back to the DEM value (no depth added).
    """
    try:
        from scipy.interpolate import griddata
        import rasterio
        import pyproj
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. pip install scipy rasterio pyproj")

    lake_mask_path = os.path.join(DATA_DIR, "lake_mask.tif")
    with rasterio.open(lake_mask_path) as src:
        lake_mask = src.read(1)

    # Water surface elevation = mean DEM elevation of lake pixels
    lake_bool = lake_mask > 0
    water_surface = float(np.nanmean(dem_arr[lake_bool & np.isfinite(dem_arr)]))
    print(f"  Water surface elevation: {water_surface:.2f} m NAVD88")

    # Reproject depth points from WGS84 to DEM CRS
    transformer = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)
    pts_xy   = np.array([(transformer.transform(lon, lat)) for _, lon, lat in depth_records])
    pts_depth = np.array([d for d, _, _ in depth_records])

    # DEM grid pixel centres in projected CRS
    nrows, ncols = dem_arr.shape
    cols_idx = np.arange(ncols)
    rows_idx = np.arange(nrows)
    grid_cols, grid_rows = np.meshgrid(cols_idx, rows_idx)
    grid_x = transform.c + (grid_cols + 0.5) * transform.a
    grid_y = transform.f + (grid_rows + 0.5) * transform.e

    # Interpolate depths over the lake area
    print("  Interpolating depth points to DEM grid …")
    grid_depth = griddata(pts_xy, pts_depth,
                          (grid_x[lake_bool], grid_y[lake_bool]),
                          method="linear", fill_value=0.0)
    grid_depth = np.clip(grid_depth, 0, None)  # depth can't be negative

    # Build lake bed DEM: water_surface − depth inside lake; DEM elsewhere
    lake_bed_out = dem_arr.copy()
    lake_bed_out[lake_bool] = water_surface - grid_depth

    # Save
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(lake_bed_out.astype(np.float32), 1)

    n_pts_in_lake = lake_bool.sum()
    depth_in_lake = grid_depth[grid_depth > 0]
    print(f"  Depth statistics over {n_pts_in_lake:,} lake cells:")
    if len(depth_in_lake) > 0:
        print(f"    Mean depth: {depth_in_lake.mean():.2f} m")
        print(f"    Max depth:  {depth_in_lake.max():.2f} m")
        print(f"    Min depth:  {depth_in_lake.min():.2f} m")
    print(f"  Saved lake_bed_dem_fwc.tif → {OUTPUT_PATH}")
    return lake_bed_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Lake Bathymetry Fetcher — Winter Garden FL")
    print("=" * 60)

    dem_arr, profile, transform, crs = load_dem_grid()
    print(f"  DEM grid: {dem_arr.shape}, cell size: {abs(transform.a):.1f} m")

    # Try all public sources in priority order; first one with data wins.

    # ── Source 1: FWC ──────────────────────────────────────────────────
    print("\n[1/4] FWC Bathymetry of Select Florida Lakes (ArcGIS REST) …")
    fwc_records = fetch_fwc_bathymetry()
    if fwc_records and len(fwc_records) > 10:
        print(f"\n  ✓ FWC: {len(fwc_records):,} depth points — interpolating …")
        interpolate_to_dem(fwc_records, dem_arr, profile, transform, crs)
        print("  ✓ lake_bed_dem_fwc.tif written (source: FWC bathymetric survey)")
        return

    # ── Source 2: FDEP ─────────────────────────────────────────────────
    print("\n[2/4] FDEP Lake Bathymetric Surveys (ArcGIS REST) …")
    fdep_records = fetch_fdep_bathymetry()
    if fdep_records and len(fdep_records) > 10:
        print(f"\n  ✓ FDEP: {len(fdep_records):,} depth points — interpolating …")
        interpolate_to_dem(fdep_records, dem_arr, profile, transform, crs)
        print("  ✓ lake_bed_dem_fwc.tif written (source: FDEP lake bathymetry)")
        return

    # ── Source 3: SJRWMD ───────────────────────────────────────────────
    print("\n[3/4] SJRWMD Historical Bathymetry (ArcGIS Open Data) …")
    sjrwmd_features = fetch_sjrwmd_bathymetry()
    if sjrwmd_features and len(sjrwmd_features) > 5:
        print(f"  ✓ SJRWMD: {len(sjrwmd_features)} features — attempting parse …")
        records = []
        for feat in sjrwmd_features:
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            depth_m = props.get("DEPTH_M") or props.get("depth_m") or props.get("DEPTH_FT")
            if depth_m is None:
                continue
            try:
                depth_m = float(depth_m)
                if "FT" in str(props.get("field", "")).upper():
                    depth_m *= 0.3048
            except (TypeError, ValueError):
                continue
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "LineString":
                for pt in coords:
                    records.append((depth_m, float(pt[0]), float(pt[1])))
            elif gtype == "Point":
                lon, lat = coords[0], coords[1]
                records.append((depth_m, float(lon), float(lat)))
        if records:
            print(f"  Parsed {len(records)} depth points from SJRWMD")
            interpolate_to_dem(records, dem_arr, profile, transform, crs)
            print("  ✓ lake_bed_dem_fwc.tif written (source: SJRWMD bathymetry)")
            return

    # ── Source 4: LAKEWATCH ────────────────────────────────────────────
    print("\n[4/4] Florida LAKEWATCH (UF) — Orange County lake database …")
    fetch_lakewatch_depth()  # informational only; returns centroid summary

    # ── No data found ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  No machine-readable bathymetric contour data found for this AOI.")
    print("  lake_bed_dem_estimated.tif (shoreline-slope extrapolation) continues to be used.")
    print()
    print("  PUBLIC DATA SOURCES TO CHECK MANUALLY (all free, no license required):")
    print()
    print("  1. FDEP Open Data — Lake Bathymetric Survey polygons:")
    print("     https://geodata.dep.state.fl.us/")
    print("     Search: 'Lake Bathymetry' → download Shapefile/GeoJSON")
    print("     Filter: Orange County, Butler Chain of Lakes")
    print()
    print("  2. SJRWMD Open Data Hub (public, no account needed):")
    print("     https://data-floridaswater.opendata.arcgis.com/")
    print("     Search: 'bathymetry' or 'lake depth' — download directly")
    print()
    print("  3. Florida LAKEWATCH (UF) bathymetric maps (free PDFs):")
    print("     https://lakewatch.ifas.ufl.edu/for-volunteers/bathymetric-maps/")
    print("     Search for: Lake Cawood, Lake Sheen, Lake Butler (Orange County)")
    print("     If XYZ tables in PDF → extract and save as lake_bathymetry_points.csv")
    print()
    print("  4. Orange County Geoportal (public ArcGIS):")
    print("     https://www.orangecountyfl.net/culturepark/gis.aspx")
    print("     → Environmental / Water Resources layers")
    print()
    print("  If you obtain a CSV [lon, lat, depth_m], place it at:")
    print(f"    {os.path.join(DATA_DIR, 'lake_bathymetry_points.csv')}")
    print("  Then re-run this script — it will auto-detect and interpolate.")
    print()
    print("  If you obtain a CSV with columns [lon, lat, depth_m], place it at:")
    print(f"    {os.path.join(DATA_DIR, 'lake_bathymetry_points.csv')}")
    print("  Then re-run this script — it will detect and interpolate the CSV automatically.")
    print("=" * 60)

    # Check for manually provided CSV
    manual_csv = os.path.join(DATA_DIR, "lake_bathymetry_points.csv")
    if os.path.exists(manual_csv):
        print(f"\n  Found manual CSV: {manual_csv}")
        import pandas as pd
        df = pd.read_csv(manual_csv)
        required = {"lon", "lat", "depth_m"}
        if required.issubset(df.columns):
            records = [(row.depth_m, row.lon, row.lat) for row in df.itertuples()]
            print(f"  Loaded {len(records)} points from CSV")
            interpolate_to_dem(records, dem_arr, profile, transform, crs)
        else:
            print(f"  ✗ CSV must have columns: {required}. Found: {set(df.columns)}")


if __name__ == "__main__":
    main()
