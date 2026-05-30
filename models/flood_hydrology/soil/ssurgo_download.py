"""
SSURGO Soil Data — Horton Infiltration & SCS Curve Number Parameters
======================================================================
Downloads USDA SSURGO soil data for the Winter Garden FL study area via
the Soil Data Access (SDA) REST API and derives the soil parameters needed
by the flood simulation engine.

Parameters extracted:
  - Hydrologic Soil Group (A/B/C/D) — HSG determines runoff potential
  - Saturated hydraulic conductivity Ksat [μm/s → mm/hr]
  - Initial infiltration rate f0 [mm/hr]  (estimated as 3–5× Ksat)
  - Decay constant k [hr⁻¹]              (Horton; texture-based look-up)
  - SCS Curve Number CN [dimensionless]  (HSG × land use)

Horton infiltration model parameters (stored per soil map unit):
  f(t) = fc + (f0 − fc) × exp(−k × t)
  where fc ≈ Ksat (final/saturated rate), f0 = initial dry rate

SCS Curve Number (NRCS TR-55 Table 2-2a, suburban residential):
  HSG A → CN 51 (low runoff potential, deep well-drained sands)
  HSG B → CN 68 (moderately low runoff potential)
  HSG C → CN 79 (moderately high, shallow or nearly impermeable layer)
  HSG D → CN 84 (high runoff potential, clay or high water table)
  Impervious/road surfaces → CN 98

Outputs (saved under soil/data/):
    ssurgo_mapunits.geojson         — soil map unit polygons with HSG + Ksat
    ssurgo_components.csv           — component-level soil properties
    cn_by_hsg.csv                   — CN lookup table (HSG × land use type)
    soil_parameters.json            — ready-to-use Horton + CN params per HSG

Usage:
    python3 soil/ssurgo_download.py
    python3 soil/ssurgo_download.py --lat 28.5652 --lon -81.5868 --radius_km 1.0
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROPERTY_LAT  = 28.521592   # 17801 Champagne Dr (28°31'17.73"N 81°39'25.13"W)
PROPERTY_LON  = -81.656981
RADIUS_KM     = 1.0

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

# Horton decay constants by USDA soil texture class [hr⁻¹]
# Higher k = faster saturation (clay soils); lower k = slower (sandy soils)
HORTON_K_BY_TEXTURE = {
    "sand":              1.5,
    "loamy sand":        1.8,
    "sandy loam":        2.0,
    "loam":              2.5,
    "silt loam":         3.0,
    "silt":              3.0,
    "sandy clay loam":   3.5,
    "clay loam":         4.0,
    "silty clay loam":   4.0,
    "sandy clay":        4.5,
    "silty clay":        5.0,
    "clay":              5.0,
    "default":           3.0,
}

# f0 multiplier over Ksat (ratio of initial to final infiltration rate)
# Based on Rawls et al. (1983) for initially dry conditions
F0_KSAT_RATIO_BY_TEXTURE = {
    "sand": 5.0, "loamy sand": 4.5, "sandy loam": 4.0,
    "loam": 3.5, "silt loam": 3.5, "silt": 3.0,
    "sandy clay loam": 3.0, "clay loam": 2.8, "silty clay loam": 2.5,
    "sandy clay": 2.5, "silty clay": 2.3, "clay": 2.0,
    "default": 3.0,
}

# SCS Curve Numbers — suburban residential (1/4 ac lots, ~25-30% impervious)
# Source: NRCS TR-55 Table 2-2a
CN_RESIDENTIAL_025_AC = {"A": 51, "B": 68, "C": 79, "D": 84}
CN_RESIDENTIAL_1_3_AC = {"A": 36, "B": 60, "C": 73, "D": 79}
CN_IMPERVIOUS         = {"A": 98, "B": 98, "C": 98, "D": 98}
CN_OPEN_SPACE_FAIR    = {"A": 49, "B": 69, "C": 79, "D": 84}
CN_WATER_BODY         = {"A":  0, "B":  0, "C":  0, "D":  0}  # no runoff from water


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def query_ssurgo_tabular(sql):
    """POST a SQL query to the USDA Soil Data Access API; return JSON result."""
    payload = {"query": sql, "format": "JSON+COLUMNNAME"}
    try:
        resp = requests.post(SDA_URL, data=payload, timeout=45)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"  SDA request failed: {exc}")
        return None


def query_ssurgo_spatial(bbox_wsen):
    """
    Query USDA SDA for soil map units intersecting the AOI bounding box.
    Returns component-level data including HSG and Ksat.
    """
    west, south, east, north = bbox_wsen

    # Step 1: Get map unit keys (mukey) for the AOI polygon via SDA spatial query
    aoi_wkt = (f"POLYGON(({west} {south}, {east} {south}, "
               f"{east} {north}, {west} {north}, {west} {south}))")

    sql_mukeys = f"SELECT DISTINCT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{aoi_wkt}')"

    print("  Querying SDA for map unit keys …")
    result = query_ssurgo_tabular(sql_mukeys)

    if result is None or not result.get("Table"):
        print("  ⚠ SDA spatial query returned no results; using Orange County defaults.")
        return None, None

    table_rows = result["Table"]
    # JSON+COLUMNNAME format: first row may be column header strings
    if table_rows and not str(table_rows[0][0]).isdigit():
        table_rows = table_rows[1:]
    mukeys = [str(row[0]) for row in table_rows]
    print(f"  Found {len(mukeys)} map unit(s)")

    if not mukeys:
        return None, None

    mukey_list = ",".join(f"'{k}'" for k in mukeys)

    # Step 2a: Get component properties (HSG) — simple query that SDA supports
    sql_components = (
        f"SELECT mu.mukey, mu.muname, co.cokey, co.compname, co.comppct_r, co.hydgrp "
        f"FROM mapunit mu INNER JOIN component co ON mu.mukey = co.mukey "
        f"WHERE mu.mukey IN ({mukey_list}) ORDER BY mu.mukey, co.comppct_r DESC"
    )
    print("  Querying component properties (HSG) …")
    comp_result = query_ssurgo_tabular(sql_components)
    if comp_result is None or not comp_result.get("Table"):
        return mukeys, None
    table = comp_result["Table"]
    cols  = [str(c) for c in table[0]]
    df    = pd.DataFrame(table[1:], columns=cols)
    print(f"  Retrieved {len(df)} component rows")

    # Step 2b: Get horizon Ksat from chorizon — use minimum over top 100cm
    # (minimum Ksat = hydraulically limiting layer; critical for Basinger/Cassia spodic horizons)
    # Columns: ksat_r [µm/s], sandtotal_r [%], claytotal_r [%]  (no "texture" column in chorizon)
    cokey_list = ",".join(str(int(float(k))) for k in df["cokey"].dropna().unique())
    if cokey_list:
        sql_horizon = (
            f"SELECT ch.cokey, ch.ksat_r, ch.sandtotal_r, ch.claytotal_r, ch.hzdept_r "
            f"FROM chorizon AS ch "
            f"WHERE ch.cokey IN ({cokey_list}) AND ch.hzdepb_r <= 100"
        )
        print("  Querying profile Ksat (chorizon, top 100 cm) …")
        hz_result = query_ssurgo_tabular(sql_horizon)
        if hz_result and hz_result.get("Table"):
            hz_table = hz_result["Table"]
            hz_df    = pd.DataFrame(hz_table[1:], columns=[str(c) for c in hz_table[0]])
            for col in ["ksat_r", "sandtotal_r", "claytotal_r"]:
                hz_df[col] = pd.to_numeric(hz_df[col], errors="coerce")
            # Min Ksat over profile = limiting horizon (Basinger spodic gives ~0.1 µm/s)
            ksat_min = hz_df.groupby("cokey")["ksat_r"].min().reset_index()
            ksat_min.columns = ["cokey", "ksat_r"]
            # Representative sand/clay from surface horizon (hzdept_r=0)
            surf = hz_df[hz_df["hzdept_r"].astype(float) == 0].drop_duplicates("cokey")
            surf = surf[["cokey", "sandtotal_r", "claytotal_r"]]
            hz_merged = ksat_min.merge(surf, on="cokey", how="left")
            df = df.merge(hz_merged, on="cokey", how="left")
            n_ksat = df["ksat_r"].notna().sum()
            print(f"  Ksat (min over top 100cm) retrieved for {n_ksat}/{len(df)} rows")
            if n_ksat > 0:
                print(f"  Ksat range: {df['ksat_r'].min():.2f}–{df['ksat_r'].max():.2f} µm/s")
        else:
            print("  ⚠ chorizon query failed — Ksat will use HSG-class defaults")

    return mukeys, df


def derive_horton_params(ksat_umps, texture_class):
    """
    Derive Horton infiltration parameters from Ksat and soil texture.

    Parameters
    ----------
    ksat_umps    : saturated hydraulic conductivity [μm/s]
    texture_class: USDA texture class string (lowercase)

    Returns
    -------
    dict with fc_mm_hr, f0_mm_hr, k_hr
    """
    # Convert Ksat: μm/s → mm/hr; handle None and NaN
    try:
        ksat_mm_hr = float(ksat_umps) * 3.6 if ksat_umps is not None else None
        if ksat_mm_hr is not None and (ksat_mm_hr != ksat_mm_hr):  # NaN check
            ksat_mm_hr = None
    except (TypeError, ValueError):
        ksat_mm_hr = None

    if ksat_mm_hr is None:
        ksat_mm_hr = 10.0   # HSG-class default when SSURGO data unavailable

    # Ensure realistic minimum (Ksat = 0 happens for water bodies)
    ksat_mm_hr = max(ksat_mm_hr, 0.1)

    tex = str(texture_class).lower().strip() if texture_class else "default"
    tex = next((k for k in HORTON_K_BY_TEXTURE if k in tex), "default")

    k_hr    = HORTON_K_BY_TEXTURE[tex]
    f0_mult = F0_KSAT_RATIO_BY_TEXTURE[tex]
    fc      = ksat_mm_hr
    f0      = fc * f0_mult

    return {"fc_mm_hr": round(fc, 2), "f0_mm_hr": round(f0, 2), "k_hr": round(k_hr, 2)}


def build_soil_parameters(components_df, mukeys):
    """Aggregate component properties to map-unit level and derive Horton + CN params."""
    params = {}

    for mukey in mukeys:
        sub = components_df[components_df["mukey"] == mukey]
        if sub.empty:
            continue

        # Dominant component (highest comppct_r)
        dom = sub.sort_values("comppct_r", ascending=False).iloc[0]

        hsg     = dom.get("hydgrp", "B")
        ksat    = dom.get("ksat_r", None)
        # Derive texture class from sand/clay % if available; fall back to "loam"
        sand_pct = dom.get("sandtotal_r", None)
        clay_pct = dom.get("claytotal_r", None)
        if sand_pct is not None and clay_pct is not None:
            try:
                s, c = float(sand_pct), float(clay_pct)
                if s >= 85:
                    texture = "sand"
                elif s >= 70 and c <= 15:
                    texture = "loamy sand"
                elif s >= 50 and c <= 20:
                    texture = "sandy loam"
                elif c >= 40:
                    texture = "clay"
                elif c >= 28:
                    texture = "clay loam"
                else:
                    texture = "loam"
            except (TypeError, ValueError):
                texture = "loam"
        else:
            texture = dom.get("texture", "loam")

        if hsg is None or str(hsg).strip() == "":
            hsg = "B"  # default
        hsg = str(hsg).strip()[0].upper()  # take first char, e.g. "A/D" → "A"
        if hsg not in "ABCD":
            hsg = "B"

        horton = derive_horton_params(ksat, texture)
        cn     = CN_RESIDENTIAL_025_AC.get(hsg, 79)

        params[mukey] = {
            "mukey":    mukey,
            "muname":   dom.get("muname", ""),
            "hsg":      hsg,
            "texture":  texture,
            **horton,
            "cn_residential_quarter_ac": cn,
            "cn_impervious":             98,
            "cn_open_space":             CN_OPEN_SPACE_FAIR.get(hsg, 79),
        }

    return params


def orange_county_defaults():
    """
    Default soil parameters for Orange County FL when SDA is unavailable.
    Dominant soils: Tavares-St. Lucie (sandy, HSG A/B) and Winder (sandy, HSG B/D).
    Source: USDA SSURGO Orange County survey area (FL617).
    """
    print("  Using Orange County FL default soil parameters (Tavares series — sandy, HSG B).")
    return {
        "dominant": {
            "mukey": "default",
            "muname": "Tavares fine sand (Orange County FL default)",
            "hsg": "B",
            "texture": "fine sand",
            "fc_mm_hr":  25.0,   # Ksat ~7 μm/s for Tavares fine sand
            "f0_mm_hr":  112.5,  # 4.5× Ksat (sandy soil, initially dry)
            "k_hr":       1.8,   # loamy sand decay constant
            "cn_residential_quarter_ac": 68,
            "cn_impervious":             98,
            "cn_open_space":             69,
        }
    }


def make_cn_lookup_table():
    """Standard CN table for all HSG groups and common land use types (TR-55)."""
    rows = []
    for hsg in ["A", "B", "C", "D"]:
        rows += [
            {"hsg": hsg, "land_use": "residential_quarter_ac",  "cn": CN_RESIDENTIAL_025_AC[hsg]},
            {"hsg": hsg, "land_use": "residential_third_ac",    "cn": CN_RESIDENTIAL_1_3_AC[hsg]},
            {"hsg": hsg, "land_use": "impervious_road_parking", "cn": CN_IMPERVIOUS[hsg]},
            {"hsg": hsg, "land_use": "open_space_fair_cover",   "cn": CN_OPEN_SPACE_FAIR[hsg]},
            {"hsg": hsg, "land_use": "water_body",              "cn": CN_WATER_BODY[hsg]},
        ]
    return pd.DataFrame(rows)


def download_mukey_polygons(mukeys, bbox_wsen, out_path):
    """
    Download SSURGO map unit polygon geometries via USDA WFS 1.0.0 (GML response).
    Parses GML to GeoJSON and filters to our AOI mukeys.
    Returns path to saved GeoJSON, or None if query fails.
    """
    import urllib.request
    import json as _json
    import xml.etree.ElementTree as ET

    west, south, east, north = bbox_wsen
    pad = 0.002
    bbox_str = f"{west-pad},{south-pad},{east+pad},{north+pad}"

    wfs_url = (
        "https://sdmdataaccess.sc.egov.usda.gov/Spatial/SDMWGS84Geographic.wfs"
        f"?SERVICE=WFS&VERSION=1.0.0&REQUEST=GetFeature"
        f"&TYPENAME=MapunitPolyExtended&BBOX={bbox_str}"
    )
    print("  Downloading SSURGO polygons via WFS 1.0.0 (GML) …")
    try:
        with urllib.request.urlopen(wfs_url, timeout=60) as resp:
            gml_bytes = resp.read()
    except Exception as e:
        print(f"  ⚠ WFS request failed: {e}; mukey_map.tif skipped")
        return None

    # Parse GML
    gml_str = gml_bytes.decode("utf-8", errors="ignore")
    try:
        root = ET.fromstring(gml_str)
    except ET.ParseError as e:
        print(f"  ⚠ GML parse error: {e}; mukey_map.tif skipped")
        return None

    NS = {
        "wfs": "http://www.opengis.net/wfs",
        "gml": "http://www.opengis.net/gml",
        "ms":  "http://mapserver.gis.umn.edu/mapserver",
    }

    def parse_coords(coord_str):
        """Parse GML coordinates string → list of [lon, lat] pairs."""
        pairs = []
        for token in coord_str.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                pairs.append([float(parts[0]), float(parts[1])])
        return pairs

    def gml_polygon_to_geojson(poly_el):
        """Convert a gml:Polygon element to a GeoJSON polygon geometry dict."""
        rings = []
        outer = poly_el.find("gml:outerBoundaryIs/gml:LinearRing/gml:coordinates", NS)
        if outer is None:
            return None
        rings.append(parse_coords(outer.text))
        for inner in poly_el.findall("gml:innerBoundaryIs/gml:LinearRing/gml:coordinates", NS):
            rings.append(parse_coords(inner.text))
        return {"type": "Polygon", "coordinates": rings}

    mukey_set = set(str(m) for m in mukeys)
    features = []

    for member in root.findall("gml:featureMember", NS):
        feat_el = None
        for child in member:
            feat_el = child
            break
        if feat_el is None:
            continue

        # Extract mukey from fid attribute (e.g., "mapunitpolyextended.323176")
        fid = feat_el.get("fid", "")
        mukey = fid.split(".")[-1] if "." in fid else ""
        # Also try child element ms:mukey
        mk_el = feat_el.find("ms:mukey", NS)
        if mk_el is not None and mk_el.text:
            mukey = mk_el.text.strip()

        if mukey not in mukey_set:
            continue

        # Parse geometry: MultiPolygon or Polygon
        coords_list = []
        multi = feat_el.find(".//gml:MultiPolygon", NS)
        if multi is not None:
            for pm in multi.findall("gml:polygonMember", NS):
                poly_el = pm.find("gml:Polygon", NS)
                if poly_el is not None:
                    g = gml_polygon_to_geojson(poly_el)
                    if g:
                        coords_list.append(g["coordinates"])
        else:
            poly_el = feat_el.find(".//gml:Polygon", NS)
            if poly_el is not None:
                g = gml_polygon_to_geojson(poly_el)
                if g:
                    coords_list.append(g["coordinates"])

        if not coords_list:
            continue

        if len(coords_list) == 1:
            geom = {"type": "Polygon", "coordinates": coords_list[0]}
        else:
            geom = {"type": "MultiPolygon", "coordinates": coords_list}

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"mukey": mukey},
        })

    if not features:
        print(f"  ⚠ WFS returned GML but no features matched our mukeys; mukey_map.tif skipped")
        return None

    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
          "features": features}
    with open(out_path, "w") as f:
        _json.dump(fc, f)
    print(f"  ✓ Saved {len(features)} polygon(s) → {os.path.basename(out_path)}")
    return out_path


def rasterize_mukey_map(geojson_path, params, dem_path, out_path):
    """
    Burn mukey integer IDs onto the DEM grid and save as mukey_map.tif.
    Also saves a mukey_legend.csv mapping integer IDs to mukey strings.
    """
    try:
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.crs import CRS
        import pyproj
        import json as _json
    except ImportError as e:
        print(f"  ⚠ Missing dependency for rasterization: {e}")
        return None

    if not os.path.exists(geojson_path):
        return None
    if not os.path.exists(dem_path):
        print(f"  ⚠ DEM not found at {dem_path}; skip rasterization")
        return None

    with open(geojson_path) as f:
        fc = _json.load(f)

    with rasterio.open(dem_path) as src:
        dem_shape    = src.shape
        dem_transform = src.transform
        dem_crs      = src.crs

    # Build mukey → integer ID mapping (sorted for reproducibility)
    all_mukeys_in_geojson = sorted({
        feat["properties"]["mukey"] for feat in fc["features"]
        if feat["properties"].get("mukey")
    })
    mukey_to_int = {mk: i + 1 for i, mk in enumerate(all_mukeys_in_geojson)}

    # Reproject geometries from EPSG:4326 → DEM CRS
    transformer = pyproj.Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)

    def reproject_geom(geom_dict):
        """Reproject a __geo_interface__ geometry dict."""
        from shapely.geometry import shape as shp_shape, mapping
        from shapely.ops import transform as shp_transform
        geom = shp_shape(geom_dict)
        return shp_transform(lambda x, y: transformer.transform(x, y), geom)

    shapes_for_rasterize = []
    for feat in fc["features"]:
        mukey = feat["properties"].get("mukey")
        if not mukey or mukey not in mukey_to_int:
            continue
        try:
            reprojected = reproject_geom(feat["geometry"])
            shapes_for_rasterize.append((reprojected.__geo_interface__, mukey_to_int[mukey]))
        except Exception:
            continue

    if not shapes_for_rasterize:
        print("  ⚠ No geometries to rasterize")
        return None

    mukey_arr = rio_rasterize(
        shapes_for_rasterize,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=0,
        dtype=np.int32,
    )

    with rasterio.open(out_path, "w",
                       driver="GTiff", height=dem_shape[0], width=dem_shape[1],
                       count=1, dtype=np.int32, crs=dem_crs,
                       transform=dem_transform, compress="lzw") as dst:
        dst.write(mukey_arr, 1)

    # Save legend CSV
    import pandas as pd
    legend = pd.DataFrame([
        {"mukey_int": v, "mukey": k,
         "muname": params.get(k, {}).get("muname", ""),
         "hsg":    params.get(k, {}).get("hsg", ""),
         "fc_mm_hr": params.get(k, {}).get("fc_mm_hr", "")}
        for k, v in mukey_to_int.items()
    ])
    legend_path = out_path.replace(".tif", "_legend.csv")
    legend.to_csv(legend_path, index=False)

    n_cells = int((mukey_arr > 0).sum())
    print(f"  ✓ Rasterized {len(shapes_for_rasterize)} polygon(s); "
          f"{n_cells:,} labelled cells → {os.path.basename(out_path)}")
    return out_path


def main(lat=PROPERTY_LAT, lon=PROPERTY_LON, radius_km=RADIUS_KM):
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"Querying SSURGO for ({lat}, {lon}), radius {radius_km} km")
    print(f"  Bounding box: {[round(x,5) for x in bbox]}")

    mukeys, components_df = query_ssurgo_spatial(bbox)

    if components_df is not None and not components_df.empty:
        components_df.to_csv(os.path.join(DATA_DIR, "ssurgo_components.csv"), index=False)
        print(f"  Saved component table → ssurgo_components.csv")
        params = build_soil_parameters(components_df, mukeys)
    else:
        params = orange_county_defaults()

    # Save parameters
    params_path = os.path.join(DATA_DIR, "soil_parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved soil parameters → {params_path}")

    # CN lookup table
    cn_df = make_cn_lookup_table()
    cn_path = os.path.join(DATA_DIR, "cn_by_hsg.csv")
    cn_df.to_csv(cn_path, index=False)
    print(f"Saved CN lookup table → {cn_path}")

    # Print summary
    print("\n── Soil Parameters Summary ──────────────────────────────────")
    for mu_id, p in params.items():
        print(f"  Map unit {mu_id}: {p.get('muname','')[:50]}")
        print(f"    HSG: {p['hsg']}  |  Texture: {p.get('texture','')}")
        print(f"    Horton: f0={p['f0_mm_hr']} mm/hr, fc={p['fc_mm_hr']} mm/hr, k={p['k_hr']} hr⁻¹")
        print(f"    CN (residential ¼ac): {p['cn_residential_quarter_ac']}")

    print("\n── Curve Number Table (TR-55) ───────────────────────────────")
    print(cn_df.pivot(index="land_use", columns="hsg", values="cn").to_string())

    # Download polygon geometries and rasterize to DEM grid
    if mukeys:
        geojson_path = os.path.join(DATA_DIR, "ssurgo_mapunits.geojson")
        mukey_tif    = os.path.join(DATA_DIR, "mukey_map.tif")
        dem_path     = os.path.join(BASE_DIR, "..", "dem", "data", "winter_garden_dem.tif")
        dem_path     = os.path.normpath(dem_path)
        print("\n── Soil Spatial Map ─────────────────────────────────────────")
        gj = download_mukey_polygons(mukeys, bbox, geojson_path)
        if gj and os.path.exists(dem_path):
            rasterize_mukey_map(gj, params, dem_path, mukey_tif)

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SSURGO soil data and derive Horton + CN parameters")
    parser.add_argument("--lat",       type=float, default=PROPERTY_LAT)
    parser.add_argument("--lon",       type=float, default=PROPERTY_LON)
    parser.add_argument("--radius_km", type=float, default=RADIUS_KM)
    args = parser.parse_args()
    main(args.lat, args.lon, args.radius_km)
