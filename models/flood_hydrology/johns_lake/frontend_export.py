"""
Digital Twin Frontend Data Export
===================================
Converts flood hydrology pipeline outputs to the format expected by the
digitaltwin frontend (https://github.com/legel/digitaltwin).

The frontend expects:
  /data/sites/{site_id}/site-bounds.json          — site center + scale
  /data/sites/{site_id}/flood-data.geojson        — flood zones (NEW)
  /data/sites/{site_id}/plantable-area-data.geojson — original planting data

Flood zones GeoJSON structure:
  FeatureCollection of Polygon features, each with:
    properties.id             — unique ID
    properties.name           — e.g., "Flood Zone 100-yr: 1–2 m depth"
    properties.description    — M-metric string for frontend parser
    properties.flood_depth_m  — max flood depth [m]
    properties.flood_risk_pct — annual exceedance probability [%]
    properties.scenario       — "flash_1hr_100yr" or "sustained_12hr_100yr"
    geometry                  — Polygon in WGS84 [lon, lat]

Annual exceedance probability:
  100-yr return period → 1%/year
  Areas outside 100-yr zone but HAND < 1m → 2–5%/year (estimated)
  HAND 1–3m, no flooding in 100-yr → 0.5–2%/year (estimated)

Usage:
    python3 frontend_export.py
    python3 frontend_export.py --output_dir /path/to/digitaltwin/data/sites/demo-site
"""

import os
import sys
import json
import uuid
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.join(BASE_DIR, "simulation", "outputs")
DEM_DIR  = os.path.join(BASE_DIR, "dem", "data")

SITE_ID  = "demo-site"
SITE_CENTER_LON = -81.6570725
SITE_CENTER_LAT =  28.5217321


def load_raster(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return arr, profile, transform, crs


def raster_to_wgs84_geojson(arr, transform, crs, threshold, label, max_polygons=50):
    """
    Threshold a raster, vectorize, reproject polygons to WGS84, return features list.
    max_polygons caps the number of returned features to keep the file size reasonable.
    """
    import rasterio.features
    from shapely.geometry import shape as shp_shape, mapping
    from shapely.ops import transform as shp_transform
    import pyproj

    mask = (arr > threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []

    shapes = list(rasterio.features.shapes(
        mask, mask=mask, transform=transform
    ))

    transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)

    features = []
    for geom_dict, val in sorted(shapes, key=lambda s: -shp_shape(s[0]).area)[:max_polygons]:
        if val != 1:
            continue
        geom = shp_shape(geom_dict)
        geom_wgs84 = shp_transform(transformer.transform, geom)
        area_ha = geom.area / 1e4
        features.append((geom_wgs84, area_ha))
    return features


def build_flood_description(depth_range, risk_pct_range, scenario_label):
    """
    Build the M-metric description string matching digitaltwin frontend format.
    M9 = Flood Risk.
    """
    m9_low, m9_high = risk_pct_range
    return (
        f"Flood Zone — {scenario_label} | "
        f"M9: Flood Risk = {m9_low}-{m9_high}%/year | "
        f"Depth range: {depth_range[0]:.1f}-{depth_range[1]:.1f} m | "
        f"Source: 2D local-inertia simulation (LISFLOOD-FP method)"
    )


def export_scenario(sim_dir, scenario_name, risk_annual_pct, output_features):
    """
    Load peak flood depth for a scenario and vectorize into depth-zone GeoJSON features.
    """
    depth_path = os.path.join(sim_dir, f"inundation_depth_{scenario_name}.tif")
    if not os.path.exists(depth_path):
        print(f"  [skip] {depth_path} not found")
        return

    arr, profile, transform, crs = load_raster(depth_path)
    arr = np.where(arr < 0, 0, arr)
    cell_m2 = abs(transform.a * transform.e)
    print(f"  {scenario_name}: max depth {arr.max():.2f}m, "
          f"flooded area (>5cm) {(arr>0.05).sum() * cell_m2/1e4:.1f} ha")

    # Depth zones: shallow, moderate, deep
    depth_zones = [
        (0.05, 0.5,  (1, 2),   "Shallow Flood Zone (5–50 cm)"),
        (0.5,  1.5,  (1, 1),   "Moderate Flood Zone (0.5–1.5 m)"),
        (1.5,  10.0, (1, 1),   "Deep Flood Zone (> 1.5 m)"),
    ]
    scenario_label = "Flash 1-hr 100-yr" if "1hr" in scenario_name else "Sustained 12-hr 100-yr"

    for d_lo, d_hi, risk_range, zone_name in depth_zones:
        zone_arr = np.where((arr >= d_lo) & (arr < d_hi), 1, 0).astype(np.uint8)
        if zone_arr.sum() == 0:
            continue

        import rasterio.features
        from shapely.geometry import shape as shp_shape
        from shapely.ops import transform as shp_transform, unary_union
        import pyproj

        shapes_list = list(rasterio.features.shapes(
            zone_arr, mask=zone_arr, transform=transform
        ))

        transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)

        polys = []
        for geom_dict, val in shapes_list:
            if val != 1:
                continue
            geom = shp_shape(geom_dict)
            if geom.area < 100:  # < 100 m² = noise
                continue
            polys.append(geom)

        if not polys:
            continue

        # Merge small polygons into one feature to keep file size manageable
        merged = unary_union(polys)
        # Simplify in projected CRS before reprojecting (~5m = 2 pixels, good for web display)
        merged = merged.simplify(5.0, preserve_topology=True)
        merged_wgs84 = shp_transform(transformer.transform, merged)
        area_ha = merged.area / 1e4

        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": merged_wgs84.__geo_interface__,
            "properties": {
                "id":             str(uuid.uuid4()),
                "name":           f"{zone_name} — {scenario_label}",
                "scenario":       scenario_name,
                "flood_depth_m":  d_hi,
                "flood_risk_pct": risk_annual_pct,
                "area_ha":        round(area_ha, 2),
                "description":    build_flood_description(
                    (d_lo, d_hi), risk_range, scenario_label
                ),
            }
        }
        output_features.append(feature)
        print(f"    {zone_name}: {area_ha:.1f} ha → added")


def export_hand_risk_zones(dem_dir, output_features):
    """
    Use HAND raster to add estimated flood risk zones for areas outside
    the 100-yr simulation extent (pluvial + fluvial background risk).
    """
    hand_path = os.path.join(dem_dir, "hand.tif")
    lake_path  = os.path.join(dem_dir, "lake_mask.tif")
    if not os.path.exists(hand_path):
        print("  [skip] HAND raster not found")
        return

    hand, profile, transform, crs = load_raster(hand_path)
    hand = np.where(np.isfinite(hand), hand, 999.0)

    lake = np.zeros_like(hand, dtype=np.uint8)
    if os.path.exists(lake_path):
        lake_arr, *_ = load_raster(lake_path)
        lake = (lake_arr > 0).astype(np.uint8)

    # Background risk zones from HAND (areas not explicitly in 100-yr simulation)
    hand_zones = [
        (0.0, 0.5, "Very High Flood Risk Zone (HAND < 0.5 m)", "5-10"),
        (0.5, 1.5, "High Flood Risk Zone (HAND 0.5–1.5 m)",    "2-5"),
        (1.5, 3.0, "Moderate Flood Risk Zone (HAND 1.5–3 m)",  "1-2"),
    ]

    import rasterio.features
    from shapely.geometry import shape as shp_shape
    from shapely.ops import transform as shp_transform, unary_union
    import pyproj

    transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)

    for h_lo, h_hi, zone_name, risk_str in hand_zones:
        zone_arr = np.where(
            (hand >= h_lo) & (hand < h_hi) & (lake == 0), 1, 0
        ).astype(np.uint8)
        if zone_arr.sum() == 0:
            continue

        shapes_list = list(rasterio.features.shapes(
            zone_arr, mask=zone_arr, transform=transform
        ))
        polys = [shp_shape(g) for g, v in shapes_list if v == 1 and shp_shape(g).area > 200]
        if not polys:
            continue

        merged = unary_union(polys)
        # Simplify in projected CRS before reprojecting (~5m tolerance for web display)
        merged = merged.simplify(5.0, preserve_topology=True)
        merged_wgs84 = shp_transform(transformer.transform, merged)
        area_ha = merged.area / 1e4

        r_lo, r_hi = risk_str.split("-")
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": merged_wgs84.__geo_interface__,
            "properties": {
                "id":             str(uuid.uuid4()),
                "name":           zone_name,
                "scenario":       "hand_background_risk",
                "flood_depth_m":  None,
                "flood_risk_pct": float(r_lo),
                "area_ha":        round(area_ha, 2),
                "description": (
                    f"{zone_name} | "
                    f"M9: Flood Risk = {risk_str}%/year | "
                    f"Source: HAND (Height Above Nearest Drainage) from USGS 3DEP LiDAR"
                ),
            }
        }
        output_features.append(feature)
        print(f"    {zone_name}: {area_ha:.1f} ha → added")


def write_site_bounds(output_dir):
    bounds = {
        "site": SITE_ID,
        "center": {
            "longitude": SITE_CENTER_LON,
            "latitude":  SITE_CENTER_LAT,
        },
        "scale_correction_factor": 0.7,
    }
    path = os.path.join(output_dir, "site-bounds.json")
    with open(path, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"  Wrote {path}")


def main(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "frontend_output", "data", "sites", SITE_ID)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("Flood Digital Twin — Frontend Data Export")
    print(f"Output directory: {output_dir}")
    print("=" * 65)

    features = []

    print("\n[1/3] Exporting 100-yr simulation flood zones …")
    for scenario, risk_pct in [
        ("flash_1hr_100yr",       1.0),
        ("sustained_12hr_100yr",  1.0),
    ]:
        export_scenario(SIM_DIR, scenario, risk_pct, features)

    print("\n[2/3] Exporting HAND-based background risk zones …")
    export_hand_risk_zones(DEM_DIR, features)

    print(f"\n[3/3] Writing GeoJSON ({len(features)} features) …")
    fc = {
        "type": "FeatureCollection",
        "name": "Winter Garden Flood Hazard Zones",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "metadata": {
            "site":          SITE_ID,
            "center_lon":    SITE_CENTER_LON,
            "center_lat":    SITE_CENTER_LAT,
            "property":      "17801 Champagne Dr, Winter Garden FL 34787",
            "pipeline":      "DeepEarth Flood Hydrology v1.0",
            "dem_source":    "USGS 3DEP LiDAR (hydro-flattened, 2.6m resolution)",
            "s2_source":     "Sentinel-2 L2A via Microsoft Planetary Computer",
            "soil_source":   "USDA SSURGO (Candler fine sand, HSG A)",
            "rainfall":      "NOAA Atlas 14 Vol 9 — Central Florida",
            "simulation":    "2D local-inertia (LISFLOOD-FP method, dt=30s, dx=2.6m)",
            "scenarios": [
                "flash_1hr_100yr: 1-hr, 100-yr ARI (1% annual exceedance), peak at 30 min",
                "sustained_12hr_100yr: 12-hr, 100-yr ARI, peak at 360 min",
            ],
            "watnet_f1":      0.854,
            "mndwi_f1":       0.743,
        },
        "features": features,
    }

    def _round_coords(obj, precision=5):
        """Recursively round all float values in GeoJSON geometry to reduce file size."""
        if isinstance(obj, dict):
            return {k: _round_coords(v, precision) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_coords(v, precision) for v in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    fc["features"] = [
        {**feat, "geometry": _round_coords(feat["geometry"])}
        for feat in fc["features"]
    ]

    out_path = os.path.join(output_dir, "flood-data.geojson")
    with open(out_path, "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"  Wrote {out_path}")

    write_site_bounds(output_dir)

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Total flood zone features: {len(features)}")

    by_scenario = {}
    for feat in features:
        sc = feat["properties"]["scenario"]
        by_scenario.setdefault(sc, 0)
        by_scenario[sc] += 1
    for sc, count in by_scenario.items():
        print(f"    {sc}: {count} depth zones")

    print(f"\n  Files written to: {output_dir}")
    print(f"  Copy to digitaltwin repo: data/sites/{SITE_ID}/")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export flood data for digital twin frontend")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: frontend_output/data/sites/demo-site/)")
    args = parser.parse_args()
    main(args.output_dir)
