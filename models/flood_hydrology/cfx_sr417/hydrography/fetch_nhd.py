"""
NHD Hydrography — Flowlines & Waterbodies for the SR417 Corridor AOI
=======================================================================
Pulls USGS NHDPlus HR flowlines (streams) and waterbodies (ponds/lakes/
wetlands) intersecting the 2x2 km AOI around 28.36687N, -81.43299W, via
pynhd. Roadway-corridor context: flowlines (existing drainage conveyance)
matter more here than at a lake-basin AOI, since this models a drainage
corridor, not a lake.

A small 2km box may legitimately contain zero mapped NHD HR flowlines —
roadside ditches and other sub-HR drainage aren't represented in NHD. If
that happens, this script reports it clearly rather than treating it as
an error.

Outputs (saved under hydrography/data/):
    nhd_flowlines.geojson    — flowline features intersecting the AOI (may be empty)
    nhd_waterbodies.geojson  — waterbody features intersecting the AOI (may be empty)
    nhd_summary.json         — counts, total length/area, bbox used

Usage:
    python3 hydrography/fetch_nhd.py
    python3 hydrography/fetch_nhd.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
"""

import os
import json
import argparse
import numpy as np
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT    = 28.36687   # CFX SR417 corridor test-landscape AOI (Lake Nona / south Orlando)
DEFAULT_LON    = -81.43299
DEFAULT_RADIUS = 1.0        # km -> 2x2 km study box


def bbox_from_center(lat, lon, radius_km):
    """Return (west, south, east, north) bounding box in EPSG:4326."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def fetch_layer(layer_key, bbox_4326):
    """
    layer_key in {"flowline_hr", "waterbody_hr"} — valid pynhd.NHD layer keys.
    Returns a GeoDataFrame (possibly empty) rather than raising, since a
    small AOI box legitimately may not intersect any mapped HR feature.
    """
    import pynhd
    print(f"  Querying NHDPlus HR '{layer_key}' …")
    try:
        nhd = pynhd.NHD(layer_key)
        gdf = nhd.bygeom(bbox_4326, geo_crs="epsg:4326")
        print(f"    ✓ {len(gdf)} feature(s) found")
        return gdf
    except Exception as exc:
        # Truncate — backend errors (e.g. transient CloudFront 5xx) can return
        # a full HTML error page as the exception message, which buries the
        # "this was a network failure, not a real zero-result" signal.
        msg = str(exc).splitlines()[0][:200]
        print(f"    ⚠ {layer_key} query failed (likely transient — retry before trusting "
              f"a zero-result as a real AOI finding): {msg}")
        return gpd.GeoDataFrame()


def summarize_flowlines(gdf):
    if gdf is None or gdf.empty:
        print("\n── Flowlines ─────────────────────────────────────────────────")
        print("  None found in AOI (consistent with unmapped roadside/sub-HR drainage).")
        return {"count": 0, "total_length_km": 0.0, "named": []}

    proj = gdf.to_crs(gdf.estimate_utm_crs())
    total_length_km = float(proj.geometry.length.sum() / 1000)
    named = sorted({n for n in gdf.get("GNIS_NAME", []) if n and str(n).strip()})

    print("\n── Flowlines ─────────────────────────────────────────────────")
    print(f"  Count: {len(gdf)}  |  Total length: {total_length_km:.2f} km")
    print(f"  Named: {named if named else '(none — unnamed/unclassified segments)'}")
    return {"count": int(len(gdf)), "total_length_km": round(total_length_km, 3), "named": named}


def summarize_waterbodies(gdf):
    if gdf is None or gdf.empty:
        print("\n── Waterbodies ───────────────────────────────────────────────")
        print("  None found in AOI.")
        return {"count": 0, "total_area_sqkm": 0.0, "features": []}

    total_area_sqkm = float(gdf["AREASQKM"].sum()) if "AREASQKM" in gdf.columns else None
    features = [
        {"name": row.get("GNIS_NAME", "") or "", "ftype": row.get("FTYPE", "") or "",
         "area_sqkm": row.get("AREASQKM", None)}
        for _, row in gdf.iterrows()
    ]

    print("\n── Waterbodies ───────────────────────────────────────────────")
    print(f"  Count: {len(gdf)}  |  Total area: {total_area_sqkm:.4f} km^2" if total_area_sqkm is not None
          else f"  Count: {len(gdf)}")
    for f in features:
        print(f"    {f['ftype']:20} {f['name'] or '(unnamed)':25} {f['area_sqkm']} km^2")
    return {"count": int(len(gdf)), "total_area_sqkm": total_area_sqkm, "features": features}


def save_outputs(flowlines_gdf, waterbodies_gdf, bbox, flow_summary, water_summary):
    flow_path  = os.path.join(DATA_DIR, "nhd_flowlines.geojson")
    water_path = os.path.join(DATA_DIR, "nhd_waterbodies.geojson")

    if flowlines_gdf is not None and not flowlines_gdf.empty:
        flowlines_gdf.to_file(flow_path, driver="GeoJSON")
        print(f"\nSaved flowlines   → {flow_path}")
    else:
        gpd.GeoDataFrame({"geometry": []}, crs="epsg:4326").to_file(flow_path, driver="GeoJSON")
        print(f"\nSaved empty flowlines placeholder → {flow_path}")

    if waterbodies_gdf is not None and not waterbodies_gdf.empty:
        waterbodies_gdf.to_file(water_path, driver="GeoJSON")
        print(f"Saved waterbodies → {water_path}")
    else:
        gpd.GeoDataFrame({"geometry": []}, crs="epsg:4326").to_file(water_path, driver="GeoJSON")
        print(f"Saved empty waterbodies placeholder → {water_path}")

    summary = {
        "bbox_wsen": list(bbox),
        "layers_queried": ["flowline_hr", "waterbody_hr"],
        "flowlines": flow_summary,
        "waterbodies": water_summary,
    }
    summary_path = os.path.join(DATA_DIR, "nhd_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary     → {summary_path}")


def main(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=DEFAULT_RADIUS):
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"Querying NHD for ({lat}, {lon}), radius {radius_km} km")
    print(f"  Bounding box: {[round(x, 5) for x in bbox]}")

    flowlines   = fetch_layer("flowline_hr", bbox)
    waterbodies = fetch_layer("waterbody_hr", bbox)

    flow_summary  = summarize_flowlines(flowlines)
    water_summary = summarize_waterbodies(waterbodies)

    save_outputs(flowlines, waterbodies, bbox, flow_summary, water_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NHDPlus HR flowlines + waterbodies for the SR417 corridor AOI")
    parser.add_argument("--lat",       type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",       type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS)
    args = parser.parse_args()
    main(args.lat, args.lon, args.radius_km)
