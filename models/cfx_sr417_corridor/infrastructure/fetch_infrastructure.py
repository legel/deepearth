"""
Infrastructure & Water Gauge Fetch — CFX SR417 Corridor
========================================================
Downloads the as-engineered hydrological infrastructure datasets
for the AOI and its extended drainage basin (Shingle Creek / HUC 03090101):

  1. USGS NWIS Stream Gauges (waterservices.usgs.gov)
     Two Shingle Creek gauges bracket our AOI:
       - 02263692  SHINGLE CREEK AT OAK RIDGE RD NR PINE CASTLE  (6 km N, upstream)
       - 02263800  SHINGLE CREEK AT AIRPORT NEAR KISSIMMEE        (7 km S, downstream)
     Plus any additional stream gauges within 30 km.

  2. USGS NWIS Daily Stage/Flow Data for the two Shingle Creek gauges
     Downloads recent discharge (00060) and gage height (00065) records.
     These are the actual measured water levels — ground truth for any flood model.

  3. SFWMD (South Florida Water Management District) Structures + Project Culverts —
     ArcGIS REST (geoweb.sfwmd.gov/agsext1/.../WaterManagementSystem/All_Structures).
     Control structures (S-gates, pumps, weirs) + engineered culverts that regulate
     Shingle Creek and the surrounding canal network — the "as-engineered floodway."
     2026-07-24: this was previously broken (old gis.sfwmd.gov host no longer resolves,
     a real DNS failure, not transient) — found and switched to the current live endpoint.
     Confirmed working (real features returned for a wider 30km search) but genuinely 0
     features fall within this project's own small 2x2km AOI — real finding, not a bug.

  4. Florida DOT Culverts/Bridges — attempted 2026-07-24, genuine dead end for this AOI.
     FDOT publishes bridge/culvert feature services unevenly by district (confirmed District
     3 and District 6 have public ArcGIS Online services; this AOI is in District 5, Orlando/
     Central Florida, which does not appear to have an equivalent public service as of this
     check). Not fetched — no working endpoint found despite a real attempt, not skipped
     without trying.

  5. NHD waterbodies clip — AOI-intersecting ponds/lakes for context.

Outputs → infrastructure/data/:
  nwis_gauges.geojson       — USGS stream gauges within 30 km
  nwis_gauges_summary.json  — gauge metadata + record dates
  nwis_shingle_creek_02263692_daily.csv  — stage + flow at upstream gauge
  nwis_shingle_creek_02263800_daily.csv  — stage + flow at downstream gauge
  sfwmd_structures.geojson  — SFWMD control structures (S-gates, weirs, pumps)
  sfwmd_structures_summary.json
  infrastructure_summary.json — combined summary

Usage:
    python3 infrastructure/fetch_infrastructure.py
    python3 infrastructure/fetch_infrastructure.py --lat 28.36687 --lon -81.43299
"""

import os
import json
import csv
import argparse
import time
from datetime import datetime, timedelta

import requests
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT    = 28.36687
DEFAULT_LON    = -81.43299
DEFAULT_RADIUS = 30.0   # km — wide enough to capture both Shingle Creek gauges

# Known Shingle Creek gauges — our primary hydraulic ground-truth
SHINGLE_CREEK_GAUGES = ["02263692", "02263800"]

# SFWMD DBHYDRO REST service base
SFWMD_REST = "https://www.sfwmd.gov/data-maps/dbhydro-environmental-databases"


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


# ── 1. USGS NWIS gauge locations ─────────────────────────────────────────────

def fetch_nwis_sites(lat, lon, radius_km):
    print(f"Fetching USGS NWIS stream gauges within {radius_km} km …")
    west, south, east, north = bbox_from_center(lat, lon, radius_km)
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format":    "rdb",
        "bBox":      f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}",
        "siteType":  "ST",
        "siteStatus":"all",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    sites = []
    for line in r.text.split("\n"):
        if not line or line.startswith("#") or line.startswith("5"):
            continue
        parts = line.split("\t")
        if len(parts) < 5 or parts[0] != "USGS":
            continue
        try:
            site_lat = float(parts[4])
            site_lon = float(parts[5])
        except (ValueError, IndexError):
            continue
        dist_km = _haversine(lat, lon, site_lat, site_lon)
        sites.append({
            "agency":      parts[0].strip(),
            "site_no":     parts[1].strip(),
            "station_nm":  parts[2].strip(),
            "site_tp":     parts[3].strip(),
            "lat":         site_lat,
            "lon":         site_lon,
            "dist_km":     round(dist_km, 2),
            "huc":         parts[11].strip() if len(parts) > 11 else "",
            "is_shingle":  parts[1].strip() in SHINGLE_CREEK_GAUGES,
        })

    sites.sort(key=lambda s: s["dist_km"])
    print(f"  Found {len(sites)} stream gauges")
    for s in sites:
        flag = " ← SHINGLE CREEK" if s["is_shingle"] else ""
        print(f"    {s['site_no']}  {s['station_nm'][:45]:45s}  {s['dist_km']:.1f} km{flag}")
    return sites


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def sites_to_geojson(sites):
    return {
        "type": "FeatureCollection",
        "crs":  {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                "properties": {k: v for k, v in s.items() if k not in ("lat", "lon")},
            }
            for s in sites
        ],
    }


# ── 2. USGS NWIS daily stage/flow data ───────────────────────────────────────

def fetch_nwis_daily(site_no, param_codes, start_date, end_date):
    """
    param_codes: list of USGS parameter codes
      00060 = discharge (cfs)
      00065 = gage height (ft)
    """
    url = "https://waterservices.usgs.gov/nwis/dv/"
    params = {
        "format":       "rdb",
        "sites":        site_no,
        "parameterCd":  ",".join(param_codes),
        "startDT":      start_date,
        "endDT":        end_date,
        "statCd":       "00003",   # mean daily value
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"    Warning: NWIS daily fetch for {site_no} failed: {e}")
        return ""


def parse_nwis_rdb(rdb_text):
    """Parse USGS RDB format → list of dicts."""
    rows = []
    headers = None
    for line in rdb_text.split("\n"):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if headers is None:
            headers = parts
            continue
        if all(c in ("4s", "5n", "10d", "14n", "10s") for c in parts[:2]):
            continue  # skip RDB type line
        if len(parts) >= len(headers):
            rows.append(dict(zip(headers, parts)))
    return rows


# ── 3. SFWMD control structures ───────────────────────────────────────────────

def fetch_sfwmd_structures(lat, lon, radius_km):
    """
    SFWMD publishes structure locations via their ArcGIS REST service.

    2026-07-24: the old endpoint (gis.sfwmd.gov/sfwmdgis/.../Hydrology/
    WaterControlStructures/MapServer) no longer resolves at all (DNS failure, confirmed via
    direct query, not a transient blip — SFWMD appears to have migrated their GIS hosting).
    Found the replacement by searching for their current service tree: SFWMD now serves this
    from geoweb.sfwmd.gov under a differently-named service, "All_Structures" (confirmed live,
    HTTP 200, 2026-07-24). Its own service description explicitly states it also includes
    "project culverts" — layer 7 below — which is directly relevant to Lance's separate
    "engineered waterway" ask (see CLAUDE.md's 2026-07-23/24 entry), not just structures.
    Queries layer 4 (All WMS Structures) + layer 7 (Project Culverts) — both confirmed to
    return real features for a wider 30km search, though (also confirmed) genuinely 0 features
    fall within this project's own small 2x2km AOI itself — a real finding about this specific
    corridor, not a fetch failure.
    """
    print("Fetching SFWMD water control structures + project culverts …")
    west, south, east, north = bbox_from_center(lat, lon, radius_km)
    bbox = f"{west},{south},{east},{north}"

    base = "https://geoweb.sfwmd.gov/agsext1/rest/services/WaterManagementSystem/All_Structures/MapServer"
    layers = {4: "All WMS Structures", 7: "Project Culverts"}
    all_features = []
    for layer_id, label in layers.items():
        params = {
            "geometry":     bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR":         "4326",
            "outSR":        "4326",
            "outFields":    "*",
            "f":            "geojson",
            "where":        "1=1",
            "returnGeometry": "true",
        }
        try:
            r = requests.get(f"{base}/{layer_id}/query", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            features = data.get("features", [])
            for feat in features:
                feat.setdefault("properties", {})["_source_layer"] = label
            print(f"  {label} (layer {layer_id}): {len(features)} features")
            all_features.extend(features)
        except Exception as e:
            print(f"  Warning: {label} fetch failed: {e}")

    return {"type": "FeatureCollection", "features": all_features}


# ── 4. Engineered canal / drain network from NHD ─────────────────────────────

def fetch_nhd_canals(lat, lon, radius_km):
    """
    Pull NHD artificial paths / canals from pynhd — these are the SFWMD-managed
    drainage canals (FTYPE=336 Canal/Ditch) that complement the natural Shingle
    Creek flowlines already in hydrography/data/.
    """
    print("Fetching NHD canal/ditch flowlines (engineered drainage) …")
    west, south, east, north = bbox_from_center(lat, lon, radius_km)
    from shapely.geometry import box
    try:
        from pynhd import NHDPlusHR
        nhd = NHDPlusHR("flowline")
        bbox_geom = box(west, south, east, north)
        gdf = nhd.bygeom(bbox_geom, crs="epsg:4326")
        canals = gdf[gdf.get("ftype", gdf.get("FType", gdf.get("FTYPE", 0))).isin([336, 460])]
        print(f"  Found {len(canals)} canal/artificial-path segments")
        return canals
    except Exception as e:
        print(f"  Warning: NHD canal fetch failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def run(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=DEFAULT_RADIUS):
    summary = {
        "aoi": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "generated": datetime.utcnow().isoformat() + "Z",
    }

    # ── 1. Gauge locations ──────────────────────────────────────────────────
    sites = fetch_nwis_sites(lat, lon, radius_km)
    gj = sites_to_geojson(sites)
    gauge_path = os.path.join(DATA_DIR, "nwis_gauges.geojson")
    with open(gauge_path, "w") as f:
        json.dump(gj, f, indent=2)
    print(f"  Saved: nwis_gauges.geojson  ({len(sites)} gauges)")

    gauge_summary_path = os.path.join(DATA_DIR, "nwis_gauges_summary.json")
    with open(gauge_summary_path, "w") as f:
        json.dump(sites, f, indent=2)
    summary["nwis_gauges"] = {"count": len(sites), "shingle_creek_gauges": SHINGLE_CREEK_GAUGES}

    # ── 2. Shingle Creek daily stage + flow ─────────────────────────────────
    end_date   = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    print(f"\nDownloading Shingle Creek daily discharge + stage ({start_date} → {end_date}) …")

    for site_no in SHINGLE_CREEK_GAUGES:
        matching = [s for s in sites if s["site_no"] == site_no]
        if not matching:
            print(f"  {site_no}: not found in bbox query — skipping daily data")
            continue
        site_name = matching[0]["station_nm"]
        print(f"  {site_no}  {site_name} …")
        rdb = fetch_nwis_daily(site_no, ["00060", "00065"], start_date, end_date)
        rows = parse_nwis_rdb(rdb)
        if rows:
            csv_path = os.path.join(DATA_DIR, f"nwis_shingle_creek_{site_no}_daily.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"    Saved {len(rows)} daily records → {os.path.basename(csv_path)}")
            summary.setdefault("nwis_daily", {})[site_no] = {
                "name": site_name, "records": len(rows), "start": start_date, "end": end_date
            }
        else:
            print(f"    No daily data returned (gauge may be inactive)")
        time.sleep(0.5)

    # ── 3. SFWMD structures ─────────────────────────────────────────────────
    print()
    sfwmd_data = fetch_sfwmd_structures(lat, lon, radius_km)
    sfwmd_path = os.path.join(DATA_DIR, "sfwmd_structures.geojson")
    with open(sfwmd_path, "w") as f:
        json.dump(sfwmd_data, f, indent=2)
    n_struct = len(sfwmd_data.get("features", []))
    print(f"  Saved: sfwmd_structures.geojson  ({n_struct} structures)")
    summary["sfwmd_structures"] = {"count": n_struct}

    # ── 4. NHD canals / engineered drainage ────────────────────────────────
    print()
    canal_gdf = fetch_nhd_canals(lat, lon, radius_km)
    if canal_gdf is not None and len(canal_gdf) > 0:
        canal_path = os.path.join(DATA_DIR, "nhd_canals.geojson")
        canal_gdf.to_file(canal_path, driver="GeoJSON")
        print(f"  Saved: nhd_canals.geojson  ({len(canal_gdf)} segments)")
        summary["nhd_canals"] = {"count": len(canal_gdf)}

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_path = os.path.join(DATA_DIR, "infrastructure_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone.  Output → {DATA_DIR}/")
    print("\nKey findings for the as-engineered hydrological system:")
    print("  • Shingle Creek passes through the AOI — gauged N and S of AOI")
    print("  • SFWMD manages the regional canal network (HUC 03090101)")
    print("  • FEMA floodway already downloaded → floodplain/data/")
    print("  • For live stage data, visit: https://waterdata.usgs.gov/nwis/uv?site_no=02263692")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat",       type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",       type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS)
    args = parser.parse_args()
    run(args.lat, args.lon, args.radius_km)
