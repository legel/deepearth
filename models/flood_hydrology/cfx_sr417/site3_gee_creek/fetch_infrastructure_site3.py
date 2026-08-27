"""
Engineered water-control structures for site3 (Gee Creek near Longwood)
============================================================================
Checks whether site3's Gee/Howell/Soldier Creek network has any mapped "as-engineered" control
structures or culverts — same question already investigated for the main AOI (2026-07-24,
found 0 features within that AOI's own 2x2km box despite 72 real SFWMD features within a wider
30km search).

**Real finding, confirmed by directly querying both agencies rather than assuming: SFWMD
(South Florida Water Management District, the agency the main AOI's own check used) genuinely
returns ZERO features anywhere near site3, even at 30km** — site3 (28.69N, -81.29W, near
Longwood/Seminole County) sits outside SFWMD's jurisdiction. The correct agency for this
location is the **St. Johns River Water Management District (SJRWMD)**, confirmed via its own
live ArcGIS Online-hosted "Water Resources Geodatabase" (WRGDB) Water_Control_Structure feature
service (found by searching ArcGIS Online for SJRWMD-owned services, then confirmed live via a
direct `?f=pjson` metadata query) — a genuinely different data source than the main AOI's own
SFWMD-only investigation used, reused here with its own dedicated fetch function below (not
copied from fetch_infrastructure.py, which only knows about SFWMD).

Usage:
    python3 site3_gee_creek/fetch_infrastructure_site3.py
"""
import os, sys, json
import requests

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "infrastructure"))
from fetch_infrastructure import fetch_sfwmd_structures, bbox_from_center  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infrastructure", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SJRWMD_URL = ("https://services.arcgis.com/s8wtJX9suxFen6TA/arcgis/rest/services/"
              "Water_Control_Structure/FeatureServer/9/query")


def fetch_sjrwmd_structures(lat, lon, radius_km):
    """SJRWMD's Water Resources Geodatabase — water control structures (weirs, pumps, culverts)
    at 1:24,000 scale. Confirmed live 2026-07-27 (this project's first-ever query against it)."""
    west, south, east, north = bbox_from_center(lat, lon, radius_km)
    params = {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*", "returnGeometry": "true", "f": "geojson",
    }
    r = requests.get(SJRWMD_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    site = get_site("site3")
    lat, lon = site["lat"], site["lon"]
    site_radius = site["radius_km"]
    wide_radius = 30.0  # same wide-search convention as the main AOI's own SFWMD check

    print(f"Querying SFWMD (wrong district, expect 0) + SJRWMD (correct district) near site3 "
          f"({site['label']})")

    # SFWMD — kept for completeness/comparison, expected to return 0 given the jurisdiction gap.
    sfwmd_wide = fetch_sfwmd_structures(lat, lon, wide_radius)
    n_sfwmd = len(sfwmd_wide.get("features", []))
    print(f"  SFWMD: {n_sfwmd} features within {wide_radius}km (expected ~0 — wrong district)")

    # SJRWMD — the real, correct-district source for this location.
    sjrwmd_wide = fetch_sjrwmd_structures(lat, lon, wide_radius)
    n_sjrwmd_wide = len(sjrwmd_wide.get("features", []))
    sjrwmd_tight = fetch_sjrwmd_structures(lat, lon, site_radius)
    n_sjrwmd_tight = len(sjrwmd_tight.get("features", []))
    print(f"  SJRWMD: {n_sjrwmd_wide} features within {wide_radius}km, "
          f"{n_sjrwmd_tight} within site3's own {site_radius}km box")
    for feat in sjrwmd_wide.get("features", []):
        p = feat["properties"]
        print(f"    · {p.get('NAME')!r} — {p.get('DESCRIPTION')}")

    out_path = os.path.join(DATA_DIR, "sjrwmd_structures_site3.geojson")
    with open(out_path, "w") as f:
        json.dump(sjrwmd_wide, f, indent=2)
    print(f"Saved → {out_path}")

    summary = {
        "site": "site3", "lat": lat, "lon": lon,
        "sfwmd_wide_search_radius_km": wide_radius, "sfwmd_wide_search_count": n_sfwmd,
        "sjrwmd_wide_search_radius_km": wide_radius, "sjrwmd_wide_search_count": n_sjrwmd_wide,
        "sjrwmd_site_box_radius_km": site_radius, "sjrwmd_site_box_count": n_sjrwmd_tight,
        "note": ("SFWMD returns 0 — site3 is outside SFWMD's jurisdiction (unlike the main AOI, "
                 "which is inside it). SJRWMD (St. Johns River WMD) is the correct agency for "
                 "this location and returns real regional structures, though 0 fall inside "
                 "site3's own small test box — same regional-vs-local-box pattern the main "
                 "AOI's SFWMD check already found."),
    }
    with open(os.path.join(DATA_DIR, "sjrwmd_structures_site3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
