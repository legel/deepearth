"""
PlanetScope Scene Catalogue — CFX SR417 Corridor
=================================================
Reads the 10 downloaded SuperDove scenes from Florida_Hydrology_PlanetScope/,
extracts metadata from each scene's manifest/metadata JSON, and writes:

  planetscope/data/catalogue.json       — full scene inventory
  planetscope/data/water_boundaries/    — per-scene water boundary GeoJSONs
                                          (WGS84, sub-pixel fitted shorelines)

Scenes are organised by precipitation category matching the GHCND NOAA analysis:
  MAX — Hurricane Ian (2022-09-30), and two other high-rain events
  AVG — 4 representative wet/dry season weeks
  MIN — 3 dry-streak baseline periods

Ground-truth link (hydrology chain):
  Max_Precipitation_1 (2022-09-30) aligns with the Hurricane Ian NWIS 02263800
  peak of 3,500 cfs / 11.43 ft on Shingle Creek — the calibration target for
  any flood model built for this AOI.

Usage:
    python3 planetscope/catalogue.py
"""

import os
import json
import shutil

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BASE_DIR)
PLANET_DIR = os.path.join(PROJ_DIR, "Florida_Hydrology_PlanetScope")
OUT_DIR    = os.path.join(BASE_DIR, "data")
WB_DIR     = os.path.join(OUT_DIR, "water_boundaries")
os.makedirs(WB_DIR, exist_ok=True)

# Precipitation category assigned by generate_target_dates.py (GHCND analysis)
CATEGORY_MAP = {
    "Florida_Hydrology_Max_Precipitation_1": {
        "category": "MAX", "rank": 1,
        "ghcnd_date": "2022-09-29", "ghcnd_mm": 345.4,
        "5day_total_mm": 381.5, "note": "Hurricane Ian peak day",
        "nwis_02263800_cfs": 3500, "nwis_02263800_ft": 11.43,
    },
    "Florida_Hydrology_Max_Precipitation_2": {
        "category": "MAX", "rank": 2,
        "ghcnd_date": "2025-05-29", "ghcnd_mm": None,
        "5day_total_mm": 133.3, "note": "Strong late wet-season event",
    },
    "Florida_Hydrology_Max_Precipitation_3": {
        "category": "MAX", "rank": 3,
        "ghcnd_date": "2024-06-30", "ghcnd_mm": None,
        "5day_total_mm": 130.8, "note": "Early wet-season rainfall pulse",
    },
    "Florida_Hydrology_Avg_Precipitation_1": {
        "category": "AVG", "rank": 1,
        "ghcnd_date": "2025-06-17", "season": "wet",
        "note": "Representative wet-season week",
    },
    "Florida_Hydrology_Avg_Precipitation_2": {
        "category": "AVG", "rank": 2,
        "ghcnd_date": "2023-09-17", "season": "wet",
        "note": "Representative wet-season week",
    },
    "Florida_Hydrology_Avg_Precipitation_3": {
        "category": "AVG", "rank": 3,
        "ghcnd_date": "2025-12-04", "season": "dry",
        "note": "Representative dry-season week",
    },
    "Florida_Hydrology_Avg_Precipitation_4": {
        "category": "AVG", "rank": 4,
        "ghcnd_date": "2023-12-23", "season": "dry",
        "note": "Representative dry-season week",
    },
    "Florida_Hydrology_Min_Precipitation_1": {
        "category": "MIN", "rank": 1,
        "ghcnd_date": "2021-02-17", "dry_streak_days": 43,
        "note": "43 consecutive dry days — baseline vegetation/soil state",
    },
    "Florida_Hydrology_Min_Precipitation_2": {
        "category": "MIN", "rank": 2,
        "ghcnd_date": "2024-11-08", "dry_streak_days": 34,
        "note": "34 consecutive dry days",
    },
    "Florida_Hydrology_Min_Precipitation_3": {
        "category": "MIN", "rank": 3,
        "ghcnd_date": "2022-04-09", "dry_streak_days": 33,
        "note": "33 consecutive dry days",
    },
}


def read_scene(scene_dir):
    """Read scene metadata from PSScene/*.json and derived README."""
    ps_dir = os.path.join(scene_dir, "PSScene")
    meta_file = next(
        (f for f in os.listdir(ps_dir) if f.endswith("_metadata.json")), None
    )
    if not meta_file:
        return {}
    with open(os.path.join(ps_dir, meta_file)) as f:
        meta = json.load(f)

    props = meta.get("properties", {})
    geom  = meta.get("geometry", {})

    # Read water stats from README if available
    readme_path = os.path.join(scene_dir, "README.md")
    water_ha = None
    water_bodies = None
    udm2_clear_pct = None
    if os.path.exists(readme_path):
        with open(readme_path) as f:
            txt = f.read()
        import re
        for line in txt.split("\n"):
            if "Open water (sub-pixel)" in line and "ha" in line:
                # README format: "| Open water (sub-pixel) | 1.8% (7.3 ha; 225 bodies) |"
                m = re.search(r'([\d.]+)\s*ha[;,]?\s*([\d]+)\s*bodies', line)
                if m:
                    water_ha     = float(m.group(1))
                    water_bodies = int(m.group(2))
            if "UDM2 clear (in-footprint)" in line:
                try:
                    val = line.split("|")[2].strip().replace("%", "")
                    udm2_clear_pct = float(val)
                except Exception:
                    pass

    return {
        "item_id":        meta.get("id", ""),
        "acquired_utc":   props.get("acquired", ""),
        "instrument":     props.get("instrument", ""),
        "satellite_id":   props.get("satellite_id", ""),
        "gsd_m":          props.get("gsd", None),
        "cloud_pct":      props.get("cloud_percent", None),
        "clear_pct":      props.get("clear_percent", None),
        "udm2_clear_pct": udm2_clear_pct,
        "view_angle_deg": props.get("view_angle", None),
        "sun_elevation":  props.get("sun_elevation", None),
        "bands":          8,
        "crs":            "EPSG:32617",
        "pixel_res_m":    3.0,
        "water_ha":       water_ha,
        "water_bodies":   water_bodies,
        "geometry":       geom,
    }


def copy_water_boundaries(scene_name, scene_dir):
    """Copy the WGS84 water boundary GeoJSON into planetscope/data/water_boundaries/."""
    src = os.path.join(scene_dir, "derived", "water_boundaries_wgs84.geojson")
    if not os.path.exists(src):
        return None
    dst = os.path.join(WB_DIR, f"{scene_name}_water_wgs84.geojson")
    shutil.copy2(src, dst)
    return dst


def run():
    print("PlanetScope Catalogue — CFX SR417 Corridor")
    print("=" * 50)

    scenes = sorted(
        d for d in os.listdir(PLANET_DIR)
        if os.path.isdir(os.path.join(PLANET_DIR, d)) and not d.startswith(".")
    )

    catalogue = []
    for scene_name in scenes:
        scene_dir = os.path.join(PLANET_DIR, scene_name)
        scene_meta = read_scene(scene_dir)
        precip_meta = CATEGORY_MAP.get(scene_name, {})

        entry = {
            "scene_name":     scene_name,
            "source_dir":     scene_dir,
            **scene_meta,
            **precip_meta,
        }

        # Copy water boundaries
        wb_dst = copy_water_boundaries(scene_name, scene_dir)
        entry["water_boundary_wgs84"] = wb_dst

        catalogue.append(entry)
        date = scene_meta.get("acquired_utc", "?")[:10]
        cat  = precip_meta.get("category", "?")
        rank = precip_meta.get("rank", "?")
        note = precip_meta.get("note", "")
        cloud = scene_meta.get("cloud_pct", "?")
        water = scene_meta.get("water_ha", "?")
        print(f"  {cat}-{rank}  {date}  cloud={cloud}%  water={water} ha  {note}")

    cat_path = os.path.join(OUT_DIR, "catalogue.json")
    with open(cat_path, "w") as f:
        json.dump(catalogue, f, indent=2, default=str)
    print(f"\nCatalogue written: {cat_path}  ({len(catalogue)} scenes)")
    print(f"Water boundaries: {WB_DIR}/")

    # Print summary table
    print("\nSummary:")
    print(f"  {'Category':<8} {'Date':<12} {'Cloud%':<8} {'Clear%':<8} {'Water ha':<10} {'Note'}")
    print("  " + "-"*70)
    for e in sorted(catalogue, key=lambda x: (x.get("category",""), x.get("rank",0))):
        print(f"  {e.get('category','?'):<8} "
              f"{e.get('acquired_utc','?')[:10]:<12} "
              f"{str(e.get('cloud_pct','?')):<8} "
              f"{str(e.get('clear_pct','?')):<8} "
              f"{str(e.get('water_ha','?')):<10} "
              f"{e.get('note','')}")
    return catalogue


if __name__ == "__main__":
    run()
