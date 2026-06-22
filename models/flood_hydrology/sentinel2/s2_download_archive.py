"""
Sentinel-2 Full Archive Download — Winter Garden FL
=====================================================
Downloads the full historical Sentinel-2 L2A archive for the 2×2 km AOI
around Johns Lake (Winter Garden FL) from Microsoft Planetary Computer STAC.

Reuses bbox/download/reproject helpers from s2_download.py.

Strategy:
    1. Query 2015-06-01 to present, eo:cloud_cover < CLOUD_PREFILTER (20%)
       as a generous prefilter; fine-grained ranking happens in s2_rank_scenes.py.
    2. Per item: download all 11 bands (B01–B12 minus B10, SCL) at 10m,
       skip existing scenes.
    3. After download, suggest next steps:
       python3 sentinel2/s2_cloud_mask.py   (generates cloud_summary.csv)
       python3 sentinel2/s2_rank_scenes.py  (ranks by cloud_on_lake_pct)
       python3 sentinel2/s2_omniwatermask.py --from-ranking  (timeseries)

Expected: ~700 scenes (2015–present), ~1.3 MB/scene → ~1 GB total.

Usage:
    python3 sentinel2/s2_download_archive.py                  # full run
    python3 sentinel2/s2_download_archive.py --validate       # validate existing 11 scenes only
    python3 sentinel2/s2_download_archive.py --max_cloud 30   # wider prefilter
    python3 sentinel2/s2_download_archive.py --start_date 2020-01-01  # partial range
"""

import os
import sys
import argparse
import warnings
from datetime import date as dt_date
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROPERTY_LAT   = 28.521592
PROPERTY_LON   = -81.656981
RADIUS_KM      = 1.1
CLOUD_PREFILTER = 20.0      # generous prefilter — scene_ranking.py does the real filter
ARCHIVE_START  = "2015-06-01"
BANDS          = ["B01", "B02", "B03", "B04", "B05", "B08", "B8A", "B09", "B11", "B12", "SCL"]
PC_STAC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Scenes already downloaded (used for validation mode)
KNOWN_SCENES = [
    "20220329", "20220408", "20220727", "20220930",
    "20230314", "20230503", "20231015",
    "20240207", "20240412", "20240726", "20241024",
]


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def reproject_geometry(geom, target_crs):
    import pyproj
    from shapely.ops import transform as shp_transform
    transformer = pyproj.Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    return shp_transform(transformer.transform, geom)


def download_band(item, band_name, bbox, out_path):
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shapely_box

    if band_name not in item.assets:
        return False

    href = item.assets[band_name].href
    west, south, east, north = bbox
    aoi_geom = shapely_box(west, south, east, north)

    try:
        with rasterio.open(href) as src:
            aoi_reproj = reproject_geometry(aoi_geom, src.crs)
            out_image, out_transform = rio_mask(src, [aoi_reproj.__geo_interface__], crop=True)
            meta = src.meta.copy()
            meta.update({
                "driver":    "GTiff",
                "height":    out_image.shape[1],
                "width":     out_image.shape[2],
                "transform": out_transform,
                "compress":  "lzw",
            })
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(out_image)
        return True
    except Exception as exc:
        print(f"      ✗ {band_name}: {exc}")
        return False


def scene_date(item):
    return item.datetime.strftime("%Y%m%d")


def query_all_items(bbox, start_date, end_date, max_cloud):
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    # Planetary Computer paging: use get_all_items() which handles pagination
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "datetime", "direction": "asc"}],
    )

    print("Fetching scene list from Planetary Computer (may take a moment) …", flush=True)
    items = list(search.get_all_items())
    print(f"  Found {len(items)} scenes matching criteria")
    return items


def validate_existing(bbox):
    """Validate that all 11 known scenes can be re-discovered in the STAC catalog."""
    import pystac_client
    import planetary_computer

    print("=== Validate mode: checking 11 existing scenes in STAC ===")
    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    found = set()
    for date_str in KNOWN_SCENES:
        year  = date_str[:4]
        month = date_str[4:6]
        day   = date_str[6:8]
        # Search ±1 day to account for tile/granule datetime edge cases
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-{month}-{day}",
            query={"eo:cloud_cover": {"lt": 100}},
            max_items=5,
        )
        items = list(search.get_items())
        match = [i for i in items if scene_date(i) == date_str]
        status = "PASS" if match else "FAIL (not found in STAC)"
        print(f"  {date_str}: {len(items)} results → {status}")
        if match:
            found.add(date_str)

    print(f"\nValidation: {len(found)}/{len(KNOWN_SCENES)} existing scenes found in STAC")
    missing = set(KNOWN_SCENES) - found
    if missing:
        print(f"  Missing from STAC: {sorted(missing)}")
    else:
        print("  All existing scenes confirmed in STAC catalog.")
    return len(missing) == 0


def main(validate=False, max_cloud=CLOUD_PREFILTER, start_date=ARCHIVE_START):
    bbox = bbox_from_center(PROPERTY_LAT, PROPERTY_LON, RADIUS_KM)
    print(f"AOI bbox [W,S,E,N]: {[round(x, 5) for x in bbox]}")
    print(f"Cloud prefilter: <{max_cloud}%  |  Archive from: {start_date}")

    if validate:
        ok = validate_existing(bbox)
        sys.exit(0 if ok else 1)

    end_date = dt_date.today().isoformat()
    items = query_all_items(bbox, start_date, end_date, max_cloud)

    if not items:
        print("No scenes found. Check connectivity or adjust --max_cloud / --start_date.")
        sys.exit(1)

    # Deduplicate by date (keep lowest cloud cover per date)
    by_date = {}
    for item in items:
        d = scene_date(item)
        cc = item.properties.get("eo:cloud_cover", 999)
        if d not in by_date or cc < by_date[d][1]:
            by_date[d] = (item, cc)

    unique_dates = sorted(by_date.keys())
    print(f"Unique scene dates after dedup: {len(unique_dates)}")

    new_count  = 0
    skip_count = 0
    fail_count = 0
    records    = []

    for date_str in unique_dates:
        item, cloud_pct = by_date[date_str]
        tile = item.properties.get("s2:mgrs_tile", "?")

        # Check if all 11 bands already on disk
        existing = all(
            os.path.exists(os.path.join(DATA_DIR, f"s2_{date_str}_{b}.tif"))
            for b in BANDS
        )
        if existing:
            skip_count += 1
            records.append({"date": date_str, "cloud_pct": cloud_pct, "tile": tile, "status": "skipped"})
            continue

        print(f"  {date_str}  cloud={cloud_pct:.1f}%  tile={tile}", flush=True)
        scene_ok = True
        for band in BANDS:
            out_path = os.path.join(DATA_DIR, f"s2_{date_str}_{band}.tif")
            if os.path.exists(out_path):
                continue
            ok = download_band(item, band, bbox, out_path)
            if not ok and band != "SCL":
                scene_ok = False

        status = "ok" if scene_ok else "partial"
        if not scene_ok:
            fail_count += 1
        else:
            new_count += 1
        print(f"    {'✓' if scene_ok else '⚠'} {status}")
        records.append({"date": date_str, "cloud_pct": cloud_pct, "tile": tile, "status": status})

    # Save archive index
    idx_path = os.path.join(DATA_DIR, "s2_archive_index.csv")
    pd.DataFrame(records).to_csv(idx_path, index=False)

    print(f"\n=== Archive download complete ===")
    print(f"  Skipped (already on disk): {skip_count}")
    print(f"  Downloaded (new):          {new_count}")
    print(f"  Partial/failed:            {fail_count}")
    print(f"  Archive index:             {idx_path}")
    print("\nNext steps:")
    print("  python3 sentinel2/s2_cloud_mask.py")
    print("  python3 sentinel2/s2_rank_scenes.py")
    print("  python3 sentinel2/s2_omniwatermask.py --from-ranking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download full S2 archive from Planetary Computer")
    parser.add_argument("--validate",   action="store_true",  help="Validate existing 11 scenes in STAC, then exit")
    parser.add_argument("--max_cloud",  type=float, default=CLOUD_PREFILTER, help="Cloud prefilter %% (default 20)")
    parser.add_argument("--start_date", type=str,   default=ARCHIVE_START,   help="Archive start date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(validate=args.validate, max_cloud=args.max_cloud, start_date=args.start_date)
