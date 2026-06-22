"""
Download additional Sentinel-2 bands needed for s2cloudless ML cloud detection.

s2cloudless requires 10 bands: B01, B02, B04, B05, B08, B8A, B09, B10(zeros), B11, B12.
We already have B02, B03, B04, B08, B11, B12, SCL.
This script downloads the missing bands: B01, B05, B8A, B09 for all existing scenes.

Usage:
    python3 sentinel2/fetch_s2cloudless_bands.py
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981
RADIUS_KM    = 1.1

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

MISSING_BANDS = ["B01", "B05", "B8A", "B09"]


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def download_band(item, band_name, bbox, out_path):
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import reproject, Resampling
    import pyproj
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform as shp_transform

    if band_name not in item.assets:
        print(f"    ⚠ Band {band_name} not in item assets")
        return False

    href = item.assets[band_name].href
    west, south, east, north = bbox

    try:
        with rasterio.open(href) as src:
            transformer = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            aoi_geom = shapely_box(west, south, east, north)
            aoi_crs = shp_transform(transformer.transform, aoi_geom)

            out_image, out_transform = rio_mask(src, [aoi_crs.__geo_interface__], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            })
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(out_image)
        return True
    except Exception as exc:
        print(f"    ✗ {band_name}: {exc}")
        return False


def main():
    import pystac_client
    import planetary_computer

    scene_csv = os.path.join(DATA_DIR, "s2_scene_index.csv")
    if not os.path.exists(scene_csv):
        sys.exit("s2_scene_index.csv not found. Run s2_download.py first.")

    df = pd.read_csv(scene_csv)
    bbox = bbox_from_center(PROPERTY_LAT, PROPERTY_LON, RADIUS_KM)

    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)

    for _, row in df.iterrows():
        date_str = str(row["date"])
        item_id  = str(row.get("item_id", ""))

        # Check which bands are already present
        needed = [b for b in MISSING_BANDS
                  if not os.path.exists(os.path.join(DATA_DIR, f"s2_{date_str}_{b}.tif"))]
        if not needed:
            print(f"  {date_str}: all s2cloudless bands already present, skipping")
            continue

        print(f"\n  {date_str}: downloading {needed} …")

        # Find the scene in STAC by date
        date_range = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}/{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        import time
        items = []
        for attempt in range(4):
            try:
                items = list(catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=date_range,
                    max_items=5,
                ).items())
                break
            except Exception as exc:
                wait = 10 * (attempt + 1)
                print(f"    Search attempt {attempt+1} failed ({exc}); retrying in {wait}s …")
                time.sleep(wait)

        # Prefer item matching the stored item_id
        item = None
        for it in items:
            if it.id == item_id or it.datetime.strftime("%Y%m%d") == date_str:
                item = it
                break
        if item is None and items:
            item = items[0]
        if item is None:
            print(f"    ✗ No STAC item found for {date_str}")
            continue

        for band in needed:
            out_path = os.path.join(DATA_DIR, f"s2_{date_str}_{band}.tif")
            ok = download_band(item, band, bbox, out_path)
            if ok:
                print(f"    ✓ {band} → {os.path.basename(out_path)}")

    print("\nDone. Run sentinel2/s2_cloud_mask.py to regenerate cloud masks with s2cloudless.")


if __name__ == "__main__":
    main()
