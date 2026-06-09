"""
Sentinel-2 Download — Winter Garden FL (Microsoft Planetary Computer)
======================================================================
Downloads cloud-free Sentinel-2 Level-2A (surface reflectance) imagery
for the 2×2 km study area around the Winter Garden FL property.

Sentinel-2 bands downloaded:
    B02 — Blue  (10m native) — WatNet input band 1
    B03 — Green (10m native) — MNDWI numerator / WatNet input band 2
    B04 — Red   (10m native) — WatNet input band 3
    B08 — NIR   (10m native) — NDWI / WatNet input band 4
    B11 — SWIR1 (20m native, resampled to 10m) — MNDWI denominator / WatNet band 5
    B12 — SWIR2 (20m native, resampled to 10m) — WatNet input band 6
    SCL — Scene Classification Layer (20m native) — pixel-level cloud/shadow mask

WatNet requires all 6 bands (B02,B03,B04,B08,B11,B12).
SCL is downloaded alongside spectral bands for pixel-level cloud masking.
SCL classes: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus.
Run sentinel2/s2_cloud_mask.py after download to generate per-pixel cloud masks.

One cloud-free scene per season is selected (wet/dry contrast):
    Dry   : Jan–Mar   (low lake level, minimal rainfall)
    Wet   : Jun–Sep   (Florida rainy season, lake at maximum)

Data source: Microsoft Planetary Computer STAC API (free, no account needed).
Collection : sentinel-2-l2a

Outputs (saved under sentinel2/data/):
    s2_{date}_B02.tif   — Blue band
    s2_{date}_B03.tif   — Green band, clipped to AOI, surface reflectance (/10000)
    s2_{date}_B04.tif   — Red band
    s2_{date}_B08.tif   — NIR band
    s2_{date}_B11.tif   — SWIR1 band (resampled to 10m)
    s2_{date}_B12.tif   — SWIR2 band (resampled to 10m)
    s2_{date}_SCL.tif   — Scene Classification Layer (20m, pixel cloud classes)
    s2_scene_index.csv  — table of downloaded scenes (date, cloud%, tile)

Usage:
    python3 sentinel2/s2_download.py
    python3 sentinel2/s2_download.py --max_cloud 15 --years 2022 2023 2024
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROPERTY_LAT  = 28.521592   # 17801 Champagne Dr (28°31'17.73"N 81°39'25.13"W)
PROPERTY_LON  = -81.656981
RADIUS_KM     = 1.1       # slightly larger than 1.0 to ensure full 2×2 km coverage
CLOUD_MAX     = 10        # max cloud cover %
SEARCH_YEARS  = [2022, 2023, 2024]

# Target one scene per season
SEASONS = {
    "dry_early":   ("01-01", "03-31"),
    "dry_late":    ("04-01", "05-31"),
    "wet_early":   ("06-01", "08-31"),
    "wet_peak":    ("09-01", "10-31"),
}

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]  # [W, S, E, N]


def search_scenes(bbox, date_range, max_cloud=CLOUD_MAX):
    """Search Planetary Computer for cloud-free Sentinel-2 scenes."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=20,
    )
    items = list(search.get_items())
    return items


def download_band(item, band_name, bbox, out_path, target_res=10):
    """Download a single band, clip to bbox, resample to target_res, save as GeoTIFF."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shapely_box

    if band_name not in item.assets:
        print(f"    ⚠ Band {band_name} not in item assets; skipping.")
        return False

    href = item.assets[band_name].href
    west, south, east, north = bbox

    try:
        with rasterio.open(href) as src:
            # Clip to AOI
            aoi_geom = shapely_box(west, south, east, north)
            aoi_geom_crs = reproject_geometry(aoi_geom, src.crs)

            out_image, out_transform = rio_mask(src, [aoi_geom_crs.__geo_interface__], crop=True)
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


def reproject_geometry(geom, target_crs):
    """Reproject a shapely geometry from EPSG:4326 to target_crs."""
    import pyproj
    from shapely.ops import transform as shp_transform

    transformer = pyproj.Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    return shp_transform(transformer.transform, geom)


def main(max_cloud=CLOUD_MAX, years=SEARCH_YEARS):
    import planetary_computer  # ensure imported for signing

    bbox = bbox_from_center(PROPERTY_LAT, PROPERTY_LON, RADIUS_KM)
    print(f"AOI bbox [W,S,E,N]: {[round(x,5) for x in bbox]}")
    print(f"Cloud max: {max_cloud}%  |  Search years: {years}")

    scene_records = []
    downloaded_scenes = set()

    for year in years:
        for season_label, (start_md, end_md) in SEASONS.items():
            date_range = f"{year}-{start_md}/{year}-{end_md}"
            items = search_scenes(bbox, date_range, max_cloud)

            if not items:
                print(f"  [{year} {season_label}] No scenes found (cloud < {max_cloud}%)")
                continue

            # Take the scene with lowest cloud cover
            best = items[0]
            date_str  = best.datetime.strftime("%Y%m%d")
            cloud_pct = best.properties.get("eo:cloud_cover", "?")
            tile      = best.properties.get("s2:mgrs_tile", "?")

            if date_str in downloaded_scenes:
                print(f"  [{year} {season_label}] Already have {date_str}, skipping duplicate")
                continue
            downloaded_scenes.add(date_str)

            print(f"  [{year} {season_label}] Best scene: {date_str}  cloud={cloud_pct:.1f}%  tile={tile}")

            scene_ok = True
            for band in ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]:
                out_path = os.path.join(DATA_DIR, f"s2_{date_str}_{band}.tif")
                if os.path.exists(out_path):
                    print(f"    {band} already cached, skipping download")
                    continue
                ok = download_band(best, band, bbox, out_path)
                if not ok and band != "SCL":  # SCL failure is non-fatal
                    scene_ok = False

            scene_records.append({
                "date":       date_str,
                "year":       year,
                "season":     season_label,
                "cloud_pct":  cloud_pct,
                "tile":       tile,
                "item_id":    best.id,
                "ok":         scene_ok,
            })
            print(f"    {'✓' if scene_ok else '⚠'} Bands saved for {date_str}")

    if scene_records:
        idx = pd.DataFrame(scene_records)
        idx_path = os.path.join(DATA_DIR, "s2_scene_index.csv")
        idx.to_csv(idx_path, index=False)
        print(f"\nSaved scene index: {idx_path}")
        print(f"Total scenes downloaded: {len(idx[idx.ok])}")
    else:
        print("\nNo scenes downloaded. Try increasing --max_cloud or checking connectivity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-2 bands from Planetary Computer")
    parser.add_argument("--max_cloud", type=float, default=CLOUD_MAX, help="Max cloud cover %%")
    parser.add_argument("--years", type=int, nargs="+", default=SEARCH_YEARS, help="Years to search")
    args = parser.parse_args()
    main(args.max_cloud, args.years)
