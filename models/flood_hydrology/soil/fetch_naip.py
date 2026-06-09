"""
NAIP Aerial Imagery Download — Winter Garden FL
================================================
Downloads NAIP (National Agriculture Imagery Program) 1m true-color + NIR
aerial photography for the 2×2 km study area.

NAIP provides sub-meter resolution 4-band (R, G, B, NIR) imagery collected
by USDA FSA over the continental US. Used here to visually verify SSURGO
soil unit boundaries against actual land cover from aerial imagery.

Data source: Microsoft Planetary Computer STAC API
Collection : naip (USDA NAIP, most recent available for Florida)
Resolution : 0.6–1.0 m (actual varies by year; recent vintages ~0.6m)
Bands      : Band 1=Red, Band 2=Green, Band 3=Blue, Band 4=NIR

Outputs (saved under soil/data/):
    naip_{year}_RGB.tif   — 3-band true-color GeoTIFF (uint8)
    naip_{year}_NIR.tif   — single-band NIR GeoTIFF (uint8)
    naip_{year}_NDVI.tif  — NDVI = (NIR-Red)/(NIR+Red), vegetation index
    naip_meta.json        — metadata: year, resolution, tile, date

Usage:
    python3 soil/fetch_naip.py
    python3 soil/fetch_naip.py --years 2022 2020
"""

import os
import sys
import json
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981
RADIUS_KM    = 1.1

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def bbox_from_center(lat, lon, radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def search_naip(bbox, years=None):
    """Search Planetary Computer for NAIP scenes."""
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        PC_STAC_URL, modifier=planetary_computer.sign_inplace)

    # Build date range for each year
    all_items = []
    search_years = years if years else list(range(2022, 2017, -1))

    for year in search_years:
        try:
            search = catalog.search(
                collections=["naip"],
                bbox=bbox,
                datetime=f"{year}-01-01/{year}-12-31",
                max_items=5,
            )
            items = list(search.get_items())
            if items:
                print(f"  NAIP {year}: {len(items)} scene(s) found")
                all_items.extend(items)
                break  # use most recent year that has coverage
        except Exception as e:
            print(f"  NAIP {year}: {e}")
            continue

    return all_items


def download_naip(item, bbox, data_dir):
    """Download NAIP scene clipped to AOI, save as GeoTIFFs."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shapely_box
    import pyproj
    from shapely.ops import transform as shp_transform

    west, south, east, north = bbox
    aoi_geom = shapely_box(west, south, east, north)

    date_str = item.datetime.strftime("%Y%m%d") if item.datetime else "unknown"
    year = date_str[:4]
    print(f"  Downloading NAIP: {date_str}")

    # NAIP on Planetary Computer: asset key is "image" (4-band RGBN)
    asset_key = None
    for k in ["image", "data", "B01", "visual"]:
        if k in item.assets:
            asset_key = k
            break
    if asset_key is None:
        print(f"  ⚠ No recognized asset key. Available: {list(item.assets.keys())}")
        asset_key = list(item.assets.keys())[0]

    href = item.assets[asset_key].href

    try:
        with rasterio.open(href) as src:
            # Reproject AOI to image CRS
            if "4326" not in str(src.crs) and "WGS" not in str(src.crs).upper():
                transformer = pyproj.Transformer.from_crs(
                    "epsg:4326", src.crs, always_xy=True)
                aoi_geom_crs = shp_transform(transformer.transform, aoi_geom)
            else:
                aoi_geom_crs = aoi_geom

            out_image, out_transform = rio_mask(
                src, [aoi_geom_crs.__geo_interface__], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            })

            print(f"  Image shape: {out_image.shape} (bands×rows×cols)")
            print(f"  Resolution: {abs(out_transform.a):.2f} m")

            n_bands = out_image.shape[0]

            # Save RGB (bands 1-3)
            rgb_path = os.path.join(data_dir, f"naip_{year}_RGB.tif")
            rgb_meta = out_meta.copy()
            rgb_meta.update(count=min(3, n_bands), dtype="uint8")
            with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
                dst.write(out_image[:min(3, n_bands)].astype(np.uint8))
            print(f"  Saved RGB → {rgb_path}")

            # Save NIR (band 4 if available)
            if n_bands >= 4:
                nir_path = os.path.join(data_dir, f"naip_{year}_NIR.tif")
                nir_meta = out_meta.copy()
                nir_meta.update(count=1, dtype="uint8")
                with rasterio.open(nir_path, "w", **nir_meta) as dst:
                    dst.write(out_image[3:4].astype(np.uint8))
                print(f"  Saved NIR → {nir_path}")

                # Compute NDVI = (NIR - Red) / (NIR + Red)
                red = out_image[0].astype(np.float32)
                nir = out_image[3].astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0.0)
                ndvi_path = os.path.join(data_dir, f"naip_{year}_NDVI.tif")
                ndvi_meta = out_meta.copy()
                ndvi_meta.update(count=1, dtype="float32")
                with rasterio.open(ndvi_path, "w", **ndvi_meta) as dst:
                    dst.write(ndvi.astype(np.float32)[np.newaxis, :, :])
                print(f"  Saved NDVI → {ndvi_path}")

            # Save metadata
            meta_info = {
                "year": year,
                "date": date_str,
                "item_id": item.id,
                "resolution_m": round(abs(out_transform.a), 3),
                "bands": n_bands,
                "crs": str(src.crs),
                "rgb_path": rgb_path,
                "nir_path": nir_path if n_bands >= 4 else None,
            }
            meta_path = os.path.join(data_dir, "naip_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta_info, f, indent=2)

            return meta_info

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None


def main(years=None):
    bbox = bbox_from_center(PROPERTY_LAT, PROPERTY_LON, RADIUS_KM)
    print(f"NAIP download for Winter Garden FL")
    print(f"AOI bbox [W,S,E,N]: {[round(x,5) for x in bbox]}")

    # Check if already downloaded
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith("naip_") and f.endswith("_RGB.tif")]
    if existing and not years:
        print(f"NAIP already downloaded: {existing}")
        print("  Use --years to force re-download.")
        return

    items = search_naip(bbox, years)
    if not items:
        print("\nNo NAIP scenes found.")
        print("  NAIP coverage for Florida varies by year.")
        print("  Try: python3 soil/fetch_naip.py --years 2022 2021 2020 2019")
        print("  Alternatively, download NAIP from USDA EarthExplorer:")
        print("  https://earthexplorer.usgs.gov/")
        return

    best_item = items[0]
    meta = download_naip(best_item, bbox, DATA_DIR)
    if meta:
        print(f"\n✓ NAIP downloaded successfully")
        print(f"  Year: {meta['year']}, Resolution: {meta['resolution_m']:.2f} m")
        print(f"  Run: python3 soil/soil_naip_overlay.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NAIP aerial imagery")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Preferred years (e.g. 2022 2021). Downloads most recent available.")
    args = parser.parse_args()
    main(args.years)
