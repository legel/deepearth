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
RADIUS_KM    = 1.8   # widened from 1.1 to capture adjacent quarter-quad tiles

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
                max_items=20,
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


def download_naip_mosaic(items, bbox, data_dir):
    """Download ALL NAIP tiles intersecting bbox, mosaic them, clip to AOI, save GeoTIFFs.

    Replaces the single-tile download_naip() to avoid black edges when the AOI
    spans multiple quarter-quad tiles.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box as shapely_box
    import pyproj
    from shapely.ops import transform as shp_transform

    west, south, east, north = bbox
    aoi_geom = shapely_box(west, south, east, north)

    date_str = items[0].datetime.strftime("%Y%m%d") if items[0].datetime else "unknown"
    year = date_str[:4]
    print(f"  Downloading {len(items)} NAIP tile(s): {year}")

    def _asset_href(item):
        for k in ["image", "data", "B01", "visual"]:
            if k in item.assets:
                return item.assets[k].href
        return next(iter(item.assets.values())).href

    open_datasets = []
    mosaic_crs = None
    for item in items:
        try:
            ds = rasterio.open(_asset_href(item))
            open_datasets.append(ds)
            if mosaic_crs is None:
                mosaic_crs = ds.crs
        except Exception as e:
            print(f"    Warning: {item.id}: {e}")

    if not open_datasets:
        print("  ✗ No tiles could be opened")
        return None

    print(f"  Mosaicking {len(open_datasets)} tile(s)…")
    mosaic, mosaic_transform = rio_merge(open_datasets)
    meta = open_datasets[0].meta.copy()
    for ds in open_datasets:
        ds.close()

    # Reproject AOI bbox to image CRS for clipping
    if "4326" not in str(mosaic_crs) and "WGS" not in str(mosaic_crs).upper():
        transformer = pyproj.Transformer.from_crs("epsg:4326", mosaic_crs, always_xy=True)
        aoi_crs = shp_transform(transformer.transform, aoi_geom)
    else:
        aoi_crs = aoi_geom

    meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width":  mosaic.shape[2],
        "transform": mosaic_transform,
    })
    with rasterio.MemoryFile() as mf:
        with mf.open(**meta) as ds:
            ds.write(mosaic)
            out_image, out_transform = rio_mask(
                ds, [aoi_crs.__geo_interface__], crop=True)
            out_meta = ds.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width":  out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",
    })

    n_bands = out_image.shape[0]
    print(f"  Mosaic shape: {out_image.shape} (bands×rows×cols), "
          f"res: {abs(out_transform.a):.2f} m")

    # Save RGB (bands 1-3)
    rgb_path = os.path.join(data_dir, f"naip_{year}_RGB.tif")
    rgb_meta = out_meta.copy()
    rgb_meta.update(count=min(3, n_bands), dtype="uint8")
    with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
        dst.write(out_image[:min(3, n_bands)].astype(np.uint8))
    print(f"  Saved RGB → {rgb_path}")

    nir_path = None
    if n_bands >= 4:
        nir_path = os.path.join(data_dir, f"naip_{year}_NIR.tif")
        nir_meta = out_meta.copy()
        nir_meta.update(count=1, dtype="uint8")
        with rasterio.open(nir_path, "w", **nir_meta) as dst:
            dst.write(out_image[3:4].astype(np.uint8))
        print(f"  Saved NIR → {nir_path}")

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

    meta_info = {
        "year": year,
        "date": date_str,
        "item_ids": [i.id for i in items],
        "n_tiles": len(items),
        "resolution_m": round(abs(out_transform.a), 3),
        "bands": n_bands,
        "crs": str(mosaic_crs),
        "rgb_path": rgb_path,
        "nir_path": nir_path,
    }
    with open(os.path.join(data_dir, "naip_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    return meta_info


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

    meta = download_naip_mosaic(items, bbox, DATA_DIR)
    if meta:
        print(f"\n✓ NAIP downloaded successfully")
        print(f"  Year: {meta['year']}, Tiles: {meta['n_tiles']}, "
              f"Resolution: {meta['resolution_m']:.2f} m")
        print(f"  Run: python3 viewer/preprocess/export_overlays.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NAIP aerial imagery")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Preferred years (e.g. 2022 2021). Downloads most recent available.")
    args = parser.parse_args()
    main(args.years)
