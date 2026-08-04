"""
NAIP Aerial Imagery Download — CFX SR417 Corridor (Lake Nona / south Orlando, FL)
==================================================================================
Downloads NAIP (National Agriculture Imagery Program) 1m true-color + NIR
aerial photography for the 2x2 km AOI centered on (28.36687, -81.43299).

NAIP provides sub-meter resolution 4-band (R, G, B, NIR) imagery collected
by USDA FSA over the continental US. Used here as the higher-resolution
land-surface counterpart to PlanetScope (which is retained for water/flood
extent) — Task 3 from the 2026-06-29 Lance meeting notes.

Data source: Microsoft Planetary Computer STAC API
Collection : naip (USDA NAIP, most recent available for Florida)
Resolution : 0.6-1.0 m (actual varies by year; recent vintages ~0.6m)
Bands      : Band 1=Red, Band 2=Green, Band 3=Blue, Band 4=NIR

Ported from models/flood_hydrology/soil/fetch_naip.py, adapted to this
project's AOI/CLI/directory conventions (imagery/data/, --lat --lon
--radius_km, bbox_from_center() copied verbatim).

Outputs (saved under imagery/data/):
    naip_{year}_RGB.tif   — 3-band true-color GeoTIFF (uint8)
    naip_{year}_NIR.tif   — single-band NIR GeoTIFF (uint8)
    naip_{year}_NDVI.tif  — NDVI = (NIR-Red)/(NIR+Red), vegetation index
    naip_meta.json        — metadata: year, resolution, tile, date

Usage:
    python3 imagery/fetch_naip.py
    python3 imagery/fetch_naip.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
    python3 imagery/fetch_naip.py --years 2022 2020
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT = 28.36687
DEFAULT_LON = -81.43299
DEFAULT_RADIUS_KM = 1.0
RADIUS_KM = 1.8   # widened beyond the AOI's 1.0 km to capture adjacent quarter-quad tiles

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

    # Build date range for each year. Upper bound was a hardcoded 2022 (real bug found
    # 2026-07-28: silently missed a real, confirmed-available 2023 0.3m acquisition for both
    # this AOI and site3, since the search loop never even checked that year) -- now derived
    # from the real current date so it can't go stale the same way again.
    all_items = []
    current_year = datetime.date.today().year
    search_years = years if years else list(range(current_year + 1, 2017, -1))

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


def _download_asset_to_local(url, dest_path, retries=5, chunk_size=1 << 20):
    """Stream a NAIP asset to a local file with retry+backoff.

    NAIP source tiles are large (multi-band, sub-meter, ~1GB+) and this network path
    showed heavy TCP packet reordering / retransmission in practice (confirmed via
    `nettop`: tens of millions of out-of-order packets over a single connection).
    GDAL's vsicurl streams tiles via scattered HTTP range requests during merge()
    — on this network that produced multiple full-run failures after 40+ minutes
    each: a mid-file "TIFFFillTile: Read error ... got 0 bytes" on one attempt, then
    "not recognized as being in a supported file format" on the immediate retry
    (almost certainly a stale GDAL VSICURL cache entry for that URL after the first
    failure). A plain sequential download to a local file — letting the OS TCP stack
    handle retransmission/reordering instead of GDAL's range-request layer — is far
    more robust here, and resuming after a failure means restarting a single download
    rather than the whole multi-tile mosaic.
    """
    import requests

    if os.path.exists(dest_path):
        print(f"    {os.path.basename(dest_path)} already downloaded — skipping")
        return dest_path

    tmp_path = dest_path + ".part"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                written = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                if total and written != total:
                    raise IOError(f"incomplete download: {written}/{total} bytes")
            os.replace(tmp_path, dest_path)
            print(f"    Downloaded {os.path.basename(dest_path)} "
                  f"({written / 1e6:.1f} MB, attempt {attempt}/{retries})")
            return dest_path
        except Exception as e:
            last_err = e
            print(f"    Download attempt {attempt}/{retries} failed for "
                  f"{os.path.basename(dest_path)}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries:
                wait = 5 * attempt
                print(f"    Retrying in {wait}s...")
                import time as _time
                _time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}")


def download_naip_mosaic(items, bbox, data_dir):
    """Download ALL NAIP tiles intersecting bbox, mosaic them, clip to AOI, save GeoTIFFs.

    Downloads every tile intersecting the bbox (not just the first) to avoid
    black edges when the AOI spans multiple quarter-quad tiles. Each tile is first
    streamed to a local temp file (see _download_asset_to_local) — mosaicking then
    happens entirely from local disk, so it never depends on the network again once
    all tiles have landed.
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

    tmp_dir = os.path.join(data_dir, "_naip_tiles_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    local_paths = []
    for item in items:
        dest = os.path.join(tmp_dir, f"{item.id}.tif")
        try:
            local_paths.append(_download_asset_to_local(_asset_href(item), dest))
        except Exception as e:
            print(f"    Warning: giving up on {item.id}: {e}")

    if not local_paths:
        print("  No tiles could be downloaded")
        return None

    open_datasets = []
    mosaic_crs = None
    for p in local_paths:
        try:
            ds = rasterio.open(p)
            open_datasets.append(ds)
            if mosaic_crs is None:
                mosaic_crs = ds.crs
        except Exception as e:
            print(f"    Warning: could not open local tile {p}: {e}")

    if not open_datasets:
        print("  No local tiles could be opened")
        return None

    print(f"  Mosaicking {len(open_datasets)} local tile(s)...")
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
    print(f"  Mosaic shape: {out_image.shape} (bands x rows x cols), "
          f"res: {abs(out_transform.a):.2f} m")

    # Save RGB (bands 1-3)
    rgb_path = os.path.join(data_dir, f"naip_{year}_RGB.tif")
    rgb_meta = out_meta.copy()
    rgb_meta.update(count=min(3, n_bands), dtype="uint8")
    with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
        dst.write(out_image[:min(3, n_bands)].astype(np.uint8))
    print(f"  Saved RGB -> {rgb_path}")

    nir_path = None
    if n_bands >= 4:
        nir_path = os.path.join(data_dir, f"naip_{year}_NIR.tif")
        nir_meta = out_meta.copy()
        nir_meta.update(count=1, dtype="uint8")
        with rasterio.open(nir_path, "w", **nir_meta) as dst:
            dst.write(out_image[3:4].astype(np.uint8))
        print(f"  Saved NIR -> {nir_path}")

        red = out_image[0].astype(np.float32)
        nir = out_image[3].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0.0)
        ndvi_path = os.path.join(data_dir, f"naip_{year}_NDVI.tif")
        ndvi_meta = out_meta.copy()
        ndvi_meta.update(count=1, dtype="float32")
        with rasterio.open(ndvi_path, "w", **ndvi_meta) as dst:
            dst.write(ndvi.astype(np.float32)[np.newaxis, :, :])
        print(f"  Saved NDVI -> {ndvi_path}")

    meta_info = {
        "year": year,
        "date": date_str,
        "item_ids": [i.id for i in items],
        "n_tiles": len(items),
        "resolution_m": round(abs(out_transform.a), 3),
        "bands": n_bands,
        "crs": str(mosaic_crs),
        "rgb_path": os.path.basename(rgb_path),
        "nir_path": os.path.basename(nir_path) if nir_path else None,
    }
    with open(os.path.join(data_dir, "naip_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    return meta_info


def main(lat=DEFAULT_LAT, lon=DEFAULT_LON, radius_km=RADIUS_KM, years=None):
    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"NAIP download for CFX SR417 corridor AOI (Lake Nona / south Orlando, FL)")
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
        print("  Try: python3 imagery/fetch_naip.py --years 2022 2021 2020 2019")
        print("  Alternatively, download NAIP from USDA EarthExplorer:")
        print("  https://earthexplorer.usgs.gov/")
        return

    meta = download_naip_mosaic(items, bbox, DATA_DIR)
    if meta:
        print(f"\nNAIP downloaded successfully")
        print(f"  Year: {meta['year']}, Tiles: {meta['n_tiles']}, "
              f"Resolution: {meta['resolution_m']:.2f} m")
        print(f"  Run: python3 viewer/preprocess/export_overlays.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NAIP aerial imagery for the CFX SR417 corridor AOI")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS_KM,
                        help="AOI half-width in km (default 1.0). NAIP tile fetch itself widens "
                             "this internally to avoid tile-boundary clipping — see RADIUS_KM.")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Preferred years (e.g. 2022 2021). Downloads most recent available.")
    args = parser.parse_args()

    # Widen the AOI radius for the tile search/clip step (avoids tile-boundary
    # clipping) unless the caller explicitly asked for a different radius via
    # --radius_km AND passed a value larger than the default AOI radius.
    search_radius_km = RADIUS_KM if args.radius_km == DEFAULT_RADIUS_KM else max(args.radius_km, RADIUS_KM)

    main(args.lat, args.lon, search_radius_km, args.years)
