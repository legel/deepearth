"""
Sentinel-2 Water Body Detection — MNDWI + Lake Time Series
===========================================================
Computes water body masks and lake area/level time series from downloaded
Sentinel-2 scenes using the Modified Normalized Difference Water Index.

MNDWI = (Green − SWIR) / (Green + SWIR)
  Band mapping: Green = B03 (10m), SWIR = B11 (20m, resampled to 10m)
  Threshold: MNDWI > 0 → water  (literature standard; McFeeters 1996 + Xu 2006)

MNDWI is preferred over NDWI for suburban/residential areas because the SWIR
band suppresses false positives from built surfaces (roads, rooftops) that
confuse the simpler NIR-based NDWI.

For each downloaded scene the script:
  1. Loads B03, B11 bands
  2. Computes MNDWI
  3. Thresholds to binary water mask
  4. Vectorizes lake polygons (connected water regions > min_area)
  5. Extracts lake water-surface elevation from DEM for each date
  6. Computes lake area [ha] and perimeter [m]

Seasonal analysis:
  - Compares wet-season (Jun–Sep) vs. dry-season (Jan–Mar) lake extents
  - Quantifies lake area change and water-level variation

Outputs (saved under sentinel2/data/):
    mndwi_{date}.tif                — MNDWI raster (float32, −1 to 1)
    water_mask_{date}.tif           — binary water mask (uint8)
    water_mask_{date}.geojson       — vectorized lake polygons
    lake_timeseries.csv             — date, lake_area_ha, water_surface_elev_m
    water_bodies_comparison.png     — side-by-side wet vs dry season maps

Usage:
    python3 sentinel2/s2_water_bodies.py
    python3 sentinel2/s2_water_bodies.py --threshold 0.0 --min_area_ha 0.1
"""

import os
import sys
import json
import argparse
import warnings
import glob
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DEM_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "dem", "data")

DEFAULT_THRESHOLD  = 0.0    # MNDWI > 0 → water
DEFAULT_MIN_AREA   = 0.05   # minimum lake area [ha] to keep
SCALE_FACTOR       = 10000  # Sentinel-2 L2A reflectance is stored ×10000


def load_band(tif_path):
    """Load a single-band GeoTIFF, return (array, profile)."""
    import rasterio
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def compute_mndwi(green, swir, scale=SCALE_FACTOR):
    """MNDWI = (Green − SWIR) / (Green + SWIR). Input arrays are DN (×10000)."""
    g = green / scale
    s = swir  / scale
    denom = g + s
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = np.where(denom > 0, (g - s) / denom, 0.0)
    return mndwi.astype(np.float32)


def compute_ndwi(green, nir, scale=SCALE_FACTOR):
    """NDWI = (Green − NIR) / (Green + NIR). Alternative to MNDWI."""
    g = green / scale
    n = nir   / scale
    denom = g + n
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(denom > 0, (g - n) / denom, 0.0)
    return ndwi.astype(np.float32)


def vectorize_water_mask(mask, profile, min_area_ha=DEFAULT_MIN_AREA, date_str=""):
    """Convert binary water mask to GeoJSON FeatureCollection."""
    import rasterio.features
    from shapely.geometry import shape as shp_shape, mapping

    cell_area_m2 = abs(profile["transform"].a * profile["transform"].e)
    min_cells = max(1, int((min_area_ha * 1e4) / cell_area_m2))

    shapes = list(rasterio.features.shapes(
        mask.astype(np.uint8),
        mask=(mask > 0).astype(np.uint8),
        transform=profile["transform"],
    ))

    features = []
    total_area_ha = 0.0
    for geom_dict, val in shapes:
        if val != 1:
            continue
        geom = shp_shape(geom_dict)
        area_ha = geom.area / 1e4 if "4326" not in str(profile.get("crs","")) else geom.area * 1.23e9 / 1e4
        # rough area estimate; precise if CRS is projected
        pixel_count = int(np.sum(rasterio.features.rasterize(
            [(geom_dict, 1)], out_shape=mask.shape,
            transform=profile["transform"], fill=0)))
        area_ha_exact = pixel_count * cell_area_m2 / 1e4
        if area_ha_exact < min_area_ha:
            continue
        total_area_ha += area_ha_exact
        features.append({
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {
                "date": date_str,
                "area_ha": round(area_ha_exact, 3),
                "pixel_count": pixel_count,
            }
        })

    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(profile.get("crs","EPSG:4326"))}},
        "features": features,
    }
    return fc, total_area_ha


def get_lake_water_surface_elevation(water_mask, profile, dem_path):
    """
    Extract mean water-surface elevation from DEM at lake pixels.
    The DEM is hydro-flattened, so all pixels within a lake have the same
    constant value = water surface elevation.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    if not os.path.exists(dem_path):
        return None

    with rasterio.open(dem_path) as dem_src:
        # Reproject/resample DEM to match Sentinel-2 grid
        dem_reproj = np.zeros_like(water_mask, dtype=np.float32)
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_reproj,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile.get("crs"),
            resampling=Resampling.bilinear,
        )

    lake_elevs = dem_reproj[water_mask > 0]
    if len(lake_elevs) == 0:
        return None
    return float(np.median(lake_elevs))


def save_tif(array, profile, out_path, dtype=None):
    import rasterio
    if dtype:
        profile = profile.copy()
        profile.update(dtype=dtype)
    profile.update(count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(profile["dtype"]), 1)


def process_all_scenes(threshold=DEFAULT_THRESHOLD, min_area_ha=DEFAULT_MIN_AREA):
    """Process all downloaded Sentinel-2 scenes in sentinel2/data/."""
    dem_path = os.path.join(DEM_DATA_DIR, "winter_garden_dem.tif")

    # Find all B03 scenes
    b03_files = sorted(glob.glob(os.path.join(DATA_DIR, "s2_*_B03.tif")))
    if not b03_files:
        sys.exit(f"No Sentinel-2 scenes found in {DATA_DIR}\n"
                 "  Run s2_download.py first.")

    print(f"Found {len(b03_files)} scene(s) to process")
    records = []

    for b03_path in b03_files:
        date_str = os.path.basename(b03_path).replace("s2_","").replace("_B03.tif","")
        b11_path = b03_path.replace("_B03.tif", "_B11.tif")
        b08_path = b03_path.replace("_B03.tif", "_B08.tif")

        if not os.path.exists(b11_path):
            print(f"  [{date_str}] Missing B11; skipping")
            continue

        print(f"\nProcessing {date_str} …")
        green, profile = load_band(b03_path)
        swir,  _       = load_band(b11_path)

        # Resample SWIR to match Green if shapes differ (B11 is 20m native)
        if swir.shape != green.shape:
            from scipy.ndimage import zoom
            scale_r = green.shape[0] / swir.shape[0]
            scale_c = green.shape[1] / swir.shape[1]
            swir = zoom(swir, (scale_r, scale_c), order=1)

        mndwi      = compute_mndwi(green, swir)
        water_mask = (mndwi > threshold).astype(np.uint8)

        # Also compute NDWI if B08 available
        if os.path.exists(b08_path):
            nir, _ = load_band(b08_path)
            ndwi = compute_ndwi(green, nir)

        water_pct = 100.0 * water_mask.mean()
        print(f"  MNDWI range: {mndwi.min():.3f} – {mndwi.max():.3f}")
        print(f"  Water pixels: {water_mask.sum():,} ({water_pct:.1f}% of AOI)")

        # Save rasters
        mndwi_path = os.path.join(DATA_DIR, f"mndwi_{date_str}.tif")
        mask_path  = os.path.join(DATA_DIR, f"water_mask_{date_str}.tif")
        save_tif(mndwi,      profile, mndwi_path, dtype="float32")
        save_tif(water_mask, profile, mask_path,  dtype="uint8")
        print(f"  Saved MNDWI → {os.path.basename(mndwi_path)}")
        print(f"  Saved mask  → {os.path.basename(mask_path)}")

        # Vectorize
        fc, total_area_ha = vectorize_water_mask(water_mask, profile, min_area_ha, date_str)
        geojson_path = mask_path.replace(".tif", ".geojson")
        with open(geojson_path, "w") as f:
            json.dump(fc, f)
        n_lakes = len(fc["features"])
        print(f"  Lakes: {n_lakes} polygon(s), {total_area_ha:.2f} ha total")

        # Water surface elevation from DEM
        wse = get_lake_water_surface_elevation(water_mask, profile, dem_path)
        if wse is not None:
            print(f"  Water surface elevation (DEM): {wse:.2f} m NAVD88")

        records.append({
            "date":              date_str,
            "n_lake_polygons":   n_lakes,
            "total_lake_area_ha": round(total_area_ha, 3),
            "water_pct_aoi":     round(water_pct, 2),
            "water_surface_elev_m": wse,
            "mndwi_mean":        round(float(mndwi.mean()), 4),
            "mndwi_max":         round(float(mndwi.max()), 4),
        })

    if not records:
        print("No scenes successfully processed.")
        return

    ts = pd.DataFrame(records).sort_values("date")
    ts_path = os.path.join(DATA_DIR, "lake_timeseries.csv")
    ts.to_csv(ts_path, index=False)
    print(f"\nSaved lake time series → {ts_path}")

    # Summary
    print("\n── Lake Time Series Summary ──────────────────────────────────")
    print(ts[["date","total_lake_area_ha","water_surface_elev_m"]].to_string(index=False))

    # Simple seasonal comparison if we have dry + wet scenes
    if len(ts) >= 2:
        lake_max  = ts.total_lake_area_ha.max()
        lake_min  = ts.total_lake_area_ha.min()
        lake_diff = lake_max - lake_min
        print(f"\n  Seasonal lake area variation: {lake_min:.2f} – {lake_max:.2f} ha ({lake_diff:.2f} ha swing)")

    make_comparison_plot(ts, DATA_DIR)


def make_comparison_plot(ts, data_dir):
    """Quick matplotlib visualization of water extent per scene."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import rasterio

        n = len(ts)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, ts.iterrows()):
            mask_path = os.path.join(data_dir, f"water_mask_{row['date']}.tif")
            if not os.path.exists(mask_path):
                continue
            with rasterio.open(mask_path) as src:
                img = src.read(1)
            ax.imshow(img, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{row['date']}\n{row['total_lake_area_ha']:.1f} ha", fontsize=9)
            ax.axis("off")

        plt.suptitle("Sentinel-2 Water Mask — Winter Garden FL", fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(data_dir, "water_bodies_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot → {out_path}")
    except Exception as exc:
        print(f"  (Plot skipped: {exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MNDWI water masks from Sentinel-2 scenes")
    parser.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD,  help="MNDWI threshold for water (default 0.0)")
    parser.add_argument("--min_area_ha",  type=float, default=DEFAULT_MIN_AREA,   help="Minimum lake area to keep [ha]")
    args = parser.parse_args()
    process_all_scenes(args.threshold, args.min_area_ha)
