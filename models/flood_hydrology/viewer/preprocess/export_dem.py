"""
Export DEM + FWC bathymetry → viewer/data/

Outputs:
  dem.bin          Float32 256×256, row 0 = north (same as raster convention)
  fwc_bed.bin      Float32 256×256, NaN outside Johns Lake (masked to largest lake component)
  geo_meta.json    Scene metadata incl. water_surface, lake centroid

Usage:
    python3 viewer/preprocess/export_dem.py
"""
import os, json
import numpy as np
import rasterio
import pandas as pd
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import label as ndi_label

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
FLOOD_DIR  = os.path.dirname(BASE_DIR)
DEM_PATH   = os.path.join(FLOOD_DIR, "dem", "data", "winter_garden_dem.tif")
FWC_PATH   = os.path.join(FLOOD_DIR, "dem", "data", "lake_bed_dem_fwc.tif")
MASK_PATH  = os.path.join(FLOOD_DIR, "dem", "data", "lake_mask.tif")
VOL_CSV    = os.path.join(FLOOD_DIR, "dem", "data", "lake_volume.csv")
OUT_DIR    = os.path.join(BASE_DIR, "data")
TARGET     = 256

os.makedirs(OUT_DIR, exist_ok=True)


def _subsample(arr, target):
    """Nearest-neighbour subsample arr (2-D) to (target × target). Preserves NaN."""
    rows, cols = arr.shape
    ri = np.round(np.linspace(0, rows - 1, target)).astype(int)
    ci = np.round(np.linspace(0, cols - 1, target)).astype(int)
    return arr[np.ix_(ri, ci)]


def _largest_component(mask):
    """Return boolean array keeping only the largest connected component of mask."""
    labeled, n = ndi_label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())[1:]
    best = np.argmax(sizes) + 1
    print(f"  Lake mask: {n} components, keeping largest ({sizes[best-1]:,} px). "
          f"Dropped: {sorted(sizes, reverse=True)[1:5]}")
    return labeled == best


def main():
    # ── DEM ─────────────────────────────────────────────────────────────
    with rasterio.open(DEM_PATH) as src:
        dem_full = src.read(1).astype(np.float32)
        transform  = src.transform
        crs        = src.crs
        nodata     = src.nodata
        rows_orig, cols_orig = src.shape
        cell_x_orig = float(abs(transform.a))
        cell_y_orig = float(abs(transform.e))
        bounds = src.bounds

    if nodata is not None:
        dem_full[dem_full == nodata] = np.nan

    # Fill tiny nodata border so wireframe has no holes
    if np.isnan(dem_full).any():
        col_means = np.nanmean(dem_full, axis=0)
        for c in range(dem_full.shape[1]):
            dem_full[np.isnan(dem_full[:, c]), c] = col_means[c]
        dem_full = np.where(np.isnan(dem_full), float(np.nanmean(dem_full)), dem_full)

    z_min    = float(np.min(dem_full))
    z_max    = float(np.max(dem_full))
    width_m  = cols_orig * cell_x_orig
    height_m = rows_orig * cell_y_orig

    # Downsample via rasterio (bilinear); from_bounds → north-up, row 0 = north, no flipud.
    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, TARGET, TARGET)
    dem_small = np.zeros((TARGET, TARGET), dtype=np.float32)
    reproject(dem_full, dem_small,
              src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs,
              resampling=Resampling.bilinear)

    dem_small.astype(np.float32).tofile(os.path.join(OUT_DIR, "dem.bin"))
    print(f"DEM:     {TARGET}×{TARGET}, z=[{z_min:.2f}, {z_max:.2f}] m")

    # ── Largest-component lake mask (868×868) ────────────────────────────
    with rasterio.open(MASK_PATH) as src:
        lake_mask_full = src.read(1)
    clean_mask = _largest_component(lake_mask_full > 0)  # bool, 868×868

    # ── FWC bathymetry masked to clean lake ─────────────────────────────
    with rasterio.open(FWC_PATH) as src:
        fwc_full = src.read(1).astype(np.float32)
        fwc_nd   = src.nodata
    if fwc_nd is not None:
        fwc_full[fwc_full == fwc_nd] = np.nan

    # Apply lake mask: NaN outside Johns Lake so JS can use isFinite() as lake test
    fwc_masked = np.where(clean_mask, fwc_full, np.nan)

    # Subsample with nearest-neighbour to preserve NaN pattern cleanly
    fwc_small = _subsample(fwc_masked, TARGET)
    fwc_small.astype(np.float32).tofile(os.path.join(OUT_DIR, "fwc_bed.bin"))
    valid_px = int(np.isfinite(fwc_small).sum())
    print(f"FWC bed: {TARGET}×{TARGET}, {valid_px:,} lake pixels, "
          f"depth range {float(np.nanmin(fwc_small)):.2f}–{float(np.nanmax(fwc_small)):.2f} m")

    # ── Water surface + lake centroid (from clean mask) ──────────────────
    water_surface = 28.74
    if os.path.exists(VOL_CSV):
        df = pd.read_csv(VOL_CSV)
        if "water_surface_m" in df.columns and len(df):
            water_surface = float(df["water_surface_m"].iloc[0])

    lake_rows, lake_cols = np.where(clean_mask)
    lake_x_center = float(lake_cols.mean() * cell_x_orig - width_m  / 2) if len(lake_cols) else 0.0
    lake_z_center = float(lake_rows.mean() * cell_y_orig - height_m / 2) if len(lake_rows) else 0.0

    meta = {
        "rows": TARGET, "cols": TARGET,
        "rows_orig": rows_orig, "cols_orig": cols_orig,
        "cell_x": width_m  / TARGET,
        "cell_y": height_m / TARGET,
        "cell_x_orig": cell_x_orig,
        "cell_y_orig": cell_y_orig,
        "z_min": z_min, "z_max": z_max,
        "width_m": width_m, "height_m": height_m,
        "water_surface":  water_surface,
        "lake_x_center":  lake_x_center,
        "lake_z_center":  lake_z_center,
        "origin_x": float(bounds.left),
        "origin_y": float(bounds.bottom),
        "crs": str(crs),
    }
    with open(os.path.join(OUT_DIR, "geo_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"geo_meta.json written  (water_surface={water_surface:.2f} m, "
          f"lake centre X={lake_x_center:.0f} Z={lake_z_center:.0f})")


if __name__ == "__main__":
    main()
