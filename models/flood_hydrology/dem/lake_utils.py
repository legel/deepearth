"""
Shared lake mask + FWC interpolation utilities.

Call get_lake_mask_and_fwc() to get:
  - lake  : boolean (dem_h × dem_w) — correct lake boundary
            (OmniWaterMask majority-vote, NHD-supplemented for uncovered pixels)
  - fwc   : float32 (dem_h × dem_w) — FWC bathymetry with unsurveyed lake pixels
            filled by linear interpolation from real survey measurements
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def get_lake_mask_and_fwc(dem, dem_t, dem_crs, dem_res,
                           data_dir, s2_data_dir=None):
    """Return (lake, fwc_filled, wse, lake_source).

    Parameters
    ----------
    dem        : (H, W) float32 array of the terrain DEM (NAVD88 m)
    dem_t      : rasterio Affine transform of the DEM grid
    dem_crs    : rasterio CRS of the DEM grid
    dem_res    : DEM pixel size in metres
    data_dir   : path to dem/data/ directory
    s2_data_dir: path to sentinel2/data/ directory (optional; auto-detected if None)
    """
    dem_h, dem_w = dem.shape

    if s2_data_dir is None:
        s2_data_dir = os.path.join(os.path.dirname(data_dir), "sentinel2", "data")

    # ── 1. OmniWaterMask majority-vote lake boundary ─────────────────────────
    owm_files = sorted(glob.glob(os.path.join(s2_data_dir, "omniwatermask_mask_*.tif")))
    if owm_files:
        n = len(owm_files)
        vote = np.zeros((dem_h, dem_w), dtype=np.float32)
        for fp in owm_files:
            with rasterio.open(fp) as src:
                out = np.zeros((dem_h, dem_w), dtype=np.float32)
                reproject(source=src.read(1).astype(np.float32), destination=out,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dem_t, dst_crs=dem_crs,
                          resampling=Resampling.nearest)
                vote += out
        lake = (vote >= n / 2).astype(bool)

        # Supplement below-majority pixels with NHD official boundary
        # (partial-vote pixels in rows 40-85 would create a notch artifact otherwise)
        nhd_path = os.path.join(data_dir, "lake_mask_nhd.tif")
        if os.path.exists(nhd_path):
            nhd_out = np.zeros((dem_h, dem_w), dtype=np.float32)
            with rasterio.open(nhd_path) as src:
                reproject(source=src.read(1).astype(np.float32), destination=nhd_out,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dem_t, dst_crs=dem_crs,
                          resampling=Resampling.nearest)
            lake = lake | ((vote < n / 2) & (nhd_out > 0))
        lake_source = f"OmniWaterMask consensus ({n} scenes) + NHD supplement"
    else:
        # Fallback: NHD only
        nhd_path = os.path.join(data_dir, "lake_mask_nhd.tif")
        if os.path.exists(nhd_path):
            nhd_out = np.zeros((dem_h, dem_w), dtype=np.float32)
            with rasterio.open(nhd_path) as src:
                reproject(source=src.read(1).astype(np.float32), destination=nhd_out,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dem_t, dst_crs=dem_crs,
                          resampling=Resampling.nearest)
            lake = nhd_out > 0
            lake_source = "NHD official (fallback)"
        else:
            s2_path = os.path.join(data_dir, "lake_mask_s2.tif")
            s2_out = np.zeros((dem_h, dem_w), dtype=np.float32)
            with rasterio.open(s2_path) as src:
                reproject(source=src.read(1).astype(np.float32), destination=s2_out,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dem_t, dst_crs=dem_crs,
                          resampling=Resampling.nearest)
            lake = s2_out > 0
            lake_source = "S2 MNDWI consensus (fallback)"

    # ── 2. Water surface elevation ───────────────────────────────────────────
    from scipy.ndimage import binary_erosion
    shore = lake & ~binary_erosion(lake, iterations=2)
    wse = float(dem[shore].mean()) if shore.any() else float(dem[lake].mean())

    # ── 3. FWC bathymetry with unsurveyed pixels interpolated ───────────────
    fwc_path = os.path.join(data_dir, "lake_bed_dem_fwc.tif")
    if not os.path.exists(fwc_path):
        fwc_path = os.path.join(data_dir, "lake_bed_dem_estimated.tif")

    with rasterio.open(fwc_path) as src:
        fwc = src.read(1).astype(np.float32)

    # Pixels where FWC ≈ DEM were never measured — the file uses DEM as filler.
    # Detect them as (dem - fwc) ≤ 0.1 m within the lake mask.
    fwc_surveyed   = lake & ((dem - fwc) > 0.1)
    fwc_unsurveyed = lake & ~fwc_surveyed

    if fwc_unsurveyed.any() and fwc_surveyed.any():
        from scipy.interpolate import griddata
        y_k, x_k = np.where(fwc_surveyed)
        z_k = fwc[fwc_surveyed]
        y_f, x_f = np.where(fwc_unsurveyed)
        z_filled = griddata((y_k, x_k), z_k, (y_f, x_f), method="linear")
        nan_mask = np.isnan(z_filled)
        if nan_mask.any():
            z_filled[nan_mask] = griddata(
                (y_k, x_k), z_k,
                (y_f[nan_mask], x_f[nan_mask]), method="nearest")
        fwc = fwc.copy()
        fwc[fwc_unsurveyed] = z_filled
        print(f"  FWC interpolated for {fwc_unsurveyed.sum()} unsurveyed lake px "
              f"({fwc_unsurveyed.sum()*dem_res**2/1e4:.1f} ha)")

    return lake, fwc, wse, lake_source
