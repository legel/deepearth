"""
Export DEM → viewer/data/

Outputs:
  dem.bin          Float32 256x256, row 0 = north (same as raster convention)
  geo_meta.json    Scene metadata (grid shape, cell size, elevation range, CRS)

Unlike flood_hydrology/viewer/preprocess/export_dem.py, there is no lake at
this AOI, so no FWC bathymetry / lake-mask / water-surface fields are
produced here.

If lidar/data/lidar_bridge_deck_1m.tif exists (see lidar/build_lidar_pointcloud.py), the DEM is
also patched at the two SR417 bridge crossings with the actual point-cloud bridge-deck surface
before export — the bare-earth DEM otherwise drops the highway ~7.5-8.4m to grade level there
(bridge-deck LiDAR returns get classified non-ground and excluded from a bare-earth DTM; see
lidar/data/BRIDGE_VALIDATION.md). This makes the viewer's actual Surface/Wireframe terrain mesh
show the real elevated overpass, not just the separate "LiDAR Bridge Correction" TIN meshes.

The correction source is deliberately the class-17 (bridge-deck) point median, gap-filled and
Gaussian-smoothed — NOT a raw max-z-per-cell DSM over a wide road buffer (an earlier version of
this correction did that and produced a visibly bumpy highway: max-z picks up guardrails/lamp
posts/signage, not just the pavement, and a hard buffer-edge mask has no blending). Using only
the actual bridge-deck classification naturally confines the correction to the two decks
themselves, with a smooth, gap-free surface and no separate road-buffer masking needed.

Usage:
    python3 viewer/preprocess/export_dem.py
"""
import os, json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
PROJ_DIR  = os.path.dirname(BASE_DIR)
DEM_PATH  = os.path.join(PROJ_DIR, "dem", "data", "sr417_corridor_dem.tif")
LIDAR_BRIDGE_DECK_PATH = os.path.join(PROJ_DIR, "lidar", "data", "lidar_bridge_deck_1m.tif")
OUT_DIR   = os.path.join(BASE_DIR, "data")
TARGET    = 256

os.makedirs(OUT_DIR, exist_ok=True)


def apply_sr417_bridge_correction(dem_small, dst_transform, crs):
    """Patch dem_small at the two SR417 bridge decks with the smoothed point-cloud
    bridge-deck-only surface (lidar_bridge_deck_1m.tif). No-ops (returns dem_small unchanged)
    if that raster isn't available yet."""
    if not os.path.exists(LIDAR_BRIDGE_DECK_PATH):
        print("  (skipping SR417 bridge correction — lidar_bridge_deck_1m.tif not found; "
              "run lidar/build_lidar_pointcloud.py first)")
        return dem_small

    with rasterio.open(LIDAR_BRIDGE_DECK_PATH) as src:
        deck_full = src.read(1).astype(np.float32)
        deck_transform, deck_crs = src.transform, src.crs

    deck_small = np.full(dem_small.shape, np.nan, dtype=np.float32)
    reproject(deck_full, deck_small,
              src_transform=deck_transform, src_crs=deck_crs,
              dst_transform=dst_transform, dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    apply_mask = np.isfinite(deck_small)
    n_cells = int(apply_mask.sum())
    if n_cells == 0:
        print("  SR417 bridge correction: 0 cells patched (bridge-deck raster empty?)")
        return dem_small

    max_diff = float(np.nanmax(deck_small[apply_mask] - dem_small[apply_mask]))
    corrected = dem_small.copy()
    corrected[apply_mask] = deck_small[apply_mask]
    print(f"  SR417 bridge correction: {n_cells} cells patched with smoothed bridge-deck "
          f"surface (max +{max_diff:.2f}m over bare-earth DEM)")
    return corrected


def main():
    with rasterio.open(DEM_PATH) as src:
        dem_full = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        rows_orig, cols_orig = src.shape
        cell_x_orig = float(abs(transform.a))
        cell_y_orig = float(abs(transform.e))
        bounds = src.bounds

    if nodata is not None:
        dem_full[dem_full == nodata] = np.nan

    if np.isnan(dem_full).any():
        col_means = np.nanmean(dem_full, axis=0)
        for c in range(dem_full.shape[1]):
            dem_full[np.isnan(dem_full[:, c]), c] = col_means[c]
        dem_full = np.where(np.isnan(dem_full), float(np.nanmean(dem_full)), dem_full)

    z_min = float(np.min(dem_full))
    z_max = float(np.max(dem_full))
    width_m = cols_orig * cell_x_orig
    height_m = rows_orig * cell_y_orig

    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, TARGET, TARGET)
    dem_small = np.zeros((TARGET, TARGET), dtype=np.float32)
    reproject(dem_full, dem_small,
              src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs,
              resampling=Resampling.bilinear)

    dem_small = apply_sr417_bridge_correction(dem_small, dst_transform, crs)
    z_min = float(min(z_min, np.min(dem_small)))
    z_max = float(max(z_max, np.max(dem_small)))

    dem_small.astype(np.float32).tofile(os.path.join(OUT_DIR, "dem.bin"))
    print(f"DEM: {TARGET}x{TARGET}, z=[{z_min:.2f}, {z_max:.2f}] m")

    meta = {
        "rows": TARGET, "cols": TARGET,
        "rows_orig": rows_orig, "cols_orig": cols_orig,
        "cell_x": width_m / TARGET,
        "cell_y": height_m / TARGET,
        "cell_x_orig": cell_x_orig,
        "cell_y_orig": cell_y_orig,
        "z_min": z_min, "z_max": z_max,
        "width_m": width_m, "height_m": height_m,
        "origin_x": float(bounds.left),
        "origin_y": float(bounds.bottom),
        "crs": str(crs),
    }
    with open(os.path.join(OUT_DIR, "geo_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"geo_meta.json written  (width={width_m:.0f}m height={height_m:.0f}m)")


if __name__ == "__main__":
    main()
