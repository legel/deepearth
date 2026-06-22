"""
Export lake voxel grid → viewer/data/voxels.bin

Binary format:
  [0:4]   magic 'VOXL'
  [4:8]   uint32  n_voxels
  [8:12]  float32 cell_x (m)
  [12:16] float32 cell_y (m)
  [16:20] float32 z_resolution (m)
  [20:24] float32 water_surface (m NAVD88)
  [24:28] float32 dem_z_min (m NAVD88)
  [28:32] uint32  n_zlayers
  [32:]   n_voxels × (row:uint16, col:uint16, z_layer:uint16) = 6 bytes each

Usage:
    python3 viewer/preprocess/export_voxels.py
"""
import os, sys, struct
import numpy as np
import rasterio
from scipy.ndimage import label as ndi_label


def _largest_component(mask):
    labeled, n = ndi_label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = np.bincount(labeled.ravel())[1:]
    best = np.argmax(sizes) + 1
    print(f"  Lake mask: {n} components, keeping largest ({sizes[best-1]:,} px), "
          f"dropping {n-1} smaller")
    return labeled == best

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOD_DIR = os.path.dirname(BASE_DIR)
DEM_DIR   = os.path.join(FLOOD_DIR, "dem")
S2_DIR    = os.path.join(FLOOD_DIR, "sentinel2", "data")
OUT_DIR   = os.path.join(BASE_DIR, "data")
Z_RES     = 0.5  # metres per z-layer

os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, DEM_DIR)


def main():
    from lake_volume import build_voxel_grid, load_tif
    from lake_utils import get_lake_mask_and_fwc

    dem, (transform, cell_m) = load_tif("winter_garden_dem.tif")
    if dem is None:
        sys.exit("DEM not found — run dem/dem_download.py first")

    with rasterio.open(os.path.join(DEM_DIR, "data", "winter_garden_dem.tif")) as src:
        dem_crs = src.crs

    print("Building lake mask + FWC bathymetry …")
    lake_bool, lake_bed, _ws, mask_label = get_lake_mask_and_fwc(
        dem, transform, dem_crs, cell_m,
        os.path.join(DEM_DIR, "data"),
        s2_data_dir=S2_DIR,
    )
    print(f"  Lake mask: {mask_label}")
    lake_bool = _largest_component(lake_bool)

    print("Building voxel grid …")
    voxels, water_surface, n_zlayers = build_voxel_grid(
        lake_bool, dem, lake_bed, cell_m, z_resolution=Z_RES
    )
    if voxels is None:
        sys.exit("Voxel grid is empty — check lake_bed raster")

    rows_idx, cols_idx, z_idx = np.where(voxels)
    n_voxels = len(rows_idx)
    dem_z_min = float(np.nanmin(dem))

    out_path = os.path.join(OUT_DIR, "voxels.bin")
    with open(out_path, "wb") as f:
        header = struct.pack(
            "<4sIfffffI",
            b"VOXL",
            n_voxels,
            float(cell_m),   # cell_x
            float(cell_m),   # cell_y
            float(Z_RES),
            float(water_surface),
            dem_z_min,
            n_zlayers,
        )
        f.write(header)
        # Pack (row, col, z) as uint16 triples, row-major per voxel
        triples = np.column_stack([
            rows_idx.astype(np.uint16),
            cols_idx.astype(np.uint16),
            z_idx.astype(np.uint16),
        ])
        f.write(triples.tobytes())

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Voxels exported: {n_voxels:,} voxels, {n_zlayers} z-layers, "
          f"cell={cell_m:.2f}m, z_res={Z_RES}m, water_surface={water_surface:.2f}m")
    print(f"  → viewer/data/voxels.bin ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
