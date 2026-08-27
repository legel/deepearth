"""
Export site3's DEM → viewer/data/ (dem_site3.bin + geo_meta_site3.json)
=========================================================================
Site3 (Gee Creek gauge-matched validation site, 37km from the original CFX AOI) needs its own
terrain — it cannot be added as just another layer to the existing scene/geo_meta.json the way
site1/site2 test-area layers are, since those sit INSIDE the original 2x2km AOI's own coordinate
space and site3 does not. This mirrors export_dem.py's own logic (same TARGET=256 downsample,
same geo_meta.json schema) but reads site3's own raw DEM instead, with no SR417 bridge
correction (not relevant here) and no cross-file alignment concern (site3 has only one DEM
tree, unlike the original AOI's two-DEM-bounds-mismatch history).

Real bug worth knowing if this ever needs debugging again: site3's DEM (unlike the original
AOI's) has an inverted affine transform (positive y-resolution) — the same issue already fixed
in lidar/droplet_flow_test.py's build_ground_surface() and simulation/flood_sim_ian.py's
load_dem_for_sim(). Fixed here the same way: sort bounds before calling from_bounds().

Usage:
    python3 viewer/preprocess/export_dem_site3.py
"""
import os, sys, json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # viewer/
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from test_sites import get_site  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "data")
TARGET = 256
os.makedirs(OUT_DIR, exist_ok=True)

DEM_PATH = os.path.join(PROJ_DIR, "site3_gee_creek", "dem", "data", "site3_dem.tif")


def main():
    site = get_site("site3")

    with rasterio.open(DEM_PATH) as src:
        dem_full = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds

    if nodata is not None:
        dem_full[dem_full == nodata] = np.nan
    if np.isnan(dem_full).any():
        col_means = np.nanmean(dem_full, axis=0)
        for c in range(dem_full.shape[1]):
            dem_full[np.isnan(dem_full[:, c]), c] = col_means[c]
        dem_full = np.where(np.isnan(dem_full), float(np.nanmean(dem_full)), dem_full)

    # Normalize orientation — site3's DEM has bounds.bottom > bounds.top (inverted transform),
    # unlike the original AOI's DEM. See module docstring.
    true_left, true_right = sorted([bounds.left, bounds.right])
    true_bottom, true_top = sorted([bounds.bottom, bounds.top])
    width_m = true_right - true_left
    height_m = true_top - true_bottom

    z_min = float(np.min(dem_full))
    z_max = float(np.max(dem_full))

    dst_transform = from_bounds(true_left, true_bottom, true_right, true_top, TARGET, TARGET)
    dem_small = np.zeros((TARGET, TARGET), dtype=np.float32)
    reproject(dem_full, dem_small,
              src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs,
              resampling=Resampling.bilinear)

    z_min = float(min(z_min, np.min(dem_small)))
    z_max = float(max(z_max, np.max(dem_small)))

    dem_small.astype(np.float32).tofile(os.path.join(OUT_DIR, "dem_site3.bin"))
    print(f"DEM: {TARGET}x{TARGET}, z=[{z_min:.2f}, {z_max:.2f}] m")

    meta = {
        "site": "site3", "site_label": site["label"],
        "rows": TARGET, "cols": TARGET,
        "cell_x": width_m / TARGET, "cell_y": height_m / TARGET,
        "z_min": z_min, "z_max": z_max,
        "width_m": width_m, "height_m": height_m,
        "origin_x": float(true_left), "origin_y": float(true_bottom),
        "crs": str(crs),
        "gauge": {
            "site_no": site["gauge_site_no"],
            "documented_drainage_area_km2": site["documented_drainage_area_km2"],
            "delineated_drainage_area_km2": site["delineated_drainage_area_km2"],
        },
    }
    with open(os.path.join(OUT_DIR, "geo_meta_site3.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"geo_meta_site3.json written (width={width_m:.0f}m height={height_m:.0f}m)")


if __name__ == "__main__":
    main()
