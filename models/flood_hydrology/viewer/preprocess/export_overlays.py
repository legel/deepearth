"""
Export overlay textures aligned to the DEM grid → viewer/data/

Outputs (512×512 PNG, north-up, covering the full DEM extent):
  naip_rgb.png   — NAIP 2021 true-color aerial (EPSG:26917 → EPSG:5070)
  ssurgo.png     — SSURGO soil types colorized by Hydrologic Soil Group
  lake_mask.png  — lake extent (blue on transparent)

Usage:
    python3 viewer/preprocess/export_overlays.py
"""
import os, sys, json
import numpy as np
import rasterio
import pandas as pd
import matplotlib.cm as mpl_cm
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy.ndimage import label as ndi_label, distance_transform_edt
from PIL import Image

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOD_DIR = os.path.dirname(BASE_DIR)
DEM_DIR   = os.path.join(FLOOD_DIR, "dem", "data")
SOIL_DIR  = os.path.join(FLOOD_DIR, "soil", "data")
OUT_DIR   = os.path.join(BASE_DIR, "data")
SIZE      = 512

os.makedirs(OUT_DIR, exist_ok=True)

NODATA_RGBA = (0, 0, 0, 0)


def get_dem_bounds():
    with rasterio.open(os.path.join(DEM_DIR, "winter_garden_dem.tif")) as src:
        return src.bounds, src.crs


def export_naip(bounds, dem_crs):
    naip_path = os.path.join(SOIL_DIR, "naip_2021_RGB.tif")
    if not os.path.exists(naip_path):
        print("  NAIP not found — skipping naip_rgb.png")
        return

    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    rgb_dst = np.zeros((3, SIZE, SIZE), dtype=np.uint8)

    with rasterio.open(naip_path) as src:
        for band in range(3):
            band_data = src.read(band + 1).astype(np.float32)
            dst_band = np.zeros((SIZE, SIZE), dtype=np.float32)
            reproject(
                band_data,
                dst_band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )
            # from_bounds gives north-up transform → row 0 = north edge; no flip needed.
            # Three.js flipY=true re-flips on GPU upload so UV(0,1) hits row 0 (north). ✓
            rgb_dst[band] = np.clip(dst_band, 0, 255).astype(np.uint8)

    img = Image.fromarray(np.moveaxis(rgb_dst, 0, -1), mode="RGB")
    out = os.path.join(OUT_DIR, "naip_rgb.png")
    img.save(out)
    print(f"  naip_rgb.png saved ({SIZE}×{SIZE})")


def export_ssurgo(bounds, dem_crs):
    mukey_path = os.path.join(SOIL_DIR, "mukey_map.tif")
    legend_path = os.path.join(SOIL_DIR, "mukey_map_legend.csv")
    if not os.path.exists(mukey_path):
        print("  SSURGO mukey_map not found — skipping ssurgo.png")
        return

    # Group mukeys by canonical soil-series name so that slope/density variants
    # of the same series share one color. Rules (order matters):
    #   "water" anywhere in name → "Water"
    #   name starts with "Candler"  → "Candler"  (covers both "Candler sand" and
    #                                               "Candler fine sand" variants)
    #   otherwise → base name before first comma
    def _canonical(muname):
        base = muname.split(',')[0].strip()
        if 'water' in base.lower():
            return 'Water'
        if base.lower().startswith('candler'):
            return 'Candler sand'
        return base

    # First pass: collect groups in CSV order (preserves deterministic color assignment)
    groups = {}   # canonical label → list of mukey_int
    if os.path.exists(legend_path):
        df = pd.read_csv(legend_path)
        for _, row in df.iterrows():
            key   = int(row["mukey_int"])
            label = _canonical(str(row.get("muname", "")).strip())
            groups.setdefault(label, []).append(key)

    # Second pass: assign one tab10 color per group; Water gets a dedicated blue
    key_to_rgba   = {}
    legend_entries = []
    color_idx = 1   # skip index 0 (blue) — reserved for Water
    for label, mukeys in groups.items():
        if label == 'Water':
            r, g, b = 40, 120, 200
        else:
            rgba_f = mpl_cm.tab10(color_idx / 10)
            r, g, b = int(rgba_f[0] * 255), int(rgba_f[1] * 255), int(rgba_f[2] * 255)
            color_idx += 1
        for mk in mukeys:
            key_to_rgba[mk] = (r, g, b, 200)             # semi-transparent in texture
        legend_entries.append({"label": label, "rgba": [r, g, b, 255]})

    with rasterio.open(mukey_path) as src:
        mukey = src.read(1).astype(np.int16)
        dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
        mukey_dst = np.zeros((SIZE, SIZE), dtype=np.float32)
        reproject(
            mukey.astype(np.float32),
            mukey_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )
    mukey_dst = mukey_dst.astype(np.int16)  # row 0 = north, no flip needed

    # Fill nodata (mukey_int=0) pixels with the nearest valid soil value so they
    # don't render as a transparent gap (dark scene background showing through).
    nodata_mask = mukey_dst == 0
    if nodata_mask.any():
        nearest = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
        mukey_dst[nodata_mask] = mukey_dst[nearest[0][nodata_mask], nearest[1][nodata_mask]]
        print(f"  Filled {int(nodata_mask.sum())} nodata pixels via nearest-neighbour")

    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for key, color in key_to_rgba.items():
        rgba[mukey_dst == key] = color
    # nodata (0) stays transparent

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(os.path.join(OUT_DIR, "ssurgo.png"))
    print(f"  ssurgo.png saved ({SIZE}×{SIZE})")

    legend_out = os.path.join(OUT_DIR, "ssurgo_legend.json")
    with open(legend_out, "w") as f:
        json.dump(legend_entries, f, indent=2)
    print(f"  ssurgo_legend.json saved ({len(legend_entries)} entries)")


def export_lake_mask(bounds, dem_crs):
    mask_path = os.path.join(DEM_DIR, "lake_mask.tif")
    if not os.path.exists(mask_path):
        print("  lake_mask.tif not found — skipping lake_mask.png")
        return

    with rasterio.open(mask_path) as src:
        mask_full = src.read(1)

    # Keep only the largest connected component (Johns Lake).
    # This drops adjacent ponds/canals that OmniWaterMask picked up.
    labeled, n = ndi_label(mask_full > 0)
    sizes = np.bincount(labeled.ravel())[1:]
    best_label = np.argmax(sizes) + 1
    clean_full = (labeled == best_label).astype(np.float32)
    print(f"  lake_mask: kept component {best_label} ({sizes[best_label-1]:,} px), "
          f"dropped {n-1} smaller bodies")

    with rasterio.open(mask_path) as src:
        dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
        mask_dst = np.zeros((SIZE, SIZE), dtype=np.float32)
        reproject(
            clean_full,
            mask_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )
    # row 0 = north, no flip needed

    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    water = mask_dst > 0.5
    rgba[water] = (30, 100, 220, 200)  # semi-transparent blue

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(os.path.join(OUT_DIR, "lake_mask.png"))
    print(f"  lake_mask.png saved ({SIZE}×{SIZE})")


def main():
    bounds, dem_crs = get_dem_bounds()
    print(f"DEM bounds: {bounds}  CRS: {dem_crs}")
    export_naip(bounds, dem_crs)
    export_ssurgo(bounds, dem_crs)
    export_lake_mask(bounds, dem_crs)
    print("Done — viewer/data/ overlays ready")


if __name__ == "__main__":
    main()
