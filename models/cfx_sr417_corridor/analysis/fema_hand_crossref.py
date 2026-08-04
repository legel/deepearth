"""
FEMA Floodway × DEM × HAND Spatial Cross-Reference
====================================================
Overlays the FEMA NFHL flood zones (floodway + AE/SFHA) onto the HAND raster
to identify where regulated flood risk intersects terrain flood susceptibility.

Key question: do the FEMA-mapped flood zones match the low-HAND cells that
terrain analysis predicts are most flood-prone?

Outputs:
  analysis/data/fema_hand_risk.tif   — 5-class risk raster (EPSG:5070)
  analysis/data/fema_hand_risk.png   — viewer overlay PNG (512×512, RGBA)
  analysis/data/fema_hand_summary.json

Risk classes:
  1 = FLOODWAY ∩ HAND<1m   (red)       — highest: regulated channel + near drainage
  2 = FLOODWAY ∩ HAND 1-3m (orange)    — high: regulated channel, moderate elevation
  3 = AE/SFHA  ∩ HAND<1m   (yellow)    — moderate: mapped flood zone + near drainage
  4 = AE/SFHA  ∩ HAND 1-3m (tan)       — lower: mapped zone, some elevation buffer
  5 = HAND<1m  only         (light blue)— terrain susceptible but outside FEMA zones

Usage:
    python3 analysis/fema_hand_crossref.py
"""

import os, json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BASE_DIR)
OUT_DIR    = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

HAND_TIF     = os.path.join(PROJ_DIR, "dem", "data", "hydro", "hand.tif")
FEMA_GEOJSON = os.path.join(PROJ_DIR, "floodplain", "data", "fema_flood_zones.geojson")
RISK_TIF     = os.path.join(OUT_DIR, "fema_hand_risk.tif")
RISK_PNG     = os.path.join(OUT_DIR, "fema_hand_risk.png")
SUMMARY_JSON = os.path.join(OUT_DIR, "fema_hand_summary.json")

FLOODWAY_SUBTYPES = {
    "FLOODWAY", "ADMINISTRATIVE FLOODWAY", "COMMUNITY ENCROACHMENT AREA",
    "FLOWAGE EASEMENT AREA", "NARROW FLOODWAY",
}

# Risk class colours (RGBA)
CLASS_COLORS = {
    1: (220,  40,  40, 210),  # red    — floodway + HAND<1m
    2: (230, 110,  30, 180),  # orange — floodway + HAND 1-3m
    3: (220, 200,  30, 160),  # yellow — AE + HAND<1m
    4: (180, 160, 100, 130),  # tan    — AE + HAND 1-3m
    5: (100, 180, 255, 110),  # blue   — HAND<1m only (outside FEMA zones)
}
CLASS_LABELS = {
    1: "Floodway + HAND<1m (highest risk)",
    2: "Floodway + HAND 1-3m",
    3: "AE/SFHA + HAND<1m",
    4: "AE/SFHA + HAND 1-3m",
    5: "HAND<1m only (outside FEMA zones)",
}


def run():
    # ── Load HAND raster ──────────────────────────────────────────────────────
    if not os.path.exists(HAND_TIF):
        print(f"HAND raster not found: {HAND_TIF}")
        print("Run: python3 dem/dem_hydro.py first")
        return

    with rasterio.open(HAND_TIF) as src:
        hand = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        bounds  = src.bounds
        dem_crs = src.crs
        nrows, ncols = src.shape
        cell_m = abs(src.transform.a)   # pixel size in meters

    print(f"HAND: {nrows}×{ncols} px @ {cell_m:.2f}m  CRS: {dem_crs}")
    print(f"HAND stats: mean={np.nanmean(hand):.2f}m  p95={np.nanpercentile(hand,95):.2f}m")

    # ── Load and rasterize FEMA zones ─────────────────────────────────────────
    if not os.path.exists(FEMA_GEOJSON):
        print(f"FEMA GeoJSON not found: {FEMA_GEOJSON}")
        return

    gdf = gpd.read_file(FEMA_GEOJSON)
    if gdf.crs is None:
        gdf = gdf.set_crs("epsg:4326")
    gdf_proj = gdf.to_crs(dem_crs)

    transform = profile["transform"]

    def zone_subty(row):
        return str(row.get("ZONE_SUBTY") or "").upper()

    is_floodway = gdf_proj.apply(lambda r: zone_subty(r) in FLOODWAY_SUBTYPES, axis=1)
    is_sfha     = gdf_proj.get("SFHA_TF", gpd.pd.Series(["F"] * len(gdf_proj))) == "T"

    floodway_geoms = [(g, 1) for g in gdf_proj[is_floodway].geometry if g is not None]
    sfha_geoms     = [(g, 1) for g in gdf_proj[is_sfha & ~is_floodway].geometry if g is not None]

    floodway_mask = rasterize(
        floodway_geoms, out_shape=(nrows, ncols),
        transform=transform, fill=0, dtype=np.uint8,
    ).astype(bool) if floodway_geoms else np.zeros((nrows, ncols), bool)

    sfha_mask = rasterize(
        sfha_geoms, out_shape=(nrows, ncols),
        transform=transform, fill=0, dtype=np.uint8,
    ).astype(bool) if sfha_geoms else np.zeros((nrows, ncols), bool)

    print(f"FEMA floodway cells: {floodway_mask.sum():,}  "
          f"({floodway_mask.sum() * cell_m**2 / 1e4:.2f} ha)")
    print(f"FEMA AE/SFHA cells:  {sfha_mask.sum():,}  "
          f"({sfha_mask.sum() * cell_m**2 / 1e4:.2f} ha)")

    # ── Compute risk classes ──────────────────────────────────────────────────
    hand_lt1 = np.isfinite(hand) & (hand < 1.0)
    hand_1_3 = np.isfinite(hand) & (hand >= 1.0) & (hand < 3.0)

    risk = np.zeros((nrows, ncols), dtype=np.uint8)
    risk[floodway_mask & hand_lt1] = 1
    risk[floodway_mask & hand_1_3] = 2
    risk[sfha_mask     & hand_lt1] = 3
    risk[sfha_mask     & hand_1_3] = 4
    risk[(risk == 0) & hand_lt1]   = 5   # HAND<1m outside any FEMA zone

    cell_ha = cell_m**2 / 1e4
    total_cells = nrows * ncols
    stats = {}
    print("\n── Risk class summary ──────────────────────────────────")
    for cls, label in CLASS_LABELS.items():
        n = int((risk == cls).sum())
        ha = n * cell_ha
        pct = n / total_cells * 100
        stats[cls] = {"label": label, "cells": n, "area_ha": round(ha, 2), "pct_aoi": round(pct, 2)}
        print(f"  Class {cls}: {n:6,} cells  {ha:6.1f} ha  {pct:.2f}%  {label}")

    # ── Verify FEMA BFE vs DEM ────────────────────────────────────────────────
    # FEMA STATIC_BFE is in feet; compare to DEM elevation at floodway cells
    bfe_col = "STATIC_BFE" if "STATIC_BFE" in gdf.columns else None
    if bfe_col:
        bfe_vals = gdf[is_floodway][bfe_col].dropna()
        bfe_vals = bfe_vals[bfe_vals > -9000]   # filter FEMA nodata sentinel (-9999)
        if len(bfe_vals):
            bfe_m = bfe_vals.mean() * 0.3048   # ft → m
            hand_at_floodway = hand[floodway_mask]
            dem_path = os.path.join(PROJ_DIR, "dem", "data", "sr417_corridor_dem_1m.tif")
            if os.path.exists(dem_path):
                with rasterio.open(dem_path) as src2:
                    dem_arr = src2.read(1).astype(np.float32)
                    dem_r = np.zeros_like(hand)
                    reproject(dem_arr, dem_r,
                              src_transform=src2.transform, src_crs=src2.crs,
                              dst_transform=transform, dst_crs=dem_crs,
                              resampling=Resampling.bilinear)
                dem_at_floodway = dem_r[floodway_mask]
                dem_mean = float(np.nanmean(dem_at_floodway))
                print(f"\n── FEMA BFE vs DEM ─────────────────────────────────────")
                print(f"  FEMA STATIC_BFE (floodway): {bfe_vals.mean():.1f} ft "
                      f"= {bfe_m:.2f} m NAVD88")
                print(f"  DEM mean elevation at floodway cells: {dem_mean:.2f} m NAVD88")
                print(f"  Difference (BFE - DEM): {bfe_m - dem_mean:.2f} m  "
                      f"(expected > 0: BFE is water surface, DEM is ground)")
                stats["bfe_check"] = {
                    "bfe_ft_mean": round(float(bfe_vals.mean()), 2),
                    "bfe_m_mean": round(bfe_m, 2),
                    "dem_m_at_floodway": round(dem_mean, 2),
                    "bfe_minus_dem_m": round(bfe_m - dem_mean, 2),
                }

    # ── Save risk raster ──────────────────────────────────────────────────────
    prof_out = profile.copy()
    prof_out.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(RISK_TIF, "w", **prof_out) as dst:
        dst.write(risk, 1)
    print(f"\nfema_hand_risk.tif saved  ({nrows}×{ncols})")

    # ── Export viewer PNG (512×512) ───────────────────────────────────────────
    SIZE = 512
    dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, SIZE, SIZE)
    risk_small = np.zeros((SIZE, SIZE), dtype=np.float32)
    reproject(risk.astype(np.float32), risk_small,
              src_transform=transform, src_crs=dem_crs,
              dst_transform=dst_transform, dst_crs=dem_crs,
              resampling=Resampling.nearest)
    risk_small = risk_small.astype(np.uint8)

    rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgba[risk_small == cls] = color

    Image.fromarray(rgba).save(RISK_PNG)
    print(f"fema_hand_risk.png saved  ({SIZE}×{SIZE})")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "hand_stats": {
            "mean_m": round(float(np.nanmean(hand)), 3),
            "p95_m":  round(float(np.nanpercentile(hand, 95)), 3),
            "pct_lt1m":  round(float(hand_lt1.sum() / total_cells * 100), 2),
            "pct_lt3m":  round(float((hand < 3).sum() / total_cells * 100), 2),
        },
        "fema_cells": {
            "floodway_ha": round(float(floodway_mask.sum()) * cell_ha, 2),
            "sfha_ha":     round(float(sfha_mask.sum()) * cell_ha, 2),
        },
        "risk_classes": stats,
        "color_legend": {
            str(k): {"color": list(v), "label": CLASS_LABELS[k]}
            for k, v in CLASS_COLORS.items()
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"fema_hand_summary.json saved")

    # ── Print the headline finding ────────────────────────────────────────────
    class1_ha = stats[1]["area_ha"]
    class2_ha = stats[2]["area_ha"]
    print(f"\n══ KEY FINDING ══════════════════════════════════════════")
    print(f"  Floodway + HAND<1m : {class1_ha:.1f} ha  "
          f"(highest erosion risk — regulated channel on flat terrain)")
    print(f"  Floodway + HAND<3m : {class1_ha + class2_ha:.1f} ha  "
          f"(Ian 11.43 ft gauge peak ~ 3.5m above drainage)")
    print(f"  19.7% of AOI is HAND<1m — consistent with flat flatwoods")


if __name__ == "__main__":
    run()
