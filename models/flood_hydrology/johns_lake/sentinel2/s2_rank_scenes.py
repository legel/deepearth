"""
Sentinel-2 Scene Ranking — Cloud Cover Analysis
================================================
Ranks existing Sentinel-2 scenes by cloud cover, focusing on
cloud-over-lake fraction to identify scenes suitable for water
extent analysis via OmniWaterMask.

Inputs:
    sentinel2/data/cloud_summary.csv  — per-scene cloud stats (from s2_cloud_mask.py)
    sentinel2/data/cloud_mask_*.tif   — per-pixel cloud masks (1=cloud/shadow)
    dem/data/lake_mask.tif            — lake extent in EPSG:5070

Outputs:
    sentinel2/data/scene_ranking.csv  — ranked scene list with tier/keep columns

Tier definitions:
    1: cloud_pct_combined < 5%                              → keep
    2: 5% ≤ cloud_pct_combined < 10% AND cloud_on_lake < 1%  → keep
    3: otherwise                                             → reject

Usage:
    python3 sentinel2/s2_rank_scenes.py
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR  = os.path.join(os.path.dirname(BASE_DIR), "dem", "data")

LAKE_MASK_PATH   = os.path.join(DEM_DIR, "lake_mask.tif")
CLOUD_SUMMARY    = os.path.join(DATA_DIR, "cloud_summary.csv")
RANKING_CSV      = os.path.join(DATA_DIR, "scene_ranking.csv")

TIER1_CLOUD_MAX  = 5.0   # % combined cloud — unconditional keep
TIER2_CLOUD_MAX  = 10.0  # % combined cloud — keep only if lake is clear
TIER2_LAKE_MAX   = 1.0   # % cloud-on-lake threshold for tier-2 promotion


def reproject_lake_mask(target_height, target_width, target_crs, target_transform):
    """Reproject lake_mask.tif (EPSG:5070) onto a target S2 pixel grid."""
    with rasterio.open(LAKE_MASK_PATH) as src:
        lake = src.read(1).astype(np.float32)
        src_crs       = src.crs
        src_transform = src.transform

    dst = np.zeros((target_height, target_width), dtype=np.float32)
    reproject(
        source=lake,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    return (dst > 0).astype(np.uint8)


def cloud_on_lake_pct(cloud_mask_path, lake_reproj):
    """Return % of lake pixels that are cloudy in the given cloud mask."""
    with rasterio.open(cloud_mask_path) as src:
        cloud = src.read(1)

    lake_px = int(lake_reproj.sum())
    if lake_px == 0:
        return 0.0
    cloudy_lake_px = int(((cloud > 0) & (lake_reproj > 0)).sum())
    return cloudy_lake_px / lake_px * 100.0


def assign_tier(comb_pct, lake_pct):
    if comb_pct < TIER1_CLOUD_MAX:
        return 1, True
    if comb_pct < TIER2_CLOUD_MAX and lake_pct < TIER2_LAKE_MAX:
        return 2, True
    return 3, False


def main():
    for path, label in [(CLOUD_SUMMARY, "cloud_summary.csv"), (LAKE_MASK_PATH, "lake_mask.tif")]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found — {label} is required.")
            sys.exit(1)

    df = pd.read_csv(CLOUD_SUMMARY, dtype={"date": str})
    df["date"] = df["date"].astype(str).str.zfill(8)
    print(f"Loaded {len(df)} scenes from cloud_summary.csv")

    lake_cache = {}  # (height, width) → reprojected lake mask
    records    = []

    for _, row in df.iterrows():
        date     = row["date"]
        # Use combined where available, fall back to SCL
        comb_pct = row.get("cloud_pct_combined")
        if pd.isna(comb_pct):
            comb_pct = float(row.get("cloud_pct_scl", 0.0))
        else:
            comb_pct = float(comb_pct)

        mask_path = os.path.join(DATA_DIR, f"cloud_mask_{date}.tif")
        if not os.path.exists(mask_path):
            print(f"  {date}: cloud mask missing, skipping")
            continue

        with rasterio.open(mask_path) as src:
            h, w   = src.height, src.width
            crs    = src.crs
            xform  = src.transform

        key = (h, w)
        if key not in lake_cache:
            lake_cache[key] = reproject_lake_mask(h, w, crs, xform)
        lake_reproj = lake_cache[key]

        lake_pct         = cloud_on_lake_pct(mask_path, lake_reproj)
        tier, keep       = assign_tier(comb_pct, lake_pct)

        records.append({
            "date":               date,
            "cloud_pct_combined": round(comb_pct,  4),
            "cloud_on_lake_pct":  round(lake_pct,  4),
            "tier":               tier,
            "keep":               keep,
        })
        print(f"  {date}: combined={comb_pct:5.2f}%  lake={lake_pct:5.2f}%  "
              f"tier={tier}  keep={keep}")

    if not records:
        print("No records produced.")
        return

    out = pd.DataFrame(records)
    out = out.sort_values(["tier", "cloud_on_lake_pct", "cloud_pct_combined"]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    # Reorder columns
    out = out[["rank", "date", "cloud_pct_combined", "cloud_on_lake_pct", "tier", "keep"]]
    out.to_csv(RANKING_CSV, index=False)

    print(f"\nSaved: {RANKING_CSV}")
    print(out.to_string(index=False))
    kept = int(out["keep"].sum())
    print(f"\nKeep: {kept}/{len(out)} scenes (tier 1 + 2)")

    # Sanity check
    print("\nPASS: scene_ranking.csv written")
    print(f"PASS: {kept} scenes flagged keep=True")
    bad_tier = out[out["tier"] > 3]
    if len(bad_tier):
        print(f"FAIL: unexpected tier values — {bad_tier}")
    else:
        print("PASS: all tier values in {{1,2,3}}")


if __name__ == "__main__":
    main()
