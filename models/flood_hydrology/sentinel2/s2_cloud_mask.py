"""
Sentinel-2 Pixel-Level Cloud Masking
=====================================
Generates per-pixel cloud/shadow masks from the Sentinel-2 SCL (Scene
Classification Layer) band, then applies them to all water detection products.

Team lead requirement: no cloudy pixels in water body detection. Scene-level
cloud filtering (≤10% cloud cover) is insufficient — individual cloudy pixels
remain within "clean" scenes and corrupt water masks and WatNet predictions.

SCL class values used (Sentinel-2 L2A specification):
    0  — No data
    1  — Defective pixel
    2  — Dark area pixels (potential shadow)
    3  — Cloud shadow
    4  — Vegetation
    5  — Not vegetated
    6  — Water
    7  — Unclassified
    8  — Cloud medium probability
    9  — Cloud high probability
    10 — Thin cirrus
    11 — Snow / ice

Mask strategy:
    CLOUD pixels : SCL ∈ {0, 1, 3, 8, 9, 10} → masked (value=1 in cloud_mask)
    CLEAR pixels : all others               → unmasked (value=0)

Outputs (saved under sentinel2/data/):
    cloud_mask_{date}.tif   — binary cloud mask (0=clear, 1=cloud/shadow)
    cloud_masked_{date}_water_mask.tif  — water mask with cloud pixels set to 0
    cloud_summary.csv       — per-scene cloud fraction statistics

Usage:
    python3 sentinel2/s2_cloud_mask.py
    python3 sentinel2/s2_cloud_mask.py --dates 20220329 20230817
"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# SCL classes to treat as cloud/shadow (mask out)
CLOUD_CLASSES = {0, 1, 3, 8, 9, 10}  # no-data, defective, cloud shadow, cloud med/high, cirrus


def build_cloud_mask_from_scl(scl_path, target_shape=None):
    """
    Load SCL GeoTIFF and build binary cloud mask at target resolution.

    Returns (cloud_mask_arr, profile, fraction_cloudy).
    cloud_mask_arr: uint8, 1=cloud/shadow, 0=clear
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(scl_path) as src:
        scl = src.read(1).astype(np.uint8)
        scl_profile = src.profile.copy()
        scl_transform = src.transform
        scl_crs = src.crs

    raw_mask = np.zeros(scl.shape, dtype=np.uint8)
    for cls in CLOUD_CLASSES:
        raw_mask |= (scl == cls).astype(np.uint8)

    if target_shape is not None and scl.shape != target_shape:
        resampled = np.zeros(target_shape, dtype=np.uint8)
        # Approximate scale from SCL (20m) to target (10m)
        scale_r = target_shape[0] / scl.shape[0]
        scale_c = target_shape[1] / scl.shape[1]
        new_transform = rasterio.transform.from_bounds(
            *rasterio.transform.array_bounds(scl.shape[0], scl.shape[1], scl_transform),
            width=target_shape[1], height=target_shape[0]
        )
        reproject(
            source=raw_mask, destination=resampled,
            src_transform=scl_transform, src_crs=scl_crs,
            dst_transform=new_transform, dst_crs=scl_crs,
            resampling=Resampling.nearest,
        )
        raw_mask = resampled
        out_transform = new_transform
    else:
        out_transform = scl_transform

    n_total = raw_mask.size
    n_cloudy = int(raw_mask.sum())
    frac = n_cloudy / n_total if n_total > 0 else 0.0

    out_profile = scl_profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=255,
                       height=raw_mask.shape[0], width=raw_mask.shape[1],
                       transform=out_transform)
    return raw_mask, out_profile, frac


def apply_cloud_mask(data_path, cloud_mask, out_path):
    """
    Apply cloud mask to an existing GeoTIFF (set cloud pixels to 0).
    Returns the masked array.
    """
    import rasterio
    from scipy.ndimage import zoom

    with rasterio.open(data_path) as src:
        arr = src.read(1)
        prof = src.profile.copy()

    # Resize cloud mask to match data array if needed
    cm = cloud_mask
    if cm.shape != arr.shape:
        scale_r = arr.shape[0] / cm.shape[0]
        scale_c = arr.shape[1] / cm.shape[1]
        cm = zoom(cm.astype(np.float32), (scale_r, scale_c), order=0).astype(np.uint8)

    masked = np.where(cm > 0, 0, arr).astype(arr.dtype)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(masked, 1)
    return masked


def main(dates=None):
    import rasterio

    # Find all SCL files
    scl_files = sorted(glob.glob(os.path.join(DATA_DIR, "s2_*_SCL.tif")))
    if not scl_files:
        print("No SCL files found. Run s2_download.py first.")
        print("  (SCL download was added — re-run s2_download.py to fetch SCL bands)")
        # Fall back: try to generate masks from existing water masks via WatNet confidence
        _warn_no_scl()
        return

    if dates:
        scl_files = [f for f in scl_files
                     if any(d in os.path.basename(f) for d in dates)]

    print(f"Processing {len(scl_files)} SCL scene(s) …")
    summary_rows = []

    for scl_path in scl_files:
        date = os.path.basename(scl_path).replace("s2_", "").replace("_SCL.tif", "")
        print(f"\n  {date}")

        # Determine target shape from B03 (10m reference band)
        b03_path = os.path.join(DATA_DIR, f"s2_{date}_B03.tif")
        target_shape = None
        if os.path.exists(b03_path):
            with rasterio.open(b03_path) as ref:
                target_shape = (ref.height, ref.width)
                ref_profile = ref.profile.copy()

        cloud_mask_arr, cm_profile, frac_cloudy = build_cloud_mask_from_scl(
            scl_path, target_shape=target_shape
        )

        # Save cloud mask
        cm_path = os.path.join(DATA_DIR, f"cloud_mask_{date}.tif")
        with rasterio.open(cm_path, "w", **cm_profile) as dst:
            dst.write(cloud_mask_arr, 1)
        print(f"    Cloud fraction: {100*frac_cloudy:.1f}% → {cm_path}")

        # Apply to water mask
        wm_path = os.path.join(DATA_DIR, f"water_mask_{date}.tif")
        if os.path.exists(wm_path):
            wm_out = os.path.join(DATA_DIR, f"water_mask_cloudmasked_{date}.tif")
            apply_cloud_mask(wm_path, cloud_mask_arr, wm_out)
            print(f"    Applied to water_mask → water_mask_cloudmasked_{date}.tif")

        # Apply to MNDWI
        mndwi_path = os.path.join(DATA_DIR, f"mndwi_{date}.tif")
        if os.path.exists(mndwi_path):
            mndwi_out = os.path.join(DATA_DIR, f"mndwi_cloudmasked_{date}.tif")
            apply_cloud_mask(mndwi_path, cloud_mask_arr, mndwi_out)
            print(f"    Applied to MNDWI    → mndwi_cloudmasked_{date}.tif")

        # Apply to WatNet mask
        wn_path = os.path.join(DATA_DIR, f"watnet_mask_{date}.tif")
        if os.path.exists(wn_path):
            wn_out = os.path.join(DATA_DIR, f"watnet_mask_cloudmasked_{date}.tif")
            apply_cloud_mask(wn_path, cloud_mask_arr, wn_out)
            print(f"    Applied to WatNet   → watnet_mask_cloudmasked_{date}.tif")

        summary_rows.append({
            "date": date,
            "cloud_pct_scl": round(100 * frac_cloudy, 2),
            "cloud_pixels": int(cloud_mask_arr.sum()),
            "total_pixels": int(cloud_mask_arr.size),
            "cloud_mask_path": cm_path,
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(DATA_DIR, "cloud_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nCloud mask summary → {summary_path}")
        print(df[["date", "cloud_pct_scl"]].to_string(index=False))
        mean_cloud = df["cloud_pct_scl"].mean()
        print(f"\nMean cloud fraction across all scenes: {mean_cloud:.1f}%")
        if mean_cloud > 5:
            print("  ⚠  Some scenes have >5% cloud coverage at pixel level.")
            print("  Consider re-downloading with stricter scene filter or excluding cloudy dates.")

    _make_cloud_overlay_viz(summary_rows)


def _warn_no_scl():
    print("\nTo download SCL bands for existing scenes, run:")
    print("  python3 sentinel2/s2_download.py")
    print("  (SCL download was added — only new/re-downloaded scenes will have SCL)")
    print("\nAlternative: use existing water_mask files filtered by scene-level cloud_pct")
    print("  in s2_scene_index.csv (column: cloud_pct)")


def _make_cloud_overlay_viz(summary_rows):
    """Quick bar chart of per-scene cloud fractions."""
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["tomato" if p > 5 else "steelblue" for p in df["cloud_pct_scl"]]
    ax.bar(df["date"], df["cloud_pct_scl"], color=colors, edgecolor="white")
    ax.axhline(5, color="orange", lw=1.5, linestyle="--", label="5% threshold")
    ax.set_xlabel("Scene date")
    ax.set_ylabel("Cloud pixel fraction [%]")
    ax.set_title("Pixel-level cloud coverage per Sentinel-2 scene\n"
                 "(from SCL band — classes: no-data, defective, cloud shadow, cloud med/high, cirrus)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "cloud_coverage_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cloud summary chart → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pixel-level cloud masks from Sentinel-2 SCL band")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Specific scene dates to process (e.g. 20220329 20230817)")
    args = parser.parse_args()
    main(args.dates)
