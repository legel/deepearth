"""
Sentinel-2 Pixel-Level Cloud Masking — Two-Layer Strategy
===========================================================
Layer 1 — SCL (Scene Classification Layer):
    Sentinel-2 L2A pixel classifier.  Classes 0,1,3,8,9,10 → masked.
    Reliable but can miss thin clouds in complex scenes.

Layer 2 — s2cloudless (Sinergise, LightGBM-based):
    ML cloud probability model using 10 S2 bands.  Requires B01, B02, B04,
    B05, B08, B8A, B09, B11, B12 (B10 is cirrus-only, set to 0 in L2A).
    Threshold 0.4 → cloud; refined by spatial smoothing (average_over=4).
    Only available for scenes that have the extra s2cloudless bands downloaded.

Final mask = SCL_cloud OR (s2cloudless_prob > 0.4).

Outputs (saved under sentinel2/data/):
    cloud_mask_{date}.tif                — binary cloud mask (0=clear, 1=cloud)
    cloud_prob_{date}.tif                — s2cloudless probability [0,1] float32
    water_mask_cloudmasked_{date}.tif    — MNDWI water mask, cloud pixels zeroed
    watnet_mask_cloudmasked_{date}.tif   — WatNet mask, cloud pixels zeroed
    prithvi_mask_cloudmasked_{date}.tif  — Prithvi mask, cloud pixels zeroed
    cloud_summary.csv                    — per-scene cloud fraction statistics

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
CLOUD_CLASSES = {1, 3, 8, 9, 10}  # defective, cloud shadow, cloud med/high, cirrus
NODATA_CLASS = 0  # no-data border pixels — excluded from cloud fraction (not cloud)


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
    nodata_mask = (scl == NODATA_CLASS).astype(np.uint8)

    if target_shape is not None and scl.shape != target_shape:
        resampled = np.zeros(target_shape, dtype=np.uint8)
        resampled_nodata = np.zeros(target_shape, dtype=np.uint8)
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
        reproject(
            source=nodata_mask, destination=resampled_nodata,
            src_transform=scl_transform, src_crs=scl_crs,
            dst_transform=new_transform, dst_crs=scl_crs,
            resampling=Resampling.nearest,
        )
        raw_mask = resampled
        nodata_mask = resampled_nodata
        out_transform = new_transform
    else:
        out_transform = scl_transform

    n_total = raw_mask.size
    n_nodata = int(nodata_mask.sum())
    n_valid = n_total - n_nodata
    n_cloudy = int(raw_mask.sum())
    # Fraction of *valid* (non-border) AOI pixels that are cloud/shadow
    frac = n_cloudy / n_valid if n_valid > 0 else 0.0

    out_profile = scl_profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=255,
                       height=raw_mask.shape[0], width=raw_mask.shape[1],
                       transform=out_transform)
    return raw_mask, out_profile, frac


def build_s2cloudless_mask(date, data_dir, target_shape, target_transform, target_crs):
    """
    Run s2cloudless ML cloud detector on 10 S2 bands for a given scene date.

    Returns (cloud_mask_uint8, cloud_prob_float32) both at target_shape resolution,
    or (None, None) if required bands are missing.

    s2cloudless band order (10 bands): B01, B02, B04, B05, B08, B8A, B09,
    B10(zeros in L2A), B11, B12.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    # Band order required by s2cloudless MODEL_BAND_IDS [0,1,3,4,7,8,9,10,11,12]
    # Mapping to Sentinel-2 band names:
    BAND_ORDER = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", None, "B11", "B12"]
    # None = B10 (cirrus, not present in L2A → fill with zeros)

    # Check that at least the key extra bands exist
    required = [b for b in BAND_ORDER if b is not None]
    missing  = [b for b in required
                if not os.path.exists(os.path.join(data_dir, f"s2_{date}_{b}.tif"))]
    if missing:
        print(f"    s2cloudless skipped ({date}): missing bands {missing}")
        return None, None

    h, w = target_shape
    stack = np.zeros((h, w, 10), dtype=np.float32)

    for i, band_name in enumerate(BAND_ORDER):
        if band_name is None:
            stack[:, :, i] = 0.0  # B10 cirrus → zeros in L2A
            continue
        path = os.path.join(data_dir, f"s2_{date}_{band_name}.tif")
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            # Reproject to target 10m grid
            dst = np.zeros(target_shape, dtype=np.float32)
            reproject(arr, dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=target_transform, dst_crs=target_crs,
                      resampling=Resampling.bilinear)
        # Normalize to [0, 1] (S2 L2A reflectance stored as DN×10000)
        stack[:, :, i] = np.clip(dst / 10000.0, 0.0, 1.0)

    # Run s2cloudless via prithvi conda env (has lightgbm with libomp resolved)
    import subprocess, tempfile
    prithvi_python = os.path.expanduser("~/miniforge3/envs/prithvi/bin/python")
    if not os.path.exists(prithvi_python):
        print("    prithvi env not found — skipping ML layer")
        return None, None

    # Write stack to a temp file, run inference in subprocess
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        stack_path = f.name
    out_path_prob = stack_path.replace(".npy", "_prob.npy")
    np.save(stack_path, stack)

    script = f"""
import numpy as np, sys
from s2cloudless import S2PixelCloudDetector
stack = np.load('{stack_path}')
det = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
probs = det.get_cloud_probability_maps(stack[None])[0]
np.save('{out_path_prob}', probs.astype(np.float32))
"""
    result = subprocess.run([prithvi_python, "-c", script],
                            capture_output=True, text=True, timeout=120)
    os.unlink(stack_path)

    if result.returncode != 0 or not os.path.exists(out_path_prob):
        print(f"    s2cloudless subprocess failed: {result.stderr[:200]}")
        return None, None

    cloud_prob = np.load(out_path_prob).astype(np.float32)
    os.unlink(out_path_prob)
    cloud_bin  = (cloud_prob > 0.4).astype(np.uint8)
    return cloud_bin, cloud_prob


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

        cloud_mask_arr, cm_profile, frac_scl = build_cloud_mask_from_scl(
            scl_path, target_shape=target_shape
        )
        print(f"    SCL cloud fraction: {100*frac_scl:.1f}%")

        # Layer 2: s2cloudless ML cloud probability
        if target_shape is not None:
            ref_transform = cm_profile["transform"]
            ref_crs       = cm_profile.get("crs") or cm_profile.get("CRS")
            if ref_crs is None and os.path.exists(b03_path):
                with rasterio.open(b03_path) as ref:
                    ref_transform = ref.transform
                    ref_crs       = ref.crs
            s2c_bin, s2c_prob = build_s2cloudless_mask(
                date, DATA_DIR, target_shape, ref_transform, ref_crs
            )
            if s2c_bin is not None:
                frac_s2c = float(s2c_bin.mean())
                print(f"    s2cloudless cloud: {100*frac_s2c:.1f}%")
                # Save cloud probability raster
                prob_path = os.path.join(DATA_DIR, f"cloud_prob_{date}.tif")
                prob_profile = cm_profile.copy()
                prob_profile.update(dtype="float32", nodata=None)
                with rasterio.open(prob_path, "w", **prob_profile) as dst:
                    dst.write(s2c_prob, 1)
                # Merge: final mask = SCL OR s2cloudless
                cloud_mask_arr = np.maximum(cloud_mask_arr, s2c_bin)
            else:
                frac_s2c = None
        else:
            frac_s2c = None

        frac_cloudy = float(cloud_mask_arr.mean())

        # Save combined cloud mask
        cm_path = os.path.join(DATA_DIR, f"cloud_mask_{date}.tif")
        with rasterio.open(cm_path, "w", **cm_profile) as dst:
            dst.write(cloud_mask_arr, 1)
        src_label = "SCL+s2cloudless" if frac_s2c is not None else "SCL only"
        print(f"    Combined ({src_label}): {100*frac_cloudy:.1f}% → {cm_path}")

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

        # Apply to Prithvi-EO-2.0 mask
        pr_path = os.path.join(DATA_DIR, f"prithvi_mask_{date}.tif")
        if os.path.exists(pr_path):
            pr_out = os.path.join(DATA_DIR, f"prithvi_mask_cloudmasked_{date}.tif")
            apply_cloud_mask(pr_path, cloud_mask_arr, pr_out)
            print(f"    Applied to Prithvi  → prithvi_mask_cloudmasked_{date}.tif")

        summary_rows.append({
            "date": date,
            "cloud_pct_scl":       round(100 * frac_scl, 2),
            "cloud_pct_s2cloudless": round(100 * frac_s2c, 2) if frac_s2c is not None else None,
            "cloud_pct_combined":  round(100 * frac_cloudy, 2),
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
