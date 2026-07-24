"""
Sentinel-2 Water Body Segmentation — WatNet CNN vs MNDWI Baseline
=================================================================
Uses WatNet (Luo et al. 2021) — a DeepLabv3+ CNN with MobileNetv2 backbone
pre-trained on global Sentinel-2 water body datasets — to segment water bodies
from Sentinel-2 imagery.  MNDWI (traditional index) is run in parallel as a
benchmark.  Both methods are compared against the DEM-derived lake mask (used
as reference) with F1, IoU, precision, and recall metrics.

WatNet input: 6 Sentinel-2 bands (B02, B03, B04, B08, B11, B12), normalized
to [0, 1] (surface reflectance / 10000), stacked as (row, col, 6) array.

Bands used:
    B02 — Blue  (10m)
    B03 — Green (10m)
    B04 — Red   (10m)
    B08 — NIR   (10m)
    B11 — SWIR1 (20m → resampled to 10m)
    B12 — SWIR2 (20m → resampled to 10m)

Reference:
    Luo, X. et al. (2021). "WatNet: a deep learning framework for surface water
    detection from Sentinel-2 imagery." ISPRS J. of Photogrammetry.
    GitHub: https://github.com/xinluo2018/WatNet

Outputs (saved under sentinel2/data/):
    watnet_mask_{date}.tif          — WatNet binary water mask
    mndwi_mask_{date}.tif          — MNDWI binary mask (threshold=0.0)
    watnet_metrics.csv             — per-scene F1/IoU/precision/recall comparison
    watnet_comparison.png          — false-color | MNDWI | WatNet | agreement per scene

Usage:
    python3 sentinel2/s2_water_segment.py
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
REPO_ROOT    = os.path.dirname(BASE_DIR)
WATNET_DIR   = os.path.join(REPO_ROOT, "WatNet")
WEIGHTS_PATH = os.path.join(WATNET_DIR, "model", "pretrained", "watnet.h5")
DEM_DATA_DIR = os.path.join(REPO_ROOT, "dem", "data")
DEM_PATH     = os.path.join(DEM_DATA_DIR, "winter_garden_dem.tif")
LAKE_MASK_PATH = os.path.join(DEM_DATA_DIR, "lake_mask.tif")

MNDWI_THRESHOLD = 0.0
SCALE_FACTOR    = 10000.0


# ── Band loading ─────────────────────────────────────────────────────────────

def load_band(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def resample_to_match(arr, target_shape):
    """Bilinear resample array to target_shape (rows, cols)."""
    from scipy.ndimage import zoom
    scale_r = target_shape[0] / arr.shape[0]
    scale_c = target_shape[1] / arr.shape[1]
    return zoom(arr, (scale_r, scale_c), order=1)


def load_scene_6band(date_str):
    """
    Load all 6 Sentinel-2 bands for a scene, return (H, W, 6) float32 array
    normalized to [0, 1] and the B03 rasterio profile for CRS/transform info.
    """
    band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
    arrays = {}
    profile = None

    for band in band_names:
        path = os.path.join(DATA_DIR, f"s2_{date_str}_{band}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing band {band} for scene {date_str}: {path}")
        arr, p = load_band(path)
        arrays[band] = arr
        if band == "B03":
            profile = p

    target_shape = arrays["B03"].shape

    # Resample 20m bands (B11, B12) to match 10m reference
    for band in ["B11", "B12"]:
        if arrays[band].shape != target_shape:
            arrays[band] = resample_to_match(arrays[band], target_shape)

    # Stack (H, W, 6) and normalize
    stack = np.stack(
        [arrays[b] for b in band_names], axis=-1
    ).astype(np.float32)
    stack = np.clip(stack / SCALE_FACTOR, 0.0, 1.0)

    return stack, profile


# ── MNDWI baseline ───────────────────────────────────────────────────────────

def compute_mndwi_mask(stack, threshold=MNDWI_THRESHOLD):
    """MNDWI = (Green − SWIR1) / (Green + SWIR1). Stack bands: [B02,B03,B04,B08,B11,B12]."""
    green = stack[:, :, 1]   # B03 index 1
    swir1 = stack[:, :, 4]   # B11 index 4
    denom = green + swir1
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = np.where(denom > 0, (green - swir1) / denom, 0.0)
    return (mndwi > threshold).astype(np.uint8), mndwi


# ── WatNet inference ─────────────────────────────────────────────────────────

def load_watnet_model():
    """Load pre-trained WatNet model from cloned repository."""
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"WatNet weights not found at {WEIGHTS_PATH}\n"
            f"Clone with: git clone --depth 1 https://github.com/xinluo2018/WatNet {WATNET_DIR}"
        )

    # Add WatNet to path so its internal imports work
    if WATNET_DIR not in sys.path:
        sys.path.insert(0, WATNET_DIR)

    import tensorflow as tf
    print(f"  TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU devices: {[g.name for g in gpus] or 'none (CPU only)'}")

    # WatNet .h5 was saved with TF 2.x / Keras 2.x. Loading via `load_model`
    # fails on Keras 3 (DepthwiseConv2D 'groups' kwarg + TFOpLambda issues).
    # Solution: build model from watnet_build.py (Keras 3 compatible port)
    # and load only the weight arrays by name from the .h5 file.
    from watnet_build import load_watnet_from_source
    model = load_watnet_from_source(WEIGHTS_PATH)
    print(f"  WatNet ready (Keras 3 compatible build + pre-trained weights)")
    return model


def watnet_predict(model, stack):
    """
    Run WatNet inference on a (H, W, 6) normalized stack.
    Uses patch-based inference (512×512 + 80px overlap) from WatNet's imgPatch utility.
    Returns binary water mask (H, W) uint8.
    """
    from utils.imgPatch import imgPatch

    patch_ins = imgPatch(stack, patch_size=512, edge_overlay=80)
    patches, starts, n_row, n_col = patch_ins.toPatch()

    import tensorflow as tf
    results = []
    for patch in patches:
        pred = model(patch[np.newaxis, :], training=False)  # (1, H, W, 1) or (1, H, W)
        results.append(np.squeeze(pred.numpy(), axis=0))

    prob_map = patch_ins.toImage(results, n_row, n_col)
    # prob_map may be (H, W, 1) — squeeze
    if prob_map.ndim == 3:
        prob_map = prob_map[:, :, 0]

    return (prob_map > 0.5).astype(np.uint8), prob_map


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred, ref):
    """Compute F1, IoU, precision, recall between two binary masks."""
    pred = pred.astype(bool)
    ref  = ref.astype(bool)
    tp = (pred & ref).sum()
    fp = (pred & ~ref).sum()
    fn = (~pred & ref).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "iou": round(iou, 4)}


def load_dem_lake_reference(target_shape, profile):
    """
    Load DEM lake mask (ground-truth reference) and reproject to match
    the Sentinel-2 grid. Returns binary mask (H, W).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    if not os.path.exists(LAKE_MASK_PATH):
        print("  DEM lake mask not found; using zeros as reference")
        return np.zeros(target_shape, dtype=np.uint8)

    with rasterio.open(LAKE_MASK_PATH) as src:
        dest = np.zeros(target_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile.get("crs"),
            resampling=Resampling.nearest,
        )
    return dest


# ── Visualization ─────────────────────────────────────────────────────────────

def make_false_color(stack):
    """NIR-Red-Green false color composite for vegetation/water contrast."""
    nir   = stack[:, :, 3]  # B08
    red   = stack[:, :, 2]  # B04
    green = stack[:, :, 1]  # B03
    rgb = np.stack([nir, red, green], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return rgb


def make_comparison_plot(records, data_dir):
    """
    5-column figure per scene: False Color | MNDWI mask | WatNet mask | Agreement map.
    Scenes are algorithm-selected (2 most agreed + 2 most discrepant + WatNet-loses case).
    Each row has a title showing date, F1 scores, and selection reason.
    """
    n = len(records)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.55})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["False Color (NIR-R-G)", "MNDWI Mask", "WatNet Mask", "Agreement"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=4)

    for row_idx, rec in enumerate(records):
        date_str = rec["date"]
        stack    = rec["stack"]
        mndwi_m  = rec["mndwi_mask"]
        watnet_m = rec["watnet_mask"]
        f1_m     = rec["mndwi_f1"]
        f1_w     = rec["watnet_f1"]
        diff     = f1_w - f1_m
        reason   = rec.get("reason", "")

        # Col 0: False color with date/F1 label overlaid as text box
        fc = make_false_color(stack)
        axes[row_idx, 0].imshow(fc)
        row_label = (f"{date_str}  MNDWI F1={f1_m:.3f}  WatNet F1={f1_w:.3f}  "
                     f"Δ={diff:+.3f}\n{reason}")
        axes[row_idx, 0].text(
            0.01, 0.99, row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=6.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88, ec="gray", lw=0.5)
        )

        # Col 1: MNDWI mask
        axes[row_idx, 1].imshow(mndwi_m, cmap="Blues", vmin=0, vmax=1)
        axes[row_idx, 1].set_xlabel(f"F1={f1_m:.3f}", fontsize=7)

        # Col 2: WatNet mask
        axes[row_idx, 2].imshow(watnet_m, cmap="Blues", vmin=0, vmax=1)
        axes[row_idx, 2].set_xlabel(f"F1={f1_w:.3f}", fontsize=7)

        # Col 3: Agreement (TP=blue, FP=red, FN=orange, TN=white)
        mndwi_b  = mndwi_m.astype(bool)
        watnet_b = watnet_m.astype(bool)
        agree_rgb = np.ones((*stack.shape[:2], 3), dtype=np.float32)
        agree_rgb[watnet_b & mndwi_b]  = [0.2, 0.4, 0.9]
        agree_rgb[watnet_b & ~mndwi_b] = [0.9, 0.2, 0.2]
        agree_rgb[~watnet_b & mndwi_b] = [1.0, 0.6, 0.1]
        axes[row_idx, 3].imshow(agree_rgb)

        for ax in axes[row_idx]:
            ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=(0.2, 0.4, 0.9), label="Both detect water"),
        mpatches.Patch(color=(0.9, 0.2, 0.2), label="WatNet only"),
        mpatches.Patch(color=(1.0, 0.6, 0.1), label="MNDWI only"),
        mpatches.Patch(color=(1.0, 1.0, 1.0), label="Both: land"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(
        "Sentinel-2 Water Segmentation — WatNet (CNN) vs MNDWI (Index)\n"
        "17801 Champagne Dr, Winter Garden FL  |  Reference: DEM lake mask\n"
        "Scenes selected by F1-diff ranking: 2 most agreed + 2 most discrepant + 1 WatNet-loses",
        fontsize=10, y=1.01
    )
    out_path = os.path.join(data_dir, "watnet_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(WATNET_DIR):
        sys.exit(
            f"WatNet not found at {WATNET_DIR}\n"
            f"Run: git clone --depth 1 https://github.com/xinluo2018/WatNet {WATNET_DIR}"
        )

    print("=" * 65)
    print("WatNet Water Body Segmentation — Winter Garden FL")
    print("=" * 65)

    print("\nLoading WatNet pre-trained model …")
    model = load_watnet_model()

    # Find all scenes with 6-band coverage
    b03_files = sorted(glob.glob(os.path.join(DATA_DIR, "s2_*_B03.tif")))
    if not b03_files:
        sys.exit(f"No S2 scenes found in {DATA_DIR} — run s2_download.py first")

    # Optionally restrict to keep==True scenes from scene_ranking.csv
    ranking_csv = os.path.join(DATA_DIR, "scene_ranking.csv")
    if "--from-ranking" in sys.argv and os.path.exists(ranking_csv):
        import pandas as _pd
        kept = set(_pd.read_csv(ranking_csv).query("keep == True")["date"].astype(str).tolist())
        print(f"--from-ranking: restricting to {len(kept)} kept scenes")
    else:
        kept = None

    # Filter to scenes with all 6 bands
    scenes = []
    for f in b03_files:
        date_str = os.path.basename(f).replace("s2_", "").replace("_B03.tif", "")
        if kept is not None and date_str not in kept:
            continue
        all_bands_present = all(
            os.path.exists(os.path.join(DATA_DIR, f"s2_{date_str}_{b}.tif"))
            for b in ["B02", "B03", "B04", "B08", "B11", "B12"]
        )
        if all_bands_present:
            scenes.append(date_str)
        else:
            print(f"  Skipping {date_str}: missing bands")

    print(f"\nFound {len(scenes)} scene(s) with full 6-band coverage")

    records = []
    plot_records = []

    for date_str in scenes:
        out_path = os.path.join(DATA_DIR, f"watnet_mask_{date_str}.tif")
        if os.path.exists(out_path) and "--rerun" not in sys.argv:
            print(f"  [{date_str}] already done — skip (use --rerun to overwrite)")
            continue
        print(f"\n── {date_str} ──────────────────────────────")

        # Load 6-band stack
        print("  Loading bands …")
        stack, profile = load_scene_6band(date_str)
        H, W, _ = stack.shape
        print(f"  Scene shape: {H}×{W} pixels")

        # MNDWI baseline
        mndwi_mask, mndwi_vals = compute_mndwi_mask(stack)
        cell_m2 = abs(profile["transform"].a * profile["transform"].e)
        mndwi_area_ha = mndwi_mask.sum() * cell_m2 / 1e4
        print(f"  MNDWI water area: {mndwi_area_ha:.1f} ha  "
              f"(mean={mndwi_vals.mean():.3f}, range [{mndwi_vals.min():.3f},{mndwi_vals.max():.3f}])")

        # WatNet inference
        print("  Running WatNet inference …")
        watnet_mask, watnet_prob = watnet_predict(model, stack)
        watnet_area_ha = watnet_mask.sum() * cell_m2 / 1e4
        print(f"  WatNet water area: {watnet_area_ha:.1f} ha  "
              f"(prob range [{watnet_prob.min():.3f},{watnet_prob.max():.3f}])")

        # Reference: DEM lake mask reprojected to S2 grid
        dem_ref = load_dem_lake_reference((H, W), profile)
        dem_area_ha = dem_ref.sum() * cell_m2 / 1e4

        has_ref = dem_ref.max() > 0
        if has_ref:
            mndwi_metrics  = compute_metrics(mndwi_mask, dem_ref)
            watnet_metrics = compute_metrics(watnet_mask, dem_ref)
            print(f"  DEM ref area: {dem_area_ha:.1f} ha")
            print(f"  MNDWI  F1={mndwi_metrics['f1']:.3f}  IoU={mndwi_metrics['iou']:.3f}  "
                  f"P={mndwi_metrics['precision']:.3f}  R={mndwi_metrics['recall']:.3f}")
            print(f"  WatNet F1={watnet_metrics['f1']:.3f}  IoU={watnet_metrics['iou']:.3f}  "
                  f"P={watnet_metrics['precision']:.3f}  R={watnet_metrics['recall']:.3f}")
            winner = "WatNet" if watnet_metrics["f1"] >= mndwi_metrics["f1"] else "MNDWI"
            print(f"  → Better: {winner}")
        else:
            mndwi_metrics  = {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
            watnet_metrics = {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
            print("  (No DEM lake mask available — metrics set to 0)")

        # Save masks
        import rasterio
        for mask, tag in [(mndwi_mask, "mndwi_mask"), (watnet_mask, "watnet_mask")]:
            p = profile.copy()
            p.update(dtype="uint8", count=1, compress="lzw")
            out_p = os.path.join(DATA_DIR, f"{tag}_{date_str}.tif")
            with rasterio.open(out_p, "w", **p) as dst:
                dst.write(mask.astype(np.uint8), 1)

        records.append({
            "date": date_str,
            "scene_pixels": H * W,
            "mndwi_area_ha": round(mndwi_area_ha, 2),
            "watnet_area_ha": round(watnet_area_ha, 2),
            "dem_ref_area_ha": round(dem_area_ha, 2),
            "mndwi_f1": mndwi_metrics["f1"],
            "mndwi_iou": mndwi_metrics["iou"],
            "mndwi_precision": mndwi_metrics["precision"],
            "mndwi_recall": mndwi_metrics["recall"],
            "watnet_f1": watnet_metrics["f1"],
            "watnet_iou": watnet_metrics["iou"],
            "watnet_precision": watnet_metrics["precision"],
            "watnet_recall": watnet_metrics["recall"],
        })

        # Collect all scenes for F1-diff based selection later
        plot_records.append({
            "date": date_str,
            "stack": stack,
            "mndwi_mask": mndwi_mask,
            "watnet_mask": watnet_mask,
            "mndwi_f1": mndwi_metrics["f1"],
            "watnet_f1": watnet_metrics["f1"],
        })

    if not records:
        print("No scenes processed.")
        return

    # Save metrics CSV
    df = pd.DataFrame(records).sort_values("date")
    csv_path = os.path.join(DATA_DIR, "watnet_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics → {csv_path}")

    # Summary table
    print("\n── Summary ──────────────────────────────────────────────────")
    cols = ["date", "mndwi_area_ha", "watnet_area_ha",
            "mndwi_f1", "watnet_f1", "mndwi_iou", "watnet_iou"]
    print(df[cols].to_string(index=False))

    # Aggregate metrics
    if df["mndwi_f1"].max() > 0:
        print(f"\n  Mean MNDWI  F1 = {df.mndwi_f1.mean():.3f}  IoU = {df.mndwi_iou.mean():.3f}")
        print(f"  Mean WatNet F1 = {df.watnet_f1.mean():.3f}  IoU = {df.watnet_iou.mean():.3f}")
        wins = (df.watnet_f1 >= df.mndwi_f1).sum()
        print(f"  WatNet wins: {wins}/{len(df)} scenes")

    # Select 5 scenes for comparison figure by F1-diff ranking
    if plot_records:
        df_rank = pd.DataFrame([
            {"date": r["date"], "diff": r["watnet_f1"] - r["mndwi_f1"], "_idx": i}
            for i, r in enumerate(plot_records)
        ])
        df_rank = df_rank.sort_values("diff")

        selected_idx = []
        reasons = {}

        # 2 most agreed (smallest absolute diff)
        for _, row in df_rank.reindex(df_rank["diff"].abs().sort_values().index).head(2).iterrows():
            if row["_idx"] not in selected_idx:
                selected_idx.append(int(row["_idx"]))
                reasons[int(row["_idx"])] = f"Most agreed — Δ={row['diff']:+.3f}"

        # 2 most discrepant (largest positive diff — WatNet >> MNDWI)
        for _, row in df_rank[df_rank["diff"] > 0].sort_values("diff", ascending=False).head(2).iterrows():
            if row["_idx"] not in selected_idx:
                selected_idx.append(int(row["_idx"]))
                reasons[int(row["_idx"])] = f"Most discrepant (WatNet >> MNDWI) — Δ={row['diff']:+.3f}"

        # 1 WatNet-loses (negative diff)
        for _, row in df_rank[df_rank["diff"] < 0].sort_values("diff").head(1).iterrows():
            if row["_idx"] not in selected_idx:
                selected_idx.append(int(row["_idx"]))
                reasons[int(row["_idx"])] = f"WatNet underperforms MNDWI — Δ={row['diff']:+.3f}"

        # Fallback: fill to 5 from remaining if needed
        for _, row in df_rank.sort_values("diff").iterrows():
            if len(selected_idx) >= 5:
                break
            if row["_idx"] not in selected_idx:
                selected_idx.append(int(row["_idx"]))
                reasons[int(row["_idx"])] = ""

        display_records = []
        for idx in selected_idx[:5]:
            rec = dict(plot_records[idx])
            rec["reason"] = reasons.get(idx, "")
            display_records.append(rec)

        make_comparison_plot(display_records, DATA_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
