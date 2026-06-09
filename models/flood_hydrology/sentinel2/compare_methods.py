"""
Fair Method Comparison — All vs S2-Observed Ground Truth
=========================================================
Re-evaluates every water segmentation method against the same reference:
    lake_mask_s2.tif  — Sentinel-2 observed lake boundary (94.4 ha)

Methods compared:
    MNDWI      — spectral index (Green-SWIR1)/(Green+SWIR1), threshold 0
    WatNet     — CNN-based DeepLabv3+/MobileNetv2 (2021), S2-trained
    Prithvi    — IBM/NASA geospatial foundation model (2024), SEN1Floods11

Outputs:
    sentinel2/method_comparison.csv    — per-scene metrics for all methods
    sentinel2/method_comparison.png    — bar chart + scatter F1 comparison
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import zoom

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR  = os.path.join(BASE_DIR, "..", "dem", "data")

METHODS = {
    "MNDWI":   "water_mask_{date}.tif",
    "WatNet":  "watnet_mask_{date}.tif",
    "Prithvi": "segformer_mask_{date}.tif",
}

# Independent ground truth options
GT_OPTIONS = {
    "s2_mndwi_consensus": os.path.join(DEM_DIR, "lake_mask_s2.tif"),   # MNDWI-derived (biased)
    "nhd_official":       os.path.join(DEM_DIR, "lake_mask_nhd.tif"),   # USGS NHD survey (independent)
}


def load_ref_mask(ref_path):
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference mask not found: {ref_path}")
    with rasterio.open(ref_path) as src:
        arr  = src.read(1).astype(np.uint8)
        meta = {
            "height": src.height, "width": src.width,
            "transform": src.transform, "crs": src.crs,
        }
        px_area_m2 = abs(src.transform.a) * abs(src.transform.e)
        epsg = src.crs.to_epsg()
        res  = src.res[0]
    area_ha = float(arr.sum() * px_area_m2 / 1e4)
    print(f"  {os.path.basename(ref_path)}: {area_ha:.1f} ha  (EPSG:{epsg}, {res:.1f}m px)")
    return arr, meta


def load_mask(path, ref_meta):
    """Reproject mask onto reference grid (CRS + transform + shape) then return binary + area."""
    from rasterio.warp import reproject, Resampling

    ref_h = ref_meta["height"]
    ref_w = ref_meta["width"]
    ref_t = ref_meta["transform"]
    ref_crs = ref_meta["crs"]

    with rasterio.open(path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_t   = src.transform
        src_crs = src.crs
        native_area_ha = float((src_arr > 0).sum() * abs(src_t.a) * abs(src_t.e) / 1e4)

    dst = np.zeros((ref_h, ref_w), dtype=np.float32)
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_t, src_crs=src_crs,
        dst_transform=ref_t, dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )
    binary = (dst > 0).astype(np.uint8)
    # Area from native resolution (before reprojection) — more accurate
    area_ha = native_area_ha
    return binary, area_ha


def metrics(pred, ref):
    p = pred.astype(bool); r = ref.astype(bool)
    tp = int((p & r).sum()); fp = int((p & ~r).sum())
    fn = int((~p & r).sum()); tn = int((~p & ~r).sum())
    prec = tp / (tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    iou  = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(prec,4), recall=round(rec,4),
                f1=round(f1,4), iou=round(iou,4))


def run_comparison(ref_arr, ref_meta, ref_label, dates):
    rows = []
    for date in dates:
        row = {"date": date}
        for method, pattern in METHODS.items():
            path = os.path.join(DATA_DIR, pattern.format(date=date))
            if not os.path.exists(path):
                continue
            pred, area = load_mask(path, ref_meta)
            m = metrics(pred, ref_arr)
            row[f"{method}_area_ha"]   = round(area, 2)
            row[f"{method}_f1"]        = m["f1"]
            row[f"{method}_iou"]       = m["iou"]
            row[f"{method}_precision"] = m["precision"]
            row[f"{method}_recall"]    = m["recall"]
            print(f"  [{date}] {method:8s}  area={area:6.1f} ha  "
                  f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
                  f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    dates = sorted({
        os.path.basename(f).replace("water_mask_","").replace(".tif","")
        for f in glob.glob(os.path.join(DATA_DIR, "water_mask_2*.tif"))
    })

    all_results = {}
    for gt_name, gt_path in GT_OPTIONS.items():
        if not os.path.exists(gt_path):
            print(f"Skipping {gt_name}: {gt_path} not found")
            continue
        print(f"\n{'='*65}")
        print(f"Ground truth: {gt_name}  ({gt_path.split('/')[-1]})")
        print(f"{'='*65}")
        ref, ref_meta = load_ref_mask(gt_path)
        print(f"Dates: {dates}\n")
        df = run_comparison(ref, ref_meta, gt_name, dates)
        all_results[gt_name] = df

        # Per-GT summary
        print(f"\nMEAN METRICS vs {gt_name}:")
        for method in METHODS:
            col = f"{method}_f1"
            if col in df.columns:
                print(f"  {method:8s}  F1={df[col].mean():.3f}  "
                      f"IoU={df[f'{method}_iou'].mean():.3f}  "
                      f"Prec={df[f'{method}_precision'].mean():.3f}  "
                      f"Rec={df[f'{method}_recall'].mean():.3f}")

    # Save per-GT CSVs
    for gt_name, df in all_results.items():
        out = os.path.join(BASE_DIR, f"method_comparison_{gt_name}.csv")
        df.to_csv(out, index=False)
        print(f"Saved → {out}")

    _plot_both(all_results)

    # Side-by-side summary table
    print("\n" + "="*70)
    print("SUMMARY: impact of ground truth choice on method ranking")
    print("="*70)
    print(f"{'Method':10s}  {'vs MNDWI-consensus':22s}  {'vs NHD official':22s}")
    print(f"{'':10s}  {'F1    IoU   Prec  Rec ':22s}  {'F1    IoU   Prec  Rec ':22s}")
    print("-"*70)
    for method in METHODS:
        parts = []
        for gt_name, df in all_results.items():
            if f"{method}_f1" in df.columns:
                parts.append(
                    f"{df[f'{method}_f1'].mean():.3f} "
                    f"{df[f'{method}_iou'].mean():.3f} "
                    f"{df[f'{method}_precision'].mean():.3f} "
                    f"{df[f'{method}_recall'].mean():.3f}"
                )
            else:
                parts.append("N/A")
        print(f"  {method:8s}  {parts[0] if len(parts)>0 else 'N/A':22s}  "
              f"{parts[1] if len(parts)>1 else 'N/A'}")


def _plot_both(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_colors = {"MNDWI": "#4C72B0", "WatNet": "#DD8452", "Prithvi": "#55A868"}
    gt_labels = {
        "s2_mndwi_consensus": "S2 MNDWI consensus (93.7 ha)\n⚠ biased: MNDWI-derived reference",
        "nhd_official":       "USGS NHD official survey (143.1 ha)\n✓ independent ground truth",
    }

    n_gt = len(all_results)
    fig, axes = plt.subplots(n_gt, 2, figsize=(16, 5 * n_gt))
    if n_gt == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Water Segmentation Method Comparison — Johns Lake, Winter Garden FL\n"
                 "Left: F1 per scene  |  Right: Mean F1 / IoU / Precision / Recall",
                 fontsize=11, fontweight="bold")

    for row_i, (gt_name, df) in enumerate(all_results.items()):
        dates = df["date"].tolist()
        x = np.arange(len(dates))
        width = 0.25
        gt_label = gt_labels.get(gt_name, gt_name)

        # Left: F1 bar chart per scene
        ax = axes[row_i, 0]
        for i, (method, color) in enumerate(method_colors.items()):
            col = f"{method}_f1"
            if col not in df.columns:
                continue
            vals = df[col].tolist()
            ax.bar(x + (i-1)*width, vals, width, label=method, color=color, alpha=0.85)
            ax.axhline(np.mean(vals), color=color, linestyle="--", lw=1.2, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([d[2:] for d in dates], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score"); ax.legend(fontsize=8)
        ax.set_title(f"F1 per scene\nGround truth: {gt_label}", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, (method, color) in enumerate(method_colors.items()):
            col = f"{method}_f1"
            if col in df.columns:
                ax.text(0.02, 0.97-i*0.08, f"{method} mean={df[col].mean():.3f}",
                        transform=ax.transAxes, fontsize=8, color=color,
                        va="top", fontweight="bold")

        # Right: mean metrics bar chart
        ax2 = axes[row_i, 1]
        metric_names = ["f1", "iou", "precision", "recall"]
        metric_labels = ["F1", "IoU", "Precision", "Recall"]
        x2 = np.arange(len(metric_names))
        for i, (method, color) in enumerate(method_colors.items()):
            means = [df[f"{method}_{m}"].mean() if f"{method}_{m}" in df.columns else 0
                     for m in metric_names]
            ax2.bar(x2 + (i-1)*0.25, means, 0.25, label=method, color=color, alpha=0.85)
            for j, v in enumerate(means):
                ax2.text(x2[j]+(i-1)*0.25, v+0.01, f"{v:.3f}", ha="center",
                         fontsize=6, color=color, fontweight="bold")
        ax2.set_xticks(x2); ax2.set_xticklabels(metric_labels)
        ax2.set_ylim(0, 1.15); ax2.set_ylabel("Score"); ax2.legend(fontsize=8)
        ax2.set_title(f"Mean metrics\nGround truth: {gt_label}", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "method_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}")


if __name__ == "__main__":
    main()
