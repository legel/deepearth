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


def load_ref_mask():
    ref_path = os.path.join(DEM_DIR, "lake_mask_s2.tif")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"S2 reference mask not found: {ref_path}")
    with rasterio.open(ref_path) as src:
        arr  = src.read(1).astype(np.uint8)
        meta = {
            "height": src.height, "width": src.width,
            "transform": src.transform, "crs": src.crs,
        }
        px_area_m2 = abs(src.transform.a) * abs(src.transform.e)
    area_ha = float(arr.sum() * px_area_m2 / 1e4)
    print(f"Reference: lake_mask_s2.tif  ({area_ha:.1f} ha, S2-observed, "
          f"{src.crs.to_epsg()} at {src.res[0]:.1f}m)")
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


def main():
    ref, ref_meta = load_ref_mask()
    ref_shape = ref.shape

    # Discover all dates with at least one method mask
    dates = sorted({
        os.path.basename(f).replace("water_mask_","").replace(".tif","")
        for f in glob.glob(os.path.join(DATA_DIR, "water_mask_2*.tif"))
    })
    print(f"Dates: {dates}\n")

    rows = []
    for date in dates:
        row = {"date": date}
        for method, pattern in METHODS.items():
            path = os.path.join(DATA_DIR, pattern.format(date=date))
            if not os.path.exists(path):
                print(f"  [{date}] {method}: missing")
                continue
            pred, area = load_mask(path, ref_meta)
            m = metrics(pred, ref)
            row[f"{method}_area_ha"]  = round(area, 2)
            row[f"{method}_f1"]       = m["f1"]
            row[f"{method}_iou"]      = m["iou"]
            row[f"{method}_precision"]= m["precision"]
            row[f"{method}_recall"]   = m["recall"]
            print(f"  [{date}] {method:8s}  area={area:6.1f} ha  "
                  f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
                  f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(BASE_DIR, "method_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # Summary
    print("\n" + "="*65)
    print(f"MEAN METRICS vs S2-observed ground truth (lake_mask_s2.tif)")
    print("="*65)
    for method in METHODS:
        col = f"{method}_f1"
        if col in df.columns:
            f1  = df[col].mean()
            iou = df[f"{method}_iou"].mean()
            rec = df[f"{method}_recall"].mean()
            pre = df[f"{method}_precision"].mean()
            print(f"  {method:8s}  F1={f1:.3f}  IoU={iou:.3f}  "
                  f"Precision={pre:.3f}  Recall={rec:.3f}")

    _plot(df)


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_colors = {"MNDWI": "#4C72B0", "WatNet": "#DD8452", "Prithvi": "#55A868"}
    dates = df["date"].tolist()
    x = np.arange(len(dates))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Water Segmentation Method Comparison\n"
                 "Ground truth: S2-observed lake_mask_s2.tif (94.4 ha)",
                 fontsize=12, fontweight="bold")

    metrics_to_plot = [
        ("f1",        "F1 Score",  axes[0,0]),
        ("iou",       "IoU",       axes[0,1]),
        ("precision", "Precision", axes[1,0]),
        ("recall",    "Recall",    axes[1,1]),
    ]

    for metric, label, ax in metrics_to_plot:
        for i, (method, color) in enumerate(method_colors.items()):
            col = f"{method}_{metric}"
            if col not in df.columns:
                continue
            vals = df[col].tolist()
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=method, color=color, alpha=0.85)
            # Mean line
            mean_val = np.mean(vals)
            ax.axhline(mean_val, color=color, linestyle="--", linewidth=1, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([d[2:] for d in dates], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Annotate means in legend area
        for i, (method, color) in enumerate(method_colors.items()):
            col = f"{method}_{metric}"
            if col in df.columns:
                ax.text(0.02, 0.97 - i*0.07,
                        f"{method} mean={df[col].mean():.3f}",
                        transform=ax.transAxes, fontsize=8,
                        color=color, va="top", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "method_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
