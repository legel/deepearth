"""
Fair Method Comparison — All Methods vs NHD Ground Truth + Water Extent Timeseries
===================================================================================
Produces two journal-quality figures:

Figure 1  — Method intercomparison on 11 manually-curated scenes (all 4 methods)
             Ground truth: lake_mask_nhd.tif (USGS NHD, independent)
             Output: sentinel2/method_comparison_selected.png

Figure 2  — Lake water extent timeseries (OWM, all kept scenes)
             OWM area (primary), MNDWI area scatter (11 pts), NHD reference line,
             Florida wet-season shading
             Output: sentinel2/water_extent_timeseries.png

Usage:
    python3 sentinel2/compare_methods.py
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR  = os.path.join(BASE_DIR, "..", "dem", "data")

METHODS = {
    "MNDWI":         "water_mask_{date}.tif",
    "WatNet":        "watnet_mask_{date}.tif",
    "Prithvi":       "prithvi_mask_{date}.tif",
    "OmniWaterMask": "omniwatermask_mask_{date}.tif",
}

# Tol bright colorblind-safe palette, one per method
METHOD_COLORS = {
    "MNDWI":         "#4477AA",
    "WatNet":        "#EE6677",
    "Prithvi":       "#228833",
    "OmniWaterMask": "#CCBB44",
}

NHD_GT_PATH = os.path.join(DEM_DIR, "lake_mask_nhd.tif")

# ── Journal style helpers ────────────────────────────────────────────────────

RC = {
    "font.family":      "DejaVu Sans",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.25,
    "grid.linewidth":   0.6,
}


def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)


# ── Core comparison helpers ──────────────────────────────────────────────────

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
    return arr, meta, area_ha


def load_mask(path, ref_meta):
    """Reproject mask onto reference grid then return binary + native area."""
    from rasterio.warp import reproject, Resampling

    ref_h, ref_w = ref_meta["height"], ref_meta["width"]
    ref_t, ref_crs = ref_meta["transform"], ref_meta["crs"]

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
    return binary, native_area_ha


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


def run_comparison(ref_arr, ref_meta, dates):
    rows = []
    for dt in dates:
        row = {"date": dt}
        for method, pattern in METHODS.items():
            path = os.path.join(DATA_DIR, pattern.format(date=dt))
            if not os.path.exists(path):
                continue
            pred, area = load_mask(path, ref_meta)
            m = metrics(pred, ref_arr)
            row[f"{method}_area_ha"]   = round(area, 2)
            row[f"{method}_f1"]        = m["f1"]
            row[f"{method}_iou"]        = m["iou"]
            row[f"{method}_precision"] = m["precision"]
            row[f"{method}_recall"]    = m["recall"]
            print(f"  [{dt}] {method:14s}  area={area:6.1f} ha  "
                  f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
                  f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Figure 1 — Method intercomparison (11 scenes, NHD GT) ───────────────────

def plot_selected_dates(df, nhd_area_ha):
    """Grouped F1 bar chart (left) + mean metric summary (right)."""
    dates = df["date"].tolist()
    xlabels = [f"{d[2:4]}-{d[4:6]}-{d[6:]}" for d in dates]
    x = np.arange(len(dates))
    n_methods = len(METHOD_COLORS)
    width = 0.18
    offsets = np.linspace(-(n_methods-1)/2 * width, (n_methods-1)/2 * width, n_methods)

    with plt.rc_context(RC):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5),
                                                  gridspec_kw={"width_ratios": [2, 1]})
        fig.suptitle(
            "Water Segmentation Method Comparison — Johns Lake, Winter Garden FL\n"
            f"Ground truth: USGS NHD official survey ({nhd_area_ha:.1f} ha, independent)  |  "
            "11 scenes (2022–2024), all methods present",
            fontsize=9, fontweight="bold", y=1.01,
        )

        # Left: F1 per scene
        for i, (method, color) in enumerate(METHOD_COLORS.items()):
            col = f"{method}_f1"
            if col not in df.columns:
                continue
            vals = df[col].tolist()
            ax_left.bar(x + offsets[i], vals, width, color=color, alpha=0.88, label=method)
            mean_f1 = np.nanmean(vals)
            ax_left.axhline(mean_f1, color=color, linestyle="--", lw=1.0, alpha=0.7)

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(xlabels, rotation=45, ha="right")
        ax_left.set_ylim(0, 1.05)
        ax_left.set_ylabel("F1 Score")
        ax_left.set_xlabel("Scene date (YY-MM-DD)")
        ax_left.set_title("F1 score per scene  (dashed = method mean)")
        _apply_style(ax_left)

        # Right: mean F1/IoU/Prec/Rec per method
        metric_keys   = ["f1", "iou", "precision", "recall"]
        metric_labels = ["F1", "IoU", "Prec", "Rec"]
        x2 = np.arange(len(metric_keys))
        for i, (method, color) in enumerate(METHOD_COLORS.items()):
            means = [df[f"{method}_{m}"].mean() if f"{method}_{m}" in df.columns else 0
                     for m in metric_keys]
            bars = ax_right.bar(x2 + offsets[i], means, width, color=color, alpha=0.88, label=method)
            for j, v in enumerate(means):
                ax_right.text(x2[j] + offsets[i], v + 0.01, f"{v:.2f}",
                              ha="center", fontsize=5.5, color=color, fontweight="bold")

        ax_right.set_xticks(x2)
        ax_right.set_xticklabels(metric_labels)
        ax_right.set_ylim(0, 1.15)
        ax_right.set_ylabel("Score")
        ax_right.set_title("Mean metrics across all scenes")
        _apply_style(ax_right)

        legend_handles = [mpatches.Patch(color=c, label=m) for m, c in METHOD_COLORS.items()]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=8, framealpha=0.95, edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout()
        out = os.path.join(BASE_DIR, "method_comparison_selected.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nFigure 1 → {out}")


# ── Method timeseries helpers (all kept scenes) ──────────────────────────────

def build_mndwi_timeseries(ranking_csv):
    """Compute MNDWI water area (ha) for all keep=True scenes from raw S2 bands."""
    if not os.path.exists(ranking_csv):
        return None
    df_rank = pd.read_csv(ranking_csv)
    kept = df_rank[df_rank["keep"] == True]["date"].astype(str).tolist()
    print(f"  Computing MNDWI area for {len(kept)} kept scenes …")
    rows = []
    for dt in kept:
        b03_path = os.path.join(DATA_DIR, f"s2_{dt}_B03.tif")
        b11_path = os.path.join(DATA_DIR, f"s2_{dt}_B11.tif")
        if not (os.path.exists(b03_path) and os.path.exists(b11_path)):
            continue
        with rasterio.open(b03_path) as src:
            b03 = src.read(1).astype(np.float32)
            px_area_m2 = abs(src.transform.a) * abs(src.transform.e)
        with rasterio.open(b11_path) as src:
            b11 = src.read(1).astype(np.float32)
            if b11.shape != b03.shape:
                from rasterio.warp import reproject, Resampling
                b11_out = np.zeros_like(b03)
                with rasterio.open(b03_path) as ref:
                    with rasterio.open(b11_path) as src11:
                        reproject(src11.read(1).astype(np.float32), b11_out,
                                  src_transform=src11.transform, src_crs=src11.crs,
                                  dst_transform=ref.transform, dst_crs=ref.crs,
                                  resampling=Resampling.bilinear)
                b11 = b11_out
        denom = b03 + b11
        with np.errstate(invalid="ignore", divide="ignore"):
            mndwi = np.where(denom > 0, (b03 - b11) / denom, 0.0)
        area_ha = float((mndwi > 0).sum() * px_area_m2 / 1e4)
        rows.append({"date": dt, "area_ha": area_ha})
    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    print(f"    → {len(df)} scenes")
    return df


def build_mask_timeseries(pattern, ranking_csv, label):
    """Read pre-computed binary mask TIFs for all keep=True scenes, return area timeseries."""
    if not os.path.exists(ranking_csv):
        return None
    df_rank = pd.read_csv(ranking_csv)
    kept = df_rank[df_rank["keep"] == True]["date"].astype(str).tolist()
    print(f"  Reading {label} masks for {len(kept)} kept scenes …")
    rows = []
    for dt in kept:
        path = os.path.join(DATA_DIR, pattern.format(date=dt))
        if not os.path.exists(path):
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.uint8)
            px_area_m2 = abs(src.transform.a) * abs(src.transform.e)
        area_ha = float(arr.sum() * px_area_m2 / 1e4)
        rows.append({"date": dt, "area_ha": area_ha})
    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    print(f"    → {len(df)} scenes")
    return df


# ── Figure 2 — Water extent timeseries (OWM, all kept scenes) ───────────────

def _wet_season_spans(year_min, year_max):
    """Return list of (start, end) as date objects for FL wet seasons."""
    spans = []
    for yr in range(year_min, year_max + 1):
        spans.append((date(yr, 6, 1), date(yr, 10, 31)))
    return spans


def plot_timeseries(timeseries_csv, ranking_csv, nhd_area_ha):
    """All-method water-area timeseries + NHD reference + season shading."""
    if not os.path.exists(timeseries_csv):
        print(f"Skipping Figure 2: {timeseries_csv} not found")
        return

    df_owm = pd.read_csv(timeseries_csv)
    df_owm["date_dt"] = pd.to_datetime(df_owm["date"].astype(str), format="%Y%m%d")
    df_owm = df_owm.sort_values("date_dt")

    print("\nBuilding method timeseries for Figure 2 …")
    df_mndwi   = build_mndwi_timeseries(ranking_csv)
    df_watnet  = build_mask_timeseries("watnet_mask_{date}.tif",        ranking_csv, "WatNet")
    df_prithvi = build_mask_timeseries("prithvi_mask_{date}.tif",       ranking_csv, "Prithvi")

    method_series = [
        ("MNDWI",         df_mndwi,   1.0, 10),
        ("WatNet",        df_watnet,  1.0, 10),
        ("Prithvi",       df_prithvi, 1.0, 10),
        ("OmniWaterMask", None,       1.4, 14),  # OWM uses separate df_owm
    ]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 4))

        year_min = df_owm["date_dt"].dt.year.min()
        year_max = df_owm["date_dt"].dt.year.max()
        for ws_start, ws_end in _wet_season_spans(year_min, year_max):
            ax.axvspan(pd.Timestamp(ws_start), pd.Timestamp(ws_end),
                       alpha=0.10, color="#4DBBFF", zorder=0)

        zorder = 3
        counts = {}
        for method, df_m, lw, ms in method_series:
            color = METHOD_COLORS[method]
            if method == "OmniWaterMask":
                df_plot = df_owm.sort_values("date_dt")
                y_col = "water_area_ha"
            else:
                if df_m is None or len(df_m) == 0:
                    continue
                df_plot = df_m.sort_values("date_dt")
                y_col = "area_ha"
            n = len(df_plot)
            counts[method] = n
            ax.plot(df_plot["date_dt"], df_plot[y_col],
                    color=color, lw=lw, alpha=0.80, zorder=zorder,
                    label=f"{method} (n={n})")
            ax.scatter(df_plot["date_dt"], df_plot[y_col],
                       color=color, s=ms, alpha=0.80, zorder=zorder + 1)
            zorder += 2

        ax.axhline(nhd_area_ha, color="#888888", linestyle="--", lw=1.0, zorder=zorder,
                   label=f"NHD reference ({nhd_area_ha:.0f} ha)")

        wet_patch = mpatches.Patch(color="#4DBBFF", alpha=0.35,
                                   label="FL wet season (Jun–Oct)")

        ax.set_xlabel("Date")
        ax.set_ylabel("Water area (ha)")
        ax.set_title(
            "Lake Water Extent Timeseries — Johns Lake, Winter Garden FL\n"
            "MNDWI | WatNet | Prithvi-EO-2.0 | OmniWaterMask  vs  NHD survey baseline",
            pad=32,
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.append(wet_patch)
        labels.append("FL wet season (Jun–Oct)")
        ax.legend(handles=handles, labels=labels,
                  loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=5, fontsize=7, framealpha=0.95, edgecolor="#cccccc",
                  bbox_transform=ax.transAxes, borderaxespad=0)

        _apply_style(ax)
        fig.autofmt_xdate(rotation=30, ha="right")
        plt.tight_layout()
        out = os.path.join(BASE_DIR, "water_extent_timeseries.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure 2 → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Discover the 11 original scenes (all 4 methods must exist)
    all_dates = sorted({
        os.path.basename(f).replace("water_mask_", "").replace(".tif", "")
        for f in glob.glob(os.path.join(DATA_DIR, "water_mask_2*.tif"))
    })
    # Filter to dates where at least MNDWI + OWM both exist
    dates = [d for d in all_dates
             if os.path.exists(os.path.join(DATA_DIR, f"omniwatermask_mask_{d}.tif"))]

    print(f"\nScenes with all methods: {len(dates)}")
    print(f"Dates: {dates}\n")

    # Load NHD reference
    print("Loading NHD ground truth:")
    ref_arr, ref_meta, nhd_area_ha = load_ref_mask(NHD_GT_PATH)

    # Run comparison vs NHD
    print(f"\n{'='*65}")
    print(f"Ground truth: NHD official  ({NHD_GT_PATH.split('/')[-1]})")
    print(f"{'='*65}")
    df_nhd = run_comparison(ref_arr, ref_meta, dates)

    # Save CSV
    csv_out = os.path.join(BASE_DIR, "method_comparison_nhd.csv")
    df_nhd.to_csv(csv_out, index=False)
    print(f"\nSaved CSV → {csv_out}")

    # Print mean metrics
    print("\nMEAN METRICS vs NHD:")
    for method in METHODS:
        col = f"{method}_f1"
        if col in df_nhd.columns:
            print(f"  {method:14s}  F1={df_nhd[col].mean():.3f}  "
                  f"IoU={df_nhd[f'{method}_iou'].mean():.3f}  "
                  f"Prec={df_nhd[f'{method}_precision'].mean():.3f}  "
                  f"Rec={df_nhd[f'{method}_recall'].mean():.3f}")

    # Figure 1 — method intercomparison
    plot_selected_dates(df_nhd, nhd_area_ha)

    # Figure 2 — water extent timeseries
    timeseries_csv = os.path.join(DATA_DIR, "water_extent_timeseries.csv")
    ranking_csv    = os.path.join(DATA_DIR, "scene_ranking.csv")
    plot_timeseries(timeseries_csv, ranking_csv, nhd_area_ha)


if __name__ == "__main__":
    main()
