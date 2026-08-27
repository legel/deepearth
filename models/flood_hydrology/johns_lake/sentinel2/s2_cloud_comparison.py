"""
Cloud Masking Comparison Figure
================================
Shows how two-layer cloud masking (SCL + s2cloudless) works across all scenes.

Two-layer strategy:
  Layer 1 — SCL (Scene Classification Layer): Sentinel-2 L2A built-in pixel
    classifier. Classes 0,1,3,8,9,10 → masked (no-data, defective, cloud shadow,
    cloud medium, cloud high, thin cirrus).
  Layer 2 — s2cloudless (Sinergise, LightGBM): ML cloud probability using 10
    S2 bands (B01,B02,B04,B05,B08,B8A,B09,B10=0,B11,B12). Threshold 0.4.
  Final mask = SCL OR s2cloudless → maximum of both layers.

Output:
    sentinel2/cloud_comparison.png  — per-scene cloud coverage overview
    sentinel2/cloud_timeline.png    — timeline of cloud % + water area per date
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_s2_rgb(date, clip_pct=(2, 98)):
    bands = {}
    for b in ["B04", "B03", "B02"]:
        p = os.path.join(DATA_DIR, f"s2_{date}_{b}.tif")
        if not os.path.exists(p):
            return None, None, None
        with rasterio.open(p) as src:
            bands[b] = src.read(1).astype(np.float32)
            t, c = src.transform, src.crs
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1) / 10000.0
    lo, hi = np.percentile(rgb, clip_pct[0]), np.percentile(rgb, clip_pct[1])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb, t, c


def load_mask(date, prefix):
    p = os.path.join(DATA_DIR, f"{prefix}_{date}.tif")
    if not os.path.exists(p):
        return None
    with rasterio.open(p) as src:
        return src.read(1)


def build_scl_mask_from_raw(date):
    """Re-derive SCL mask from raw SCL band (without cloud_mask file)."""
    scl_path = os.path.join(DATA_DIR, f"s2_{date}_SCL.tif")
    if not os.path.exists(scl_path):
        return None
    CLOUD_CLASSES = {0, 1, 3, 8, 9, 10}
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    mask = np.zeros(scl.shape, np.uint8)
    for c in CLOUD_CLASSES:
        mask |= (scl == c)
    return mask


def main():
    dates = sorted({
        os.path.basename(f).split("_B04.tif")[0].replace("s2_", "")
        for f in glob.glob(os.path.join(DATA_DIR, "s2_*_B04.tif"))
    })
    print(f"Scenes: {dates}")

    # ── Figure 1: grid of scenes with RGB + cloud overlay ───────────────────
    n = len(dates)
    n_cols = 4   # per row: RGB, SCL mask, combined mask, cloud-masked water
    fig = plt.figure(figsize=(5 * n_cols, 3.5 * n + 0.8))
    gs = gridspec.GridSpec(n + 1, n_cols, figure=fig,
                           hspace=0.35, wspace=0.05,
                           height_ratios=[0.12] + [1] * n)

    # Column headers
    col_titles = [
        "Sentinel-2 True Color (RGB)",
        "Layer 1: SCL Cloud Mask\n(classes 0,1,3,8,9,10)",
        "Layer 2: s2cloudless ML\n(LightGBM, threshold 0.4)",
        "Final Combined Mask\n(SCL OR s2cloudless)",
    ]
    for ci, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, ci])
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=8.5,
                fontweight="bold", transform=ax.transAxes, wrap=True)
        ax.axis("off")

    summary_rows = []
    for ri, date in enumerate(dates):
        rgb, t, crs = load_s2_rgb(date)
        if rgb is None:
            for ci in range(n_cols):
                ax = fig.add_subplot(gs[ri + 1, ci])
                ax.text(0.5, 0.5, f"{date}\n(missing)", ha="center", va="center", fontsize=7)
                ax.axis("off")
            continue

        H, W = rgb.shape[:2]

        def _reproject_to_rgb(arr, src_t, src_crs, dst_t, dst_crs, H, W):
            if arr is None:
                return np.zeros((H, W), dtype=np.uint8)
            if arr.shape == (H, W):
                return arr
            out = np.zeros((H, W), dtype=np.float32)
            reproject(source=arr.astype(np.float32), destination=out,
                      src_transform=src_t, src_crs=src_crs,
                      dst_transform=dst_t, dst_crs=dst_crs,
                      resampling=Resampling.nearest)
            return (out > 0.5).astype(np.uint8)

        # Combined mask: prefer pre-computed cloud_mask_{date}.tif
        combined_raw = load_mask(date, "cloud_mask")
        if combined_raw is not None and combined_raw.shape != (H, W):
            b02_path = os.path.join(DATA_DIR, f"s2_{date}_B02.tif")
            with rasterio.open(b02_path) as src:
                b02_t, b02_crs = src.transform, src.crs
            combined = _reproject_to_rgb(combined_raw, b02_t, b02_crs, t, crs, H, W)
        elif combined_raw is not None:
            combined = combined_raw.astype(np.uint8)
        else:
            combined = np.zeros((H, W), np.uint8)

        # SCL-only mask (Layer 1 alone for comparison)
        scl_raw = build_scl_mask_from_raw(date)
        scl_path = os.path.join(DATA_DIR, f"s2_{date}_SCL.tif")
        if scl_raw is not None and scl_raw.shape != (H, W):
            with rasterio.open(scl_path) as src:
                scl_t_file, scl_crs_file = src.transform, src.crs
            scl_mask = _reproject_to_rgb(scl_raw, scl_t_file, scl_crs_file, t, crs, H, W)
        else:
            scl_mask = scl_raw if scl_raw is not None else np.zeros((H, W), np.uint8)

        # s2cloudless mask (Layer 2 alone for comparison)
        cloud_prob_raw = load_mask(date, "cloud_prob")
        if cloud_prob_raw is not None and cloud_prob_raw.shape != (H, W):
            b02_path = os.path.join(DATA_DIR, f"s2_{date}_B02.tif")
            with rasterio.open(b02_path) as src:
                b02_t, b02_crs = src.transform, src.crs
            s2cl_mask = _reproject_to_rgb((cloud_prob_raw > 0.4).astype(np.uint8),
                                           b02_t, b02_crs, t, crs, H, W)
        elif cloud_prob_raw is not None:
            s2cl_mask = (cloud_prob_raw > 0.4).astype(np.uint8)
        else:
            s2cl_mask = np.zeros((H, W), np.uint8)

        # Water mask
        water = load_mask(date, "water_mask")
        water_clear = np.where(combined == 0, water, 0) if water is not None else None

        # Statistics
        total_px = H * W
        scl_pct  = scl_mask.sum() / total_px * 100
        s2cl_pct = s2cl_mask.sum() / total_px * 100
        comb_pct = combined.sum() / total_px * 100
        water_ha = float(water.sum() * abs(t.a) * abs(t.e) / 1e4) if water is not None else np.nan
        summary_rows.append({
            "date": date,
            "cloud_pct_scl": round(scl_pct, 2),
            "cloud_pct_s2cloudless": round(s2cl_pct, 2),
            "cloud_pct_combined": round(comb_pct, 2),
            "water_ha_mndwi": round(water_ha, 2),
        })
        print(f"  {date}  SCL={scl_pct:.1f}%  s2cl={s2cl_pct:.1f}%  comb={comb_pct:.1f}%")

        # Overlay cloud mask on RGB (red tint)
        def overlay_mask_on_rgb(rgb_img, mask, color=(1, 0, 0), alpha=0.5):
            out = rgb_img.copy()
            m = mask.astype(bool)
            for ch, cv in enumerate(color):
                out[m, ch] = rgb_img[m, ch] * (1 - alpha) + cv * alpha
            return out

        rgb_scl      = overlay_mask_on_rgb(rgb, scl_mask, color=(1, 0.2, 0))
        rgb_s2cl     = overlay_mask_on_rgb(rgb, s2cl_mask, color=(0.8, 0, 0.8))
        rgb_combined = overlay_mask_on_rgb(rgb, combined, color=(1, 0.1, 0.1))

        date_str = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        for ci, (img, overlay_pct, subtitle) in enumerate([
            (rgb,          None,      f"{date_str}"),
            (rgb_scl,      scl_pct,   f"SCL: {scl_pct:.1f}% cloud"),
            (rgb_s2cl,     s2cl_pct,  f"s2cloudless: {s2cl_pct:.1f}% cloud"),
            (rgb_combined, comb_pct,  f"Combined: {comb_pct:.1f}% cloud"),
        ]):
            ax = fig.add_subplot(gs[ri + 1, ci])
            ax.imshow(img, origin="upper")
            ax.set_title(subtitle, fontsize=7.5, pad=2)
            ax.axis("off")
        # Add water area annotation on the RGB panel
        if water_ha > 0:
            ax0 = fig.add_subplot(gs[ri + 1, 0])
            ax0.text(0.02, 0.03, f"MNDWI water: {water_ha:.0f} ha",
                     transform=ax0.transAxes, fontsize=6.5, color="cyan",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5))

    # Legend
    legend_handles = [
        mpatches.Patch(color=(1, 0.2, 0), alpha=0.7, label="SCL cloud/shadow (Layer 1)"),
        mpatches.Patch(color=(0.8, 0, 0.8), alpha=0.7, label="s2cloudless ML cloud (Layer 2)"),
        mpatches.Patch(color=(1, 0.1, 0.1), alpha=0.7, label="Combined final mask"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Sentinel-2 Two-Layer Cloud Masking — Johns Lake, Winter Garden FL\n"
        "Layer 1: SCL pixel classifier (built-in L2A)  |  "
        "Layer 2: s2cloudless LightGBM ML model (10-band)  |  "
        "Final = SCL OR s2cloudless",
        fontsize=10, fontweight="bold", y=0.995
    )

    out1 = os.path.join(BASE_DIR, "cloud_comparison.png")
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out1}")

    # ── Figure 2: Timeline bar chart ─────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(BASE_DIR, "cloud_summary.csv"), index=False)

    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig2.suptitle(
        "Cloud Coverage Timeline — Sentinel-2 Scenes, Johns Lake FL\n"
        "All 11 scenes pre-filtered to ≤10% scene-level cloud cover at download",
        fontsize=10, fontweight="bold"
    )
    x = np.arange(len(df))
    xtick_labels = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in df["date"]]

    # Top: stacked cloud fractions
    ax_top.bar(x, df["cloud_pct_scl"], label="SCL (Layer 1)", color="#E74C3C", alpha=0.85)
    ax_top.bar(x, (df["cloud_pct_combined"] - df["cloud_pct_scl"]).clip(0),
               bottom=df["cloud_pct_scl"],
               label="s2cloudless adds (Layer 2)", color="#8E44AD", alpha=0.85)
    ax_top.set_ylabel("Cloud-masked pixels (%)", fontsize=9)
    ax_top.legend(fontsize=8.5, loc="upper right")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.set_title("Per-scene cloud coverage after two-layer masking", fontsize=9)
    for i, row in df.iterrows():
        ax_top.text(i, row["cloud_pct_combined"] + 0.3,
                    f"{row['cloud_pct_combined']:.1f}%", ha="center", fontsize=6.5)

    # Bottom: MNDWI water area
    ax_bot.bar(x, df["water_ha_mndwi"], color="#2E86C1", alpha=0.85, label="MNDWI water area")
    ax_bot.set_ylabel("MNDWI water area (ha)", fontsize=9)
    ax_bot.legend(fontsize=8.5, loc="upper right")
    ax_bot.grid(axis="y", alpha=0.3)
    ax_bot.set_title("MNDWI-derived water area per scene (cloud pixels excluded)", fontsize=9)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(xtick_labels, rotation=40, ha="right", fontsize=8)
    for i, row in df.iterrows():
        if not np.isnan(row["water_ha_mndwi"]):
            ax_bot.text(i, row["water_ha_mndwi"] + 1,
                        f"{row['water_ha_mndwi']:.0f}", ha="center", fontsize=6.5)

    plt.tight_layout()
    out2 = os.path.join(BASE_DIR, "cloud_timeline.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")
    print("\nCloud summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
