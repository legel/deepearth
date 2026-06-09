"""
Spatial Comparison Visualization — Water Boundary Outlines on S2 RGB
=====================================================================
Shows the actual lake boundary detected by each method overlaid on
Sentinel-2 true-color imagery, with the USGS NHD official polygon
as ground truth reference.

Outputs:
    sentinel2/boundary_comparison.png  — 4 scenes × 1 panel each,
        RGB + NHD outline + MNDWI/WatNet/Prithvi outlines
    sentinel2/boundary_comparison_grid.png  — compact 2×2 scene grid
"""

import os
import numpy as np
import glob
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from scipy.ndimage import binary_dilation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR  = os.path.join(BASE_DIR, "..", "dem", "data")

# Methods: label, mask filename pattern, color
METHODS = [
    ("NHD official\n(ground truth)", None,                         "#FF3333", 2.5),
    ("MNDWI",                        "water_mask_{date}.tif",       "#4477CC", 1.5),
    ("WatNet (CNN)",                  "watnet_mask_{date}.tif",      "#FF8800", 1.5),
    ("Prithvi-EO-2.0",               "segformer_mask_{date}.tif",   "#22AA55", 1.5),
]

# Representative scenes — pick clear-sky ones
DISPLAY_DATES = ["20220329", "20230314", "20240412", "20231015"]


def load_s2_rgb(date):
    """Load S2 RGB (B04/B03/B02), return (H,W,3) float32 [0,1], transform, crs."""
    bands = {}
    for b in ["B04", "B03", "B02"]:
        p = os.path.join(DATA_DIR, f"s2_{date}_{b}.tif")
        if not os.path.exists(p):
            return None, None, None
        with rasterio.open(p) as src:
            bands[b] = src.read(1).astype(np.float32)
            t, c = src.transform, src.crs
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1) / 10000.0
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
    return rgb, t, c


def load_mask_reprojected(path, target_shape, target_transform, target_crs):
    """Reproject any mask to the S2 RGB grid. Returns binary (H,W) uint8."""
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        dst = np.zeros(target_shape, dtype=np.float32)
        reproject(source=arr, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=target_transform, dst_crs=target_crs,
                  resampling=Resampling.nearest)
    return (dst > 0).astype(np.uint8)


def mask_to_boundary(mask, thickness=1):
    """Convert binary mask to boundary (outline) via dilation - erosion."""
    dilated = binary_dilation(mask, iterations=thickness)
    return (dilated.astype(np.int8) - mask.astype(np.int8)).clip(0).astype(bool)


def make_boundary_rgba(boundary, color_hex, alpha=0.9):
    """Return RGBA image with boundary pixels in given color."""
    rgba = np.zeros((*boundary.shape, 4), dtype=np.float32)
    c = to_rgba(color_hex, alpha=alpha)
    rgba[boundary] = c
    return rgba


def plot_scene(ax, date, rgb, t, crs):
    """Draw one scene: RGB background + all method boundaries."""
    ax.imshow(rgb, origin="upper", extent=[0, rgb.shape[1], rgb.shape[0], 0])

    # NHD ground truth
    nhd_path = os.path.join(DEM_DIR, "lake_mask_nhd.tif")
    if os.path.exists(nhd_path):
        nhd = load_mask_reprojected(nhd_path, rgb.shape[:2], t, crs)
        if nhd is not None:
            boundary = mask_to_boundary(nhd, thickness=3)
            rgba = make_boundary_rgba(boundary, "#FF3333", alpha=1.0)
            ax.imshow(rgba, origin="upper",
                      extent=[0, rgb.shape[1], rgb.shape[0], 0], zorder=10)

    # Method masks
    for label, pattern, color, _ in METHODS[1:]:  # skip NHD, already drawn
        if pattern is None:
            continue
        path = os.path.join(DATA_DIR, pattern.format(date=date))
        mask = load_mask_reprojected(path, rgb.shape[:2], t, crs)
        if mask is not None:
            boundary = mask_to_boundary(mask, thickness=2)
            rgba = make_boundary_rgba(boundary, color, alpha=0.85)
            ax.imshow(rgba, origin="upper",
                      extent=[0, rgb.shape[1], rgb.shape[0], 0], zorder=11)

    # Date label
    ax.set_title(f"Scene {date[:4]}-{date[4:6]}-{date[6:]}", fontsize=9, pad=3)
    ax.axis("off")


def main():
    # -----------------------------------------------------------------------
    # Figure 1: 1×4 strip for 4 scenes
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(DISPLAY_DATES), figsize=(5 * len(DISPLAY_DATES), 5))
    fig.suptitle(
        "Water Boundary Comparison — Johns Lake, Winter Garden FL\n"
        "ROI = western bay of Johns Lake (143 ha of 1044 ha full lake)\n"
        "Red = NHD official survey (ground truth)  |  "
        "Blue = MNDWI  |  Orange = WatNet  |  Green = Prithvi-EO-2.0",
        fontsize=9, fontweight="bold"
    )

    for ax, date in zip(axes, DISPLAY_DATES):
        rgb, t, crs = load_s2_rgb(date)
        if rgb is None:
            ax.set_title(f"{date}\n(missing)", fontsize=8)
            ax.axis("off")
            continue
        plot_scene(ax, date, rgb, t, crs)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#FF3333", label="NHD official (ground truth)"),
        mpatches.Patch(color="#4477CC", label="MNDWI"),
        mpatches.Patch(color="#FF8800", label="WatNet (CNN, 2021)"),
        mpatches.Patch(color="#22AA55", label="Prithvi-EO-2.0 (transformer, 2024)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out1 = os.path.join(BASE_DIR, "boundary_comparison.png")
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out1}")

    # -----------------------------------------------------------------------
    # Figure 2: 2×2 grid with F1 annotations
    # -----------------------------------------------------------------------
    import pandas as pd
    nhd_csv = os.path.join(BASE_DIR, "method_comparison_nhd_official.csv")
    df_nhd = pd.read_csv(nhd_csv) if os.path.exists(nhd_csv) else None

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(
        "Water Segmentation Boundary Comparison — vs USGS NHD Official Ground Truth\n"
        "Johns Lake western bay, Winter Garden FL  |  AOI covers 143 ha of 1044 ha full lake",
        fontsize=10, fontweight="bold"
    )

    for ax, date in zip(axes2.ravel(), DISPLAY_DATES):
        rgb, t, crs = load_s2_rgb(date)
        if rgb is None:
            ax.axis("off"); continue
        plot_scene(ax, date, rgb, t, crs)

        # Add per-method F1 annotation from NHD comparison
        if df_nhd is not None:
            row = df_nhd[df_nhd["date"] == int(date)]
            if not row.empty:
                txt_lines = ["vs NHD:"]
                for col, label, color in [
                    ("MNDWI_f1",   "MNDWI",   "#4477CC"),
                    ("WatNet_f1",  "WatNet",  "#FF8800"),
                    ("Prithvi_f1", "Prithvi", "#22AA55"),
                ]:
                    if col in row.columns:
                        val = row[col].values[0]
                        txt_lines.append(f"{label}: F1={val:.3f}")

                y_pos = 0.98
                for line in txt_lines:
                    color = "white" if line == "vs NHD:" else (
                        "#4477CC" if "MNDWI" in line else
                        "#FF8800" if "WatNet" in line else "#22AA55"
                    )
                    ax.text(0.02, y_pos, line, transform=ax.transAxes,
                            fontsize=7.5, color=color, va="top",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="black", alpha=0.55, ec="none"))
                    y_pos -= 0.09

    legend_handles = [
        mpatches.Patch(color="#FF3333", label="NHD official (ground truth)"),
        mpatches.Patch(color="#4477CC", label="MNDWI"),
        mpatches.Patch(color="#FF8800", label="WatNet (CNN, 2021)"),
        mpatches.Patch(color="#22AA55", label="Prithvi-EO-2.0 (IBM/NASA, 2024)"),
    ]
    fig2.legend(handles=legend_handles, loc="lower center", ncol=4,
                fontsize=9, framealpha=0.9,
                bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out2 = os.path.join(BASE_DIR, "boundary_comparison_grid.png")
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")


if __name__ == "__main__":
    main()
