"""
Visual QC for the segmentation — NAIP against the class map, side by side
=========================================================================
Class-area percentages can look entirely reasonable while the classification is wrong, so this
renders crops of the NAIP orthophoto next to the classes assigned to it, plus the canopy-height
model that drives the tree/grass split. Anything obviously broken — trees on roofs, roads
classified as water, a canopy layer offset from the imagery — shows up immediately here and in
no summary statistic.

Crops are chosen to cover the mix that matters rather than at random: the gauge, a residential
block, and the densest-canopy and most-impervious cells in the domain.

Usage:
    python3 segmentation/qc_preview.py --site site3
    python3 segmentation/qc_preview.py --site site3 --size 900
"""
import os
import sys
import argparse
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from segment_naip import CLASSES   # noqa: E402

COLORS = {
    0: "#000000",   # nodata
    1: "#2b6cb0",   # water
    2: "#c05621",   # building_roof
    3: "#4a5568",   # road_paved
    4: "#a0aec0",   # impervious_other
    5: "#22543d",   # tree_canopy
    6: "#68a357",   # shrub_scrub
    7: "#9ae6b4",   # grass_turf
    8: "#d6bd8c",   # bare_soil
    9: "#4fd1c5",   # wetland_marsh
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3")
    ap.add_argument("--size", type=int, default=800, help="crop size in NAIP pixels")
    args = ap.parse_args()

    rgb_p = os.path.join(PROJ_DIR, "site3_gee_creek", "imagery", "data", "naip_2021_RGB.tif")
    lc_p = os.path.join(DATA_DIR, f"landcover_0.6m_{args.site}.tif")
    chm_p = os.path.join(DATA_DIR, f"chm_2m_{args.site}.tif")
    for p in (rgb_p, lc_p, chm_p):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    from test_sites import get_site
    site = get_site(args.site)

    rgb_src = rasterio.open(rgb_p)
    lc_src = rasterio.open(lc_p)
    chm_src = rasterio.open(chm_p)
    S = args.size

    # Locate the gauge in NAIP pixel space.
    from pyproj import Transformer
    tr = Transformer.from_crs("epsg:4326", rgb_src.crs, always_xy=True)
    gx, gy = tr.transform(site["gauge_lon"], site["gauge_lat"])
    gcol, grow = ~rgb_src.transform * (gx, gy)

    # Find the most-canopy and most-impervious neighbourhoods from a decimated read of the
    # class map, so the crops are chosen by the data rather than by eye.
    lc_small = lc_src.read(1, out_shape=(lc_src.height // 16, lc_src.width // 16),
                           resampling=Resampling.mode)
    from scipy.ndimage import uniform_filter
    tree_d = uniform_filter((lc_small == 5).astype(np.float32), 24)
    imp_d = uniform_filter(np.isin(lc_small, [2, 3, 4]).astype(np.float32), 24)
    tr_r, tr_c = np.unravel_index(int(np.argmax(tree_d)), tree_d.shape)
    im_r, im_c = np.unravel_index(int(np.argmax(imp_d)), imp_d.shape)

    crops = [
        ("gauge 02234400", int(grow) - S // 2, int(gcol) - S // 2),
        ("densest canopy", tr_r * 16 - S // 2, tr_c * 16 - S // 2),
        ("most impervious", im_r * 16 - S // 2, im_c * 16 - S // 2),
    ]

    codes = sorted(CLASSES)
    cmap = ListedColormap([COLORS[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)

    fig, axes = plt.subplots(len(crops), 3, figsize=(15, 5 * len(crops)))
    for i, (label, r0, c0) in enumerate(crops):
        r0 = int(np.clip(r0, 0, rgb_src.height - S))
        c0 = int(np.clip(c0, 0, rgb_src.width - S))
        win = Window(c0, r0, S, S)
        wt = window_transform(win, rgb_src.transform)

        rgb = rgb_src.read(window=win).transpose(1, 2, 0)
        lc = lc_src.read(1, window=win)
        chm = np.full((S, S), np.nan, dtype=np.float32)
        reproject(chm_src.read(1).astype(np.float32), chm,
                  src_transform=chm_src.transform, src_crs=chm_src.crs,
                  dst_transform=wt, dst_crs=rgb_src.crs, resampling=Resampling.bilinear)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"NAIP 0.6 m — {label}", fontsize=10)
        axes[i, 1].imshow(lc, cmap=cmap, norm=norm, interpolation="nearest")
        axes[i, 1].set_title("surface class", fontsize=10)
        im = axes[i, 2].imshow(chm, cmap="viridis", vmin=0, vmax=25)
        axes[i, 2].set_title("LiDAR canopy height [m]", fontsize=10)
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])

    handles = [Patch(facecolor=COLORS[c], label=CLASSES[c]) for c in codes if c != 0]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"Segmentation QC — {args.site}  ({S*0.6:.0f} m crops)", fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.98])

    out = os.path.join(DATA_DIR, f"qc_preview_{args.site}.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {os.path.relpath(out, PROJ_DIR)}")
    for label, r0, c0 in crops:
        print(f"  crop: {label}")


if __name__ == "__main__":
    main()
