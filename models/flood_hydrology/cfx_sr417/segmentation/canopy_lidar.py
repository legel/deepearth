"""
Canopy height and cover from the 2018 LiDAR point cloud
=======================================================
Tree canopy is absent from this project's physics mesh entirely: the surface is bare-earth DEM
plus LiDAR building roofs, with no vegetation in the flow surface at all. This script recovers
canopy from the point cloud that is already on disk, rather than inferring it from nadir NAIP.

A CORRECTION TO THE HANDOFF NOTE, measured rather than assumed
--------------------------------------------------------------
`NEXT_STEPS.md` says "the 2018 point cloud carries vegetation returns (classes 3/4/5), so canopy
height is recoverable directly" and points at `load_cached_points(classification_filter=...)`.
**Classes 3/4/5 do not exist in this data.** Across 48,055,738 points sampled from 7 of site3's
31 cached tiles, the classification histogram is:

    class  1 (unclassified)  61.32 %
    class  2 (ground)        32.63 %
    class  6 (building)       5.30 %
    class  7 (low noise)      0.25 %
    class  9 (water)          0.45 %
    class 17/18 (bridge)      0.00 %
    class 20                  0.04 %

`lidar/data/classification_histogram.json` shows the same for the main AOI. This acquisition
(FL_Peninsular_FDEM_2018_D18_LID2019) was classified to ground/building/water only; everything
else — including every vegetation return — was left in class 1. Filtering on 3/4/5 returns
nothing, so canopy has to come from class-1 returns normalised against a ground surface. That is
the standard normalised-DSM construction anyway, and it is what this script does.

Method
------
Canopy height model = (return elevation) - (bare-earth DEM at that location), accumulated onto a
2 m grid in the DEM's own CRS. The bare-earth DEM is the right ground reference here because it
is derived from this same 2018 acquisition, so the two surfaces share a datum by construction.

Per cell, the statistics are built from THRESHOLD COUNTS rather than a per-cell maximum:

  * `canopy_cover`  = returns above CANOPY_MIN_HEIGHT_M / all returns in the cell
  * `canopy_height` = the highest threshold still exceeded by >= HEIGHT_PERCENTILE of returns,
                      i.e. a discretised p98 height

A per-cell max would be the more familiar CHM, but it takes its value from a single return and
so is set by whatever noise survived classification. The p98-style statistic needs several
returns to agree before it reports a height, is computed entirely with `np.bincount` (fast,
unlike `np.maximum.at`), and is more than precise enough for what consumes it: roughness and
interception classes, not centimetres.

Checkpointed per tile, same reason every other site3 LiDAR script here is: reading all 31 cached
tiles in one uninterrupted process is exactly the pattern that kept getting killed during the
2026-07-27 site3 build (see `lidar/cache_bbox_points.py`). Each tile's accumulator is written to
disk as it completes; a re-run skips whatever is already done.

Usage:
    python3 segmentation/canopy_lidar.py --site site3
    python3 segmentation/canopy_lidar.py --site site3 --force   # ignore checkpoints
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

from build_lidar_pointcloud import LAS_CRS, DEM_CRS, FT_TO_M  # noqa: E402

# Grid resolution for the canopy products. Finer than the 5 m solver grid on purpose: this layer
# also feeds the 0.6 m NAIP classification (segment_naip.py), where its job is to separate tree
# canopy from lawn — two covers that look nearly identical to a nadir sensor and have very
# different roughness. At ~24 returns/m2 native density a 2 m cell holds ~100 returns, enough for
# the percentile statistic below to mean something.
CELL_M = 2.0

# Height thresholds the per-cell return counts are accumulated against [m above ground].
HEIGHT_BINS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0])

# A return has to clear this to count as canopy rather than ground clutter / low shrub. 2 m is
# the conventional cutoff and also clears the tallest thing a flood on this terrain would meet.
CANOPY_MIN_HEIGHT_M = 2.0

# Fraction of a cell's returns that must exceed a threshold before that threshold is reported as
# the cell's canopy height — the "p98" in the docstring.
HEIGHT_PERCENTILE = 0.02

# Returns above this are dropped as noise before anything is accumulated. The tallest trees in
# central Florida are ~35 m; the raw cached z ranges run to +564 ft / -426 ft, which is high-noise
# (class 7/18 covers only part of it), so an absolute gate is still needed.
MAX_PLAUSIBLE_HEIGHT_M = 45.0

# ASPRS noise classes — dropped outright, they are not returns from any real surface.
NOISE_CLASSES = (7, 18)

# Classes that count toward a cell's return TOTAL but can never count as canopy. Keeping them in
# the denominator is what makes the cover fraction meaningful: a roof cell has plenty of returns,
# all of them non-canopy, so it correctly reports cover ~= 0. Dropping them instead would leave
# the cell with only its handful of stray class-1 returns — which sit at roof height — and the
# cell would report near-total canopy cover at 4 m, i.e. every unmapped building in the domain
# would be classified as a tree. Buildings are 5.3 % of returns here and OSM does not map all of
# them, so this is a real effect, not a theoretical one.
NON_CANOPY_CLASSES = (2, 6, 9)   # ground, building, water


def build_grid(dem_path, cell_m):
    """Target grid in the DEM's own CRS. site3's DEM has a POSITIVE y-resolution (bottom > top),
    the inverted-affine quirk this project has now hit in four separate places — so bounds are
    sorted before use rather than assumed north-up."""
    with rasterio.open(dem_path) as src:
        b, crs = src.bounds, src.crs
    left, right = sorted([b.left, b.right])
    bottom, top = sorted([b.bottom, b.top])
    width = int((right - left) / cell_m)
    height = int((top - bottom) / cell_m)
    transform = from_bounds(left, bottom, right, top, width, height)
    return {"crs": crs, "transform": transform, "width": width, "height": height,
            "bounds": (left, bottom, right, top)}


def load_ground_surface(dem_path, grid):
    """Bare-earth DEM resampled onto the canopy grid. Resampling.average is right here — unlike
    the solver's DEM downsampling (which uses min() to preserve breached drainage paths), this is
    a reference surface for subtracting heights, where the mean ground level under a 2 m cell is
    exactly what is wanted."""
    with rasterio.open(dem_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        out = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
        reproject(src.read(1).astype(np.float32), out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  src_nodata=nodata, dst_nodata=np.nan,
                  resampling=Resampling.average)
    return out


def accumulate_tile(npz_path, grid, ground, to_dem):
    """Return (n_total, n_above) count arrays for one cached tile, flattened over the grid.

    n_total: returns landing in each cell (noise/water excluded)
    n_above: shape (len(HEIGHT_BINS), n_cells) — returns exceeding each height threshold
    """
    d = np.load(npz_path)
    x, y, z = d["x"], d["y"], d["z"]
    cls = d["classification"]
    if len(x) == 0:
        return None

    keep = ~np.isin(cls, NOISE_CLASSES)
    x, y, z, cls = x[keep], y[keep], z[keep], cls[keep]
    if len(x) == 0:
        return None
    # Ground/building/water returns stay in the denominator but are barred from the canopy
    # counts — see NON_CANOPY_CLASSES.
    canopy_ok = ~np.isin(cls, NON_CANOPY_CLASSES)

    # Cached coordinates are EPSG:2881 US survey feet (see cache_bbox_points.py); z likewise.
    X, Y = to_dem.transform(x, y)
    z_m = z.astype(np.float64) * FT_TO_M

    inv = ~grid["transform"]
    col, row = inv * (X, Y)
    col = col.astype(np.int64)
    row = row.astype(np.int64)
    inside = (row >= 0) & (row < grid["height"]) & (col >= 0) & (col < grid["width"])
    if not inside.any():
        return None
    row, col, z_m, canopy_ok = row[inside], col[inside], z_m[inside], canopy_ok[inside]

    flat = row * grid["width"] + col
    g = ground.reshape(-1)[flat]
    valid = np.isfinite(g)
    flat, z_m, g, canopy_ok = flat[valid], z_m[valid], g[valid], canopy_ok[valid]

    h = z_m - g
    ok = (h > -2.0) & (h < MAX_PLAUSIBLE_HEIGHT_M)   # -2 m allows real DEM/return disagreement
    flat, h, canopy_ok = flat[ok], h[ok], canopy_ok[ok]
    if len(flat) == 0:
        return None

    n_cells = grid["height"] * grid["width"]
    n_total = np.bincount(flat, minlength=n_cells).astype(np.int32)
    flat_c, h_c = flat[canopy_ok], h[canopy_ok]
    n_above = np.empty((len(HEIGHT_BINS), n_cells), dtype=np.int32)
    for i, thr in enumerate(HEIGHT_BINS):
        m = h_c > thr
        n_above[i] = np.bincount(flat_c[m], minlength=n_cells)
    return n_total, n_above


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3")
    ap.add_argument("--cell-size", type=float, default=CELL_M)
    ap.add_argument("--force", action="store_true", help="ignore per-tile checkpoints")
    args = ap.parse_args()

    if args.site != "site3":
        sys.exit("only site3 has a cached LiDAR bbox today; see lidar/cache_bbox_points.py")

    site_dir = os.path.join(PROJ_DIR, "site3_gee_creek")
    dem_path = os.path.join(site_dir, "dem", "data", "site3_dem.tif")
    cache_dir = os.path.join(site_dir, "lidar", "data", "bbox_cache")
    ckpt_dir = os.path.join(DATA_DIR, "canopy_checkpoints")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for p in (dem_path, cache_dir):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}")

    grid = build_grid(dem_path, args.cell_size)
    n_cells = grid["height"] * grid["width"]
    print("=" * 74)
    print(f"Canopy height / cover from LiDAR — {args.site}")
    print("=" * 74)
    print(f"  grid   : {grid['height']}x{grid['width']} @ {args.cell_size:.1f} m  {grid['crs']}")
    print(f"  ground : {os.path.relpath(dem_path, PROJ_DIR)}")

    ground = load_ground_surface(dem_path, grid)
    print(f"  ground surface: {100 * np.isfinite(ground).mean():.1f} % of cells have elevation, "
          f"range {np.nanmin(ground):.2f}-{np.nanmax(ground):.2f} m")

    to_dem = Transformer.from_crs(LAS_CRS, DEM_CRS, always_xy=True)
    tiles = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    print(f"  tiles  : {len(tiles)} cached\n")

    total = np.zeros(n_cells, dtype=np.int64)
    above = np.zeros((len(HEIGHT_BINS), n_cells), dtype=np.int64)

    for i, t in enumerate(tiles, 1):
        ck = os.path.join(ckpt_dir, os.path.basename(t) + f".{args.cell_size:g}m.npz")
        if os.path.exists(ck) and not args.force:
            c = np.load(ck)
            total += c["n_total"]
            above += c["n_above"]
            print(f"  [{i}/{len(tiles)}] {os.path.basename(t)[:52]}  (cached)")
            continue
        print(f"  [{i}/{len(tiles)}] {os.path.basename(t)[:52]} …", flush=True)
        res = accumulate_tile(t, grid, ground, to_dem)
        if res is None:
            n_total = np.zeros(n_cells, dtype=np.int32)
            n_above = np.zeros((len(HEIGHT_BINS), n_cells), dtype=np.int32)
        else:
            n_total, n_above = res
        tmp = ck + ".tmp.npz"
        np.savez_compressed(tmp, n_total=n_total, n_above=n_above)
        os.replace(tmp, ck)     # atomic, same pattern as the mesh-build checkpoints
        total += n_total
        above += n_above

    # ── derive the products ───────────────────────────────────────────────────
    total_f = total.astype(np.float32)
    has_returns = total > 0

    idx_canopy = int(np.argmin(np.abs(HEIGHT_BINS - CANOPY_MIN_HEIGHT_M)))
    cover = np.zeros(n_cells, dtype=np.float32)
    cover[has_returns] = above[idx_canopy][has_returns] / total_f[has_returns]

    # Highest threshold still exceeded by >= HEIGHT_PERCENTILE of the cell's returns.
    height = np.zeros(n_cells, dtype=np.float32)
    for i, thr in enumerate(HEIGHT_BINS):
        frac = np.zeros(n_cells, dtype=np.float32)
        frac[has_returns] = above[i][has_returns] / total_f[has_returns]
        height = np.where(frac >= HEIGHT_PERCENTILE, np.float32(thr), height)

    cover = cover.reshape(grid["height"], grid["width"])
    height = height.reshape(grid["height"], grid["width"])
    density = (total_f / (args.cell_size ** 2)).reshape(grid["height"], grid["width"])

    nodata_mask = ~has_returns.reshape(grid["height"], grid["width"])
    cover[nodata_mask] = 0.0
    height[nodata_mask] = 0.0

    profile = {"driver": "GTiff", "height": grid["height"], "width": grid["width"], "count": 1,
               "dtype": "float32", "crs": grid["crs"], "transform": grid["transform"],
               "compress": "lzw", "nodata": None}
    for name, arr in [("chm", height), ("canopy_cover", cover), ("return_density", density)]:
        p = os.path.join(DATA_DIR, f"{name}_{args.cell_size:g}m_{args.site}.tif")
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"\n  wrote {os.path.relpath(p, PROJ_DIR)}")

    treed = cover > 0.20
    summary = {
        "site": args.site,
        "cell_size_m": args.cell_size,
        "grid_shape": [grid["height"], grid["width"]],
        "crs": str(grid["crs"]),
        "n_tiles": len(tiles),
        "total_returns_used": int(total.sum()),
        "mean_return_density_per_m2": float(density[~nodata_mask].mean()) if (~nodata_mask).any() else 0.0,
        "cells_with_returns_pct": float(100 * has_returns.mean()),
        "canopy_cover_gt20pct_area_pct": float(100 * treed.mean()),
        "mean_canopy_height_where_treed_m": float(height[treed].mean()) if treed.any() else 0.0,
        "max_canopy_height_m": float(height.max()),
        "height_bins_m": HEIGHT_BINS.tolist(),
        "method": "class-1 returns normalised against the bare-earth DEM; discretised p98 height",
        "classification_note": ("ASPRS vegetation classes 3/4/5 are ABSENT from this acquisition "
                                "— canopy comes from class-1 unclassified returns, see module "
                                "docstring"),
    }
    sp = os.path.join(DATA_DIR, f"canopy_summary_{args.site}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "-" * 74)
    print(f"  returns used            : {summary['total_returns_used']:,}")
    print(f"  mean return density     : {summary['mean_return_density_per_m2']:.1f} /m2")
    print(f"  cells with any return   : {summary['cells_with_returns_pct']:.1f} %")
    print(f"  canopy cover > 20 %     : {summary['canopy_cover_gt20pct_area_pct']:.1f} % of area")
    print(f"  mean height where treed : {summary['mean_canopy_height_where_treed_m']:.1f} m")
    print(f"  max height              : {summary['max_canopy_height_m']:.1f} m")
    print(f"  wrote {os.path.relpath(sp, PROJ_DIR)}")
    print("-" * 74)


if __name__ == "__main__":
    main()
