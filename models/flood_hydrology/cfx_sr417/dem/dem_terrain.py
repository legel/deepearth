"""
Terrain Variables Analysis — CFX SR417 Corridor
================================================
Computes standard terrain derivatives for slope-stability and erosion analysis:

  Geomorphometric (pure geometry — zero routing assumptions):
    slope_deg       — gradient steepness (degrees)
    aspect_deg      — flow-face azimuth (degrees CW from N; -1 = flat)
    hillshade       — shaded relief (visualization)
    curvature_profile — curvature along steepest descent (+concave/erosion, -convex/deposition)
    curvature_plan    — curvature perpendicular to descent (+convergent, -divergent)
    tpi_30m         — Topographic Position Index, 30 m window (micro-drainage)
    tpi_100m        — Topographic Position Index, 100 m window (landscape position)
    tri             — Terrain Ruggedness Index (Riley et al. 1999)
    roughness       — local elevation range (max-min in 3×3 window)

  All derived from the raw DEM. No pit filling, no flow routing, no assumptions.

Hydrological conditioning and flow-routing derivatives (flow direction, accumulation,
HAND, watershed) are in dem_hydro.py — these DO involve methodological choices and are
separated deliberately so the terrain variables can be trusted independently.

DEM source: USGS 3DEP via py3dep.
Preferred: sr417_corridor_dem_1m.tif (1 m LiDAR — available for this AOI).
Fallback:  sr417_corridor_dem.tif (3 m, already downloaded).
The 1m DEM is dramatically better for detecting subtle slope breaks and drainage swales
on this flat (14 m relief over 2 km) Central Florida terrain. Florida has statewide
USGS Quality Level 1 LiDAR; the 1m 3DEP product comes directly from the raw point cloud.

Methods:
  richdem  — slope, aspect, curvature (Zevenbergen-Thorne 1987 / Horn 1981 algorithms,
             industry standard, consistent with ArcGIS Spatial Analyst and GRASS r.slope.aspect)
  scipy    — TPI, TRI, roughness (focal statistics over rectangular windows)

Outputs → dem/data/terrain/:
  *.tif     — GeoTIFF for each variable (EPSG matches input DEM, nodata = NaN)
  *.png     — 512×512 PNG for viewer integration (matched to DEM grid)
  terrain_summary.json — per-variable statistics (min/max/mean/p5/p95)
  terrain_overview.png  — 9-panel matplotlib summary figure

Usage:
    python3 dem/dem_terrain.py
    python3 dem/dem_terrain.py --dem dem/data/sr417_corridor_dem_1m.tif
    python3 dem/dem_terrain.py --no-png     (skip viewer PNGs, only GeoTIFFs)
"""

import os
import json
import argparse
import warnings
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter, generic_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(DATA_DIR, "terrain")
os.makedirs(OUT_DIR, exist_ok=True)

# Preferred 1m DEM; fall back to 3m if not yet downloaded
_DEM_1M = os.path.join(DATA_DIR, "sr417_corridor_dem_1m.tif")
_DEM_3M = os.path.join(DATA_DIR, "sr417_corridor_dem.tif")
DEFAULT_DEM = _DEM_1M if os.path.exists(_DEM_1M) else _DEM_3M

PNG_SIZE = 512   # viewer overlay resolution


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_dem(dem_path):
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
        res_x, res_y = src.res          # (x_res, y_res) in CRS units
        crs = src.crs
    if nodata is not None:
        z[z == nodata] = np.nan
    # richdem needs a finite masked array; replace NaN edges with nearest valid
    # (only for richdem calls; we re-apply NaN mask to outputs afterward)
    return z, profile, abs(res_x), abs(res_y), crs


def save_tif(arr, profile, out_path, nodata=np.nan):
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=nodata, compress="deflate")
    arr_out = arr.astype(np.float32)
    arr_out[~np.isfinite(arr_out)] = nodata
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr_out, 1)


def save_png(arr, out_path, cmap, vmin=None, vmax=None, invalid_color=(0, 0, 0, 0)):
    """Export a 512×512 RGBA PNG matched to the DEM grid extent (for viewer overlays)."""
    from PIL import Image
    from skimage.transform import resize as sk_resize
    from matplotlib import colormaps

    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        Image.fromarray(np.zeros((PNG_SIZE, PNG_SIZE, 4), dtype=np.uint8)).save(out_path)
        return
    if vmin is None:
        vmin = float(np.percentile(valid, 2))
    if vmax is None:
        vmax = float(np.percentile(valid, 98))
    if vmin == vmax:
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cm   = colormaps.get_cmap(cmap)
    rgba_f = cm(norm(arr))           # (H, W, 4) float
    rgba_f[~np.isfinite(arr)] = [c / 255 for c in invalid_color[:3]] + [0.0]
    rgba8 = (rgba_f * 255).astype(np.uint8)

    # Resize to PNG_SIZE×PNG_SIZE
    # skimage resize expects float; we resize then re-quantise
    rgba_r = sk_resize(rgba8.astype(np.float32), (PNG_SIZE, PNG_SIZE, 4),
                       order=1, anti_aliasing=True, preserve_range=True)
    Image.fromarray(rgba_r.astype(np.uint8)).save(out_path)


# ── Terrain computations ─────────────────────────────────────────────────────

def compute_richdem(z, profile, res_m, nodata_val=-9999.0):
    """Slope (deg), aspect (deg), profile curvature, plan curvature via richdem."""
    import richdem as rd

    # richdem cannot handle NaN — fill edges with a sentinel nodata
    z_fill = z.copy()
    z_fill[~np.isfinite(z_fill)] = nodata_val

    rda = rd.rdarray(z_fill, no_data=nodata_val, geotransform=(
        profile["transform"].c,
        profile["transform"].a,
        0,
        profile["transform"].f,
        0,
        profile["transform"].e,
    ))

    slope_rd  = rd.TerrainAttribute(rda, attrib="slope_degrees")
    aspect_rd = rd.TerrainAttribute(rda, attrib="aspect")
    curv_p    = rd.TerrainAttribute(rda, attrib="curvature")   # profile
    curv_pl   = rd.TerrainAttribute(rda, attrib="planform_curvature")

    def _to_numpy(rdarray_out):
        arr = np.array(rdarray_out, dtype=np.float32)
        arr[arr == rdarray_out.no_data] = np.nan
        arr[~np.isfinite(z)] = np.nan
        return arr

    return (
        _to_numpy(slope_rd),
        _to_numpy(aspect_rd),
        _to_numpy(curv_p),
        _to_numpy(curv_pl),
    )


def compute_hillshade(z, res_m, azimuth=315.0, altitude=45.0):
    """Horn (1981) hillshade — same formula as ArcGIS/QGIS."""
    from scipy.ndimage import sobel
    az_rad  = np.radians(360 - azimuth + 90)
    alt_rad = np.radians(altitude)

    # Gradient via Sobel (central differences, weighted)
    dz_dx = sobel(z, axis=1) / (8 * res_m)
    dz_dy = sobel(z, axis=0) / (8 * res_m)

    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dy, dz_dx)

    hs = (np.cos(alt_rad) * np.cos(slope) +
          np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    hs = np.clip(hs, 0, 1).astype(np.float32)
    hs[~np.isfinite(z)] = np.nan
    return hs


def compute_tpi(z, res_m, window_m):
    """TPI = z - mean(annular neighborhood). window_m sets the outer radius."""
    px = max(1, int(round(window_m / res_m)))
    mean_z = uniform_filter(np.where(np.isfinite(z), z, 0.0), size=2 * px + 1)
    tpi = z - mean_z
    tpi[~np.isfinite(z)] = np.nan
    return tpi.astype(np.float32)


def compute_tri(z):
    """Riley et al. (1999) TRI = sqrt(sum of squared differences from 8 neighbours)."""
    def _tri_cell(block):
        c = block[4]
        if not np.isfinite(c):
            return np.nan
        diffs = block - c
        diffs[4] = 0.0
        return float(np.sqrt(np.nansum(diffs ** 2)))

    tri = generic_filter(z, _tri_cell, size=3, mode="nearest").astype(np.float32)
    tri[~np.isfinite(z)] = np.nan
    return tri


def compute_roughness(z):
    """Roughness = max - min in a 3×3 window (GDAL definition)."""
    def _range(block):
        finite = block[np.isfinite(block)]
        if len(finite) == 0:
            return np.nan
        return float(finite.max() - finite.min())

    rough = generic_filter(z, _range, size=3, mode="nearest").astype(np.float32)
    rough[~np.isfinite(z)] = np.nan
    return rough


# ── Statistics ───────────────────────────────────────────────────────────────

def stats(arr, name):
    valid = arr[np.isfinite(arr)].astype(np.float64)
    if len(valid) == 0:
        return {"name": name, "n_valid": 0}
    return {
        "name":  name,
        "n_valid": int(len(valid)),
        "min":   float(valid.min()),
        "p5":    float(np.percentile(valid, 5)),
        "mean":  float(valid.mean()),
        "median":float(np.median(valid)),
        "p95":   float(np.percentile(valid, 95)),
        "max":   float(valid.max()),
        "std":   float(valid.std()),
    }


# ── Multi-panel overview figure ───────────────────────────────────────────────

def make_overview(z, layers, res_m, out_path):
    items = [
        ("Elevation (m NAVD88)", z,               "terrain",    None,  None),
        ("Slope (°)",            layers["slope"],  "YlOrRd",     0,     35),
        ("Aspect (° from N)",    layers["aspect"], "hsv",        0,     360),
        ("Hillshade",            layers["hillshade"],"gray",     0,     1),
        ("Profile Curvature",    layers["curv_profile"],"RdBu_r",None,  None),
        ("Plan Curvature",       layers["curv_plan"],  "RdBu_r", None,  None),
        ("TPI 30 m",             layers["tpi_30m"],    "RdBu_r", None,  None),
        ("TPI 100 m",            layers["tpi_100m"],   "RdBu_r", None,  None),
        ("TRI",                  layers["tri"],         "magma",  None,  None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"Terrain Variables — CFX SR417 Corridor AOI\n"
        f"{z.shape[0]}×{z.shape[1]} px @ {res_m:.1f} m/px  |  "
        f"z {np.nanmin(z):.1f}–{np.nanmax(z):.1f} m NAVD88",
        color="white", fontsize=12, y=0.98
    )
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.flat, items):
        valid = arr[np.isfinite(arr)]
        v0 = float(np.percentile(valid, 2)) if vmin is None else vmin
        v1 = float(np.percentile(valid, 98)) if vmax is None else vmax
        im = ax.imshow(arr, cmap=cmap, vmin=v0, vmax=v1, origin="upper",
                       aspect="equal", interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
        ax.set_facecolor("#0d1117")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  terrain_overview.png saved")


# ── Colourmap choices (scientifically standard) ───────────────────────────────
# elevation → terrain (classic hypsometric tint: blue-green low, brown/white high)
# slope     → YlOrRd (white=flat, red=steep; good for erosion risk)
# aspect    → hsv (circular; 0=N, 90=E, 180=S, 270=W)
# hillshade → gray (black=shadow, white=illuminated)
# curvature → RdBu_r (red=concave/erosional, blue=convex/depositional)
# TPI       → RdBu_r (red=ridge, blue=valley)
# TRI/rough → magma (low=uniform, high=complex)

LAYER_PNGS = [
    # (key,             png_name,                cmap,      vmin, vmax,  opacity)
    ("elevation",       "terrain_elevation.png",  "terrain", None, None, 0.80),
    ("slope",           "terrain_slope.png",      "YlOrRd",  0,    35,   0.80),
    ("aspect",          "terrain_aspect.png",     "hsv",     0,    360,  0.75),
    ("hillshade",       "terrain_hillshade.png",  "gray",    0,    1,    0.85),
    ("curv_profile",    "terrain_curvature.png",  "RdBu_r",  None, None, 0.75),
    ("tpi_30m",         "terrain_tpi.png",        "RdBu_r",  None, None, 0.75),
    ("tri",             "terrain_tri.png",        "magma",   None, None, 0.75),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dem_path, make_pngs=True):
    print(f"\nTerrain Variables — {os.path.basename(dem_path)}")
    print("=" * 60)

    z, profile, res_x, res_y, crs = load_dem(dem_path)
    res_m = (res_x + res_y) / 2  # isotropic approximation
    print(f"Grid       : {z.shape[0]}×{z.shape[1]} px @ {res_m:.2f} m  CRS: {crs}")
    print(f"Elevation  : {np.nanmin(z):.2f}–{np.nanmax(z):.2f} m NAVD88")
    print(f"Output dir : {OUT_DIR}")

    # ── Richdem-based derivatives ──────────────────────────────────────────
    print("\nComputing slope / aspect / curvature (richdem) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, aspect, curv_profile, curv_plan = compute_richdem(z, profile, res_m)

    # ── Hillshade ──────────────────────────────────────────────────────────
    print("Computing hillshade …")
    hillshade = compute_hillshade(z, res_m)

    # ── TPI ────────────────────────────────────────────────────────────────
    print("Computing TPI (30 m / 100 m windows) …")
    tpi_30  = compute_tpi(z, res_m, window_m=30)
    tpi_100 = compute_tpi(z, res_m, window_m=100)

    # ── TRI / Roughness ────────────────────────────────────────────────────
    print("Computing TRI and roughness …")
    tri      = compute_tri(z)
    roughness = compute_roughness(z)

    layers = {
        "elevation":    z,
        "slope":        slope,
        "aspect":       aspect,
        "hillshade":    hillshade,
        "curv_profile": curv_profile,
        "curv_plan":    curv_plan,
        "tpi_30m":      tpi_30,
        "tpi_100m":     tpi_100,
        "tri":          tri,
        "roughness":    roughness,
    }

    # ── Save GeoTIFFs ──────────────────────────────────────────────────────
    print("\nSaving GeoTIFFs …")
    tif_names = {
        "slope":        "slope_deg.tif",
        "aspect":       "aspect_deg.tif",
        "hillshade":    "hillshade.tif",
        "curv_profile": "curvature_profile.tif",
        "curv_plan":    "curvature_plan.tif",
        "tpi_30m":      "tpi_30m.tif",
        "tpi_100m":     "tpi_100m.tif",
        "tri":          "tri.tif",
        "roughness":    "roughness.tif",
    }
    for key, fname in tif_names.items():
        out_tif = os.path.join(OUT_DIR, fname)
        save_tif(layers[key], profile, out_tif)
        print(f"  {fname}")

    # ── Statistics ─────────────────────────────────────────────────────────
    summary = {
        "dem_path":   dem_path,
        "dem_res_m":  res_m,
        "crs":        str(crs),
        "grid_shape": list(z.shape),
        "variables":  {k: stats(v, k) for k, v in layers.items()},
    }
    summary_path = os.path.join(OUT_DIR, "terrain_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  terrain_summary.json")

    # ── Overview figure ────────────────────────────────────────────────────
    print("\nGenerating terrain_overview.png …")
    make_overview(z, layers, res_m, os.path.join(OUT_DIR, "terrain_overview.png"))

    # ── Viewer PNGs ────────────────────────────────────────────────────────
    if make_pngs:
        from skimage.transform import resize as sk_resize  # noqa: confirm available
        print("\nSaving viewer PNGs (512×512) …")
        for key, png_name, cmap, vmin, vmax, _ in LAYER_PNGS:
            out_png = os.path.join(OUT_DIR, png_name)
            save_png(layers[key], out_png, cmap, vmin, vmax)
            print(f"  {png_name}")

    # ── Summary to console ─────────────────────────────────────────────────
    print("\nKey statistics:")
    for key in ["slope", "tri", "tpi_100m"]:
        s = summary["variables"][key]
        print(f"  {key:18s}: mean={s['mean']:.2f}  p95={s['p95']:.2f}  max={s['max']:.2f}")

    print(f"\nDone.  Output → {OUT_DIR}/")
    return layers, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem",    default=DEFAULT_DEM, help="Path to DEM GeoTIFF")
    parser.add_argument("--no-png", action="store_true",  help="Skip viewer PNG export")
    args = parser.parse_args()

    if not os.path.exists(args.dem):
        print(f"DEM not found: {args.dem}")
        if args.dem == DEFAULT_DEM and not os.path.exists(_DEM_1M):
            print(f"Run first:  python3 dem/dem_download.py --resolution 1")
        import sys; sys.exit(1)

    run(args.dem, make_pngs=not args.no_png)
