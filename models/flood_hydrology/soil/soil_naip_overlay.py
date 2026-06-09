"""
NAIP + SSURGO Soil Verification Overlay
=========================================
Overlays NAIP true-color aerial imagery with SSURGO HSG soil units and
Sentinel-2 MNDWI water boundary to visually verify soil data accuracy.

Team lead requirement: "overlay NAIP soil data satellite images with our
current soil data to verify". This script creates the verification figure.

Panels
------
1. NAIP true-color aerial (1m resolution) with SSURGO soil unit boundaries
2. SSURGO Hydrologic Soil Group map with NAIP as background at 30% opacity
3. NDVI from NAIP NIR — shows vegetation vs impervious vs water patterns
4. Composite: SSURGO HSG + MNDWI water mask + NAIP hillshade background

Output: soil/naip_soil_overlay.png

Usage:
    python3 soil/soil_naip_overlay.py
    (Run soil/fetch_naip.py first to download NAIP imagery)
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
S2_DIR   = os.path.join(BASE_DIR, "..", "sentinel2", "data")
OUT_PATH = os.path.join(BASE_DIR, "naip_soil_overlay.png")

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981

# SSURGO HSG color scheme (standard hydrologic soil group colors)
HSG_COLORS = {
    "A": "#2ecc71",   # green — high infiltration, low runoff
    "B": "#f39c12",   # orange — moderate infiltration
    "C": "#e74c3c",   # red — slow infiltration, high runoff
    "D": "#8e44ad",   # purple — very slow, highest runoff
    "A/D": "#27ae60",
    "B/D": "#d35400",
    "C/D": "#c0392b",
    "W":   "#3498db",  # water
    "":    "#95a5a6",  # unknown
}


def load_raster(fpath, dtype=np.float32):
    import rasterio
    if not os.path.exists(fpath):
        return None, None
    with rasterio.open(fpath) as src:
        arr = src.read()
        meta = {"transform": src.transform, "crs": src.crs, "profile": src.profile}
    return arr, meta


def reproject_raster_to_target(src_path, target_path, resampling_method="bilinear"):
    """Reproject src raster to match target raster's CRS, transform, and shape."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.enums import Resampling as RS

    rs_map = {"bilinear": RS.bilinear, "nearest": RS.nearest, "cubic": RS.cubic}
    rs = rs_map.get(resampling_method, RS.bilinear)

    with rasterio.open(target_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    with rasterio.open(src_path) as src:
        n_bands = src.count
        out_arr = np.zeros((n_bands, *ref_shape), dtype=np.float32)
        for b in range(1, n_bands + 1):
            reproject(
                source=src.read(b).astype(np.float32),
                destination=out_arr[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=rs,
            )
    return out_arr, ref_profile


def latlon_to_pixel(lat, lon, transform, crs):
    """Convert lat/lon to pixel row/col in raster space."""
    import pyproj
    if "4326" in str(crs) or "WGS" in str(crs).upper():
        x, y = lon, lat
    else:
        t = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)
        x, y = t.transform(lon, lat)
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def overlay_ssurgo_on_ax(ax, ssurgo_path, transform, crs, shape, alpha=0.5):
    """Draw SSURGO soil unit polygons as colored patches on ax."""
    if not os.path.exists(ssurgo_path):
        return []
    try:
        with open(ssurgo_path) as f:
            data = json.load(f)
    except Exception:
        return []

    import pyproj
    use_transform = "4326" not in str(crs) and "WGS" not in str(crs).upper()
    if use_transform:
        proj = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)

    patches_added = {}
    for feat in data.get("features", []):
        geom  = feat.get("geometry", {})
        props = feat.get("properties", {})
        hsg   = str(props.get("hydgrp", props.get("hsg", props.get("HSG", "")))).strip()
        hsg   = hsg if hsg else ""
        color = HSG_COLORS.get(hsg, HSG_COLORS.get("", "#95a5a6"))

        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        rings = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        for poly in rings:
            for ring in poly:
                coords = np.array(ring)
                if use_transform:
                    xs, ys = proj.transform(coords[:, 0], coords[:, 1])
                else:
                    xs, ys = coords[:, 0], coords[:, 1]
                cols_px = (xs - transform.c) / transform.a
                rows_px = (ys - transform.f) / transform.e
                ax.fill(cols_px, rows_px, color=color, alpha=alpha, linewidth=0)
                ax.plot(cols_px, rows_px, color="white", lw=0.5, alpha=0.6)

        if hsg not in patches_added:
            patches_added[hsg] = mpatches.Patch(color=color, label=f"HSG {hsg}" if hsg else "Unknown")

    return list(patches_added.values())


def main():
    import rasterio

    # Find NAIP data
    naip_meta_path = os.path.join(DATA_DIR, "naip_meta.json")
    if not os.path.exists(naip_meta_path):
        sys.exit("NAIP not downloaded. Run: python3 soil/fetch_naip.py")

    with open(naip_meta_path) as f:
        naip_meta = json.load(f)

    rgb_path = naip_meta.get("rgb_path") or os.path.join(DATA_DIR, f"naip_{naip_meta['year']}_RGB.tif")
    nir_path = naip_meta.get("nir_path")
    ndvi_path = os.path.join(DATA_DIR, f"naip_{naip_meta['year']}_NDVI.tif")

    if not os.path.exists(rgb_path):
        sys.exit(f"NAIP RGB not found at {rgb_path}. Re-run fetch_naip.py.")

    print(f"NAIP: {rgb_path}")
    print(f"NAIP year: {naip_meta['year']}, resolution: {naip_meta.get('resolution_m','?')} m")

    # Load NAIP RGB as display array
    with rasterio.open(rgb_path) as src:
        rgb = src.read()         # shape (3, H, W)
        naip_transform = src.transform
        naip_crs = src.crs
        naip_shape = (src.height, src.width)

    # Normalize to 0-1 for display
    rgb_display = np.moveaxis(rgb.astype(np.float32), 0, -1)  # (H, W, 3)
    p_lo = np.percentile(rgb_display, 2)
    p_hi = np.percentile(rgb_display, 98)
    rgb_display = np.clip((rgb_display - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)

    nrows, ncols = naip_shape

    # Property location
    prop_row, prop_col = latlon_to_pixel(
        PROPERTY_LAT, PROPERTY_LON, naip_transform, naip_crs)
    prop_row = np.clip(prop_row, 0, nrows - 1)
    prop_col = np.clip(prop_col, 0, ncols - 1)

    # Reproject MNDWI water mask to NAIP grid (for overlay)
    mndwi_files = sorted(glob.glob(os.path.join(S2_DIR, "water_mask_*.tif")))
    water_freq = np.zeros(naip_shape, dtype=np.float32)
    for wf in mndwi_files:
        try:
            from rasterio.warp import reproject as warp_reproject, Resampling
            with rasterio.open(wf) as src:
                out = np.zeros(naip_shape, dtype=np.float32)
                warp_reproject(
                    source=src.read(1).astype(np.float32), destination=out,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=naip_transform, dst_crs=naip_crs,
                    resampling=Resampling.nearest)
            water_freq += (out > 0.5)
        except Exception:
            pass

    water_persistent = water_freq >= max(1, len(mndwi_files) * 0.5)

    # SSURGO soil map
    ssurgo_path = os.path.join(DATA_DIR, "ssurgo_mapunits.geojson")

    # NDVI
    ndvi_arr = None
    if os.path.exists(ndvi_path):
        with rasterio.open(ndvi_path) as src:
            ndvi_raw = src.read(1).astype(np.float32)
            if ndvi_raw.shape != naip_shape:
                from scipy.ndimage import zoom
                ndvi_arr = zoom(ndvi_raw, (naip_shape[0]/ndvi_raw.shape[0],
                                           naip_shape[1]/ndvi_raw.shape[1]), order=1)
            else:
                ndvi_arr = ndvi_raw

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle(
        f"NAIP Aerial Imagery + SSURGO Soil Verification — Winter Garden FL\n"
        f"NAIP {naip_meta['year']} ({naip_meta.get('resolution_m','?')} m) | "
        f"SSURGO soil units | S2 persistent water boundary",
        fontsize=12, fontweight="bold")

    # ── Panel 1: NAIP RGB + SSURGO boundaries ───────────────────────────────
    ax = axes[0]
    ax.imshow(rgb_display, origin="upper", extent=[0, ncols, nrows, 0])
    soil_patches = overlay_ssurgo_on_ax(
        ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.35)
    if water_persistent.any():
        ax.contour(water_persistent.astype(np.float32), levels=[0.5],
                   colors=["cyan"], linewidths=1.8, origin="upper")
        soil_patches.append(mpatches.Patch(color="cyan", label="S2 water boundary"))
    ax.plot(prop_col, prop_row, "r*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    soil_patches.append(mpatches.Patch(color="red", label="Property"))
    ax.legend(handles=soil_patches, fontsize=7, loc="lower right",
              framealpha=0.85, ncol=2)
    ax.set_title(f"NAIP {naip_meta['year']} True-Color\n+ SSURGO HSG boundaries + S2 water",
                 fontsize=10)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ax.text(5, nrows - 8, f"Resolution: {naip_meta.get('resolution_m','?')} m",
            color="white", fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    # ── Panel 2: SSURGO HSG choropleth on NAIP background ───────────────────
    ax = axes[1]
    ax.imshow(rgb_display, origin="upper", extent=[0, ncols, nrows, 0], alpha=0.4)
    soil_patches2 = overlay_ssurgo_on_ax(
        ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.65)
    if water_persistent.any():
        ax.contour(water_persistent.astype(np.float32), levels=[0.5],
                   colors=["white"], linewidths=1.5, origin="upper")
    ax.plot(prop_col, prop_row, "r*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    ax.legend(handles=soil_patches2 + [mpatches.Patch(color="red", label="Property")],
              fontsize=8, loc="lower right", framealpha=0.9, ncol=2)
    ax.set_title("SSURGO Hydrologic Soil Groups\n"
                 "(A=green/fast, B=orange, C=red, D=purple/slow)\n"
                 "on NAIP background", fontsize=9)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")

    # Panel 3: NDVI or NAIP NIR channel
    ax = axes[2]
    if ndvi_arr is not None:
        im3 = ax.imshow(ndvi_arr, cmap="RdYlGn", vmin=-0.3, vmax=0.8,
                        origin="upper", extent=[0, ncols, nrows, 0])
        cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
        cb3.set_label("NDVI (NIR−Red)/(NIR+Red)", fontsize=8)
        ax.set_title("NAIP NDVI\n(dark green = dense vegetation,\n"
                     "red/yellow = bare soil/impervious, blue = water)", fontsize=9)
        # SSURGO boundaries overlay
        overlay_ssurgo_on_ax(ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.15)
        if water_persistent.any():
            ax.contour(water_persistent.astype(np.float32), levels=[0.5],
                       colors=["blue"], linewidths=1.5, origin="upper")
    else:
        ax.imshow(rgb_display, origin="upper", extent=[0, ncols, nrows, 0])
        ax.set_title("NAIP True-Color (NIR/NDVI not available)", fontsize=10)
    ax.plot(prop_col, prop_row, "r*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {OUT_PATH}")
    print(f"\nVerification guide:")
    print(f"  • Do SSURGO soil unit boundaries match visible land cover in NAIP?")
    print(f"  • Does the S2 water boundary (cyan) align with the visible lake in NAIP?")
    print(f"  • NDVI green areas should match A/B HSG (permeable soils with vegetation)")
    print(f"  • Low NDVI (impervious) areas should align with C/D HSG (slow drainage)")


if __name__ == "__main__":
    main()
