"""
NAIP + SSURGO Soil Verification Overlay
=========================================
Overlays NAIP true-color aerial imagery with SSURGO HSG soil units and
Sentinel-2 MNDWI water boundary to visually verify soil data accuracy.

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

# Distinct colors for individual soil series (used in panel 2 for unit-level detail)
MUNAME_COLORS = [
    "#e41a1c",  # 0 Basinger fine sand  — red
    "#d4a843",  # 1 Candler fine sand   — sandy golden/tan (not blue)
    "#4daf4a",  # 2 Candler sand        — green
    "#984ea3",  # 3 Cassia sand         — purple
    "#ff7f00",  # 4 Florahome fine sand — orange
    "#a65628",  # 5 Orlando fine sand   — brown
    "#f781bf",  # 6 (fallback)          — pink
    "#999999",  # 7                     — grey
    "#66c2a5",  # 8                     — teal
    "#fc8d62",  # 9                     — salmon
    "#e6d800",  # 10                    — yellow (was blue-grey)
]


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


def _naip_bbox_latlon(transform, crs, shape):
    """Return the NAIP image extent as a shapely box in EPSG:4326 (lon/lat)."""
    import shapely.geometry as sg
    import pyproj
    nrows, ncols = shape
    # Four corners in NAIP CRS (projected metres)
    left   = transform.c
    right  = transform.c + transform.a * ncols
    top    = transform.f
    bottom = transform.f + transform.e * nrows   # negative step → bottom < top
    to_ll = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    xs, ys = to_ll.transform([left, right, right, left],
                              [top,  top,  bottom, bottom])
    return sg.box(min(xs), min(ys), max(xs), max(ys))


def _extract_rings(clipped_geom):
    """Yield (coords_array,) for every exterior ring in a (Multi)Polygon."""
    import shapely.geometry as sg
    if clipped_geom.is_empty:
        return
    if clipped_geom.geom_type == "Polygon":
        yield np.array(clipped_geom.exterior.coords)
    elif clipped_geom.geom_type == "MultiPolygon":
        for p in clipped_geom.geoms:
            yield np.array(p.exterior.coords)
    elif hasattr(clipped_geom, "geoms"):
        for g in clipped_geom.geoms:
            yield from _extract_rings(g)


def overlay_ssurgo_on_ax(ax, ssurgo_path, transform, crs, shape, alpha=0.5,
                         use_muname=False):
    """Draw SSURGO soil unit polygons clipped to NAIP bounds.

    Clips every polygon to the NAIP image extent (via shapely intersection)
    before drawing — prevents oversized lake/watershed polygons from flooding
    the entire panel with one colour.
    """
    if not os.path.exists(ssurgo_path):
        return []
    try:
        with open(ssurgo_path) as f:
            data = json.load(f)
    except Exception:
        return []

    # Build mukey → (HSG, muname) lookup
    mukey_to_hsg  = {}
    mukey_to_name = {}
    params_path = os.path.join(DATA_DIR, "soil_parameters.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            soil_params = json.load(f)
        for mukey, vals in soil_params.items():
            hsg = str(vals.get("hsg", "")).strip()
            if hsg:
                mukey_to_hsg[str(mukey)] = hsg
            muname = str(vals.get("muname", "")).strip()
            if muname:
                mukey_to_name[str(mukey)] = muname.split(",")[0].strip()

    all_names = sorted(set(mukey_to_name.values()))
    name_color_map = {name: MUNAME_COLORS[i % len(MUNAME_COLORS)]
                      for i, name in enumerate(all_names)}

    import pyproj, shapely.geometry as sg
    # Clip box in SSURGO coordinates (EPSG:4326 lon/lat)
    naip_clip_box = _naip_bbox_latlon(transform, crs, shape)

    # Projector: EPSG:4326 → NAIP CRS (projected metres)
    proj = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)

    patches_added = {}
    for feat in data.get("features", []):
        geom  = feat.get("geometry", {})
        props = feat.get("properties", {})
        mukey = str(props.get("mukey", ""))

        hsg = str(props.get("hydgrp", props.get("hsg", props.get("HSG", "")))).strip()
        if not hsg or hsg.upper() in ("NONE", "NULL", "NAN"):
            hsg = mukey_to_hsg.get(mukey, "")

        if use_muname:
            muname = mukey_to_name.get(mukey, "Unknown")
            color = "#3498db" if muname == "Water" else name_color_map.get(muname, "#95a5a6")
            legend_key = muname
        else:
            color = HSG_COLORS.get(hsg, HSG_COLORS.get("", "#95a5a6"))
            legend_key = f"HSG {hsg}" if hsg else "Unknown"

        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        # Clip to NAIP bounds before drawing
        try:
            feat_shape = sg.shape(geom)
            clipped = feat_shape.intersection(naip_clip_box)
        except Exception:
            continue
        if clipped.is_empty:
            continue

        drawn = False
        for ring_coords in _extract_rings(clipped):
            if len(ring_coords) < 3:
                continue
            # Convert lon/lat → NAIP CRS metres → pixel coords
            xs, ys = proj.transform(ring_coords[:, 0], ring_coords[:, 1])
            cols_px = (xs - transform.c) / transform.a
            rows_px = (ys - transform.f) / transform.e
            ax.fill(cols_px, rows_px, color=color, alpha=alpha, linewidth=0)
            ax.plot(cols_px, rows_px, color="white", lw=0.5, alpha=0.6)
            drawn = True

        if drawn and legend_key not in patches_added:
            patches_added[legend_key] = mpatches.Patch(color=color, label=legend_key)

    return list(patches_added.values())


def main():
    import rasterio

    # Find NAIP data
    naip_meta_path = os.path.join(DATA_DIR, "naip_meta.json")
    if not os.path.exists(naip_meta_path):
        sys.exit("NAIP not downloaded. Run: python3 soil/fetch_naip.py")

    with open(naip_meta_path) as f:
        naip_meta = json.load(f)

    rgb_name = naip_meta.get("rgb_path") or f"naip_{naip_meta['year']}_RGB.tif"
    rgb_path = os.path.join(DATA_DIR, os.path.basename(rgb_name))
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

    # Water boundary: compute NDWI directly from NAIP NIR/Green bands.
    # Using NAIP's own spectral bands gives a boundary perfectly aligned
    # with the visible lake in the aerial image — no reprojection artefacts.
    from rasterio.warp import reproject as warp_reproject, Resampling
    from scipy.ndimage import binary_opening, binary_fill_holes, gaussian_filter

    nir_name = naip_meta.get("nir_path") or f"naip_{naip_meta['year']}_NIR.tif"
    nir_path = os.path.join(DATA_DIR, os.path.basename(nir_name))
    if nir_path and os.path.exists(nir_path):
        with rasterio.open(rgb_path) as src:
            g_band = src.read(2).astype(np.float32)   # band 2 = Green
            px_size = abs(src.transform.a)
        with rasterio.open(nir_path) as src:
            nir_band_raw = src.read(1).astype(np.float32)
        ndwi = (g_band - nir_band_raw) / (g_band + nir_band_raw + 1e-9)
        # Threshold + morphological cleaning
        from scipy.ndimage import label as nd_label
        water_raw = ndwi > 0.2
        water_opened = binary_opening(water_raw, iterations=3)
        water_filled = binary_fill_holes(water_opened)
        # Keep only connected components ≥ 1 ha (removes retention ponds, shadows)
        min_px_1ha = int(1e4 / px_size ** 2)
        labeled_cc, n_cc = nd_label(water_filled)
        water_persistent = np.zeros_like(water_filled)
        for comp_id in range(1, n_cc + 1):
            if (labeled_cc == comp_id).sum() >= min_px_1ha:
                water_persistent |= (labeled_cc == comp_id)
        ndwi_available = True
        n_components_kept = int(water_persistent.any())  # count kept
        print(f"NAIP NDWI water: {water_persistent.sum() * px_size**2 / 1e4:.1f} ha "
              f"({n_cc} raw components → filtered to ≥1 ha bodies)")
    else:
        # Fall back: reproject S2 consensus mask
        ndwi = None
        ndwi_available = False
        lake_mask_path = os.path.join(S2_DIR, "..", "dem", "data", "lake_mask_s2.tif")
        if os.path.exists(lake_mask_path):
            with rasterio.open(lake_mask_path) as src:
                out = np.zeros(naip_shape, dtype=np.float32)
                warp_reproject(
                    source=src.read(1).astype(np.float32), destination=out,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=naip_transform, dst_crs=naip_crs,
                    resampling=Resampling.bilinear)
            out_smooth = gaussian_filter(out, sigma=6.0)
            water_persistent = binary_fill_holes(out_smooth > 0.35)
        else:
            water_persistent = np.zeros(naip_shape, dtype=bool)

    # Secondary boundary: OmniWaterMask majority-vote consensus reprojected to NAIP.
    # OmniWaterMask masks are in EPSG:32617 (UTM zone 17N) — same UTM zone as NAIP
    # (EPSG:26917), so reprojection artefacts are minimal.
    s2_water_boundary = None
    owm_files = sorted(glob.glob(os.path.join(S2_DIR, "omniwatermask_mask_*.tif")))
    if owm_files:
        vote_stack = np.zeros(naip_shape, dtype=np.float32)
        for fp in owm_files:
            with rasterio.open(fp) as src:
                out_owm = np.zeros(naip_shape, dtype=np.float32)
                warp_reproject(
                    source=src.read(1).astype(np.float32), destination=out_owm,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=naip_transform, dst_crs=naip_crs,
                    resampling=Resampling.nearest)
                vote_stack += out_owm
        owm_consensus = (vote_stack >= len(owm_files) / 2).astype(bool)
        # Where OmniWaterMask is below majority threshold (including partial-vote
        # pixels that cause notch artifacts) and NAIP NDWI confirms water, fill in.
        if ndwi_available:
            no_s2_coverage = (vote_stack < len(owm_files) / 2)
            owm_consensus = owm_consensus | (no_s2_coverage & water_persistent)
        s2_water_boundary = owm_consensus
        print(f"OmniWaterMask consensus: {owm_consensus.sum() * px_size**2 / 1e4:.1f} ha")

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
        f"NAIP Aerial Imagery + SSURGO Soil Verification — Winter Garden FL (Johns Lake)\n"
        f"NAIP {naip_meta['year']} | {naip_meta.get('resolution_m','?')} m resolution | "
        f"SSURGO soil map units from USDA Web Soil Survey | "
        f"Water boundary from NAIP NDWI (Green−NIR)/(Green+NIR)",
        fontsize=10, fontweight="bold")

    def _add_water_boundary(ax, water_mask, color="cyan", lw=1.8, linestyle="solid", label="NAIP NDWI water"):
        """Draw water boundary contour, suppressing the straight frame-edge artifact.

        When the lake extends beyond the NAIP boundary, contour() would draw a
        straight line along the image edge (row 0 / col 0).  Zeroing the 5-pixel
        border strip ensures only interior water/land transitions are drawn.
        """
        if water_mask is None or not water_mask.any():
            return None
        # Suppress contour at image border (prevents straight-line frame artifact)
        interior = water_mask.copy().astype(np.float32)
        interior[:3,  :] = 0
        interior[-3:, :] = 0
        interior[:,  :3] = 0
        interior[:, -3:] = 0
        if not interior.any():
            return None
        x_px = np.arange(water_mask.shape[1])
        y_px = np.arange(water_mask.shape[0])
        ax.contour(x_px, y_px, interior, levels=[0.5],
                   colors=[color], linewidths=[lw], linestyles=[linestyle])
        return mpatches.Patch(color=color, label=label)

    # ── Panel 1: NAIP RGB + SSURGO HSG boundaries + water ───────────────────
    ax = axes[0]
    ax.imshow(rgb_display, origin="upper", extent=[0, ncols, nrows, 0])
    soil_patches = overlay_ssurgo_on_ax(
        ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.35)
    # Primary: NAIP NDWI water boundary (perfectly aligned with aerial)
    p = _add_water_boundary(ax, water_persistent, color="cyan", lw=2.0,
                            label="NAIP NDWI water boundary")
    if p: soil_patches.append(p)
    # Secondary: S2 consensus mask boundary (dashed, for cross-check)
    if ndwi_available and s2_water_boundary is not None:
        p2 = _add_water_boundary(ax, s2_water_boundary, color="yellow", lw=1.2,
                                  linestyle="dashed", label="S2 consensus boundary")
        if p2: soil_patches.append(p2)
    ax.plot(prop_col, prop_row, "r*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    soil_patches.append(mpatches.Patch(color="red", label="Property pin"))
    ax.set_xlim(0, ncols); ax.set_ylim(nrows, 0)
    ax.legend(handles=soil_patches, fontsize=6.5, loc="lower right",
              framealpha=0.88, ncol=2)
    ax.set_title(f"NAIP {naip_meta['year']} True-Color + SSURGO HSG\n"
                 f"Cyan = NAIP NDWI water  |  Yellow dashed = OmniWaterMask consensus",
                 fontsize=9)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ax.text(5, nrows - 8, f"NAIP {naip_meta['year']} | {naip_meta.get('resolution_m','?')} m",
            color="white", fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    # ── Panel 2: SSURGO soil series names on NAIP background ───────────────
    ax = axes[1]
    ax.imshow(rgb_display, origin="upper", extent=[0, ncols, nrows, 0], alpha=0.4)
    soil_patches2 = overlay_ssurgo_on_ax(
        ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.55,
        use_muname=True)
    p = _add_water_boundary(ax, water_persistent, color="cyan", lw=1.8,
                            label="NAIP NDWI water boundary")
    if p: soil_patches2.append(p)
    ax.set_xlim(0, ncols); ax.set_ylim(nrows, 0)
    ax.plot(prop_col, prop_row, "r*", markersize=10, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    soil_patches2.append(mpatches.Patch(color="red", label="Property pin"))
    ax.legend(handles=soil_patches2, fontsize=6, loc="lower right",
              framealpha=0.9, ncol=1)
    ax.set_title("SSURGO Soil Map Units — Individual Series Names\n"
                 "Verify: do unit boundaries match visible land cover in NAIP?",
                 fontsize=9)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")

    # ── Panel 3: NAIP NDWI map — directly shows water vs land ──────────────
    ax = axes[2]
    if ndwi_available:
        # NDWI: blue = water (>0), brown/red = dry land (<0)
        im3 = ax.imshow(ndwi, cmap="RdYlBu", vmin=-0.4, vmax=0.6,
                        origin="upper", extent=[0, ncols, nrows, 0])
        cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
        cb3.set_label("NDWI (Green−NIR)/(Green+NIR)\nBlue=water  |  Red=dry land/vegetation", fontsize=7.5)
        ax.set_title("NAIP NDWI — Water Index\n"
                     "Blue ≥ 0.2 = open water  |  Red < 0 = vegetation/impervious",
                     fontsize=9)
        # SSURGO boundaries thin overlay
        overlay_ssurgo_on_ax(ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.12)
        p = _add_water_boundary(ax, water_persistent, color="black", lw=1.2,
                                label="NDWI > 0.2 threshold")
        if p: ax.legend(handles=[p], fontsize=7, loc="lower right")
    elif ndvi_arr is not None:
        im3 = ax.imshow(ndvi_arr, cmap="RdYlGn", vmin=-0.3, vmax=0.8,
                        origin="upper", extent=[0, ncols, nrows, 0])
        cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
        cb3.set_label("NDVI (NIR−Red)/(NIR+Red)", fontsize=8)
        ax.set_title("NAIP NDVI\n(green = vegetation, red/yellow = bare/impervious)", fontsize=9)
        overlay_ssurgo_on_ax(ax, ssurgo_path, naip_transform, naip_crs, naip_shape, alpha=0.15)
        p = _add_water_boundary(ax, water_persistent, color="blue", lw=1.5, label="Water boundary")
        if p: ax.legend(handles=[p], fontsize=7, loc="lower right")
    else:
        ax.imshow(rgb_display, origin="upper", extent=[0, ncols, nrows, 0])
        ax.set_title("NAIP True-Color (NIR not available)", fontsize=10)
    ax.set_xlim(0, ncols); ax.set_ylim(nrows, 0)
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
