"""
DEM Visualization — 17801 Champagne Dr, Winter Garden FL
=========================================================
Standalone visualization of the USGS 3DEP DEM and all derived layers.
Produces dem_visualization.png with seven clearly labeled panels.

Panels
------
1. Hillshade with property pin
2. Elevation color-coded relative to lake water surface (0 = water surface,
   negative = would be submerged, positive = above water)
3. Lake mask overlay on hillshade — S2-authoritative lake boundary + estimated bed depth
4. Flow accumulation (log scale) — drainage network
5. Elevation histogram with lake surface and mean marked
6. N–S cross-section through property + lake
7. (wide, bottom row) DEM filled contours at 1 m intervals with S2 lake boundary
   and SSURGO soil unit boundaries — geospatial accuracy check

Usage:
    python3 dem/dem_visualize.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
import rasterio
from rasterio.transform import xy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(BASE_DIR, "dem_visualization.png")

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981


def load(fname, dtype=np.float32):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None, None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(dtype)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    nodata = profile.get("nodata", None)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, (transform, crs, profile)


def latlon_to_rowcol(lat, lon, transform, crs):
    """Convert geographic coordinates to pixel row/col."""
    import pyproj
    if "4326" in str(crs) or "WGS" in str(crs).upper():
        x, y = lon, lat
    else:
        transformer = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def hillshade(dem, azimuth=315, altitude=45):
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    nodata_mask = ~np.isfinite(dem)
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    hs = ls.hillshade(filled, vert_exag=3, dx=2.6, dy=2.6)
    hs[nodata_mask] = np.nan
    return hs


def pct_clip(arr, lo=2, hi=98):
    a, b = np.nanpercentile(arr, lo), np.nanpercentile(arr, hi)
    return np.clip((arr - a) / (b - a + 1e-9), 0, 1)


def main():
    print("Loading rasters …")
    dem, meta     = load("winter_garden_dem.tif")
    # Prefer S2-authoritative lake mask (confirmed open water from satellite imagery)
    # over DEM-derived mask which may include false positives (flat roads, rooftops)
    _mask_src = ("lake_mask_s2.tif"
                 if os.path.exists(os.path.join(DATA_DIR, "lake_mask_s2.tif"))
                 else "lake_mask.tif")
    lake_mask, _  = load(_mask_src)
    # Prefer FWC bathymetric survey; fall back to shoreline-slope estimate
    _bed_src  = ("lake_bed_dem_fwc.tif"
                 if os.path.exists(os.path.join(DATA_DIR, "lake_bed_dem_fwc.tif"))
                 else "lake_bed_dem_estimated.tif")
    lake_bed, _   = load(_bed_src)
    print(f"  Lake mask source: {_mask_src}")
    print(f"  Lake bed source: {_bed_src}")
    flow_acc, _   = load("flow_acc.tif")

    if dem is None:
        sys.exit("DEM not found. Run dem_download.py first.")

    transform, crs, profile = meta
    cell_m = abs(transform.a)
    nrows, ncols = dem.shape

    # Property pixel location
    prop_row, prop_col = latlon_to_rowcol(PROPERTY_LAT, PROPERTY_LON, transform, crs)
    prop_row = np.clip(prop_row, 0, nrows - 1)
    prop_col = np.clip(prop_col, 0, ncols - 1)
    print(f"  Property pixel: row={prop_row}, col={prop_col}")

    # Water surface reference elevation (mean of lake mask cells in original DEM)
    lake_bool = (lake_mask == 1) & np.isfinite(dem)
    if lake_bool.sum() > 0:
        water_surface_elev = float(np.nanmean(dem[lake_bool]))
    else:
        water_surface_elev = float(np.nanmean(dem))
    print(f"  Water surface reference: {water_surface_elev:.2f} m NAVD88")
    print(f"  DEM is hydro-flattened: lakes shown at SURFACE elevation, not bed")
    print(f"  Lake bed DEM: estimated by shoreline slope extrapolation")

    # Elevation relative to water surface
    dem_rel = dem - water_surface_elev

    hs = hillshade(dem)

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    ax_contour = fig.add_subplot(gs[2, :])   # bottom row spanning all 3 columns

    fig.suptitle(
        "DEM Verification — 17801 Champagne Dr, Winter Garden FL\n"
        f"2×2 km AOI | USGS 3DEP 3m | NAVD88 hydro-flattened | "
        f"Water surface ref = {water_surface_elev:.1f} m",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Hillshade with property pin ────────────────────────────────
    ax = axes[0, 0]
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.plot(prop_col, prop_row, marker="*", color="red", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, label="17801 Champagne Dr", zorder=5)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Hillshade (vert. exag. ×3)\nwith property location", fontsize=10)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ax.text(5, nrows - 10, f"Cell: {cell_m:.1f} m | Grid: {nrows}×{ncols}",
            color="white", fontsize=7, va="bottom")

    # ── Panel 2: Elevation relative to water surface ─────────────────────────
    ax = axes[0, 1]
    vmax = max(abs(np.nanpercentile(dem_rel, 2)), abs(np.nanpercentile(dem_rel, 98)))
    im = ax.imshow(dem_rel, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax.plot(prop_col, prop_row, "r*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.6, zorder=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Elevation relative to\nlake surface [m]", fontsize=8)
    ax.set_title("Elevation relative to lake water surface\nBlue = below, Red = above", fontsize=10)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    # Add 0-line annotation
    ax.text(5, 15, "0 = lake water surface\n< 0: would be submerged\n> 0: above water",
            color="black", fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── Panel 3: Lake mask — surface vs bed comparison ───────────────────────
    ax = axes[0, 2]
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, origin="upper", alpha=0.6)
    # Show the difference: original DEM (flat surface) vs lake bed
    if lake_bed is not None:
        depth = np.where(lake_bool, water_surface_elev - lake_bed, np.nan)
        depth_plot = np.where(lake_bool, np.clip(depth, 0, 10), np.nan)
        im3 = ax.imshow(depth_plot, cmap="Blues", vmin=0, vmax=5, origin="upper", alpha=0.85)
        cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
        cb3.set_label("Estimated lake depth [m]\n(water surface − bed)", fontsize=8)
    else:
        ax.imshow(np.where(lake_bool, 1, np.nan), cmap="Blues", origin="upper", alpha=0.7)
    ax.plot(prop_col, prop_row, "r*", markersize=12, markeredgecolor="white",
            markeredgewidth=0.6, zorder=5)
    _bed_label = "FWC survey" if "fwc" in _bed_src else "Estimated (shoreline slope)"
    _mask_label = "S2-authoritative" if "s2" in _mask_src else "DEM-derived"
    ax.set_title(f"Lake mask ({_mask_label}) + depth\n"
                 f"Bed source: {_bed_label}", fontsize=9)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")

    # ── Panel 4: Flow accumulation ───────────────────────────────────────────
    ax = axes[1, 0]
    if flow_acc is not None:
        log_acc = np.log10(np.clip(flow_acc, 1, None)).astype(np.float32)
        log_acc[~np.isfinite(dem)] = np.nan
        # 100-cell border + 98th-percentile clip — removes D8 edge artifact
        B = 100
        log_acc[:B, :] = np.nan; log_acc[-B:, :] = np.nan
        log_acc[:, :B] = np.nan; log_acc[:, -B:] = np.nan
        pct98 = float(np.nanpercentile(log_acc, 98))
        log_acc = np.clip(log_acc, 0, pct98)
        im4 = ax.imshow(log_acc, cmap="plasma", origin="upper", vmin=0, vmax=pct98)
        cb4 = fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
        cb4.set_label("log₁₀(upstream cells)", fontsize=8)
        ax.set_title(f"Flow accumulation (log scale)\nMax: {int(np.nanmax(flow_acc)):,} cells upstream", fontsize=10)
    ax.plot(prop_col, prop_row, "w*", markersize=12, markeredgecolor="red",
            markeredgewidth=0.6, zorder=5)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")

    # ── Panel 5: Elevation histogram ─────────────────────────────────────────
    ax = axes[1, 1]
    valid_dem = dem[np.isfinite(dem)].ravel()
    ax.hist(valid_dem, bins=80, color="steelblue", edgecolor="none", alpha=0.8, density=True)
    ax.axvline(water_surface_elev, color="royalblue", lw=2,
               label=f"Lake surface: {water_surface_elev:.1f} m")
    ax.axvline(float(np.nanmean(dem)), color="orange", lw=1.5, linestyle="--",
               label=f"Domain mean: {float(np.nanmean(dem)):.1f} m")
    prop_elev = float(dem[prop_row, prop_col]) if np.isfinite(dem[prop_row, prop_col]) else float(np.nanmean(dem))
    ax.axvline(prop_elev, color="red", lw=1.5, linestyle=":",
               label=f"Property elev: {prop_elev:.1f} m")
    ax.set_xlabel("Elevation [m NAVD88]")
    ax.set_ylabel("Density")
    ax.set_title("Elevation distribution\n(all 753k terrain cells)", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 6: N–S cross-section through property ──────────────────────────
    ax = axes[1, 2]
    col_slice = prop_col
    rows = np.arange(nrows)
    elev_profile = dem[:, col_slice]
    bed_profile  = lake_bed[:, col_slice] if lake_bed is not None else None
    mask_profile = lake_bool[:, col_slice]
    dist_km = rows * cell_m / 1000

    ax.fill_between(dist_km, elev_profile, alpha=0.25, color="saddlebrown", label="Terrain")
    ax.plot(dist_km, elev_profile, color="saddlebrown", lw=1)
    if bed_profile is not None:
        ax.fill_between(dist_km,
                        np.where(mask_profile, bed_profile, np.nan),
                        np.where(mask_profile, elev_profile, np.nan),
                        color="steelblue", alpha=0.6, label="Lake water body")
        ax.plot(dist_km, np.where(mask_profile, elev_profile, np.nan),
                color="royalblue", lw=1.5, label="Lake surface (DEM flat)")
        ax.plot(dist_km, np.where(mask_profile, bed_profile, np.nan),
                color="navy", lw=1, linestyle="--", label="Est. lake bed")
    ax.axhline(water_surface_elev, color="royalblue", lw=0.8, linestyle=":",
               alpha=0.5, label=f"Water surface ref {water_surface_elev:.1f} m")
    prop_dist = prop_row * cell_m / 1000
    ax.axvline(prop_dist, color="red", lw=1.5, linestyle="--", label="Property")
    ax.set_xlabel("Distance N→S [km]")
    ax.set_ylabel("Elevation [m NAVD88]")
    ax.set_title(f"N–S cross-section (col={col_slice}, through property)\n"
                 "Shows lake surface flatness vs extrapolated bed", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(bottom=max(0, float(np.nanmin(dem)) - 2))

    # ── Panel 7: DEM Filled Contours — geospatial accuracy check ────────────
    ax = ax_contour
    valid_dem = np.where(np.isfinite(dem), dem, np.nan)
    z_min = np.nanpercentile(valid_dem, 1)
    z_max = np.nanpercentile(valid_dem, 99)
    levels = np.arange(np.floor(z_min), np.ceil(z_max) + 1.0, 1.0)
    cf = ax.contourf(valid_dem, levels=levels, cmap="terrain", origin="upper", extend="both")
    ax.contour(valid_dem, levels=levels, colors="white", linewidths=0.3, alpha=0.4, origin="upper")
    cb7 = fig.colorbar(cf, ax=ax, fraction=0.015, pad=0.01)
    cb7.set_label("Elevation [m NAVD88]", fontsize=8)
    # S2 lake boundary (white contour line at mask=0.5)
    if lake_mask is not None:
        ax.contour(lake_mask.astype(np.float32), levels=[0.5],
                   colors=["cyan"], linewidths=2.0, origin="upper",
                   linestyles="solid")
        mpatches_lake = mpatches.Patch(color="cyan", label=f"S2 lake boundary ({_mask_label})")
    else:
        mpatches_lake = None
    # SSURGO soil unit boundaries (optional)
    _ssurgo_path = os.path.join(BASE_DIR, "..", "soil", "data", "ssurgo_mapunits.geojson")
    _ssurgo_plotted = False
    if os.path.exists(_ssurgo_path):
        try:
            with open(_ssurgo_path) as f:
                ssurgo = json.load(f)
            from rasterio.transform import rowcol
            import pyproj
            _transformer = pyproj.Transformer.from_crs("epsg:4326", str(crs), always_xy=True) \
                if "4326" not in str(crs) else None
            for feat in ssurgo.get("features", []):
                geom = feat.get("geometry", {})
                if geom.get("type") not in ("Polygon", "MultiPolygon"):
                    continue
                rings = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                         else [geom["coordinates"]])
                for poly in rings:
                    for ring in poly:
                        coords = np.array(ring)
                        if _transformer:
                            xs, ys = _transformer.transform(coords[:, 0], coords[:, 1])
                        else:
                            xs, ys = coords[:, 0], coords[:, 1]
                        cols_s = (xs - transform.c) / transform.a
                        rows_s = (ys - transform.f) / transform.e
                        ax.plot(cols_s, rows_s, color="gold", lw=0.7, alpha=0.7)
            _ssurgo_plotted = True
        except Exception as e:
            print(f"  SSURGO overlay skipped: {e}")
    # Property pin
    ax.plot(prop_col, prop_row, marker="*", color="red", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5, label="Property")
    legend_handles = [
        mpatches.Patch(color="red", label="Property"),
    ]
    if mpatches_lake:
        legend_handles.append(mpatches_lake)
    if _ssurgo_plotted:
        legend_handles.append(mpatches.Patch(color="gold", label="SSURGO soil units"))
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_title(
        "DEM Contours at 1 m intervals — Geospatial Accuracy Check\n"
        "Cyan = S2 lake boundary (verify contours hug the lake shore; misalignment = DEM/CRS error)",
        fontsize=10)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)  # origin=upper: row 0 at top
    ax.text(5, nrows - 10, f"Cell: {cell_m:.1f} m | Contour interval: 1 m | "
            f"Elev range: {z_min:.1f}–{z_max:.1f} m NAVD88",
            color="white", fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {OUT_PATH}")
    print(f"\nKey facts:")
    print(f"  DEM type        : hydro-flattened (lakes = water SURFACE elevation, NOT bed)")
    print(f"  Lake surface    : {water_surface_elev:.2f} m NAVD88")
    print(f"  Property elev   : {prop_elev:.2f} m NAVD88 ({prop_elev - water_surface_elev:+.1f} m above lake)")
    print(f"  Domain elev     : {float(np.nanmin(dem)):.1f} – {float(np.nanmax(dem)):.1f} m")
    print(f"  Lake cells      : {int(lake_bool.sum()):,} ({100*lake_bool.sum()/np.isfinite(dem).sum():.1f}% of domain)")


if __name__ == "__main__":
    main()
