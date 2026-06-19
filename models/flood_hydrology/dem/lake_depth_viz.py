"""
Lake Depth & Volume Visualization — FWC Bathymetry + DEM
=========================================================
Uses real FWC bathymetric survey data for the lake bed elevation.
NO assumptions or extrapolated depth estimates — only measured data.

Physics:
    Water surface elevation (WSE) = mean DEM elevation at S2 lake mask shoreline
    Lake depth(x,y)  = max(0, WSE − FWC_lakebed(x,y))    [metres]
    Lake volume       = Σ depth(x,y) × cell_area_m²       [m³]
    (standard voxel integration; each grid cell is a vertical column of water)

Data sources:
    winter_garden_dem.tif       — USGS 3DEP 1-arc-second DEM (2.6m resampled)
    lake_bed_dem_fwc.tif        — FWC bathymetric survey (NAVD88 metres)
    lake_mask_s2.tif            — S2 MNDWI consensus lake boundary
    winter_garden_dem_merged.tif — DEM + FWC bathymetry merged raster

Outputs:
    dem/lake_depth_viz.png       — 6-panel figure
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import rasterio
from scipy.ndimage import gaussian_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PNG  = os.path.join(BASE_DIR, "lake_depth_viz.png")


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        return arr, src.transform, src.crs, src.res[0]


def main():
    # ── Load DEM ────────────────────────────────────────────────────────────
    dem, dem_t, dem_crs, dem_res = load(os.path.join(DATA_DIR, "winter_garden_dem.tif"))

    # Lake boundary + interpolated FWC from shared utility
    import sys as _sys; _sys.path.insert(0, BASE_DIR)
    from lake_utils import get_lake_mask_and_fwc
    S2_DATA = os.path.join(BASE_DIR, "..", "sentinel2", "data")
    lake, fwc, wse, lake_source = get_lake_mask_and_fwc(
        dem, dem_t, dem_crs, dem_res, DATA_DIR, s2_data_dir=S2_DATA)
    print(f"Lake WSE (water surface elevation): {wse:.3f} m NAVD88")
    print(f"Lake boundary source: {lake_source}")

    # Depth grid (water column height at each lake pixel, 0 outside)
    depth = np.where(lake & ~np.isnan(fwc), np.maximum(0.0, wse - fwc), 0.0)

    # Volume
    cell_area = dem_res ** 2
    volume_m3 = float(depth.sum() * cell_area)
    lake_area_ha = float(lake.sum() * cell_area / 1e4)
    mean_depth = float(depth[lake].mean())
    max_depth  = float(depth[lake].max())
    print(f"Lake area: {lake_area_ha:.1f} ha")
    print(f"Mean depth: {mean_depth:.2f} m  |  Max depth: {max_depth:.2f} m")
    print(f"Volume: {volume_m3:,.0f} m³  ({volume_m3/1e6:.3f} million m³)")

    # FWC lake bed stats
    fwc_lake = fwc[lake & ~np.isnan(fwc)]
    print(f"FWC bed elev (lake pixels): min={fwc_lake.min():.2f}  "
          f"mean={fwc_lake.mean():.2f}  max={fwc_lake.max():.2f} m NAVD88")

    # Hypsometric curve: water area vs WSE (elevation)
    hyps_path = os.path.join(DATA_DIR, "lake_hypsometric_curve.csv")
    if os.path.exists(hyps_path):
        import pandas as pd
        hyps = pd.read_csv(hyps_path)
    else:
        hyps = None

    # Cross-section through the deepest point
    max_row, max_col = np.unravel_index(depth.argmax(), depth.shape)
    cs_row = depth[max_row, :]           # east-west transect
    cs_col = depth[:, max_col]           # north-south transect
    cs_bed_row = np.where(lake[max_row, :], fwc[max_row, :], np.nan)
    cs_bed_col = np.where(lake[:, max_col], fwc[:, max_col], np.nan)

    # Lake contour array: zero the 3-pixel border so contour() never draws a
    # straight line along the frame edge if the lake extends to row/col 0.
    lake_contour = lake.astype(float)
    lake_contour[:3, :] = 0; lake_contour[-3:, :] = 0
    lake_contour[:, :3] = 0; lake_contour[:, -3:] = 0

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Lake Depth & Volume — Johns Lake Western Bay, Winter Garden FL\n"
        "FWC Bathymetric Survey Data (no assumptions)  |  "
        f"WSE = {wse:.2f} m NAVD88  |  Area = {lake_area_ha:.1f} ha (OmniWaterMask)  |  "
        f"Mean depth = {mean_depth:.2f} m  |  Max depth = {max_depth:.2f} m  |  "
        f"Volume = {volume_m3/1e6:.3f} million m³",
        fontsize=11, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                          left=0.06, right=0.97, top=0.91, bottom=0.06)

    # ── Panel 1: DEM terrain map ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dem, cmap="terrain", origin="upper",
                     vmin=dem[lake | (~lake)].min(), vmax=dem.max())
    # Lake boundary overlay
    from matplotlib.contour import QuadContourSet
    ax1.contour(lake_contour, levels=[0.5], colors=["cyan"], linewidths=1.5)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("Elevation (m NAVD88)", fontsize=8)
    ax1.set_title("USGS 3DEP Terrain DEM\n(cyan = S2 lake boundary)", fontsize=9)
    ax1.set_xlabel("col (px)"); ax1.set_ylabel("row (px)")
    ax1.plot(max_col, max_row, "r+", markersize=10, lw=2, label=f"Deepest point")
    ax1.legend(fontsize=7)

    # ── Panel 2: FWC lake bed elevation ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fwc_display = np.where(lake, fwc, np.nan)
    # Use a masked array so NaN (outside lake) is transparent
    fwc_masked = np.ma.masked_invalid(fwc_display)
    im2 = ax2.imshow(fwc_masked, cmap="YlOrRd_r", origin="upper",
                     vmin=fwc_lake.min(), vmax=wse)
    ax2.imshow(dem, cmap="gray", origin="upper", alpha=0.3,
               vmin=dem.min(), vmax=dem.max())
    ax2.contour(lake_contour, levels=[0.5], colors=["black"], linewidths=1.0)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("FWC Lake Bed (m NAVD88)", fontsize=8)
    ax2.set_title(f"FWC Bathymetric Survey — Lake Bed Elevation\n"
                  f"Dark red = shallow ({wse:.1f} m = surface) | Yellow = deepest ({fwc_lake.min():.1f} m)",
                  fontsize=8.5)
    ax2.set_xlabel("col (px)"); ax2.set_ylabel("row (px)")

    # ── Panel 3: Depth map ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    depth_display = np.where(lake, depth, np.nan)
    depth_masked = np.ma.masked_invalid(depth_display)
    cmap_depth = matplotlib.colormaps["Blues"]
    im3 = ax3.imshow(depth_masked, cmap=cmap_depth, origin="upper",
                     vmin=0, vmax=max_depth)
    ax3.imshow(dem, cmap="gray", origin="upper", alpha=0.25,
               vmin=dem.min(), vmax=dem.max())
    # Depth contours at 0.5m intervals
    depth_smooth = gaussian_filter(depth_display if not np.all(np.isnan(depth_display))
                                   else np.zeros_like(depth), sigma=1.5)
    depths_to_contour = np.arange(0.5, max_depth, 0.5)
    if len(depths_to_contour):
        CS = ax3.contour(np.where(lake, depth_smooth, 0), levels=depths_to_contour,
                         colors=["navy"], linewidths=0.7, alpha=0.6)
        ax3.clabel(CS, fmt="%.1fm", fontsize=6, inline=True)
    ax3.contour(lake_contour, levels=[0.5], colors=["black"], linewidths=1.2)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02)
    cb3.set_label("Water Depth (m)", fontsize=8)
    ax3.set_title(f"Lake Water Depth Map (WSE − FWC bed)\n"
                  f"Contours at 0.5m intervals  |  Max depth = {max_depth:.2f} m",
                  fontsize=8.5)
    ax3.set_xlabel("col (px)"); ax3.set_ylabel("row (px)")
    ax3.plot(max_col, max_row, "r+", markersize=10, lw=2)

    # ── Panel 4: E-W cross-section ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    dist_ew = np.arange(len(cs_row)) * dem_res
    wse_line = np.where(lake[max_row, :], wse, np.nan)
    ax4.fill_between(dist_ew, cs_bed_row, wse_line,
                     where=~np.isnan(cs_bed_row),
                     color="#2E86C1", alpha=0.6, label=f"Water column (max {cs_row.max():.2f} m)")
    ax4.plot(dist_ew, cs_bed_row, color="#8B4513", lw=1.5, label="FWC lake bed")
    ax4.axhline(wse, color="cyan", lw=1.2, linestyle="--", label=f"WSE = {wse:.2f} m")
    ax4.set_xlabel("Distance E-W (m)"); ax4.set_ylabel("Elevation (m NAVD88)")
    ax4.set_title(f"E-W Cross-Section at Deepest Point (row {max_row})\n"
                  f"Max depth along transect: {cs_row.max():.2f} m", fontsize=8.5)
    ax4.legend(fontsize=7.5); ax4.grid(alpha=0.3)
    ax4.set_ylim(fwc_lake.min() - 0.5, wse + 0.8)

    # ── Panel 5: N-S cross-section ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    dist_ns = np.arange(len(cs_col)) * dem_res
    wse_col = np.where(lake[:, max_col], wse, np.nan)
    ax5.fill_between(dist_ns, cs_bed_col, wse_col,
                     where=~np.isnan(cs_bed_col),
                     color="#2E86C1", alpha=0.6, label=f"Water column")
    ax5.plot(dist_ns, cs_bed_col, color="#8B4513", lw=1.5, label="FWC lake bed")
    ax5.axhline(wse, color="cyan", lw=1.2, linestyle="--", label=f"WSE = {wse:.2f} m")
    ax5.set_xlabel("Distance N-S (m)"); ax5.set_ylabel("Elevation (m NAVD88)")
    ax5.set_title(f"N-S Cross-Section at Deepest Point (col {max_col})\n"
                  f"Max depth along transect: {cs_col.max():.2f} m", fontsize=8.5)
    ax5.legend(fontsize=7.5); ax5.grid(alpha=0.3)
    ax5.set_ylim(fwc_lake.min() - 0.5, wse + 0.8)

    # ── Panel 6: Hypsometric curve or volume summary ─────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if hyps is not None and "elevation_m" in hyps.columns:
        elev_col = "elevation_m"
        area_col = [c for c in hyps.columns if "area" in c.lower()][0]
        ax6.plot(hyps[area_col], hyps[elev_col], color="#2E86C1", lw=2, marker="o",
                 markersize=3, label="Area-elevation curve")
        ax6.axhline(wse, color="cyan", lw=1.2, linestyle="--", label=f"Current WSE {wse:.2f} m")
        ax6.axhline(fwc_lake.min(), color="#8B4513", lw=1.0, linestyle=":",
                    label=f"Min lake bed {fwc_lake.min():.2f} m")
        ax6.set_xlabel("Lake area (ha)"); ax6.set_ylabel("Elevation (m NAVD88)")
        ax6.set_title("Hypsometric Curve\n(lake area vs. water surface elevation)", fontsize=8.5)
        ax6.legend(fontsize=7.5); ax6.grid(alpha=0.3)
    else:
        # Depth histogram
        depth_vals = depth[lake & (depth > 0)]
        ax6.hist(depth_vals, bins=30, color="#2E86C1", alpha=0.8, edgecolor="navy")
        ax6.axvline(mean_depth, color="red", lw=1.5, linestyle="--",
                    label=f"Mean = {mean_depth:.2f} m")
        ax6.axvline(max_depth, color="orange", lw=1.5, linestyle=":",
                    label=f"Max = {max_depth:.2f} m")
        ax6.set_xlabel("Water depth (m)"); ax6.set_ylabel("Grid cells (count)")
        ax6.set_title("Depth Distribution (FWC bathymetry)\n"
                      f"Volume = {volume_m3/1e6:.3f} M m³  |  Area = {lake_area_ha:.1f} ha",
                      fontsize=8.5)
        ax6.legend(fontsize=7.5); ax6.grid(alpha=0.3)

        # Summary text box
        summary = (
            f"Lake: Johns Lake western bay AOI\n"
            f"Boundary: OmniWaterMask consensus\n"
            f"WSE: {wse:.3f} m NAVD88\n"
            f"Bed min: {fwc_lake.min():.3f} m NAVD88\n"
            f"Bed mean: {fwc_lake.mean():.3f} m NAVD88\n"
            f"Mean depth: {mean_depth:.2f} m\n"
            f"Max depth: {max_depth:.2f} m\n"
            f"Lake area: {lake_area_ha:.1f} ha\n"
            f"Volume: {volume_m3/1e6:.3f} M m³\n"
            f"Bed source: FWC bathymetric survey"
        )
        ax6.text(0.97, 0.97, summary, transform=ax6.transAxes,
                 fontsize=7.5, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.95))

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {OUT_PNG}")


if __name__ == "__main__":
    main()
