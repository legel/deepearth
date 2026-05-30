"""
NLCD 2021 Impervious Surface — Winter Garden FL
================================================
Downloads the NLCD 2021 developed impervious surface descriptor raster (30 m)
for the 2×2 km study area around 17801 Champagne Dr, Winter Garden FL, and
resamples it to the DEM grid (2.6 m) for use in CN correction.

Data source:
  USGS MRLC NLCD 2021 Impervious Descriptor — web tile service via
  the USGS TNM (The National Map) API or direct COG access via AWS S3.

Output:
  soil/data/nlcd_impervious.tif   — % impervious per DEM cell (0–100 %)
  soil/data/nlcd_impervious.png   — quick-look visualization

Usage:
    python3 soil/fetch_nlcd.py
"""

import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR  = os.path.join(BASE_DIR, "..", "dem", "data")
DEM_DIR  = os.path.normpath(DEM_DIR)

PROPERTY_LAT =  28.521592
PROPERTY_LON = -81.656981
# 2×2 km study area bbox (WGS84), 1 km radius from property
BBOX_WSEN = (-81.66723, 28.51258, -81.64673, 28.53060)

OUTPUT_TIF = os.path.join(DATA_DIR, "nlcd_impervious.tif")
OUTPUT_PNG = os.path.join(DATA_DIR, "nlcd_impervious.png")


def load_dem_profile():
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio not found. pip install rasterio")
    dem_path = os.path.join(DEM_DIR, "winter_garden_dem.tif")
    if not os.path.exists(dem_path):
        sys.exit(f"DEM not found: {dem_path}. Run dem_download.py first.")
    with rasterio.open(dem_path) as src:
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        shape     = (src.height, src.width)
    return profile, transform, crs, shape


# ── Source: MRLC WCS (OGC Web Coverage Service) ─────────────────────────────

MRLC_WCS_BASE = "https://www.mrlc.gov/geoserver/mrlc_display/ows"
# WCS 1.0.0 COVERAGE name (without workspace prefix — 2.0.1 ID has __ prefix)
NLCD_COVERAGE_NAME = "NLCD_2021_Impervious_L48"

def fetch_nlcd_wcs(bbox_wsen, dem_crs, output_path):
    """
    Download NLCD 2021 impervious raster via MRLC WCS 1.0.0 at ~30 m resolution.
    Values 0–100 = % impervious; 255 = nodata (open water / unclassified).
    Returns True if successful.
    """
    import urllib.request
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.warp import reproject, Resampling
    except ImportError as e:
        print(f"  ⚠ Missing dependency: {e}")
        return False

    west, south, east, north = bbox_wsen
    pad = 0.006  # ~650 m padding so all DEM cells are covered after reprojection
    w2, s2, e2, n2 = west - pad, south - pad, east + pad, north + pad

    # Native resolution ~30 m; request at ~30 m for our ~4.5 km box (~0.00027°/cell)
    width  = max(100, int((e2 - w2) / 0.00027))
    height = max(100, int((n2 - s2) / 0.00027))

    base_url = (
        f"{MRLC_WCS_BASE}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        f"&COVERAGE={NLCD_COVERAGE_NAME}"
        f"&BBOX={w2},{s2},{e2},{n2}"
        f"&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326"
        f"&FORMAT=GeoTIFF&WIDTH={width}&HEIGHT={height}"
    )
    print(f"  WCS 1.0.0 request ({width}×{height} px): {NLCD_COVERAGE_NAME}")
    try:
        with urllib.request.urlopen(base_url, timeout=60) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"  ✗ WCS request failed: {e}")
        return False

    if not raw[:4] == b"II*\x00" and not raw[:4] == b"MM\x00*" and b"<" in raw[:200]:
        print(f"  ✗ WCS returned non-TIFF response (likely XML error): {raw[:200]}")
        return False

    # Load the raw GeoTIFF bytes
    try:
        with MemoryFile(raw) as memfile:
            with memfile.open() as src_nlcd:
                nlcd_arr = src_nlcd.read(1).astype(np.float32)
                nlcd_nodata = src_nlcd.nodata
                if nlcd_nodata is not None:
                    nlcd_arr[nlcd_arr == nlcd_nodata] = np.nan
                nlcd_profile  = src_nlcd.profile.copy()
                nlcd_transform = src_nlcd.transform
                nlcd_crs      = src_nlcd.crs
        print(f"  ✓ Downloaded NLCD tile: {nlcd_arr.shape}, "
              f"range {np.nanmin(nlcd_arr):.0f}–{np.nanmax(nlcd_arr):.0f} %")
    except Exception as e:
        print(f"  ✗ Could not read downloaded GeoTIFF: {e}")
        return False

    # Resample to DEM grid using rasterio.warp
    _, dem_transform, dem_crs, dem_shape = load_dem_profile()
    out_arr = np.full(dem_shape, np.nan, dtype=np.float32)

    reproject(
        source=nlcd_arr,
        destination=out_arr,
        src_transform=nlcd_transform,
        src_crs=nlcd_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    out_arr = np.clip(out_arr, 0, 100)
    profile, _, _, _ = load_dem_profile()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_arr, 1)

    n_cells = int(np.isfinite(out_arr).sum())
    imp_frac = float(np.nanmean(out_arr[out_arr > 50]))
    print(f"  ✓ Saved NLCD impervious → {os.path.basename(output_path)}")
    print(f"    Valid cells: {n_cells:,}  |  Mean impervious: {np.nanmean(out_arr):.1f}%")
    print(f"    Cells >50% impervious: {int((out_arr > 50).sum()):,} "
          f"({100*float((out_arr > 50).sum()) / n_cells:.1f}% of domain)")
    return True


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_nlcd(tif_path, out_png):
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError as e:
        print(f"  ⚠ Cannot visualize: {e}")
        return

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and not (nodata != nodata):  # skip if nodata is NaN
            arr[arr == nodata] = np.nan
        raster_crs       = src.crs
        raster_transform = src.transform
    nrows, ncols = arr.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NLCD 2021 Impervious Surface — Winter Garden FL AOI",
                 fontsize=11, fontweight="bold")

    # Panel 1: impervious % map
    ax = axes[0]
    cmap = plt.colormaps["RdYlGn_r"].copy()
    cmap.set_bad("lightblue")
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=100, origin="upper")
    plt.colorbar(im, ax=ax, label="Impervious [%]", fraction=0.046)
    ax.set_title("Impervious surface fraction\n(30 m NLCD 2021, resampled to 2.6 m)", fontsize=9)
    ax.set_xlim(0, ncols); ax.set_ylim(nrows, 0)

    # Property marker — reproject WGS84 → raster CRS → pixel coords
    try:
        import pyproj
        from rasterio.transform import rowcol as _rowcol
        _tr = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        _px, _py = _tr.transform(PROPERTY_LON, PROPERTY_LAT)
        _r_pix, _c_pix = _rowcol(raster_transform, _px, _py)
        ax.plot(_c_pix, _r_pix, "b*", ms=14, markeredgecolor="white",
                markeredgewidth=0.8, label="Property", zorder=5)
        ax.legend(fontsize=8)
    except Exception:
        pass
    ax.axis("off")

    # Panel 2: histogram
    ax2 = axes[1]
    valid = arr[np.isfinite(arr)]
    bins = np.arange(0, 105, 5)
    ax2.hist(valid, bins=bins, color="steelblue", edgecolor="white", lw=0.4)
    ax2.set_xlabel("Impervious fraction [%]")
    ax2.set_ylabel("Cell count")
    ax2.set_title("Distribution of impervious cover\nacross 2×2 km AOI", fontsize=9)
    pct_high = 100 * float((valid > 50).sum()) / len(valid)
    ax2.axvline(50, color="red", ls="--", lw=1.2, label=f">50% impervious: {pct_high:.1f}%")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    mean_imp = float(np.nanmean(valid))
    ax2.text(0.98, 0.95, f"Mean: {mean_imp:.1f}%\nMax: {float(np.nanmax(valid)):.0f}%",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization → {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NLCD 2021 Impervious Surface Fetch — Winter Garden FL")
    print("=" * 60)

    _, _, dem_crs, _ = load_dem_profile()

    print("\n[1/1] MRLC WCS — NLCD 2021 Impervious …")
    success = fetch_nlcd_wcs(BBOX_WSEN, dem_crs, OUTPUT_TIF)

    if success and os.path.exists(OUTPUT_TIF):
        print("\nGenerating visualization …")
        visualize_nlcd(OUTPUT_TIF, OUTPUT_PNG)
        print("\n✓ NLCD impervious raster ready.")
        print("  Add it to verify_soil.png or use in CN correction:")
        print(f"  {OUTPUT_TIF}")
    else:
        print("\n  WCS download failed. Manual alternative:")
        print("  1. Visit https://www.mrlc.gov/viewer/")
        print("  2. Download NLCD 2021 Impervious for Orange County, FL")
        print("  3. Place GeoTIFF at:")
        print(f"     {OUTPUT_TIF}")
        print("  4. Re-run this script — it will detect and resample it.")

        # Check for manually placed file
        if os.path.exists(OUTPUT_TIF):
            print(f"\n  Found existing {OUTPUT_TIF} — generating visualization.")
            visualize_nlcd(OUTPUT_TIF, OUTPUT_PNG)


if __name__ == "__main__":
    main()
