"""
NLCD 2021 Impervious Surface — CFX SR417 Corridor
====================================================
Downloads the NLCD 2021 developed impervious surface descriptor raster (30 m)
for the 2x2 km CFX SR417 corridor test-landscape AOI (28.36687N, -81.43299W,
near Lake Nona / south Orlando, FL) and resamples it onto the project's 1m
LiDAR DEM grid for use as the "roads and buildings" mask — i.e. the
complement of the natural-ground / landcover layer requested by Lance
(everything that ISN'T impervious is the natural-ground surface that the
soil/vegetation layers in this project already describe).

Ported as-is from models/flood_hydrology/soil/fetch_nlcd.py (Winter Garden
digital twin), adapted only for this project's AOI coordinates and DEM path.

Data source:
  USGS MRLC NLCD 2021 Impervious Descriptor — WCS 1.0.0 at
  https://www.mrlc.gov/geoserver/mrlc_display/ows, coverage
  NLCD_2021_Impervious_L48.

Output:
  soil/data/nlcd_impervious.tif   — % impervious per DEM cell (0-100 %)
  soil/data/nlcd_impervious.png   — quick-look visualization

Usage:
    python3 soil/fetch_nlcd.py
    python3 soil/fetch_nlcd.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
"""

import os
import sys
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Shared site registry (added 2026-08-04) ──────────────────────────────────
# Makes `--site <name>` resolve lat/lon/radius AND the output directory from the ONE registry
# (site_registry.py -> lidar/test_sites.py) instead of hand-typed coordinates. Purely additive:
# with no --site flag this script behaves exactly as it always has. See site_registry.py's
# docstring for why (INTERNSHIP_AUDIT_2026-08-03.md §4: site3's data existed on disk with no
# script that could reproduce it, because coordinates were typed by hand per-invocation).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import site_registry  # noqa: E402

DEM_DIR  = os.path.join(BASE_DIR, "..", "dem", "data")
DEM_DIR  = os.path.normpath(DEM_DIR)

PROPERTY_LAT  = 28.36687   # CFX SR417 corridor test-landscape AOI (Lake Nona / south Orlando)
PROPERTY_LON  = -81.43299
RADIUS_KM     = 1.0

# Reference grid: the project's full-resolution 1m LiDAR DEM (2608x2609, EPSG:5070).
# Per project convention, the 3m DEM is archived and all current work aligns to the
# 1m DEM — this mirrors the original script's use of the raw (not hydro-conditioned)
# project DEM as the resampling target.
DEM_FILENAME = "sr417_corridor_dem_1m.tif"

OUTPUT_TIF = os.path.join(DATA_DIR, "nlcd_impervious.tif")
OUTPUT_PNG = os.path.join(DATA_DIR, "nlcd_impervious.png")


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def load_dem_profile():
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio not found. pip install rasterio")
    dem_path = os.path.join(DEM_DIR, DEM_FILENAME)
    if not os.path.exists(dem_path):
        sys.exit(f"DEM not found: {dem_path}. Run dem/dem_download.py first.")
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
    Values 0-100 = % impervious; 255 = nodata (open water / unclassified).
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

    # Native resolution ~30 m; request at ~30 m for our ~4.5 km box (~0.00027 deg/cell)
    width  = max(100, int((e2 - w2) / 0.00027))
    height = max(100, int((n2 - s2) / 0.00027))

    base_url = (
        f"{MRLC_WCS_BASE}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        f"&COVERAGE={NLCD_COVERAGE_NAME}"
        f"&BBOX={w2},{s2},{e2},{n2}"
        f"&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326"
        f"&FORMAT=GeoTIFF&WIDTH={width}&HEIGHT={height}"
    )
    print(f"  WCS 1.0.0 request ({width}x{height} px): {NLCD_COVERAGE_NAME}")
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
                nlcd_transform = src_nlcd.transform
                nlcd_crs      = src_nlcd.crs
        print(f"  ✓ Downloaded NLCD tile: {nlcd_arr.shape}, "
              f"range {np.nanmin(nlcd_arr):.0f}-{np.nanmax(nlcd_arr):.0f} %")
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
    print(f"  ✓ Saved NLCD impervious → {os.path.basename(output_path)}")
    print(f"    Valid cells: {n_cells:,}  |  Mean impervious: {np.nanmean(out_arr):.1f}%")
    print(f"    Cells >50% impervious: {int((out_arr > 50).sum()):,} "
          f"({100*float((out_arr > 50).sum()) / n_cells:.1f}% of domain)")
    return True


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_nlcd(tif_path, out_png, lat, lon):
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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
    fig.suptitle("NLCD 2021 Impervious Surface — CFX SR417 Corridor AOI",
                 fontsize=11, fontweight="bold")

    # Panel 1: impervious % map
    ax = axes[0]
    cmap = plt.colormaps["RdYlGn_r"].copy()
    cmap.set_bad("lightblue")
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=100, origin="upper")
    plt.colorbar(im, ax=ax, label="Impervious [%]", fraction=0.046)
    ax.set_title("Impervious surface fraction\n(30 m NLCD 2021, resampled to 1m DEM grid)", fontsize=9)
    ax.set_xlim(0, ncols); ax.set_ylim(nrows, 0)

    # AOI center marker — reproject WGS84 -> raster CRS -> pixel coords
    try:
        import pyproj
        from rasterio.transform import rowcol as _rowcol
        _tr = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        _px, _py = _tr.transform(lon, lat)
        _r_pix, _c_pix = _rowcol(raster_transform, _px, _py)
        ax.plot(_c_pix, _r_pix, "b*", ms=14, markeredgecolor="white",
                markeredgewidth=0.8, label="AOI center", zorder=5)
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
    ax2.set_title("Distribution of impervious cover\nacross 2x2 km AOI", fontsize=9)
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

def main(lat=PROPERTY_LAT, lon=PROPERTY_LON, radius_km=RADIUS_KM):
    print("=" * 60)
    print("NLCD 2021 Impervious Surface Fetch — CFX SR417 Corridor")
    print("=" * 60)

    bbox = bbox_from_center(lat, lon, radius_km)
    print(f"AOI: ({lat}, {lon}), radius {radius_km} km")
    print(f"  Bounding box: {[round(x,5) for x in bbox]}")

    _, _, dem_crs, _ = load_dem_profile()

    print("\n[1/1] MRLC WCS — NLCD 2021 Impervious …")
    success = fetch_nlcd_wcs(bbox, dem_crs, OUTPUT_TIF)

    if success and os.path.exists(OUTPUT_TIF):
        print("\nGenerating visualization …")
        visualize_nlcd(OUTPUT_TIF, OUTPUT_PNG, lat, lon)
        print("\n✓ NLCD impervious raster ready.")
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
            visualize_nlcd(OUTPUT_TIF, OUTPUT_PNG, lat, lon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NLCD 2021 impervious surface and resample to DEM grid")
    parser.add_argument("--lat",       type=float, default=PROPERTY_LAT)
    parser.add_argument("--lon",       type=float, default=PROPERTY_LON)
    parser.add_argument("--radius_km", type=float, default=RADIUS_KM)
    site_registry.add_site_arg(parser)
    args = site_registry.resolve(parser.parse_args(), category="soil")
    if args.site_data_root:
        # Rebind the module-level DATA_DIR so every function writing output lands in the
        # selected site's own tree instead of the main AOI's (the exact clobbering
        # fetch_naip_site3.py's docstring warns about — both share e.g. naip_2021_RGB.tif).
        globals()["DATA_DIR"] = args.site_data_dir
    main(args.lat, args.lon, args.radius_km)
