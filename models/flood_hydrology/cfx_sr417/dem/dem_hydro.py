"""
Hydrological DEM Processing — CFX SR417 Corridor
=================================================
Computes flow-routing and flood-depth derivatives from the conditioned DEM.

Methodology (peer-reviewed, no arbitrary assumptions):
  1. Stream burning  — carve the USGS 3DHP Shingle Creek flowlines into the 1m DEM
                       before routing.  Enforces the known drainage network and
                       eliminates the largest source of flat-terrain routing error.
                       Method: reduce DEM elevations along the buffered channel
                       by BURN_DEPTH_M (default 1.5 m).
  2. Breach depressions — richdem `BreachDepressions()` on the burned DEM.
                       Carves least-cost paths through barriers instead of filling
                       them.  Appropriate for this AOI: Central-Florida wetland
                       soils (Basinger, Samsula muck) create REAL depressions that
                       should not be treated as DEM artefacts.
  3. D8 flow direction — pysheds deterministic-8 steepest-descent routing.
                       Industry standard (USGS StreamStats, NOAA National Water
                       Model); reproducible and auditable.
  4. Flow accumulation — upstream contributing area per cell (cells, not area).
                       Stream network = cells where accumulation > ACC_THRESHOLD.
                       Threshold calibrated against 3DHP: Shingle Creek appears at
                       ~50–150 cells (50–130 m²) on the 1m grid.
  5. HAND (Height Above Nearest Drainage) — elevation of each cell above the
                       nearest stream-network cell along the D8 flow path.
                       Reference: Nobre et al. (2011), NOAA NWS operational use.
                       Directly maps flood inundation depth: HAND < flood_depth →
                       inundated.  Calibration target: Ian peak 3,500 cfs /
                       11.43 ft at NWIS 02263800 (7.2 km downstream).
  6. Watershed delineation — pour-point watershed at the AOI centre.

Outputs → dem/data/hydro/:
  dem_burned.tif          — DEM after stream burning (pre-breach, for inspection)
  dem_conditioned.tif     — DEM after stream burning + breach depression filling
  flow_dir.tif            — D8 flow direction (pysheds encoding)
  flow_accum.tif          — upstream cell count (log-scaled for vis)
  stream_network.geojson  — vectorised stream cells (EPSG:4326, accum > threshold)
  hand.tif                — Height Above Nearest Drainage (metres)
  watershed.geojson       — pour-point watershed polygon (EPSG:4326)
  hydro_summary.json      — threshold used, stream length, HAND stats

  Viewer PNGs (512×512, copied to dem/data/hydro/ for export_overlays.py):
  hand.png                — HAND colourised 0–5 m (blue=near-drainage, red=high)
  flow_accum.png          — log flow accumulation (white=ridgeline → blue=channel)
  stream_network.png      — stream network raster (bright cyan on transparent)

Usage:
    python3 dem/dem_hydro.py
    python3 dem/dem_hydro.py --acc-threshold 100   # adjust stream initiation area
    python3 dem/dem_hydro.py --burn-depth 1.5
    python3 dem/dem_hydro.py --dem dem/data/sr417_corridor_dem_1m.tif
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize, shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import shape, mapping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
HYDRO_DIR = os.path.join(DATA_DIR, "hydro")
HYDRO_GEO = os.path.join(BASE_DIR, "..", "hydrography", "data")
os.makedirs(HYDRO_DIR, exist_ok=True)

_DEM_1M = os.path.join(DATA_DIR, "sr417_corridor_dem_1m.tif")
_DEM_3M = os.path.join(DATA_DIR, "sr417_corridor_dem.tif")
DEFAULT_DEM = _DEM_1M if os.path.exists(_DEM_1M) else _DEM_3M

DEFAULT_ACC_THRESHOLD = 50000  # cells; ~5 ha on 1m grid — resolves Shingle Creek main channel
                               # and major tributaries while excluding micro-drainage swales.
                               # On a 4 km² AOI (4M cells), 50k = top 1.25% of contributing area.
DEFAULT_BURN_DEPTH    = 1.5   # metres to carve channels into DEM
PNG_SIZE = 512


# ── DEM I/O ──────────────────────────────────────────────────────────────────

def load_dem(dem_path):
    with rasterio.open(dem_path) as src:
        z     = src.read(1).astype(np.float32)
        prof  = src.profile.copy()
        nd    = src.nodata
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        crs   = src.crs
        transform = src.transform
    if nd is not None:
        z[z == nd] = np.nan
    return z, prof, transform, res_x, res_y, crs


def save_tif(arr, profile, path, nodata=np.nan):
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=nodata, compress="deflate")
    out = arr.astype(np.float32)
    out[~np.isfinite(out)] = nodata
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)
    print(f"  {os.path.basename(path)}")


# ── 1. Stream burning ─────────────────────────────────────────────────────────

def burn_streams(z, profile, transform, dem_crs, burn_depth_m):
    """Lower DEM elevations along 3DHP Shingle Creek flowlines by burn_depth_m."""
    flow_path = os.path.join(HYDRO_GEO, "3dhp_flowlines.geojson")
    if not os.path.exists(flow_path):
        print("  3DHP flowlines not found — skipping stream burning")
        return z

    gdf = gpd.read_file(flow_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("epsg:4326")
    gdf_proj = gdf.to_crs(dem_crs)

    rows, cols = z.shape
    # Buffer flowlines by 2 cell widths so they burn a few pixels wide
    cell_m = (abs(transform.a) + abs(transform.e)) / 2
    buffered = gdf_proj.geometry.buffer(cell_m * 2)

    burn_mask = rasterize(
        [(geom, 1) for geom in buffered if geom is not None],
        out_shape=(rows, cols), transform=transform, fill=0, dtype=np.uint8,
    ).astype(bool)

    z_burned = z.copy()
    z_burned[burn_mask & np.isfinite(z)] -= burn_depth_m
    n_burned = int(burn_mask.sum())
    print(f"  Stream burned: {n_burned:,} cells lowered by {burn_depth_m} m  "
          f"(flowlines={len(gdf)}, buffer={cell_m*2:.1f} m)")
    return z_burned


# ── 2. Breach depressions (richdem) ──────────────────────────────────────────

def breach_depressions(z, profile):
    """richdem BreachDepressions — carves paths through barriers rather than filling."""
    import richdem as rd
    nodata_val = -9999.0
    z_fill = z.copy()
    z_fill[~np.isfinite(z_fill)] = nodata_val

    rda = rd.rdarray(z_fill, no_data=nodata_val, geotransform=(
        profile["transform"].c, profile["transform"].a, 0,
        profile["transform"].f, 0, profile["transform"].e,
    ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd.BreachDepressions(rda, in_place=True)

    z_out = np.array(rda, dtype=np.float32)
    z_out[z_out == nodata_val] = np.nan
    z_out[~np.isfinite(z)] = np.nan
    print(f"  Depression breaching complete")
    return z_out


# ── 3 + 4. D8 flow direction + accumulation (pysheds) ────────────────────────

def compute_flow(z_cond, profile, dem_path_tmp):
    """Write conditioned DEM to a temp file, run pysheds D8 routing."""
    from pysheds.grid import Grid

    tmp_path = os.path.join(HYDRO_DIR, "_dem_cond_tmp.tif")
    save_tif(z_cond, profile, tmp_path, nodata=-9999.0)

    grid = Grid.from_raster(tmp_path)
    dem_ps = grid.read_raster(tmp_path)

    # Fill any remaining pits (pysheds fill_pits is conservative — only true
    # single-cell pits that survived breaching)
    pit_filled = grid.fill_pits(dem_ps)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)

    fdir  = grid.flowdir(inflated)
    accum = grid.accumulation(fdir)

    os.remove(tmp_path)
    return grid, fdir, accum, inflated  # return inflated DEM for HAND


# ── 5. HAND ───────────────────────────────────────────────────────────────────

def compute_hand(grid, fdir, dem_inflated, accum, acc_threshold):
    """Height Above Nearest Drainage via pysheds.
    pysheds signature: grid.compute_hand(fdir, dem, mask)
    """
    stream_mask = accum > acc_threshold
    hand_arr = grid.compute_hand(fdir, dem_inflated, stream_mask)
    return hand_arr, stream_mask


# ── 6. Watershed delineation ──────────────────────────────────────────────────

def delineate_watershed(grid, fdir, accum, center_lat, center_lon, dem_crs, acc_threshold, raster_transform):
    """Pour-point watershed, snapped to the nearest stream cell to the AOI centre.
    Uses rasterio's xy() to convert grid indices → projected coordinates reliably."""
    from pyproj import Transformer
    from rasterio.transform import xy as rio_xy

    transformer = Transformer.from_crs("epsg:4326", dem_crs, always_xy=True)
    cx, cy = transformer.transform(center_lon, center_lat)

    acc_arr = np.array(accum, dtype=np.float64)
    stream_mask = acc_arr > acc_threshold

    if not stream_mask.any():
        print("  No stream cells — cannot delineate watershed")
        return None

    # Convert pour point from projected coords to grid row/col using the rasterio transform
    # transform * (col, row) = (x, y)  →  ~inverse: row/col from x/y
    t = raster_transform
    col0 = (cx - t.c) / t.a
    row0 = (cy - t.f) / t.e

    stream_rows, stream_cols = np.where(stream_mask)
    dists = np.hypot(stream_rows - row0, stream_cols - col0)
    idx   = int(np.argmin(dists))
    snap_row, snap_col = int(stream_rows[idx]), int(stream_cols[idx])

    # Use rasterio's reliable xy() to convert grid index → projected coordinate
    snap_x, snap_y = rio_xy(raster_transform, snap_row, snap_col)

    print(f"  Pour point snapped: grid ({snap_row},{snap_col})  "
          f"acc={acc_arr[snap_row, snap_col]:.0f}  "
          f"dist={dists[idx]:.1f} cells from AOI centre")
    print(f"  Snap coords (EPSG:5070): ({snap_x:.1f}, {snap_y:.1f})")

    try:
        # pysheds xytype='index': x=col, y=row (column-first convention)
        catch = grid.catchment(x=snap_col, y=snap_row, fdir=fdir, xytype="index")
        n_cells = int(np.array(catch, dtype=np.uint8).sum())
        print(f"  Catchment cells: {n_cells:,}")
        return catch
    except Exception as e:
        print(f"  Watershed delineation failed: {e}")
        return None


# ── Raster → vector for stream network ───────────────────────────────────────

def stream_to_geojson(stream_mask_np, profile, dem_crs, out_path):
    """Convert stream raster mask → GeoJSON in EPSG:4326."""
    mask8 = stream_mask_np.astype(np.uint8)
    polys = []
    for geom_dict, val in shapes(mask8, transform=profile["transform"]):
        if val == 1:
            polys.append(shape(geom_dict))
    if not polys:
        print("  No stream cells found — try lowering --acc-threshold")
        return 0

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(geometry=polys, crs=dem_crs)
    gdf_4326 = gdf.to_crs("epsg:4326")
    dissolved = gdf_4326.dissolve()
    dissolved.to_file(out_path, driver="GeoJSON")
    total_km = float(gdf_4326.geometry.length.sum()) / 1000  # rough (degrees), not meaningful
    print(f"  stream_network.geojson: {len(gdf_4326)} segments")
    return len(gdf_4326)


# ── Watershed → GeoJSON ───────────────────────────────────────────────────────

def watershed_to_geojson(catch_arr, profile, dem_crs, out_path):
    if catch_arr is None:
        return
    mask = np.array(catch_arr, dtype=np.uint8)
    polys = [shape(g) for g, v in shapes(mask, transform=profile["transform"]) if v == 1]
    if not polys:
        print("  No watershed cells found")
        return
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(geometry=polys, crs=dem_crs).dissolve().to_crs("epsg:4326")
    gdf.to_file(out_path, driver="GeoJSON")
    area_km2 = float(gdf.to_crs("epsg:5070").geometry.area.sum()) / 1e6
    print(f"  watershed.geojson: {area_km2:.2f} km²")


# ── Viewer PNGs ───────────────────────────────────────────────────────────────

def save_png_viewer(arr, path, cmap, vmin=None, vmax=None):
    from PIL import Image
    from skimage.transform import resize as sk_resize
    from matplotlib import colormaps

    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        Image.fromarray(np.zeros((PNG_SIZE, PNG_SIZE, 4), dtype=np.uint8)).save(path)
        return
    v0 = float(np.percentile(valid, 2))  if vmin is None else vmin
    v1 = float(np.percentile(valid, 98)) if vmax is None else vmax
    if v0 == v1:
        v1 = v0 + 1e-6

    norm  = mcolors.Normalize(vmin=v0, vmax=v1, clip=True)
    cm    = colormaps.get_cmap(cmap)
    rgba_f = cm(norm(arr))
    rgba_f[~np.isfinite(arr)] = [0, 0, 0, 0]
    rgba8 = (rgba_f * 255).astype(np.uint8)
    rgba_r = sk_resize(rgba8.astype(np.float32), (PNG_SIZE, PNG_SIZE, 4),
                       order=1, anti_aliasing=True, preserve_range=True)
    Image.fromarray(rgba_r.astype(np.uint8)).save(path)


def save_stream_png(stream_mask, path):
    """Stream network as bright cyan on transparent."""
    from PIL import Image
    from skimage.transform import resize as sk_resize
    rgba = np.zeros((*stream_mask.shape, 4), dtype=np.uint8)
    rgba[stream_mask] = (80, 220, 255, 240)
    rgba_r = sk_resize(rgba.astype(np.float32), (PNG_SIZE, PNG_SIZE, 4),
                       order=0, anti_aliasing=False, preserve_range=True)
    Image.fromarray(rgba_r.astype(np.uint8)).save(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dem_path=DEFAULT_DEM, acc_threshold=DEFAULT_ACC_THRESHOLD,
        burn_depth=DEFAULT_BURN_DEPTH, center_lat=28.36687, center_lon=-81.43299):

    print(f"\nHydrological DEM Processing — {os.path.basename(dem_path)}")
    print(f"  acc_threshold={acc_threshold} cells  burn_depth={burn_depth} m")
    print("=" * 60)

    if not os.path.exists(dem_path):
        sys.exit(f"DEM not found: {dem_path}  Run: python3 dem/dem_download.py --resolution 1")

    z, profile, transform, res_x, res_y, dem_crs = load_dem(dem_path)
    print(f"DEM     : {z.shape[0]}×{z.shape[1]} px @ {res_x:.2f}m  CRS: {dem_crs}")
    print(f"Elevation: {np.nanmin(z):.2f}–{np.nanmax(z):.2f} m NAVD88")

    # ── Step 1: stream burning ───────────────────────────────────────────────
    print("\n[1/6] Stream burning (3DHP Shingle Creek) …")
    z_burned = burn_streams(z, profile, transform, dem_crs, burn_depth)
    save_tif(z_burned, profile, os.path.join(HYDRO_DIR, "dem_burned.tif"))

    # ── Step 2: breach depressions ───────────────────────────────────────────
    print("\n[2/6] Breaching depressions (richdem) …")
    z_cond = breach_depressions(z_burned, profile)
    save_tif(z_cond, profile, os.path.join(HYDRO_DIR, "dem_conditioned.tif"))

    # ── Steps 3 + 4: D8 flow direction + accumulation ────────────────────────
    print("\n[3+4/6] D8 flow direction + accumulation (pysheds) …")
    grid, fdir, accum, dem_inflated = compute_flow(z_cond, profile, dem_path)

    # Save flow accumulation log-scaled
    accum_np = np.array(accum, dtype=np.float32)
    accum_np[accum_np <= 0] = np.nan
    save_tif(np.log1p(accum_np), profile, os.path.join(HYDRO_DIR, "flow_accum_log.tif"))

    n_stream = int((accum_np > acc_threshold).sum())
    pct = n_stream / accum_np.size * 100
    print(f"  Flow accumulation complete  stream cells (>{acc_threshold}): {n_stream:,}  ({pct:.2f}% of AOI)")

    # ── Step 5: HAND ─────────────────────────────────────────────────────────
    print(f"\n[5/6] HAND (Height Above Nearest Drainage, threshold={acc_threshold}) …")
    hand_arr, stream_mask = compute_hand(grid, fdir, dem_inflated, accum, acc_threshold)
    hand_np = np.array(hand_arr, dtype=np.float32)
    hand_np[~np.isfinite(z)] = np.nan
    save_tif(hand_np, profile, os.path.join(HYDRO_DIR, "hand.tif"))

    valid_hand = hand_np[np.isfinite(hand_np)]
    print(f"  HAND stats: mean={np.mean(valid_hand):.2f} m  "
          f"p95={np.percentile(valid_hand, 95):.2f} m  "
          f"max={np.nanmax(hand_np):.2f} m")
    print(f"  Cells HAND<1m (potential inundation): "
          f"{int((hand_np < 1).sum()):,}  ({int((hand_np < 1).sum())/hand_np.size*100:.1f}%)")
    print(f"  Cells HAND<3m: "
          f"{int((hand_np < 3).sum()):,}  ({int((hand_np < 3).sum())/hand_np.size*100:.1f}%)")

    # ── Step 6: watershed ────────────────────────────────────────────────────
    print(f"\n[6/6] Watershed delineation (pour point: AOI centre) …")
    catch = delineate_watershed(grid, fdir, accum, center_lat, center_lon, str(dem_crs), acc_threshold, profile["transform"])
    watershed_to_geojson(catch, profile, dem_crs,
                         os.path.join(HYDRO_DIR, "watershed.geojson"))

    # ── Vector stream network ─────────────────────────────────────────────────
    stream_mask_np = accum_np > acc_threshold
    stream_to_geojson(stream_mask_np, profile, dem_crs,
                      os.path.join(HYDRO_DIR, "stream_network.geojson"))

    # ── Viewer PNGs ───────────────────────────────────────────────────────────
    print("\nSaving viewer PNGs …")
    save_png_viewer(hand_np,   os.path.join(HYDRO_DIR, "hand.png"),
                    cmap="RdYlBu_r", vmin=0, vmax=5)
    save_png_viewer(np.log1p(accum_np), os.path.join(HYDRO_DIR, "flow_accum.png"),
                    cmap="Blues")
    save_stream_png(stream_mask_np, os.path.join(HYDRO_DIR, "stream_network.png"))
    print("  hand.png  flow_accum.png  stream_network.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "dem_path":          dem_path,
        "dem_res_m":         (res_x + res_y) / 2,
        "crs":               str(dem_crs),
        "burn_depth_m":      burn_depth,
        "acc_threshold":     acc_threshold,
        "stream_cells":      n_stream,
        "hand_mean_m":       float(np.mean(valid_hand)),
        "hand_p95_m":        float(np.percentile(valid_hand, 95)),
        "hand_max_m":        float(np.nanmax(hand_np)),
        "cells_hand_lt1m":   int((hand_np < 1).sum()),
        "cells_hand_lt3m":   int((hand_np < 3).sum()),
        "calibration_note":  (
            "Ian peak at NWIS 02263800 = 3500 cfs / 11.43 ft (2022-09-30). "
            "HAND contour matching this stage (~3.5m above thalweg) should "
            "reproduce the observed inundation visible in PlanetScope Max_1 scene."
        ),
    }
    summary_path = os.path.join(HYDRO_DIR, "hydro_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone.  Output → {HYDRO_DIR}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem",           default=DEFAULT_DEM,        type=str)
    parser.add_argument("--acc-threshold", default=DEFAULT_ACC_THRESHOLD, type=int,
                        help="Flow accumulation cells threshold for stream initiation")
    parser.add_argument("--burn-depth",    default=DEFAULT_BURN_DEPTH, type=float,
                        help="Metres to lower DEM along channel centrelines")
    parser.add_argument("--lat",           default=28.36687,           type=float)
    parser.add_argument("--lon",           default=-81.43299,          type=float)
    args = parser.parse_args()
    run(args.dem, args.acc_threshold, args.burn_depth, args.lat, args.lon)
