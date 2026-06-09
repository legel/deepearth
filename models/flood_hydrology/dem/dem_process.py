"""
DEM Processing — Hydrologic Derivatives & Lake Bed Estimation
=============================================================
Processes the downloaded USGS 3DEP DEM to extract all terrain inputs
needed by the flood simulation engine.

Processing steps
----------------
1. Fill pits / depressions         (removes spurious sinks in lidar)
2. Resolve flats                   (ensures flow direction through flat areas)
3. D8 flow direction               (eight-direction steepest-descent)
4. Flow accumulation               (upstream contributing cells)
5. HAND — Height Above Nearest Drainage
   (proxy flood depth for quick inundation extent estimation)
6. Stream network extraction       (accumulation threshold → stream pixels)
7. Watershed delineation           (contributing area to property pour point)
8. Lake identification             (hydro-flattened flat regions = lakes)
9. Lake bed DEM estimation         (linear slope extrapolation inward)

Outputs (saved under dem/data/):
    flow_dir.tif         — D8 flow direction (pysheds integer encoding)
    flow_acc.tif         — flow accumulation [cell count]
    hand.tif             — HAND [meters above nearest drainage]
    streams.tif          — binary stream mask (accumulation ≥ threshold)
    streams.geojson      — vectorized stream network
    watershed.tif        — binary watershed mask (draining to property)
    watershed.geojson    — watershed polygon
    lake_mask.tif        — binary lake mask (hydro-flattened flat regions)
    lake_mask.geojson    — lake polygons with water-surface elevation
    lake_bed_dem.tif     — estimated lake bed elevation (extrapolated)

Usage:
    python3 dem/dem_process.py
    python3 dem/dem_process.py --dem dem/data/winter_garden_dem.tif
"""

import os
import sys
import json
import argparse
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_DEM        = os.path.join(DATA_DIR, "winter_garden_dem.tif")
ACC_THRESHOLD      = 300   # cells; stream if upstream area ≥ this
FLAT_TOL_M         = 0.30  # elevation std threshold to classify a region as "lake" (hydro-flat)
LAKE_MIN_CELLS     = 2000  # minimum cell count for a lake polygon (~1.5 ha at 2.6m cell)
PROPERTY_LAT       = 28.521592   # 17801 Champagne Dr (28°31'17.73"N 81°39'25.13"W)
PROPERTY_LON       = -81.656981
S2_DATA_DIR        = os.path.join(BASE_DIR, "..", "sentinel2", "data")
S2_DATA_DIR        = os.path.normpath(S2_DATA_DIR)
# Lake surface elevation band — cells within this range + confirmed by S2 are true lake
LAKE_ELEV_MAX_M    = 32.0  # exclude upland cells (roads, rooftops) above this elevation


def build_s2_lake_mask(dem_arr, dem_transform, dem_crs, dem_shape):
    """
    Build a clean lake mask using Sentinel-2 persistent water + DEM elevation filter.

    Strategy:
      1. Load all 11 S2 water_mask_*.tif scenes, reproject to DEM grid.
      2. Keep cells classified as water in >=7/11 scenes (persistent water).
      3. Restrict to cells with DEM elevation < LAKE_ELEV_MAX_M to exclude
         suburban flat surfaces (roads, rooftops) that D8 confuses with lakes.
      4. The result is a clean binary mask of the true lake surface.

    Returns (lake_mask_bool, water_surface_elev) or (None, None) if S2 data missing.
    """
    import glob
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
    except ImportError:
        return None, None

    s2_files = sorted(glob.glob(os.path.join(S2_DATA_DIR, "water_mask_*.tif")))
    if not s2_files:
        print("  ⚠ No Sentinel-2 water masks found — skipping lake pre-conditioning")
        return None, None

    print(f"  Loading {len(s2_files)} Sentinel-2 water masks for lake pre-conditioning …")
    stack = np.zeros((len(s2_files), *dem_shape), dtype=np.uint8)
    for i, fpath in enumerate(s2_files):
        try:
            with rasterio.open(fpath) as src:
                out = np.zeros(dem_shape, dtype=np.float32)
                reproject(source=src.read(1).astype(np.float32), destination=out,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dem_transform, dst_crs=dem_crs,
                          resampling=Resampling.nearest)
            stack[i] = (out > 0.5).astype(np.uint8)
        except Exception:
            pass

    freq = stack.sum(axis=0)
    # >=7/11 = persistent open water; <LAKE_ELEV_MAX_M removes upland flat surfaces
    lake_bool = (freq >= 7) & (dem_arr < LAKE_ELEV_MAX_M) & np.isfinite(dem_arr)
    n = int(lake_bool.sum())
    if n < 1000:
        print(f"  ⚠ Only {n} lake cells after S2+elevation filter — skipping pre-conditioning")
        return None, None

    water_surface_elev = float(np.nanpercentile(dem_arr[lake_bool], 10))
    print(f"  ✓ S2 lake mask: {n:,} cells, water surface (10th-pct) = {water_surface_elev:.2f} m")
    return lake_bool, water_surface_elev


def precondition_lake_surface(dem_arr, dem_transform, dem_crs, dem_shape):
    """
    Pre-flatten the true lake surface to a single elevation before D8 routing.

    D8 on a hydro-flattened DEM with a large flat lake creates artificial diagonal
    channels through the lake (resolve_flats artifact). Setting the lake to one
    uniform elevation makes resolve_flats route all flow cleanly toward the outlet.

    Returns the pre-conditioned DEM array (same shape/dtype as input).
    Saves dem/data/lake_mask_s2.tif as the authoritative lake mask.
    """
    import rasterio

    lake_bool, water_surface_elev = build_s2_lake_mask(dem_arr, dem_transform, dem_crs, dem_shape)
    if lake_bool is None:
        print("  Using original DEM (no S2-based pre-conditioning)")
        return dem_arr.copy()

    dem_precond = dem_arr.copy()
    dem_precond[lake_bool] = water_surface_elev
    print(f"  Pre-conditioned {int(lake_bool.sum()):,} lake cells → {water_surface_elev:.2f} m")

    # Save the authoritative lake mask derived from S2 + elevation filter
    lake_mask_s2_path = os.path.join(DATA_DIR, "lake_mask_s2.tif")
    with rasterio.open(DEFAULT_DEM) as ref:
        prof = ref.profile.copy()
    prof.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(lake_mask_s2_path, "w", **prof) as dst:
        dst.write(lake_bool.astype(np.uint8), 1)
    print(f"  Saved authoritative lake mask → lake_mask_s2.tif")

    return dem_precond


def load_dem(dem_path):
    from pysheds.grid import Grid
    if not os.path.exists(dem_path):
        sys.exit(f"DEM not found: {dem_path}\n  Run dem_download.py first.")
    print(f"Loading DEM : {dem_path}")
    grid = Grid.from_raster(dem_path)
    dem  = grid.read_raster(dem_path)
    print(f"  Shape     : {dem.shape}  ({dem.shape[0] * dem.shape[1]:,} cells)")
    print(f"  CRS       : {dem.crs}")
    print(f"  Elevation : {float(dem.min()):.1f} – {float(dem.max()):.1f} m")
    return grid, dem


def condition_dem(grid, dem):
    """Fill pits and resolve flats."""
    print("Conditioning DEM …")
    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)
    print("  ✓ Pits filled, depressions removed, flats resolved")
    return inflated


def compute_flow(grid, conditioned):
    """D8 flow direction and flow accumulation."""
    print("Computing flow direction and accumulation …")
    fdir = grid.flowdir(conditioned)
    acc  = grid.accumulation(fdir)
    print(f"  ✓ Max accumulation: {int(acc.max()):,} cells")
    return fdir, acc


def compute_hand(grid, dem_conditioned, fdir, acc, threshold):
    """Height Above Nearest Drainage — flood zone proxy."""
    print(f"Computing HAND (stream threshold: {threshold} cells) …")
    streams = acc > threshold
    hand    = grid.compute_hand(fdir, dem_conditioned, streams)
    print(f"  ✓ HAND range: 0 – {float(hand.max()):.1f} m")
    print(f"  Cells within 1m of drainage: {int((hand < 1).sum()):,}  "
          f"({100*(hand<1).mean():.1f}% of domain)")
    return hand, streams.astype(np.uint8)


def delineate_watershed(grid, fdir, lat, lon):
    """Delineate watershed draining to the property location."""
    import pyproj
    from shapely.geometry import Point

    print(f"Delineating watershed for ({lat}, {lon}) …")

    # Convert property lat/lon to raster CRS pixel
    crs_str = str(grid.crs)
    if "4326" in crs_str or "WGS" in crs_str.upper():
        # Already geographic; snap to nearest grid cell
        x, y = lon, lat
    else:
        transformer = pyproj.Transformer.from_crs("epsg:4326", grid.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

    try:
        catch = grid.catchment(x=x, y=y, fdir=fdir, xytype="coordinate")
        n_cells = int(catch.sum())
        cell_size_m = abs(grid.affine[0])
        area_km2 = n_cells * (cell_size_m ** 2) / 1e6
        print(f"  ✓ Watershed: {n_cells:,} cells, ~{area_km2:.2f} km²")
        return catch.astype(np.uint8)
    except Exception as exc:
        print(f"  ⚠ Watershed delineation failed: {exc}; skipping.")
        return None


def identify_lakes(grid, dem, flat_tol=FLAT_TOL_M, min_cells=LAKE_MIN_CELLS):
    """
    Identify hydro-flattened lake regions from the lidar DEM.

    Lakes in a hydro-flattened DEM are flat AND topographically low relative
    to their local neighborhood. This two-criterion filter avoids false
    positives from roads, parking lots, and residential flat areas:
      1. Local elevation std dev < flat_tol (flat surface)
      2. Local elevation < neighborhood 20th percentile (topographically low)
      3. Connected region size >= min_cells
    """
    from scipy import ndimage
    from scipy.ndimage import percentile_filter

    print("Identifying lake regions (hydro-flattened areas) …")
    arr = np.array(dem, dtype=np.float32)
    nodata = dem.nodata if hasattr(dem, 'nodata') else -9999.0
    valid = np.isfinite(arr) & (arr != nodata)

    # Criterion 1: Local std dev < flat_tol in a 5×5 window
    arr_filled = np.where(valid, arr, float(np.nanmean(arr)))
    local_mean = ndimage.uniform_filter(arr_filled, size=5)
    local_sq   = ndimage.uniform_filter(arr_filled**2, size=5)
    local_std  = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
    is_flat = local_std < flat_tol

    # Criterion 2: Cell elevation is at or below local 20th percentile
    # over a 51×51 cell neighborhood (~130m at 2.6m) — lakes are valley-bottom features
    neighborhood_low = percentile_filter(arr_filled, percentile=20, size=51)
    is_low = arr_filled <= (neighborhood_low + flat_tol)

    lake_mask = is_flat & is_low & valid

    # Remove small regions
    labeled, n_features = ndimage.label(lake_mask)
    if n_features == 0:
        print("  ✓ No lake regions detected; lake model disabled")
        return lake_mask.astype(np.uint8), labeled.astype(np.int32), np.array([], dtype=int)

    sizes = ndimage.sum(lake_mask, labeled, range(1, n_features + 1))
    large = np.where(np.array(sizes) >= min_cells)[0] + 1

    # Safety cap: discard any lake that exceeds 30% of domain (likely misclassification)
    domain_size = valid.sum()
    large = np.array([lid for lid in large
                      if ndimage.sum(lake_mask, labeled, lid) < 0.30 * domain_size],
                     dtype=int)

    lake_mask = np.isin(labeled, large)
    n_lakes = len(large)
    lake_cells = int(lake_mask.sum())
    cell_m = abs(grid.affine[0])
    lake_area_ha = lake_cells * cell_m**2 / 1e4
    print(f"  ✓ Found {n_lakes} lake region(s), {lake_area_ha:.1f} ha total")
    return lake_mask.astype(np.uint8), labeled.astype(np.int32), large


def estimate_lake_bed(dem, lake_mask_arr, labeled, large, cell_m, buffer_cells=5):
    """
    Estimate lake bed elevation by extrapolating surrounding terrain slope inward.

    Method
    ------
    For each lake polygon:
      1. Sample terrain elevation in a buffer zone just outside the shoreline.
      2. Fit a linear trend (plane) through these shoreline samples.
      3. Project the plane inward across the lake polygon.
      4. Clip to ensure lake bed ≤ water surface elevation.

    This approximates lake bathymetry where no survey exists.
    """
    from scipy import ndimage
    from scipy.stats import linregress

    print("Estimating lake bed elevations …")
    arr = np.array(dem, dtype=np.float32)
    lake_bed = arr.copy()  # start with water-surface (flat) elevation

    rows, cols = np.indices(arr.shape)

    for lake_id in large:
        region = (labeled == lake_id)
        water_surface_elev = float(arr[region].mean())

        # Dilate mask to get buffer ring around lake
        dilated = ndimage.binary_dilation(region, iterations=buffer_cells)
        ring = dilated & ~region

        if ring.sum() < 10:
            continue  # too small, skip

        r_ring = rows[ring]
        c_ring = cols[ring]
        z_ring = arr[ring]

        # Fit plane: z = a*r + b*c + intercept using least squares
        A = np.column_stack([r_ring, c_ring, np.ones(len(r_ring))])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_ring, rcond=None)
        a, b, intercept = coeffs

        # Project plane into lake interior
        r_lake = rows[region]
        c_lake = cols[region]
        z_extrapolated = a * r_lake + b * c_lake + intercept

        # Lake bed must be ≤ water surface and ≥ some reasonable minimum
        # Assume max lake depth ≤ 10m for Florida lakes (shallow glacial absence)
        z_bed = np.clip(z_extrapolated, water_surface_elev - 10.0, water_surface_elev - 0.1)
        lake_bed[region] = z_bed

        depth_mean = float((water_surface_elev - z_bed).mean())
        depth_max  = float((water_surface_elev - z_bed).max())
        print(f"  Lake {lake_id}: water surface {water_surface_elev:.1f}m, "
              f"est. depth {depth_mean:.1f}m avg / {depth_max:.1f}m max")

    return lake_bed


def merge_dem_bathymetry(dem_path, lake_mask_path=None, out_path=None):
    """
    Merge terrain DEM with lake bathymetry into a single continuous raster.

    Standard approach (USACE / NHD Plus):
      - Above lake surface  → use lidar DEM elevation (terrain)
      - Below lake surface  → substitute bathymetric survey depth converted to
                              elevation NAVD88:  bed_elev = survey depth file values

    Priority for bathymetry source:
      lake_bed_dem_fwc.tif      (FWC survey — highest priority)
      lake_bed_dem_survey.tif   (SJRWMD/FDEP survey)
      lake_bed_dem_estimated.tif (shoreline-slope estimate — fallback, warns user)

    The output winter_garden_dem_merged.tif can be fed directly into flood_sim.py.
    """
    import rasterio
    import warnings

    dem_dir = os.path.dirname(dem_path)

    # Resolve bathymetry source
    bed_candidates = [
        ("lake_bed_dem_fwc.tif",       "FWC survey"),
        ("lake_bed_dem_survey.tif",     "SJRWMD/FDEP survey"),
        ("lake_bed_dem_estimated.tif",  "shoreline-slope estimate — consider getting real survey data"),
    ]
    bed_path = None
    bed_label = None
    for fname, label in bed_candidates:
        candidate = os.path.join(dem_dir, fname)
        if os.path.exists(candidate):
            bed_path = candidate
            bed_label = label
            break

    if bed_path is None:
        print("  merge_dem_bathymetry: no lake bed file found — returning unmodified DEM")
        return dem_path

    if "estimate" in bed_label:
        warnings.warn(
            "merge_dem_bathymetry: using ESTIMATED lake bed (shoreline-slope extrapolation). "
            "Run dem/fetch_sjrwmd_bathymetry.py to get real survey data.",
            UserWarning, stacklevel=2)
    print(f"  Merging DEM + bathymetry [{bed_label}]")

    # Resolve lake mask
    if lake_mask_path is None:
        lake_mask_path = (os.path.join(dem_dir, "lake_mask_s2.tif")
                          if os.path.exists(os.path.join(dem_dir, "lake_mask_s2.tif"))
                          else os.path.join(dem_dir, "lake_mask.tif"))

    if out_path is None:
        out_path = os.path.join(dem_dir, "winter_garden_dem_merged.tif")

    with rasterio.open(dem_path) as src:
        dem_arr  = src.read(1).astype(np.float32)
        dem_prof = src.profile.copy()
        nd = src.nodata or -9999.0

    with rasterio.open(bed_path) as src:
        bed_arr = src.read(1).astype(np.float32)
        bed_nd  = src.nodata
        if bed_nd is not None:
            bed_arr[bed_arr == bed_nd] = np.nan

    with rasterio.open(lake_mask_path) as src:
        lake_mask_arr = src.read(1).astype(np.uint8)
        if lake_mask_arr.shape != dem_arr.shape:
            from scipy.ndimage import zoom as _zoom
            lake_mask_arr = (_zoom(lake_mask_arr.astype(np.float32),
                                   (dem_arr.shape[0] / lake_mask_arr.shape[0],
                                    dem_arr.shape[1] / lake_mask_arr.shape[1]),
                                   order=0) > 0.5).astype(np.uint8)

    lake_bool = (lake_mask_arr == 1) & np.isfinite(bed_arr)

    # Merge: DEM everywhere, bed elevation inside lake
    merged = dem_arr.copy()
    merged[lake_bool] = bed_arr[lake_bool]
    # Ensure no lake cell is above the water surface (sanity check)
    water_surface = float(np.nanmean(dem_arr[lake_bool])) if lake_bool.any() else 0.0
    merged[lake_bool] = np.minimum(merged[lake_bool], water_surface - 0.05)

    dem_prof.update(dtype="float32", nodata=nd, compress="lzw")
    with rasterio.open(out_path, "w", **dem_prof) as dst:
        dst.write(merged.astype(np.float32), 1)

    n_merged = int(lake_bool.sum())
    print(f"  Merged {n_merged:,} lake cells | Saved → {out_path}")
    return out_path


def save_raster(grid, array, out_path, dtype=None):
    """Write a numpy array as a GeoTIFF using the grid's affine transform."""
    import rasterio
    from rasterio.transform import from_bounds

    h, w = array.shape
    transform = grid.affine
    crs_str = str(grid.crs)

    if dtype is None:
        dtype = array.dtype

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype=dtype,
        crs=crs_str,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(array.astype(dtype), 1)


def raster_to_geojson(array, grid, out_path, value=1, simplify_tol=0.00005):
    """Vectorize a binary raster mask to GeoJSON polygons."""
    import rasterio.features
    import json

    transform = grid.affine
    crs_str   = str(grid.crs)
    shapes = list(rasterio.features.shapes(
        array.astype(np.uint8), mask=(array > 0).astype(np.uint8),
        transform=transform))

    features = [{"type": "Feature",
                 "geometry": geom,
                 "properties": {"value": val}}
                for geom, val in shapes if val == value]

    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs_str}},
          "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f)
    print(f"  Saved {len(features)} feature(s) → {out_path}")


def main(dem_path=DEFAULT_DEM, acc_threshold=ACC_THRESHOLD):
    import rasterio

    grid, dem = load_dem(dem_path)
    cell_m = abs(grid.affine[0])

    # Build the S2-based authoritative lake mask for post-processing and display.
    # D8 routing runs on the original hydro-flattened DEM (no modification) to avoid
    # breaking flow paths. After routing, lake cells (which ARE the drainage) have
    # flow_acc set to NaN and HAND set to 0 — removing D8 streak artifacts inside
    # the lake body that result from resolve_flats on the large flat water surface.
    print("\nBuilding S2-based lake mask …")
    dem_arr = np.array(dem, dtype=np.float32)
    with rasterio.open(dem_path) as ref:
        dem_transform = ref.transform
        dem_crs = ref.crs
    lake_s2_bool, water_surface_elev = build_s2_lake_mask(dem_arr, dem_transform, dem_crs, dem.shape)
    if lake_s2_bool is not None:
        lake_mask_s2_path = os.path.join(DATA_DIR, "lake_mask_s2.tif")
        with rasterio.open(dem_path) as ref:
            prof = ref.profile.copy()
        prof.update(dtype="uint8", count=1, nodata=0)
        with rasterio.open(lake_mask_s2_path, "w", **prof) as dst:
            dst.write(lake_s2_bool.astype(np.uint8), 1)
        print(f"  Saved → lake_mask_s2.tif ({int(lake_s2_bool.sum()):,} cells at {water_surface_elev:.2f} m)")

    conditioned = condition_dem(grid, dem)
    fdir, acc   = compute_flow(grid, conditioned)
    hand, streams = compute_hand(grid, conditioned, fdir, acc, acc_threshold)

    watershed = delineate_watershed(grid, fdir, PROPERTY_LAT, PROPERTY_LON)

    lake_mask, labeled, large = identify_lakes(grid, dem)

    if len(large) > 0:
        lake_bed_arr = estimate_lake_bed(dem, lake_mask, labeled, large, cell_m)
    else:
        lake_bed_arr = np.array(dem, dtype=np.float32)

    # Post-process flow_acc and HAND:
    # 1. Lake cells (S2 mask): flow_acc → NaN (lake is the receiving body, not contributing
    #    area); HAND → 0 (lake surface IS the drainage, by definition).
    # 2. HAND NaN cells from pysheds (disconnected boundary cells): fill by nearest neighbor
    #    so no NaN holes appear in the display — boundary cells are masked anyway.
    from scipy import ndimage as _ndi
    acc_arr  = np.array(acc,  dtype=np.float32)
    hand_arr = np.array(hand, dtype=np.float32)
    if lake_s2_bool is not None:
        acc_arr[lake_s2_bool]  = np.nan
        hand_arr[lake_s2_bool] = 0.0
        print(f"  Applied lake mask: flow_acc NaN, HAND=0 for {int(lake_s2_bool.sum()):,} lake cells")
    # Fill remaining NaN in HAND (pysheds boundary disconnects) with nearest valid value
    _nan_mask = np.isnan(hand_arr)
    if _nan_mask.sum() > 0:
        _ind = _ndi.distance_transform_edt(_nan_mask, return_distances=False, return_indices=True)
        hand_arr[_nan_mask] = hand_arr[tuple(_ind[:, _nan_mask])]
        print(f"  Filled {int(_nan_mask.sum()):,} HAND NaN cells with nearest-neighbor")

    # Save all derivatives
    print("\nSaving outputs …")
    save_raster(grid, np.array(fdir, dtype=np.int16),  os.path.join(DATA_DIR, "flow_dir.tif"),               dtype=np.int16)
    save_raster(grid, acc_arr,                          os.path.join(DATA_DIR, "flow_acc.tif"),               dtype=np.float32)
    save_raster(grid, hand_arr,                         os.path.join(DATA_DIR, "hand.tif"),                   dtype=np.float32)
    save_raster(grid, streams,                          os.path.join(DATA_DIR, "streams.tif"),                dtype=np.uint8)
    save_raster(grid, lake_mask,                        os.path.join(DATA_DIR, "lake_mask.tif"),              dtype=np.uint8)
    save_raster(grid, lake_bed_arr,                     os.path.join(DATA_DIR, "lake_bed_dem_estimated.tif"), dtype=np.float32)
    if watershed is not None:
        save_raster(grid, watershed, os.path.join(DATA_DIR, "watershed.tif"), dtype=np.uint8)
        raster_to_geojson(watershed, grid, os.path.join(DATA_DIR, "watershed.geojson"))

    raster_to_geojson(streams,  grid, os.path.join(DATA_DIR, "streams.geojson"))
    raster_to_geojson(lake_mask, grid, os.path.join(DATA_DIR, "lake_mask.geojson"))

    # Build merged DEM (terrain + bathymetry) for use by flood_sim.py
    print("\nBuilding merged DEM + bathymetry raster …")
    merge_dem_bathymetry(
        dem_path=DEFAULT_DEM,
        lake_mask_path=os.path.join(DATA_DIR, "lake_mask_s2.tif")
        if lake_s2_bool is not None else None,
    )

    print("\n── DEM Processing Complete ──────────────────────────────────")
    print(f"  Cell size          : {cell_m:.1f} m")
    print(f"  Stream threshold   : {acc_threshold} cells ({acc_threshold*cell_m**2/1e4:.2f} ha)")
    print(f"  Max HAND (land)    : {float(np.nanmax(hand_arr)):.1f} m")
    print(f"  Lake mask (S2)     : {int(lake_s2_bool.sum()) if lake_s2_bool is not None else 0:,} cells")
    print(f"  Output directory   : {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DEM into hydrologic derivatives")
    parser.add_argument("--dem",           type=str, default=DEFAULT_DEM,  help="Input DEM GeoTIFF path")
    parser.add_argument("--acc_threshold", type=int, default=ACC_THRESHOLD, help="Flow accumulation threshold for streams")
    args = parser.parse_args()
    main(args.dem, args.acc_threshold)
