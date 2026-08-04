"""
Raw LiDAR Point Cloud Ingestion + Bridge-Crossing Validation — CFX SR417 Corridor
==================================================================================
Reads the raw USGS 3DEP LiDAR point cloud (LAZ tiles, FL_Peninsular_FDEM_2018_D19_DRRA
project — same 2018 acquisition already confirmed as this project's DEM source), filters to
the AOI, and answers a specific question raised by inspecting the existing bare-earth DEM:
SR417 is an elevated, limited-access toll expressway (Central Florida GreeneWay). Sampling
the current bare-earth `dem_conditioned.tif` along its centerline shows a ~7-8m elevation
DROP down to grade level in a ~40m-wide notch at both places SR417 crosses a surface street
(Town Loop Boulevard and John Young Parkway/CR423) within the AOI, before jumping back up.
A real overpass would not do this — it's the classic signature of a bare-earth DTM where
bridge-deck LiDAR returns get classified as non-ground and stripped out, leaving only the
ground visible underneath the bridge.

This script builds, per crossing, a first-return/all-points DSM (captures whatever is
physically highest — bridge deck, vegetation, etc.) and compares it against the bare-earth
DEM to confirm/quantify the artifact, and produces a small 2.5D Delaunay TIN mesh of each
crossing's immediate neighborhood (not the whole AOI — the full-AOI ground surface already
matches the existing DEM; the point cloud's value here is specifically the two bridges).

Usage:
    python3 lidar/build_lidar_pointcloud.py
    python3 lidar/build_lidar_pointcloud.py --lat 28.36687 --lon -81.43299 --radius_km 1.0
"""
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
import laspy
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.ops import unary_union, linemerge
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM = 28.36687, -81.43299, 1.0

DEM_COND = os.path.join(PROJ_DIR, "dem", "data", "hydro", "dem_conditioned.tif")
GEO_META = os.path.join(PROJ_DIR, "viewer", "data", "geo_meta.json")

VERT_EXAG = 8   # must match viewer/static/js/terrain.js's VERT_EXAG exactly

# LAS files from this acquisition don't carry a machine-readable CRS VLR (parse_crs() -> None);
# coordinate magnitudes (~5.1e5, ~1.47e6) match Florida East state plane in US survey feet —
# the standard distribution CRS for this FDEM/3DEP project. Confirmed empirically: transforming
# the AOI center through EPSG:2881 (NAD83 / Florida East ftUS) lands at (516921, 1466334),
# squarely inside a tile's own header bounds ([510000-515000] x [1465000-1470000]); EPSG:6437
# (a different Florida zone/datum realization) does not match and was ruled out this way.
LAS_CRS  = "EPSG:2881"
DEM_CRS  = "EPSG:5070"

FT_TO_M = 0.3048006096012192   # US survey foot

# ASPRS LAS classification codes relevant here
CLASS_NAMES = {
    0: "created_never_classified", 1: "unclassified", 2: "ground",
    3: "low_vegetation", 4: "medium_vegetation", 5: "high_vegetation",
    6: "building", 7: "low_point_noise", 9: "water", 10: "rail",
    11: "road_surface", 13: "wire_guard", 14: "wire_conductor",
    17: "bridge_deck", 18: "high_noise",
}

CROSSINGS = {
    "town_loop_blvd":  {"lon": -81.4329, "lat": 28.3666,
                         "label": "SR417 x Town Loop Boulevard"},
    "john_young_pkwy": {"lon": -81.4258, "lat": 28.3727,
                         "label": "SR417 x John Young Parkway (CR423)"},
}
CROSSING_BUFFER_M = 60.0   # half-width of the local mesh/analysis window per crossing


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def load_points_in_bbox(lon_min, lat_min, lon_max, lat_max):
    """Read every LAZ tile in RAW_DIR, keep only points inside the AOI bbox, return
    a dict of numpy arrays in EPSG:5070 meters (matching the project's DEM CRS)."""
    to_las = Transformer.from_crs("EPSG:4326", LAS_CRS, always_xy=True)
    xmin_ft, ymin_ft = to_las.transform(lon_min, lat_min)
    xmax_ft, ymax_ft = to_las.transform(lon_max, lat_max)
    xmin_ft, xmax_ft = sorted([xmin_ft, xmax_ft])
    ymin_ft, ymax_ft = sorted([ymin_ft, ymax_ft])

    laz_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.laz")))
    if not laz_files:
        raise FileNotFoundError(f"No .laz files found in {RAW_DIR}")

    xs, ys, zs, cls, rn, nr = [], [], [], [], [], []
    for path in laz_files:
        print(f"  reading {os.path.basename(path)} …")
        las = laspy.read(path)
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        mask = (x >= xmin_ft) & (x <= xmax_ft) & (y >= ymin_ft) & (y <= ymax_ft)
        n = int(mask.sum())
        print(f"    {len(x):,} points in tile → {n:,} in AOI bbox")
        if n == 0:
            continue
        xs.append(x[mask]); ys.append(y[mask]); zs.append(z[mask])
        cls.append(np.asarray(las.classification)[mask])
        rn.append(np.asarray(las.return_number)[mask])
        nr.append(np.asarray(las.number_of_returns)[mask])

    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    classification = np.concatenate(cls)
    return_number  = np.concatenate(rn)
    num_returns    = np.concatenate(nr)

    # z is stored in feet (same CRS vertical units); reproject planimetric x,y (also feet)
    # directly into EPSG:5070 meters via pyproj (handles the unit conversion itself).
    z_m = z * FT_TO_M
    to_dem_ft = Transformer.from_crs(LAS_CRS, DEM_CRS, always_xy=True)
    X, Y = to_dem_ft.transform(x, y)

    return {"x": X, "y": Y, "z": z_m, "classification": classification,
            "return_number": return_number, "num_returns": num_returns}


def classification_histogram(classification):
    vals, counts = np.unique(classification, return_counts=True)
    hist = {}
    for v, c in zip(vals, counts):
        hist[CLASS_NAMES.get(int(v), f"class_{int(v)}")] = int(c)
    return hist


def rasterize_max(x, y, z, xmin, ymin, xmax, ymax, cell=1.0):
    """First-return/all-points DSM: max z per grid cell (captures the topmost surface —
    bridge deck, canopy, roofline — same principle as a standard USGS DSM product)."""
    ncols = int(np.ceil((xmax - xmin) / cell))
    nrows = int(np.ceil((ymax - ymin) / cell))
    col = np.clip(((x - xmin) / cell).astype(int), 0, ncols - 1)
    row = np.clip(((ymax - y) / cell).astype(int), 0, nrows - 1)
    grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    flat_idx = row * ncols + col
    order = np.argsort(z)
    flat_idx_sorted = flat_idx[order]
    z_sorted = z[order]
    grid_flat = grid.reshape(-1)
    grid_flat[flat_idx_sorted] = z_sorted   # last write per cell wins -> since sorted
                                              # ascending, this keeps the MAX z per cell
    return grid, from_origin(xmin, ymax, cell, cell)


def rasterize_median(x, y, z, xmin, ymin, xmax, ymax, cell=1.0):
    """Per-cell median z — more robust than max() for a 'clean deck surface' estimate, since
    max() picks up guardrails/lamp posts/signage (anything tall in the cell), not just the
    pavement itself."""
    ncols = int(np.ceil((xmax - xmin) / cell))
    nrows = int(np.ceil((ymax - ymin) / cell))
    col = np.clip(((x - xmin) / cell).astype(int), 0, ncols - 1)
    row = np.clip(((ymax - y) / cell).astype(int), 0, nrows - 1)
    flat_idx = row * ncols + col
    med = pd.DataFrame({"idx": flat_idx, "z": z}).groupby("idx")["z"].median()
    grid = np.full(nrows * ncols, np.nan, dtype=np.float32)
    grid[med.index.values] = med.values.astype(np.float32)
    return grid.reshape(nrows, ncols), from_origin(xmin, ymax, cell, cell)


def fill_and_smooth(grid, dilate_iters=10, sigma=1.5):
    """Nearest-neighbor-fill NaN gaps (so smoothing doesn't propagate NaNs), Gaussian-smooth
    to remove per-point/per-cell noise, then re-mask back down to a slightly dilated version
    of the original footprint (a small margin for a smoother edge blend, not the whole grid).

    dilate_iters was 2 (~2m) until this was traced as the root cause of a real bug: the raw
    class-17 (bridge-deck) points rasterize into two dense parallel bands — the highway's two
    carriageways — separated by a consistent ~10-16m median-strip gap with NO class-17 points
    at all (confirmed by inspecting the raw valid-cell mask at full resolution). A 2-cell dilate
    can't bridge that gap, so it survived into the saved raster as a real hole; when
    export_dem.py's apply_sr417_bridge_correction() reprojects this onto the viewer's much
    coarser 256x256 grid, cells whose interpolation kernel straddles that hole go back to NaN
    (falls back to bare-earth ~26m) right next to cells that land solidly on a deck band
    (~34m) — a checkerboard of correct/uncorrected cells instead of one continuous elevated
    roadway, which is what the team-lead flagged as the highway's slope/scale "looking wrong"
    at the two crossings. 10 cells (~10m) safely bridges the median gap without merging
    unrelated, far-apart classification noise elsewhere in the tile."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation
    valid = np.isfinite(grid)
    if not valid.any():
        return grid
    _, ind = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    filled = grid[tuple(ind)]
    smoothed = gaussian_filter(filled, sigma=sigma)
    keep = binary_dilation(valid, iterations=dilate_iters)
    return np.where(keep, smoothed, np.nan)


def build_bridge_deck_surface(pts, out_path):
    """Rasterize ONLY the bridge-deck-classified (ASPRS class 17) points into a smooth 1m
    surface — naturally confined to the two bridge decks themselves (no road-corridor buffer
    needed), median-per-cell (robust to guardrail/lamp-post outliers), gap-filled + smoothed.
    Returns (grid, transform, profile-like dict) or None if no bridge-deck points exist."""
    mask = pts["classification"] == 17
    n = int(mask.sum())
    if n == 0:
        print("  No bridge-deck (class 17) points found — skipping bridge-deck surface")
        return None
    x, y, z = pts["x"][mask], pts["y"][mask], pts["z"][mask]
    xmin, ymin, xmax, ymax = x.min() - 5, y.min() - 5, x.max() + 5, y.max() + 5
    grid, transform = rasterize_median(x, y, z, xmin, ymin, xmax, ymax, cell=1.0)
    n_valid_before = int(np.isfinite(grid).sum())
    grid = fill_and_smooth(grid)
    n_valid_after = int(np.isfinite(grid).sum())
    print(f"  Bridge-deck surface: {n} class-17 points → {n_valid_before} cells directly, "
          f"{n_valid_after} after gap-fill+smooth")

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1, "nodata": np.nan,
        "width": grid.shape[1], "height": grid.shape[0],
        "transform": transform, "crs": DEM_CRS, "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(grid.astype(np.float32), 1)
    print(f"  {os.path.basename(out_path)}")
    return grid, transform


def sample_grid(grid, transform, x, y):
    row, col = rasterio.transform.rowcol(transform, x, y)
    if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        v = grid[row, col]
        return None if np.isnan(v) else float(v)
    return None


def sr417_line_5070():
    roads = gpd.read_file(os.path.join(PROJ_DIR, "infrastructure", "data", "roads.geojson"))
    sr417 = roads[roads["ref"].astype(str).str.contains("417", na=False)]
    merged = linemerge(unary_union(sr417.geometry.values))
    gdf = gpd.GeoDataFrame(geometry=[merged], crs="EPSG:4326").to_crs(DEM_CRS)
    line = gdf.geometry.values[0]
    return list(line.geoms) if line.geom_type == "MultiLineString" else [line]


def profile_crossing(segs, cx, cy, dem_arr, dem_tf, dsm_arr, dsm_tf, span_m=150, step_m=10):
    """Sample bare-earth DEM vs point-cloud DSM along the SR417 centerline through a crossing,
    at span_m on either side of the along-line point nearest (cx,cy). This is the real test —
    a single point sampled exactly at the crossing coordinate tends to land on the cross-street's
    own grade (directly under the bridge), not on the highway surface itself."""
    best = None
    for si, seg in enumerate(segs):
        d = seg.project(Point(cx, cy))
        dist = seg.interpolate(d).distance(Point(cx, cy))
        if best is None or dist < best[0]:
            best = (dist, si, d)
    _, si, d0 = best
    seg = segs[si]
    rows = []
    for dd in range(-span_m, span_m + 1, step_m):
        d = d0 + dd
        if d < 0 or d > seg.length:
            continue
        p = seg.interpolate(d)
        z_dem = sample_grid(dem_arr, dem_tf, p.x, p.y)
        z_dsm = sample_grid(dsm_arr, dsm_tf, p.x, p.y)
        diff = (z_dsm - z_dem) if (z_dem is not None and z_dsm is not None) else None
        rows.append({"offset_m": dd, "dem_bare_earth_z_m": z_dem,
                     "point_cloud_dsm_z_m": z_dsm, "dsm_minus_dem_m": diff})
    return rows


def build_crossing_mesh(pts, cx, cy, buffer_m, out_path, label, geo_meta):
    """2.5D Delaunay TIN of all points within buffer_m of (cx,cy). Exports an OBJ mesh in the
    viewer's own scene-space convention (see viewer/static/js/terrain.js) so it can be added
    directly to the Three.js scene with zero extra transform:
        scene_x = real_X - origin_x - width_m/2
        scene_y = (real_Z_elev - z_min) * VERT_EXAG
        scene_z = origin_y + height_m/2 - real_Y
    """
    d2 = (pts["x"] - cx) ** 2 + (pts["y"] - cy) ** 2
    mask = d2 <= buffer_m ** 2
    n = int(mask.sum())
    if n < 4:
        print(f"    [{label}] only {n} points in window — skipping mesh")
        return None
    x, y, z = pts["x"][mask], pts["y"][mask], pts["z"][mask]
    cls = pts["classification"][mask]

    tri = Delaunay(np.column_stack([x, y]))

    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min, exag = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"], VERT_EXAG
    sx = x - ox - w / 2
    sy = (z - z_min) * exag
    sz = oy + h / 2 - y

    with open(out_path, "w") as fh:
        fh.write(f"# {label} — 2.5D Delaunay TIN from raw LiDAR ({n} points)\n")
        fh.write(f"# scene-space coords (viewer convention, VERT_EXAG={exag})\n")
        for xi, yi, zi in zip(sx, sy, sz):
            fh.write(f"v {xi:.3f} {yi:.3f} {zi:.3f}\n")
        for simplex in tri.simplices:
            a, b, c = simplex + 1   # OBJ is 1-indexed
            fh.write(f"f {a} {b} {c}\n")

    print(f"    [{label}] {n} points, {len(tri.simplices)} triangles → {os.path.basename(out_path)}")

    return {
        "n_points": n,
        "n_triangles": len(tri.simplices),
        "z_min": float(z.min()), "z_max": float(z.max()),
        "classification_histogram": classification_histogram(cls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--radius_km", type=float, default=DEFAULT_RADIUS_KM)
    args = ap.parse_args()

    lon_min, lat_min, lon_max, lat_max = bbox_from_center(args.lat, args.lon, args.radius_km)

    print("=" * 66)
    print("Raw LiDAR point cloud — SR417 corridor bridge-crossing validation")
    print("=" * 66)
    print(f"  AOI bbox (EPSG:4326): {lon_min:.5f},{lat_min:.5f} .. {lon_max:.5f},{lat_max:.5f}")

    print("\n[1/4] Loading + filtering point cloud …")
    pts = load_points_in_bbox(lon_min, lat_min, lon_max, lat_max)
    n_total = len(pts["x"])
    print(f"  Total points in AOI: {n_total:,}")

    hist = classification_histogram(pts["classification"])
    print("  Classification histogram:")
    for name, count in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"    {name:28s} {count:>10,}  ({100*count/n_total:.2f}%)")
    has_bridge_class = hist.get("bridge_deck", 0) > 0
    print(f"  Bridge-deck (class 17) points present: {has_bridge_class}")

    print("\n[2/4] Rasterizing DSM (max-z per 1m cell, all points) …")
    xmin, ymin = float(pts["x"].min()), float(pts["y"].min())
    xmax, ymax = float(pts["x"].max()), float(pts["y"].max())
    dsm, dsm_transform = rasterize_max(pts["x"], pts["y"], pts["z"], xmin, ymin, xmax, ymax, cell=1.0)

    ground_mask = pts["classification"] == 2
    dtm_pc, dtm_pc_transform = rasterize_max(
        pts["x"][ground_mask], pts["y"][ground_mask], pts["z"][ground_mask],
        xmin, ymin, xmax, ymax, cell=1.0,
    )

    print("\n  Building smoothed bridge-deck-only surface (class 17 median, gap-filled) …")
    build_bridge_deck_surface(pts, os.path.join(DATA_DIR, "lidar_bridge_deck_1m.tif"))

    with rasterio.open(DEM_COND) as src:
        dem_arr = src.read(1)
        dem_transform = src.transform

    with open(GEO_META) as fh:
        geo_meta = json.load(fh)

    print("\n[3/4] Comparing DSM vs existing bare-earth DEM along SR417 through each crossing …")
    segs = sr417_line_5070()
    results = {}
    to_dem_from_wgs84 = Transformer.from_crs("EPSG:4326", DEM_CRS, always_xy=True)
    for key, info in CROSSINGS.items():
        cx, cy = to_dem_from_wgs84.transform(info["lon"], info["lat"])

        # Single point exactly at the crossing coordinate tends to land on the cross-street's
        # own grade (under the bridge) — the real test is the profile along SR417 itself.
        profile = profile_crossing(segs, cx, cy, dem_arr, dem_transform, dsm, dsm_transform)
        diffs = [r["dsm_minus_dem_m"] for r in profile if r["dsm_minus_dem_m"] is not None]
        max_diff = max(diffs) if diffs else None

        print(f"  {info['label']}:")
        for r in profile:
            print(f"    offset {r['offset_m']:+4d}m  DEM={r['dem_bare_earth_z_m']!s:>8}  "
                  f"DSM={r['point_cloud_dsm_z_m']!s:>8}  diff={r['dsm_minus_dem_m']!s:>8}")
        print(f"    → max DSM-over-DEM within the profile: {max_diff:+.2f} m"
              if max_diff is not None else "    → no valid samples")

        mesh_path = os.path.join(DATA_DIR, f"bridge_mesh_{key}.obj")
        mesh_info = build_crossing_mesh(pts, cx, cy, CROSSING_BUFFER_M, mesh_path, info["label"], geo_meta)

        results[key] = {
            "label": info["label"], "lon": info["lon"], "lat": info["lat"],
            "profile": profile,
            "max_dsm_minus_dem_m": max_diff,
            "mesh": mesh_info,
        }

    print("\n[4/4] Saving outputs → lidar/data/ …")
    dem_profile = rasterio.open(DEM_COND).profile
    dsm_profile = dem_profile.copy()
    dsm_profile.update(height=dsm.shape[0], width=dsm.shape[1], transform=dsm_transform,
                        dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(os.path.join(DATA_DIR, "lidar_dsm_1m.tif"), "w", **dsm_profile) as dst:
        dst.write(dsm.astype(np.float32), 1)
    print("  lidar_dsm_1m.tif")

    with open(os.path.join(DATA_DIR, "classification_histogram.json"), "w") as fh:
        json.dump({"total_points": n_total, "histogram": hist}, fh, indent=2)
    print("  classification_histogram.json")

    with open(os.path.join(DATA_DIR, "bridge_crossing_validation.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("  bridge_crossing_validation.json")

    print("\n══ COMPLETE ══════════════════════════════════════════════")
    for key, r in results.items():
        diff = r["max_dsm_minus_dem_m"]
        if diff is not None:
            verdict = "CONFIRMED bridge artifact" if diff > 2.0 else "no significant difference"
            print(f"  {r['label']}: max DSM-DEM = {diff:+.2f} m  → {verdict}")


if __name__ == "__main__":
    main()
