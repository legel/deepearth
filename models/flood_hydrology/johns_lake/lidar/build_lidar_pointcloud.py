"""
Raw LiDAR point-cloud loader — Johns Lake AOI
=================================================
Reads the raw USGS 3DEP LiDAR point cloud (LAZ tiles downloaded via download_laz_tiles.py)
and filters to the AOI. Adapted from cfx_sr417_corridor's build_lidar_pointcloud.py, but the
CRS handling is genuinely different here and can't just reuse a single hardcoded constant:
this AOI's 12-tile query returns TWO different LiDAR acquisitions —
`FL_LAKECO_2007` (older, Lake County 2007 project) and `FL_Peninsular_2018_D18_LID2019`
(newer, 2018/2019) — confirmed by directly inspecting tile filenames and headers, not assumed.

Real CRS finding, checked directly via laspy (not guessed): the 2018 Peninsular tiles carry
real embedded CRS metadata (`las.header.parse_crs()` returns a full WKT) — EPSG:6438,
"NAD83(2011) / Florida East (ftUS)". The older 2007 LAKECO tiles carry NO CRS metadata
(parse_crs() returns None, same "distribution project predates VLR-embedded CRS" issue
cfx_sr417_corridor already documented for its own 2018 tiles). Their raw X coordinates
(440000-445000) exactly match the 2018 tile's X range, and their Y range (1515000-1520000) is
immediately adjacent/contiguous with the 2018 tile's Y range (1525000-1529999) — strong
evidence (not proof) they share the same Florida East (ftUS) planimetric system. Falls back to
EPSG:6438 for any tile with no embedded CRS, per-tile (not a single global assumption) so a
tile that DOES carry real metadata always uses its own declared CRS instead.

Usage (as a library — no __main__, mirrors cfx_sr417_corridor's own module-import pattern):
    from build_lidar_pointcloud import load_points_in_bbox, bbox_from_center
"""
import os, glob
import numpy as np
import laspy
from pyproj import Transformer, CRS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")

DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM = 28.521592, -81.656981, 1.0

GEO_META = os.path.join(PROJ_DIR, "viewer", "data", "geo_meta.json")
VERT_EXAG = 8   # must match viewer/static/js/terrain.js's default VERT_EXAG

FALLBACK_LAS_CRS = "EPSG:6438"   # NAD83(2011) / Florida East (ftUS) — see docstring above
DEM_CRS = "EPSG:5070"
FT_TO_M = 0.3048006096012192   # US survey foot

CLASS_NAMES = {
    0: "created_never_classified", 1: "unclassified", 2: "ground",
    3: "low_vegetation", 4: "medium_vegetation", 5: "high_vegetation",
    6: "building", 7: "low_point_noise", 9: "water", 10: "rail",
    11: "road_surface", 13: "wire_guard", 14: "wire_conductor",
    17: "bridge_deck", 18: "high_noise",
}


def bbox_from_center(lat, lon, radius_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    dlat = radius_km / km_per_deg_lat
    dlon = radius_km / km_per_deg_lon
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def load_points_in_bbox(lon_min, lat_min, lon_max, lat_max):
    """Read every LAZ tile in RAW_DIR, keep only points inside the AOI bbox, return
    a dict of numpy arrays in EPSG:5070 meters. Each tile's own CRS is detected
    independently (see module docstring) rather than assuming one global constant."""
    laz_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.laz")))
    if not laz_files:
        raise FileNotFoundError(f"No .laz files found in {RAW_DIR}")

    xs, ys, zs, cls, rn, nr = [], [], [], [], [], []
    for path in laz_files:
        print(f"  reading {os.path.basename(path)} …")
        las = laspy.read(path)

        tile_crs = las.header.parse_crs()
        if tile_crs is None:
            tile_crs = CRS.from_user_input(FALLBACK_LAS_CRS)
            print(f"    no embedded CRS — using fallback {FALLBACK_LAS_CRS}")
        else:
            print(f"    embedded CRS: {tile_crs.name}")

        to_las = Transformer.from_crs("EPSG:4326", tile_crs, always_xy=True)
        xmin_ft, ymin_ft = to_las.transform(lon_min, lat_min)
        xmax_ft, ymax_ft = to_las.transform(lon_max, lat_max)
        xmin_ft, xmax_ft = sorted([xmin_ft, xmax_ft])
        ymin_ft, ymax_ft = sorted([ymin_ft, ymax_ft])

        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        mask = (x >= xmin_ft) & (x <= xmax_ft) & (y >= ymin_ft) & (y <= ymax_ft)
        n = int(mask.sum())
        print(f"    {len(x):,} points in tile → {n:,} in AOI bbox")
        if n == 0:
            continue

        # Reproject THIS tile's kept points to EPSG:5070 meters using its own detected CRS,
        # before concatenating — the whole reason per-tile detection matters here (a single
        # global transform would be wrong for whichever acquisition doesn't match it).
        to_dem = Transformer.from_crs(tile_crs, DEM_CRS, always_xy=True)
        X, Y = to_dem.transform(x[mask], y[mask])
        # z unit: US survey foot for both acquisitions (confirmed via the 2018 tile's own WKT
        # VERT_CS unit; the 2007 tile is assumed to match, same fallback-CRS reasoning above).
        Z = z[mask] * FT_TO_M

        xs.append(X); ys.append(Y); zs.append(Z)
        cls.append(np.asarray(las.classification)[mask])
        rn.append(np.asarray(las.return_number)[mask])
        nr.append(np.asarray(las.number_of_returns)[mask])

    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    classification = np.concatenate(cls)
    return_number  = np.concatenate(rn)
    num_returns    = np.concatenate(nr)

    return {"x": x, "y": y, "z": z, "classification": classification,
            "return_number": return_number, "num_returns": num_returns}


def classification_histogram(classification):
    vals, counts = np.unique(classification, return_counts=True)
    return {CLASS_NAMES.get(int(v), f"class_{int(v)}"): int(c) for v, c in zip(vals, counts)}
