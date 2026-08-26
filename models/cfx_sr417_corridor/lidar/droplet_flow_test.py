"""
Droplet-based rainfall/runoff test — dense test area (DEM/LiDAR fusion)
==========================================================================
A deliberately minimal 3D water-flow prototype:
Lagrangian particle tracing across a real 3D surface — NOT the grid-based 2.5D shallow-water
solver used elsewhere in this project (simulation/flood_sim_ian.py).

v2 — DEM/LiDAR fusion. v1 built one Delaunay mesh
from raw ground+building LiDAR points together, which let droplets get numerically "stuck" on
rooftops (a false local minimum from LiDAR point noise in what should be a sloped, shedding
roof) — physically wrong; water essentially never just sits on a raised structure. v2 fixes
this with two distinct surfaces, fused with an explicit transition rule instead of one mesh:

  - GROUND = the project's own hydro-conditioned DEM (dem/data/hydro/dem_conditioned.tif),
    not raw LiDAR ground points. The DEM is the modeled, smoothed bare-earth surface — using
    it directly (rather than noisier raw point returns) is preferable because it is
    "less noise ... in how water flows on the ground." Droplets CAN settle here — that's a
    real depression in the modeled terrain, a real puddle.
  - BUILDINGS = per-building local meshes built from LiDAR building-classified (class 6)
    points only — the actual measured roof geometry, which the DEM excludes entirely.
    Droplets CANNOT settle here: reaching a local minimum on a roof (a valley, a numerical
    LiDAR-noise dip) is treated as reaching a gutter/drain point, not a puddle — the droplet
    is immediately handed down to the ground DEM at that (x,y) and continues flowing there.
    Reaching the edge of a roof footprint does the same thing.

A droplet's initial state (on a specific building's roof, or on the ground) is decided once,
at seeding, by whether its starting (x,y) falls inside a building footprint polygon. Once a
droplet transitions off a roof onto the ground, it stays on the ground for the rest of its run
— it does not re-ascend onto a roof it happens to pass under.

Physics on either surface (same minimal model as v1 — no depth field, no droplet-droplet
interaction, no infiltration): per triangle, downhill direction = gravity (0,0,-1) projected
onto that triangle's own plane.

Usage:
    python3 lidar/droplet_flow_test.py
    python3 lidar/droplet_flow_test.py --n-droplets 3000 --max-steps 400 --step-m 0.4
"""
import os, sys, json, struct, argparse
import numpy as np
import rasterio
import geopandas as gpd
from scipy.spatial import Delaunay
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lidar_pointcloud import (
    bbox_from_center, load_points_in_bbox, GEO_META, VERT_EXAG, DEM_CRS,
)
from cache_bbox_points import load_cached_points  # noqa: E402 — see test_sites.py's
                                                     # bbox_cache_dir docstring
from test_sites import SITES, get_site

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_COND = os.path.join(PROJ_DIR, "dem", "data", "hydro", "dem_conditioned.tif")
BUILDINGS_PATH = os.path.join(PROJ_DIR, "infrastructure", "data", "buildings.geojson")

# The dense test area identified 2026-07-07 — 41 buildings, mean ground slope 1.78°
# (real roof pitch is steeper still, not captured by the bare-earth slope raster).
# Kept as the module-level default (= test_sites.SITES["site1"]) for backward compatibility —
# see test_sites.py for the full site registry (site1 + site2, the retention-pond site added
# 2026-07-20).
TEST_LAT, TEST_LON, TEST_RADIUS_KM = 28.363316587590315, -81.4315738078251, 0.08

GRAVITY = np.array([0.0, 0.0, -1.0])
MIN_BUILDING_POINTS = 8   # skip buildings with too few LiDAR returns to mesh sensibly


def downhill_directions(verts, simplices):
    """Per-triangle gravity-projected-onto-plane downhill unit direction (dx,dy) + magnitude.
    Same formula used for both the ground and building meshes."""
    v0, v1, v2 = verts[simplices[:, 0]], verts[simplices[:, 1]], verts[simplices[:, 2]]
    normal = np.cross(v1 - v0, v2 - v0)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    flip = normal[:, 2] < 0
    normal[flip] *= -1

    g_dot_n = normal @ GRAVITY
    downhill_3d = GRAVITY[None, :] - g_dot_n[:, None] * normal
    downhill_xy = downhill_3d[:, :2]
    mag = np.linalg.norm(downhill_xy, axis=1)
    unit = np.where(mag[:, None] > 1e-9, downhill_xy / (mag[:, None] + 1e-12), 0.0)
    return unit, mag


class Surface:
    """A 2.5D Delaunay mesh + its per-triangle downhill field, plus barycentric height
    interpolation. Used for both the ground DEM mesh and each building's roof mesh."""
    def __init__(self, x, y, z):
        self.tri = Delaunay(np.column_stack([x, y]))
        self.verts = np.column_stack([x, y, z])
        self.simplices = self.tri.simplices
        self.downhill_unit, self.downhill_mag = downhill_directions(self.verts, self.simplices)

    def simplex_of(self, xy):
        return self.tri.find_simplex(xy)

    def z_at(self, xy, simplex_idx):
        z = np.full(len(xy), np.nan, dtype=np.float32)
        valid = simplex_idx >= 0
        if not valid.any():
            return z
        T = self.tri.transform[simplex_idx[valid]]
        delta = xy[valid] - T[:, 2, :]
        bary_01 = np.einsum("nij,nj->ni", T[:, :2, :], delta)
        bary_2 = 1.0 - bary_01.sum(axis=1)
        bary = np.column_stack([bary_01, bary_2])
        tri_verts_z = self.verts[self.simplices[simplex_idx[valid]], 2]
        z[valid] = np.sum(bary * tri_verts_z, axis=1)
        return z


def build_ground_surface(lon_min, lat_min, lon_max, lat_max, dem_cond_path=None, decimate=1):
    """Ground surface = the project's own hydro-conditioned DEM, cropped to the test area —
    the modeled bare-earth ground, not raw LiDAR ground returns.

    dem_cond_path: added 2026-07-27 for site3 (a genuinely different location, 37km from the
    original AOI, with its own conditioned DEM) — defaults to DEM_COND (the original AOI's
    file) for backward compatibility with site1/site2, which correctly still want that file.

    decimate: added 2026-07-27 for site3 — at native ~0.88m DEM resolution, site3's 6x6km box
    has 46.78 MILLION valid cells, and a full Delaunay triangulation over all of them (inside
    Surface.__init__ below) is neither tractable in this environment nor actually useful — the
    rest of this project already uses coarser resolutions for larger areas (the full-AOI Ian
    solver runs at 5m, not native 0.88m). decimate=N keeps every Nth grid point in each
    dimension BEFORE triangulating — decimate=1 (default) is a no-op, exactly today's behavior
    for site1/site2's already-small native-resolution triangle counts."""
    dem_cond_path = dem_cond_path or DEM_COND
    from pyproj import Transformer
    to5070 = Transformer.from_crs("EPSG:4326", DEM_CRS, always_xy=True)
    # 2026-07-21 alignment fix #1 (clipping): DEM_CRS (EPSG:5070, national-scale Albers) has
    # real meridian-convergence skew at this AOI's longitude — transforming only the 2 opposite
    # corners of the lon/lat box (as this used to do) produces a slightly rotated
    # parallelogram, not the true axis-aligned bbox, silently clipping off a real strip of
    # terrain (measured: ~24m on the west edge at site1's scale). Fixed by transforming all 4
    # corners and taking the true min/max — this is the WINDOW rasterio reads, wide enough to
    # never miss real terrain, but it is NOT yet the final point selection (see #2 below).
    corners = [to5070.transform(lon, lat) for lon, lat in
               [(lon_min, lat_min), (lon_max, lat_min), (lon_min, lat_max), (lon_max, lat_max)]]
    xmin, xmax = min(c[0] for c in corners), max(c[0] for c in corners)
    ymin, ymax = min(c[1] for c in corners), max(c[1] for c in corners)

    with rasterio.open(dem_cond_path) as src:
        # Compute the window via the raster's own inverse transform directly, rather than
        # rasterio.windows.from_bounds() (which assumes the standard north-up, negative-y-
        # resolution convention and raises "Bounds and transform are inconsistent" otherwise).
        # Added 2026-07-27: site3's own DEM (py3dep, for reasons not fully understood — the
        # original AOI's DEM does NOT have this issue at the same resolution/pipeline) came back
        # with a genuinely inverted transform (positive y-resolution, bottom>top) — this makes
        # the window computation robust to either orientation instead of special-casing site3.
        inv = ~src.transform
        col_a, row_a = inv * (xmin, ymin)
        col_b, row_b = inv * (xmax, ymax)
        col_off, col_stop = sorted([col_a, col_b])
        row_off, row_stop = sorted([row_a, row_b])
        window = rasterio.windows.Window(col_off=col_off, row_off=row_off,
                                          width=col_stop - col_off, height=row_stop - row_off)
        arr = src.read(1, window=window)
        win_transform = src.window_transform(window)

    if decimate > 1:
        # Subsample BEFORE triangulating (see the decimate docstring above) — scale the
        # transform to match so each kept point's real-world (x,y) is still computed correctly,
        # not just taking every Nth array element without adjusting its coordinate mapping.
        arr = arr[::decimate, ::decimate]
        win_transform = win_transform * rasterio.Affine.scale(decimate, decimate)

    rows, cols = arr.shape
    col_idx, row_idx = np.meshgrid(np.arange(cols), np.arange(rows))
    gx, gy = rasterio.transform.xy(win_transform, row_idx.ravel(), col_idx.ravel())
    gx, gy, gz = np.asarray(gx), np.asarray(gy), arr.ravel().astype(np.float64)
    valid = np.isfinite(gz)

    # 2026-07-21 alignment fix #2 (rotation): fix #1 above only widened the READ WINDOW so no
    # real terrain gets missed — it does NOT make this mesh's actual shape match the point
    # cloud's. The DEM raster is natively gridded axis-aligned to EPSG:5070's own Albers grid,
    # but load_points_in_bbox() filters LiDAR points against an axis-aligned box in EPSG:2881
    # (Florida State Plane, near-true-north at this location, confirmed negligible skew
    # earlier). Because Albers grid-north and true/State-Plane north differ by a REAL meridian
    # convergence angle at this AOI's longitude (measured ~9deg empirically via min-area-rect
    # on the exported point cloud vs. mesh; predicted 8.78deg from Albers' own cone-constant
    # formula n=(sin(29.5)+sin(45.5))/2, angle=n*(lon-(-96)) — matches almost exactly), an
    # axis-aligned-in-Albers rectangle and an axis-aligned-in-State-Plane rectangle covering
    # "the same" lon/lat box are NOT the same shape — one is rotated relative to the other by
    # that convergence angle. This is why the point cloud's and this mesh's "squares" had
    # visibly non-parallel edges even after fix #1 closed the size/clipping gap. Fixed by
    # masking grid points to the TRUE lon/lat box (not just the Albers-aligned window) — this
    # gives the ground mesh the same true-north-aligned footprint shape the point cloud already
    # has, eliminating the rotation instead of just the clipping.
    to4326 = Transformer.from_crs(DEM_CRS, "EPSG:4326", always_xy=True)
    glon, glat = to4326.transform(gx, gy)
    glon, glat = np.asarray(glon), np.asarray(glat)
    in_aoi = (glon >= lon_min) & (glon <= lon_max) & (glat >= lat_min) & (glat <= lat_max)
    valid = valid & in_aoi

    print(f"  Ground DEM grid: {rows}x{cols} = {rows*cols} cells ({valid.sum()} valid, "
          f"true-AOI-shape masked)")
    return Surface(gx[valid], gy[valid], gz[valid])


def build_building_surfaces(pts, test_area_bounds, buildings_path=None, max_points_per_building=None):
    """One local Delaunay mesh per OSM building footprint, from that building's own LiDAR
    building-classified (class 6) points — the measured roof geometry the DEM excludes.

    buildings_path: added 2026-07-27 for site3 — see build_ground_surface's dem_cond_path
    docstring for the same reasoning. Defaults to BUILDINGS_PATH for site1/site2.

    max_points_per_building: added 2026-07-27 for site3 — its 10,739 meshed buildings average
    ~7,477 roof triangles EACH (80.3M total), far denser than a roof shape needs and big enough
    to make OBJ export (a verbose ASCII format) and viewer rendering impractical. None (default)
    keeps every candidate point, exactly today's behavior for site1/site2. When set, each
    building's own candidate points are subsampled (evenly, not randomly, to keep spatial
    coverage of the roof) down to at most this many before triangulating."""
    buildings_path = buildings_path or BUILDINGS_PATH
    buildings = gpd.read_file(buildings_path).to_crs(DEM_CRS)
    xmin, ymin, xmax, ymax = test_area_bounds
    from shapely.geometry import box
    local = buildings[buildings.intersects(box(xmin, ymin, xmax, ymax))].reset_index(drop=True)

    bldg_mask = pts["classification"] == 6
    bx, by, bz = pts["x"][bldg_mask], pts["y"][bldg_mask], pts["z"][bldg_mask]

    # Spatial index for per-building point lookup — added 2026-07-27. The old approach (each
    # building re-masking the FULL building-point array with a boolean rectangle test) never
    # mattered at site1/site2's scale (37-52 buildings) but is a real O(n_buildings * n_points)
    # cost — directly timed at ~7 MINUTES for site3's 11,241 buildings against 52M points before
    # writing this fix, not guessed. A cKDTree lets each building query only its own local
    # neighborhood (radius = its own bounding-box half-diagonal + margin) instead of scanning
    # every point every time; the exact rectangle test still runs, just against that much
    # smaller per-building candidate set instead of the whole cloud.
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([bx, by])) if len(bx) else None

    surfaces = []
    polys = []
    n_skipped = 0
    for _, row in local.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        if tree is None:
            n_skipped += 1
            continue
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        radius = float(np.hypot(maxx - minx, maxy - miny)) / 2 + 1.0  # +1m margin
        cand_idx = np.asarray(tree.query_ball_point([cx, cy], r=radius), dtype=np.int64)
        if len(cand_idx) == 0:
            n_skipped += 1
            continue
        cx_pts, cy_pts = bx[cand_idx], by[cand_idx]
        exact = (cx_pts >= minx) & (cx_pts <= maxx) & (cy_pts >= miny) & (cy_pts <= maxy)
        n = int(exact.sum())
        if n < MIN_BUILDING_POINTS:
            n_skipped += 1
            continue
        final_idx = cand_idx[exact]
        if max_points_per_building is not None and len(final_idx) > max_points_per_building:
            stride = len(final_idx) // max_points_per_building
            final_idx = final_idx[::stride]
        surfaces.append(Surface(bx[final_idx], by[final_idx], bz[final_idx]))
        polys.append(row.geometry)

    print(f"  Building roofs meshed: {len(surfaces)} / {len(local)} "
          f"({n_skipped} skipped — too few LiDAR building points)")
    return surfaces, polys


def run_droplets_fused(ground, buildings, building_polys, n_droplets, max_steps, step_m, seed=0):
    rng = np.random.default_rng(seed)
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)

    cand = rng.uniform([gxmin, gymin], [gxmax, gymax], size=(n_droplets * 2, 2))
    on_ground_init = ground.simplex_of(cand) >= 0
    cand = cand[on_ground_init][:n_droplets]
    n = len(cand)

    # Decide each droplet's starting surface: a building roof if its seed point falls inside
    # that building's footprint (rain landing directly on a roof), else the ground.
    building_id = np.full(n, -1, dtype=np.int32)   # -1 = on ground
    for bi, poly in enumerate(building_polys):
        inside = np.array([poly.contains(Point(p)) for p in cand])
        building_id[inside] = bi
    n_on_roof = int((building_id >= 0).sum())
    print(f"  Seeded {n} droplets: {n_on_roof} land on a roof, {n - n_on_roof} land on open ground")

    xy = cand.copy()
    active = np.ones(n, dtype=bool)
    paths = [[] for _ in range(n)]
    settle_reason = np.full(n, "", dtype=object)
    roof_falls = np.zeros(n, dtype=np.int32)   # how many times each droplet fell off a roof

    def ground_z(xy_pts):
        return ground.z_at(xy_pts, ground.simplex_of(xy_pts))

    for step in range(max_steps):
        idx_active = np.where(active)[0]
        if len(idx_active) == 0:
            break

        z_now = np.full(n, np.nan)
        # Ground-state droplets
        gmask = active & (building_id < 0)
        if gmask.any():
            s = ground.simplex_of(xy[gmask])
            z_now[gmask] = ground.z_at(xy[gmask], s)
        # Roof-state droplets, grouped by building
        for bi, surf in enumerate(buildings):
            bmask = active & (building_id == bi)
            if not bmask.any():
                continue
            s = surf.simplex_of(xy[bmask])
            z_now[bmask] = surf.z_at(xy[bmask], s)

        for i in idx_active:
            if np.isfinite(z_now[i]):
                paths[i].append((float(xy[i, 0]), float(xy[i, 1]), float(z_now[i])))

        # ── Advance ground-state droplets ──────────────────────────────────────
        gmask = active & (building_id < 0)
        if gmask.any():
            gi = np.where(gmask)[0]
            s = ground.simplex_of(xy[gi])
            left = s < 0
            settle_reason[gi[left]] = "left_mesh"
            active[gi[left]] = False
            stay = ~left
            if stay.any():
                local_mag = ground.downhill_mag[s[stay]]
                settled = local_mag < 1e-6
                still_idx = gi[stay]
                settle_reason[still_idx[settled]] = "local_min"
                active[still_idx[settled]] = False
                move = still_idx[~settled]
                move_s = s[stay][~settled]
                xy[move] += ground.downhill_unit[move_s] * step_m

        # ── Advance roof-state droplets (per building) — never "settle" on a roof ──────────
        for bi, surf in enumerate(buildings):
            bmask = active & (building_id == bi)
            if not bmask.any():
                continue
            bi_idx = np.where(bmask)[0]
            s = surf.simplex_of(xy[bi_idx])
            fell_off_edge = s < 0
            local_mag = np.zeros(len(bi_idx))
            on_roof = ~fell_off_edge
            if on_roof.any():
                local_mag[on_roof] = surf.downhill_mag[s[on_roof]]
            reached_valley = on_roof & (local_mag < 1e-6)
            transition = fell_off_edge | reached_valley   # both mean: drop down to the ground

            trans_idx = bi_idx[transition]
            if len(trans_idx):
                roof_falls[trans_idx] += 1
                building_id[trans_idx] = -1   # now on the ground for good — (x,y) unchanged,
                                               # only the elevation source switches to the DEM

            move_mask = on_roof & ~reached_valley
            if move_mask.any():
                move_idx = bi_idx[move_mask]
                move_s = s[move_mask]
                xy[move_idx] += surf.downhill_unit[move_s] * step_m

    settle_reason[active] = "max_steps"
    n_steps_done = step + 1
    n_fell = int((roof_falls > 0).sum())
    print(f"  Ran {n_steps_done} steps; {n_fell} droplets fell off a roof onto the ground "
          f"during the run")
    print(f"  Final: local_min={int((settle_reason=='local_min').sum())} "
          f"left_mesh={int((settle_reason=='left_mesh').sum())} "
          f"max_steps={int((settle_reason=='max_steps').sum())}")
    return paths, settle_reason


def export_mesh_obj(verts_list, simplices_list, out_path, geo_meta, colors_list=None):
    """Export the combined ground + all building meshes as one OBJ (one visual surface for
    the viewer), in the same scene-space convention as the bridge-crossing meshes.

    colors_list: optional, added 2026-07-27 for site3 — same length/order as verts_list, each
    entry an (n,3) uint8 RGB array (e.g. from export_full_pointcloud.py's color_by_naip()) to
    write as real per-vertex NAIP color instead of a flat material tint, same "color the mesh
    from the aerial imagery it was built alongside" idea site1/site2's point clouds already use.
    Writes the extended "v x y z r g b" OBJ convention (values 0-1), which three.js's OBJLoader
    parses into a vertex-color attribute automatically. None (the default) preserves the exact
    original plain "v x y z" output for site1/site2 — no behavior change for existing callers.
    """
    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]

    total_v, total_t = sum(len(v) for v in verts_list), sum(len(s) for s in simplices_list)
    with open(out_path, "w") as fh:
        fh.write(f"# Dense test area — fused DEM ground + LiDAR roof meshes "
                  f"({total_v} verts, {total_t} triangles)\n")
        vertex_offset = 0
        for i, (verts, simplices) in enumerate(zip(verts_list, simplices_list)):
            sx = verts[:, 0] - ox - w / 2
            sy = (verts[:, 2] - z_min) * VERT_EXAG
            sz = oy + h / 2 - verts[:, 1]
            if colors_list is not None:
                rgb01 = colors_list[i].astype(np.float64) / 255.0
                for (xi, yi, zi), (r, g, b) in zip(zip(sx, sy, sz), rgb01):
                    fh.write(f"v {xi:.3f} {yi:.3f} {zi:.3f} {r:.4f} {g:.4f} {b:.4f}\n")
            else:
                for xi, yi, zi in zip(sx, sy, sz):
                    fh.write(f"v {xi:.3f} {yi:.3f} {zi:.3f}\n")
            for tri_idx in simplices:
                a, b, c = tri_idx + 1 + vertex_offset
                fh.write(f"f {a} {b} {c}\n")
            vertex_offset += len(verts)
    kb = os.path.getsize(out_path) / 1024
    print(f"  {os.path.basename(out_path)}: {total_v} verts, {total_t} triangles ({kb/1024:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES), default="site1",
                     help="named test site (see test_sites.py) — sets lat/lon/radius_km "
                          "unless individually overridden below, and suffixes output "
                          "filenames for any site other than site1 (kept unsuffixed for "
                          "backward compatibility with the existing viewer wiring)")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--radius_km", type=float, default=None)
    ap.add_argument("--n-droplets", type=int, default=3000)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--step-m", type=float, default=0.4)
    args = ap.parse_args()

    site = get_site(args.site)
    lat = args.lat if args.lat is not None else site["lat"]
    lon = args.lon if args.lon is not None else site["lon"]
    radius_km = args.radius_km if args.radius_km is not None else site["radius_km"]
    suffix = "" if args.site == "site1" else f"_{args.site}"

    print("=" * 66)
    print(f"Droplet-based rainfall/runoff test — DEM ground / LiDAR roofs fusion  [{args.site}: {site['label']}]")
    print("=" * 66)

    lon_min, lat_min, lon_max, lat_max = bbox_from_center(lat, lon, radius_km)

    # Data-source path overrides — added 2026-07-27 for site3, a genuinely different location
    # (37km from the original AOI) with its own DEM/buildings files. site.get(...) returns None
    # for site1/site2 (no such keys in their SITES entries), so build_ground_surface/
    # build_building_surfaces fall back to the original AOI's module-level constants exactly as
    # before — this is purely additive, not a behavior change for the existing sites.
    dem_cond_path = site.get("dem_cond_path")
    buildings_path = site.get("buildings_path")

    print("\n[1/5] Building ground surface from the DEM (modeled bare earth) …")
    ground = build_ground_surface(lon_min, lat_min, lon_max, lat_max, dem_cond_path=dem_cond_path,
                                   decimate=site.get("ground_decimate", 1))
    print(f"  {len(ground.simplices):,} ground triangles  "
          f"downhill mag: mean={ground.downhill_mag.mean():.3f} max={ground.downhill_mag.max():.3f}")

    print("\n[2/5] Loading LiDAR building points + building roof meshes …")
    bbox_cache_dir = site.get("bbox_cache_dir")
    if bbox_cache_dir:
        pts = load_cached_points(lon_min, lat_min, lon_max, lat_max, bbox_cache_dir)
    else:
        pts = load_points_in_bbox(lon_min, lat_min, lon_max, lat_max)
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)
    buildings, building_polys = build_building_surfaces(
        pts, (gxmin, gymin, gxmax, gymax), buildings_path=buildings_path,
        max_points_per_building=site.get("roof_max_points"))
    n_roof_tri = sum(len(b.simplices) for b in buildings)
    print(f"  {n_roof_tri:,} roof triangles across {len(buildings)} buildings")

    with open(GEO_META) as fh:
        geo_meta = json.load(fh)

    print("\n[3/5] Exporting the fused mesh (ground + roofs) …")
    verts_list = [ground.verts] + [b.verts for b in buildings]
    simplices_list = [ground.simplices] + [b.simplices for b in buildings]
    export_mesh_obj(verts_list, simplices_list,
                     os.path.join(DATA_DIR, f"dense_test_area_mesh{suffix}.obj"), geo_meta)

    print("\n[4/5] Tracing droplets (roofs shed to the ground, never settle on a roof) …")
    paths, settle_reason = run_droplets_fused(ground, buildings, building_polys,
                                               args.n_droplets, args.max_steps, args.step_m)

    print("\n[5/5] Exporting scene-space paths for the viewer …")
    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]

    reason_code = {"local_min": 0, "left_mesh": 1, "max_steps": 2}
    out_path = os.path.join(DATA_DIR, f"droplet_paths{suffix}.bin")
    n_written = 0
    with open(out_path, "wb") as fh:
        fh.write(b"DROP")
        fh.write(struct.pack("<I", len(paths)))
        for path, reason in zip(paths, settle_reason):
            pts_arr = np.array(path, dtype=np.float32)
            if len(pts_arr) < 2:
                fh.write(struct.pack("<I", 0))
                fh.write(struct.pack("<B", reason_code.get(reason, 2)))
                continue
            sx = pts_arr[:, 0] - ox - w / 2
            sy = (pts_arr[:, 2] - z_min) * VERT_EXAG
            sz = oy + h / 2 - pts_arr[:, 1]
            scene = np.column_stack([sx, sy, sz]).astype(np.float32).reshape(-1)
            fh.write(struct.pack("<I", len(pts_arr)))
            fh.write(scene.tobytes())
            fh.write(struct.pack("<B", reason_code.get(reason, 2)))
            n_written += 1

    kb = os.path.getsize(out_path) / 1024
    print(f"  {os.path.basename(out_path)}: {n_written} droplet paths written  ({kb:.0f} KB)")

    summary = {
        "site": args.site, "site_label": site["label"],
        "n_droplets_seeded": len(paths), "n_paths_written": n_written,
        "n_ground_triangles": len(ground.simplices), "n_roof_triangles": n_roof_tri,
        "n_buildings_meshed": len(buildings),
        "max_steps": args.max_steps, "step_m": args.step_m,
        "settle_counts": {r: int((settle_reason == r).sum())
                           for r in ["local_min", "left_mesh", "max_steps"]},
        "test_area": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "architecture": "v2: DEM ground surface + per-building LiDAR roof meshes, "
                         "droplets shed off roofs (edge or local min) to the ground, "
                         "never settle on a raised surface",
    }
    with open(os.path.join(DATA_DIR, f"droplet_paths_summary{suffix}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  droplet_paths_summary{suffix}.json")


if __name__ == "__main__":
    main()
