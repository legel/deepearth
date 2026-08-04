"""
Unstructured-mesh 3D local-inertial shallow-water solver — dense test area
============================================================================
Generalizes simulation/flood_sim_ian.py's Bates et al. (2010) local-inertia scheme
(grid-based, 2.5D: one depth value per square cell) to the fused ground-DEM +
per-building LiDAR-roof triangulated mesh built by lidar/droplet_flow_test.py. Same
physics family — explicit mass conservation + semi-implicit-friction momentum equation,
Froude-number flux limiter, Horton infiltration — but flux now moves between triangles
of arbitrary size/orientation via the mesh's real edge-adjacency graph. This is what
makes it 3D: water flows across a sloped roof triangle, off its eave, and down onto the
ground triangle below it, rather than being confined to one height value per (x,y) grid
cell the way the flat 2.5D grid solver is.

Architecture (see lidar/droplet_flow_test.py's docstring for the same ground/roof split,
applied there to single Lagrangian point-droplets instead of a continuous depth field):
  - GROUND triangles = the project's conditioned DEM, cropped to the test area.
  - ROOF triangles = per-building meshes from real LiDAR building-classified (class 6)
    points, one independent Delaunay mesh per building footprint.
  - Every triangle-triangle edge (ground-ground, roof-roof, AND roof-eave-to-ground) is
    treated as one uniform Bates-flux edge in a single combined edge list — this is the
    literal mechanism for "water sheds off the roof and lands on the ground below" in a
    mass-conserving depth field: a roof triangle at a building's edge (eave) is connected
    by a real flux edge to whichever ground triangle sits beneath that eave's (x,y)
    midpoint, so water crosses onto the ground the same way it crosses any other edge —
    driven by the real elevation drop, damped by friction — no ad hoc "handoff" rule
    needed (droplet_flow_test.py needed one because a single point has no way to "flow
    across" an edge; a continuous depth field does this naturally through the flux term).
  - Only the ground mesh's own OUTER boundary (the edge of the ~160m test-area box) is
    genuinely open/transmissive — water reaching it is treated as real runoff leaving the
    simulated area (critical-flow outflow sink), same role as flood_sim_ian.py's
    transmissive domain edges, implemented differently because the boundary here is an
    irregular mesh hull rather than a rectangle.

Uses the SAME ground+roof construction functions as lidar/droplet_flow_test.py (same
test area, same LiDAR points, same buildings), so the resulting triangle order is
IDENTICAL to lidar/data/dense_test_area_mesh.obj's face order — the existing exported
mesh geometry is reused as-is by the viewer; this script only exports a per-triangle
depth-over-time binary that indexes into that same face order. No second mesh export.

Deliberate simplifications (demo-scale prototype, not a rigorous CFD build):
  - Ground infiltration uses the single project-wide Horton mean (soil/data/soil_parameters
    .json) rather than per-cell SSURGO sampling — sub-160m soil variation isn't resolved by
    SSURGO's own map-unit polygons anyway at this scale.
  - Rain is a synthetic trapezoidal burst (not tied to Hurricane Ian's real hyetograph) —
    this test area is orders of magnitude smaller than the full-AOI Ian run, so reusing
    that 336mm/72hr event here would be physically disproportionate; the point of this
    script is mesh-resolution water behavior, not another Ian calibration run.
  - Edge "length" L in the Bates equation uses full 3D centroid-to-centroid distance
    (not horizontal-only) so steep roof/wall triangles get a physically real, not
    horizontally-foreshortened, travel distance; both L and edge width W are floored at
    L_FLOOR_M to guard against near-degenerate slivers in the raw LiDAR-derived roof mesh
    blowing up the 1/L term — the implicit friction term (same as Bates 2010 relies on
    for CFL-violation robustness) does the rest of the stabilizing work.

Usage:
    python3 simulation/mesh_shallow_water.py
    python3 simulation/mesh_shallow_water.py --peak-rain-mm-hr 100 --rain-duration-min 4 \
        --total-min 8 --dt 0.15 --frame-interval-s 3
"""
import os, sys, json, time, struct, argparse
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import shapely
from shapely.geometry import box, Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
ROADS_PATH = os.path.join(PROJ_DIR, "infrastructure", "data", "roads.geojson")
WATERBODIES_PATH = os.path.join(PROJ_DIR, "hydrography", "data", "3dhp_waterbodies.geojson")

sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from droplet_flow_test import (  # noqa: E402
    build_ground_surface, build_building_surfaces,
    TEST_LAT, TEST_LON, TEST_RADIUS_KM,
)
from build_lidar_pointcloud import (  # noqa: E402
    bbox_from_center, load_points_in_bbox, GEO_META, VERT_EXAG, DEM_CRS,
)
from cache_bbox_points import load_cached_points  # noqa: E402 — see test_sites.py's
                                                     # bbox_cache_dir docstring
from test_sites import SITES, get_site  # noqa: E402

sys.path.insert(0, BASE_DIR)
from flood_sim_ian import (  # noqa: E402  (reuse validated constants)
    HORTON, horton_rate, G, MIN_DEPTH, ROAD_BUFFER_M, ROAD_BUFFER_DEFAULT_M,
    MUKEY_MAP, MUKEY_LEGEND, SOIL_JSON, AMC3_FACTOR, NLCD_IMPERVIOUS_PATH,
)

GROUND_MANNING_N = 0.040   # same flatwoods/mixed-cover value as flood_sim_ian.py
ROOF_MANNING_N   = 0.015   # smooth roofing material (shingle/metal) — sheds much faster
CFL_ALPHA        = 0.3     # same conservative factor as flood_sim_ian.py
L_FLOOR_M        = 0.10    # min edge centroid-distance / width used in flux calc
DEPTH_THR        = 0.01    # m — "wet" threshold for reporting
MAX_DEPTH_CAP    = 3.0     # m — last-resort numerical safety rail, not a physical limit;
                            # any step that hits it is logged so a real blowup isn't silently
                            # masked (see the per-cell Courant fix above for the actual cause
                            # this was originally guarding against)
AREA_MIN_ACTIVE  = 1e-3    # m^2 — triangles smaller than this (near-duplicate raw LiDAR
                            # points producing near-zero-area slivers, seen down to 1e-11 m^2
                            # in the roof point clouds) are excluded from the flux network
                            # entirely rather than area-floored: dividing flux by a floored-but-
                            # still-tiny area was amplifying it into an instant NaN blowup.
                            # Excluded triangles just stay permanently dry (h=0) — geometrically
                            # negligible, invisible in the rendered mesh either way.
POND_OUTLET_ORIFICE_D_M = 0.10   # m — 4-inch bleeder orifice, a typical small FL residential/
                                   # roadside stormwater-detention-pond low-flow control riser
                                   # size. Added 2026-07-24 per Lance's "engineered waterway"
                                   # ask: site2's pond previously behaved as a passive natural
                                   # depression (same generic edge-flux as everywhere else) —
                                   # a real detention pond releases through a SIZED, ENGINEERED
                                   # outlet that deliberately holds back peak flow, not however
                                   # fast the terrain allows. Demo-scale caveat: real bleeder
                                   # orifices are sized to drain a retained volume over ~24-72hrs
                                   # (SFWMD design criteria) — this project's synthetic rain
                                   # events are only ~8 minutes, so the outlet's effect within
                                   # one demo run is necessarily small BY DESIGN, not a bug.
POND_OUTLET_CD   = 0.61          # standard sharp-edged-orifice discharge coefficient
ROOF_POND_MAX_M  = 0.02    # m — per Lance's team-lead feedback on the earlier droplet tracer
                            # ("water should rarely, if ever, just sit on raised surfaces"):
                            # a real roof can hold a thin film before finding a gutter/downspout,
                            # but a LiDAR-noise pit shouldn't become a standing puddle. Depth
                            # above this on any roof triangle is drained straight to the ground
                            # triangle beneath it each step (see roof_below in build_combined_mesh)
                            # — the continuous-depth-field equivalent of the droplet tracer's
                            # "never settle on a roof, hand off to the DEM" rule.

# ── Physics-driven tracer particles (see run_flow_tracers) ───────────────────────────
# 2026-07-21: replaces droplet_flow_test.py's role — same visual (a drawn path + moving point,
# 3-way settle/runoff/still-moving color coding) but motion comes from this solver's own
# reconstructed velocity field instead of a fixed 0.4m-per-step kinematic walk.
TRACER_SUB_DT           = 1.0 / 30.0   # s — MUST match dropletFlow.js's BASE_STEPS_PER_SECOND
                                         # exactly: each exported path point represents this
                                         # much real simulated time, so client playback speed
                                         # (which just steps through points at 30/s) reads as
                                         # the true relative speed — fast where real velocity is
                                         # fast, slow/stopped where it's genuinely stalled.
TRACER_MAX_STEPS        = 500           # ~16.7s of animation per particle before "max_steps"
TRACER_SETTLE_SPEED     = 0.02          # m/s — below this counts toward "settled"
TRACER_SETTLE_PATIENCE  = 20            # consecutive slow sub-steps before calling it settled
TRACER_SKY_FALL_STEPS   = 40            # ~1.3s visual lead-in falling from the sky
TRACER_SKY_HEIGHT_M     = 15.0          # real-world meters above the landing point to fall from


# ── Impervious / water-body classification (ground triangles only — roofs are already ────
# ── 100% impervious by construction everywhere in this solver) ───────────────────────────

def compute_ground_impervious_mask(ground_xy, roads_path=None):
    """Boolean per ground-triangle-centroid, True under an OSM road (buffered by
    ROAD_BUFFER_M's real per-highway-type width, same table flood_sim_ian.py uses for the
    full-AOI solver). Team-lead meeting note: "classify hard materials so no infiltration" —
    already implemented for the full-AOI solver (apply_impervious_mask in flood_sim_ian.py)
    but this small-area mesh solver only ever zeroed infiltration on BUILDINGS (via is_roof),
    never on roads/driveways running through the ground mesh itself — a real gap, fixed here.

    roads_path: added 2026-07-27 for site3 — see droplet_flow_test.build_ground_surface's
    dem_cond_path docstring for the same reasoning. Defaults to ROADS_PATH for site1/site2."""
    roads_path = roads_path or ROADS_PATH
    roads = gpd.read_file(roads_path).to_crs(DEM_CRS)
    pad = box(ground_xy[:, 0].min() - 50, ground_xy[:, 1].min() - 50,
              ground_xy[:, 0].max() + 50, ground_xy[:, 1].max() + 50)
    roads = roads[roads.geometry.intersects(pad)]
    if len(roads) == 0:
        return np.zeros(len(ground_xy), dtype=bool)
    widths = roads["highway"].map(ROAD_BUFFER_M).fillna(ROAD_BUFFER_DEFAULT_M)
    buffered = roads.geometry.buffer(widths.values)
    # Real, MEASURED scaling bug found 2026-07-27 at site3's scale (1.46M ground triangles,
    # 3,193 road segments — never exercised before at more than tens of thousands of ground
    # triangles / a few hundred segments): the straightforward `union.buffer().union_all()`
    # then `GeoSeries.within(single_geometry)` approach hung for 16+ minutes on the union+
    # within step alone. Profiled directly (not guessed): union_all() itself is fast (0.48s,
    # only 14 disjoint polygon parts) and shapely.prepare() on that union does NOT fix it —
    # a `.within()` test against 200,000 sample points still hadn't finished after 2:39+
    # minutes even prepared (this geopandas/shapely combination's scalar-geometry `.within()`
    # broadcast doesn't appear to get real indexed lookup here, root cause not fully pinned
    # down). The actual fix, confirmed by direct profiling: build a `shapely.STRtree` over the
    # INDIVIDUAL buffered polygons (not their union) and bulk-query it — 1.45s for 200,000
    # points, ~10.6s extrapolated for the full 1,461,971 ground triangles. ~230x faster than
    # the union+within path was projected to be, and actually finishes.
    if len(buffered) == 0:
        return np.zeros(len(ground_xy), dtype=bool)
    tree = shapely.STRtree(buffered.values)
    pts = gpd.points_from_xy(ground_xy[:, 0], ground_xy[:, 1])
    hit_pt_idx, _ = tree.query(pts, predicate="within")
    mask = np.zeros(len(ground_xy), dtype=bool)
    mask[np.unique(hit_pt_idx)] = True
    return mask


def load_spatial_horton_points(ground_xy, mukey_map_path=None, mukey_legend_path=None,
                                soil_json_path=None):
    """Point-sampled equivalent of flood_sim_ian.load_spatial_horton() — that function is
    raster-shaped (reprojects mukey_map.tif onto a regular grid), which doesn't apply here since
    this solver's ground triangles are irregular Delaunay centroids, not grid cells. Added
    2026-07-24: this small-area mesh solver previously used a single project-wide Horton mean
    everywhere non-impervious (see module docstring's "Deliberate simplifications") — real gap
    per Lance's repeated ask to show different soil types absorbing water differently, not just
    a binary hard/soft split. Reuses the exact same mukey_map.tif + legend + soil_parameters.json
    + AMC3_FACTOR the full-AOI Ian solver already uses (same soil survey, no new data needed) —
    just samples it at each ground triangle's own (x,y) instead of reprojecting onto a grid.
    Returns None (caller falls back to the scalar HORTON dict) if the required files are missing.

    mukey_map_path/mukey_legend_path/soil_json_path: added 2026-07-27 for site3 — see
    droplet_flow_test.build_ground_surface's dem_cond_path docstring for the same reasoning.
    Default to the module-level MUKEY_MAP/MUKEY_LEGEND/SOIL_JSON constants for site1/site2.
    """
    mukey_map_path = mukey_map_path or MUKEY_MAP
    mukey_legend_path = mukey_legend_path or MUKEY_LEGEND
    soil_json_path = soil_json_path or SOIL_JSON
    if not (os.path.exists(mukey_map_path) and os.path.exists(mukey_legend_path)
            and os.path.exists(soil_json_path)):
        return None

    with rasterio.open(mukey_map_path) as src:
        mukey_arr = src.read(1)
        transform = src.transform
        raster_crs = src.crs

    xy = ground_xy
    if str(raster_crs) != str(DEM_CRS):
        from pyproj import Transformer
        tr = Transformer.from_crs(DEM_CRS, raster_crs, always_xy=True)
        gx, gy = tr.transform(xy[:, 0], xy[:, 1])
    else:
        gx, gy = xy[:, 0], xy[:, 1]

    rows, cols = rasterio.transform.rowcol(transform, gx, gy)
    rows = np.clip(np.asarray(rows), 0, mukey_arr.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, mukey_arr.shape[1] - 1)
    mukey_codes = mukey_arr[rows, cols]

    legend = pd.read_csv(mukey_legend_path).set_index("mukey_int")["mukey"].astype(str).to_dict()
    with open(soil_json_path) as fh:
        soil = json.load(fh)

    n = len(ground_xy)
    fc_mm_hr = np.full(n, HORTON["fc"], dtype=np.float32)
    f0_mm_hr = np.full(n, HORTON["f0"], dtype=np.float32)
    k_arr    = np.full(n, HORTON["k"],  dtype=np.float32)

    for code in np.unique(mukey_codes):
        mukey_str = legend.get(int(code))
        params    = soil.get(mukey_str) if mukey_str else None
        if not params or "fc_mm_hr" not in params:
            continue   # leave the scalar-mean fallback for unmapped codes
        mask = mukey_codes == code
        fc_mm_hr[mask] = params["fc_mm_hr"] * AMC3_FACTOR
        f0_mm_hr[mask] = fc_mm_hr[mask][0] * 2.5
        k_arr[mask]    = params["k_hr"]

    n_units = len(np.unique(mukey_codes))
    print(f"  Spatial Horton (point-sampled): {n_units} soil units across {n} ground triangles  "
          f"(fc_eff range {fc_mm_hr.min():.1f}–{fc_mm_hr.max():.1f} mm/hr)")

    return {"f0": f0_mm_hr, "fc": fc_mm_hr, "k": k_arr}   # mm/hr, mm/hr, hr^-1 — SI conversion
                                                            # happens in run_sim, same as the
                                                            # scalar HORTON dict's convention


def load_nlcd_impervious_fraction_points(ground_xy, already_hard_mask, nlcd_path=None):
    """Point-sampled equivalent of flood_sim_ian.apply_nlcd_graded_impervious() — same NLCD
    layer, same "graded, not binary" reasoning (see that function's docstring), just sampled at
    ground triangle centroids instead of reprojected onto a raster grid. already_hard_mask (Tg,)
    marks triangles the binary OSM-road mask already zeroed — NLCD doesn't touch those (a real
    road is 100% impervious regardless of what its 30m NLCD pixel reports). Returns an all-1.0
    (no reduction) array if nlcd_impervious.tif isn't available, so callers can multiply
    unconditionally without a None-check.

    nlcd_path: added 2026-07-27 for site3 — see droplet_flow_test.build_ground_surface's
    dem_cond_path docstring for the same reasoning. Defaults to NLCD_IMPERVIOUS_PATH.
    """
    nlcd_path = nlcd_path or NLCD_IMPERVIOUS_PATH
    n = len(ground_xy)
    if not os.path.exists(nlcd_path):
        return np.ones(n, dtype=np.float64)

    with rasterio.open(nlcd_path) as src:
        nlcd_arr = src.read(1)
        transform = src.transform
        raster_crs = src.crs

    if str(raster_crs) != str(DEM_CRS):
        from pyproj import Transformer
        tr = Transformer.from_crs(DEM_CRS, raster_crs, always_xy=True)
        gx, gy = tr.transform(ground_xy[:, 0], ground_xy[:, 1])
    else:
        gx, gy = ground_xy[:, 0], ground_xy[:, 1]

    rows, cols = rasterio.transform.rowcol(transform, gx, gy)
    rows = np.clip(np.asarray(rows), 0, nlcd_arr.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, nlcd_arr.shape[1] - 1)
    pct = nlcd_arr[rows, cols]
    imperv_frac = np.clip(np.nan_to_num(pct, nan=0.0) / 100.0, 0.0, 1.0)

    grade = np.where(already_hard_mask, 1.0, 1.0 - imperv_frac)
    n_graded = int((~already_hard_mask).sum())
    if n_graded:
        print(f"  NLCD graded impervious: mean {100*imperv_frac[~already_hard_mask].mean():.1f}% "
              f"impervious fraction across {n_graded} non-road ground triangles "
              f"(mean infiltration-capacity reduction "
              f"{100*(1-grade[~already_hard_mask]).mean():.1f}%)")
    return grade


def compute_lake_mask(ground_xy, pond_id3dhp):
    """Boolean per ground-triangle-centroid, True inside the named 3DHP waterbody polygon
    (site2's retention pond, test_sites.py's pond_id3dhp). None (not an all-False array) when
    the site has no water body, so callers can tell "no lake here" apart from "lake found, but
    it happens to cover zero of this mesh's triangles." A real water body doesn't infiltrate
    into itself either, so this mask is also folded into the impervious classification."""
    if not pond_id3dhp:
        return None
    wb = gpd.read_file(WATERBODIES_PATH).to_crs(DEM_CRS)
    match = wb[wb["id3dhp"] == pond_id3dhp]
    if match.empty:
        print(f"  WARNING: pond_id3dhp={pond_id3dhp!r} not found in {WATERBODIES_PATH}")
        return None
    poly = match.geometry.iloc[0]
    shapely.prepare(poly)   # same fix as compute_ground_impervious_mask, same reasoning
    pts = gpd.GeoSeries(gpd.points_from_xy(ground_xy[:, 0], ground_xy[:, 1]), crs=DEM_CRS)
    return pts.within(poly).to_numpy()


# ── Mesh topology (per-triangle edge adjacency, from scipy Delaunay) ──────────────────

def build_mesh_topology(surface):
    """Per-triangle area/centroid + internal edge list (i,j,L,W) + boundary edge list
    (owner triangle, length, midpoint xy) for one Surface (ground, or one building roof).
    Edges touching a near-zero-area sliver triangle (see AREA_MIN_ACTIVE) are dropped."""
    tri = surface.tri
    simplices = surface.simplices
    verts = surface.verts
    neighbors = tri.neighbors
    T = len(simplices)

    centroid_xy = verts[simplices].mean(axis=1)[:, :2]
    centroid_z = verts[simplices, 2].mean(axis=1)
    v0xy, v1xy, v2xy = verts[simplices[:, 0], :2], verts[simplices[:, 1], :2], verts[simplices[:, 2], :2]
    area = 0.5 * np.abs((v1xy[:, 0] - v0xy[:, 0]) * (v2xy[:, 1] - v0xy[:, 1])
                         - (v2xy[:, 0] - v0xy[:, 0]) * (v1xy[:, 1] - v0xy[:, 1]))
    active = area > AREA_MIN_ACTIVE
    area = np.maximum(area, AREA_MIN_ACTIVE)

    ei_all, ej_all, L_all, W_all = [], [], [], []
    bnd_owner, bnd_len, bnd_xy = [], [], []
    arangeT = np.arange(T)
    for k in range(3):
        nb = neighbors[:, k]
        va = simplices[:, (k + 1) % 3]
        vb = simplices[:, (k + 2) % 3]
        elen = np.linalg.norm(verts[va, :2] - verts[vb, :2], axis=1)
        has_nb = nb >= 0

        pick = has_nb & (nb > arangeT)   # dedupe: process each undirected edge once
        pick &= active & np.where(has_nb, active[np.clip(nb, 0, T - 1)], False)
        i_idx, j_idx = arangeT[pick], nb[pick]
        ci = np.column_stack([centroid_xy[i_idx], centroid_z[i_idx]])
        cj = np.column_stack([centroid_xy[j_idx], centroid_z[j_idx]])
        d3 = np.linalg.norm(ci - cj, axis=1)
        ei_all.append(i_idx); ej_all.append(j_idx); L_all.append(d3); W_all.append(elen[pick])

        bnd = ~has_nb & active
        bnd_owner.append(arangeT[bnd])
        bnd_len.append(elen[bnd])
        bnd_xy.append(0.5 * (verts[va[bnd], :2] + verts[vb[bnd], :2]))

    return dict(
        T=T, area=area, z=centroid_z, xy=centroid_xy,
        edges=dict(i=np.concatenate(ei_all), j=np.concatenate(ej_all),
                   L=np.maximum(np.concatenate(L_all), L_FLOOR_M),
                   W=np.maximum(np.concatenate(W_all), 1e-3)),
        boundary=dict(owner=np.concatenate(bnd_owner),
                      length=np.maximum(np.concatenate(bnd_len), 1e-3),
                      xy=np.concatenate(bnd_xy, axis=0)),
    )


def build_combined_mesh(ground, buildings):
    """Fuse the ground topology and every roof's topology into ONE global triangle/edge
    system: triangle indices [0,Tg) = ground, [Tg,Tg+Tr) = roofs (per-building order,
    matching lidar/droplet_flow_test.py's export order exactly). Each roof's eave
    (boundary) edges become real flux edges to the ground triangle beneath them — the
    mechanism that lets water cross from a roof onto the ground it sits above."""
    g = build_mesh_topology(ground)
    Tg = g["T"]

    area_parts, z_parts, xy_parts, roof_flag_parts = [g["area"]], [g["z"]], [g["xy"]], [np.zeros(Tg, dtype=bool)]
    ei_parts, ej_parts, L_parts, W_parts = [g["edges"]["i"]], [g["edges"]["j"]], [g["edges"]["L"]], [g["edges"]["W"]]
    building_id_parts = [np.full(Tg, -1, dtype=np.int32)]   # -1 = ground; per-building index
                                                              # for roof triangles — lets a single
                                                              # building's own water volume be
                                                              # isolated for the one-house cm^3
                                                              # sanity check (see main())

    roof_below_ground = []   # per GLOBAL roof triangle index -> ground triangle beneath its
                              # own centroid (not just eave/boundary triangles) — the anti-
                              # ponding drain path (see POND_MAX_M in run_sim): Lance's original
                              # feedback on the droplet tracer was that a false local minimum
                              # from LiDAR point noise on a roof (a "gutter" that isn't real)
                              # must not become a puddle — the droplet tracer enforced that by
                              # handing a single point straight to the ground on any roof local
                              # min; a continuous depth field needs the equivalent: any roof
                              # triangle ponding past a small physical cap dumps its excess onto
                              # the ground triangle directly beneath it, same as a real gutter/
                              # downspout would, rather than accumulating indefinitely.
    roof_below_idx = []      # matching global roof triangle indices (only where a ground
                              # triangle was actually found beneath)

    offset = Tg
    for bi, surf in enumerate(buildings):
        t = build_mesh_topology(surf)
        T = t["T"]
        area_parts.append(t["area"]); z_parts.append(t["z"]); xy_parts.append(t["xy"])
        roof_flag_parts.append(np.ones(T, dtype=bool))
        building_id_parts.append(np.full(T, bi, dtype=np.int32))
        ei_parts.append(t["edges"]["i"] + offset)
        ej_parts.append(t["edges"]["j"] + offset)
        L_parts.append(t["edges"]["L"])
        W_parts.append(t["edges"]["W"])

        # Eave edges: connect each roof-boundary triangle to the ground triangle under it.
        bxy = t["boundary"]["xy"]
        gsimp = ground.simplex_of(bxy)
        valid = gsimp >= 0
        if valid.any():
            roof_tri = t["boundary"]["owner"][valid] + offset
            ground_tri = gsimp[valid]
            ci = np.column_stack([t["boundary"]["xy"][valid],
                                   t["z"][t["boundary"]["owner"][valid]]])
            cj = np.column_stack([g["xy"][ground_tri], g["z"][ground_tri]])
            L_eave = np.maximum(np.linalg.norm(ci - cj, axis=1), L_FLOOR_M)
            W_eave = np.maximum(t["boundary"]["length"][valid], 1e-3)
            ei_parts.append(roof_tri); ej_parts.append(ground_tri)
            L_parts.append(L_eave); W_parts.append(W_eave)

        gsimp_all = ground.simplex_of(t["xy"])
        valid_all = gsimp_all >= 0
        roof_below_idx.append(np.where(valid_all)[0] + offset)
        roof_below_ground.append(gsimp_all[valid_all])

        offset += T

    T_total = sum(len(a) for a in area_parts)

    return dict(
        T=T_total, Tg=Tg,
        area=np.concatenate(area_parts), z=np.concatenate(z_parts), xy=np.concatenate(xy_parts, axis=0),
        is_roof=np.concatenate(roof_flag_parts),
        building_id=np.concatenate(building_id_parts),
        edges=dict(i=np.concatenate(ei_parts), j=np.concatenate(ej_parts),
                   L=np.concatenate(L_parts), W=np.concatenate(W_parts)),
        ground_open_boundary=g["boundary"],   # test-area-box perimeter — real runoff exit
        roof_below=dict(roof_tri=np.concatenate(roof_below_idx),
                         ground_tri=np.concatenate(roof_below_ground)),
    )


# ── Rain forcing ────────────────────────────────────────────────────────────────────

def synthetic_rain_rate_ms(t_s, rain_duration_s, peak_mm_hr, ramp_frac=0.15):
    """Trapezoidal demo-scale rain burst — see module docstring for why this test area
    doesn't reuse the full-AOI Hurricane Ian hyetograph. Returns rain rate in m/s."""
    ramp = rain_duration_s * ramp_frac
    if t_s < ramp:
        mm_hr = peak_mm_hr * (t_s / ramp)
    elif t_s < rain_duration_s - ramp:
        mm_hr = peak_mm_hr
    elif t_s < rain_duration_s:
        mm_hr = peak_mm_hr * ((rain_duration_s - t_s) / ramp)
    else:
        mm_hr = 0.0
    return mm_hr / 1000.0 / 3600.0


# ── Solver ──────────────────────────────────────────────────────────────────────────

def run_sim(mesh, dt_target, total_s, rain_duration_s, peak_mm_hr, frame_interval_s, verbose=True,
            ground_impervious_mask=None, lake_mask=None, ground_horton=None,
            ground_nlcd_grade=None, pond_outlet_diameter_m=None):
    T = mesh["T"]
    Tg = mesh["Tg"]
    area, z, is_roof, xy = mesh["area"], mesh["z"], mesh["is_roof"], mesh["xy"]
    i_all, j_all, L_all, W_all = (mesh["edges"]["i"], mesh["edges"]["j"],
                                   mesh["edges"]["L"], mesh["edges"]["W"])
    L_min = float(L_all.min())

    # Real 3D unit direction i->j for every edge (mesh geometry is static, computed once) —
    # used to reconstruct an actual per-triangle velocity VECTOR from the scalar edge fluxes
    # at each saved frame, for physics-driven tracer advection (see run_flow_tracers below).
    # This is a visualization-purpose reconstruction, not a rigorously derived physical
    # quantity the way the depth field h is (h is exactly mass-conserving; this velocity
    # estimate is a standard "face-flux-to-cell-vector" approximation, common in unstructured
    # hydraulic post-processing, sound enough to drive believable tracer motion but not
    # claimed as a validated new physical output).
    edge_dx = xy[j_all, 0] - xy[i_all, 0]
    edge_dy = xy[j_all, 1] - xy[i_all, 1]
    edge_dz = z[j_all] - z[i_all]
    edge_dir_x, edge_dir_y, edge_dir_z = edge_dx / L_all, edge_dy / L_all, edge_dz / L_all

    # Per-edge-length CFL alone isn't enough on this mesh: many small roof-eave edges can
    # converge onto ONE small ground triangle within a single step (each individually
    # Froude-capped, but their sum still overfilling that one small cell) — a real FV
    # Courant condition needs each triangle's total incident edge "capacity" too, i.e. an
    # effective cell radius = area / (sum of its edges' widths), not just neighbor distance.
    edge_width_sum = (np.bincount(i_all, weights=W_all, minlength=T)
                       + np.bincount(j_all, weights=W_all, minlength=T))
    has_edges = edge_width_sum > 0
    cell_scale = np.where(has_edges, area / np.maximum(edge_width_sum, 1e-9), np.inf)
    cell_scale_min = float(cell_scale[has_edges].min()) if has_edges.any() else L_min
    # A tiny minority of triangles (dense LiDAR clusters where many small edges converge on
    # one small cell) have a pathologically small cell_scale that would collapse dt toward
    # zero for the WHOLE mesh if used directly — floor it at L_FLOOR_M and let the per-edge
    # Froude cap + MAX_DEPTH_CAP safety rail (see run loop) absorb the resulting local
    # under-resolution instead. A demo-scale trade of local accuracy in a few tiny cells for
    # a tractable runtime, not a correctness claim about those specific cells.
    L_min = max(min(L_min, cell_scale_min), L_FLOOR_M)

    n_arr = np.where(is_roof, ROOF_MANNING_N, GROUND_MANNING_N)
    n_edge = 0.5 * (n_arr[i_all] + n_arr[j_all])

    # Impervious = roofs (always) + OSM roads + the water body itself, if this site has one —
    # none of these infiltrate. ground_impervious_mask/lake_mask are (Tg,) — ground-only —
    # since roofs are already 100% covered by is_roof.
    is_impervious = is_roof.copy()
    if ground_impervious_mask is not None:
        is_impervious[:Tg] |= ground_impervious_mask
    if lake_mask is not None:
        is_impervious[:Tg] |= lake_mask

    # Soil-type-specific Horton params (mm/hr, mm/hr, hr^-1) — added 2026-07-24. ground_horton
    # is (Tg,) point-sampled per-triangle values (see load_spatial_horton_points) if the caller
    # supplied them, else every ground triangle falls back to the same project-wide scalar mean
    # HORTON used before this change. Roof triangles always use the scalar mean too (irrelevant —
    # is_impervious zeroes them regardless; roofs were never meant to have "soil").
    f0_mm_hr = np.full(T, HORTON["f0"], dtype=np.float64)
    fc_mm_hr = np.full(T, HORTON["fc"], dtype=np.float64)
    k_hr     = np.full(T, HORTON["k"],  dtype=np.float64)
    if ground_horton is not None:
        f0_mm_hr[:Tg] = ground_horton["f0"]
        fc_mm_hr[:Tg] = ground_horton["fc"]
        k_hr[:Tg]     = ground_horton["k"]

    # NLCD graded impervious % (added 2026-07-24) — scales capacity DOWN further for ground
    # triangles NLCD reports as partly hard-surfaced (driveways, compacted soil near buildings)
    # that the binary OSM road mask doesn't cover. ground_nlcd_grade is (Tg,) in [0,1]; roofs get
    # grade=1 (no-op — is_impervious already zeroes them). See load_nlcd_impervious_fraction_points.
    nlcd_grade_full = np.ones(T, dtype=np.float64)
    if ground_nlcd_grade is not None:
        nlcd_grade_full[:Tg] = ground_nlcd_grade

    f0_si = np.where(is_impervious, 0.0, f0_mm_hr / 1000 / 3600 * nlcd_grade_full)
    fc_si = np.where(is_impervious, 0.0, fc_mm_hr / 1000 / 3600 * nlcd_grade_full)
    k_si = np.where(is_impervious, 0.0, k_hr / 3600)

    bnd_owner = mesh["ground_open_boundary"]["owner"]
    bnd_len = mesh["ground_open_boundary"]["length"]
    roof_below_tri = mesh["roof_below"]["roof_tri"]
    roof_below_ground = mesh["roof_below"]["ground_tri"]

    h = np.zeros(T, dtype=np.float64)
    q = np.zeros(len(i_all), dtype=np.float64)   # persistent momentum state, one per edge
    h_max = np.zeros(T, dtype=np.float64)

    cum_rain_vol = 0.0
    cum_infil_vol = 0.0
    cum_outflow_vol = 0.0

    frames_h, frames_vel, frame_times = [], [], []   # frames_vel: (T,3) float32 per saved frame
    # Lake "level" = area-weighted mean depth over the lake-mask ground triangles. This works
    # directly because the ground DEM already represents the pond as a flat bare-earth-DTM fill
    # at today's real water-surface elevation (confirmed by direct sampling — see test_sites.py)
    # — so h itself, restricted to those triangles, literally IS the rise above today's level,
    # no separate datum bookkeeping needed. Tracked every step (not just at saved frames) for a
    # smooth hydrograph-quality curve; cheap (one float per step).
    lake_level_ts, lake_time_ts = [], []
    lake_area = float(area[:Tg][lake_mask].sum()) if lake_mask is not None else 0.0
    lake_tri_idx = np.where(lake_mask)[0] if lake_mask is not None else None   # global indices
    pond_outlet_area_m2 = (np.pi * (pond_outlet_diameter_m / 2) ** 2
                           if (lake_mask is not None and pond_outlet_diameter_m) else None)
    cum_pond_outlet_vol = 0.0

    t_s = 0.0
    last_frame_t = -1e9
    step = 0
    t0 = time.time()
    n_capped_total = [0]

    while t_s < total_s:
        h_max_local = float(h.max())
        dt = min(dt_target, CFL_ALPHA * L_min / np.sqrt(G * h_max_local)) if h_max_local > MIN_DEPTH else dt_target
        dt = min(dt, total_s - t_s)

        # ── Edge fluxes (Bates et al. 2010 local inertia, one unified edge list) ──────
        eta = z + h
        hf = np.maximum(np.maximum(eta[i_all], eta[j_all]) - np.maximum(z[i_all], z[j_all]), 0.0)
        num = q - G * hf * dt * (eta[j_all] - eta[i_all]) / L_all
        denom = 1.0 + G * dt * n_edge ** 2 * np.abs(q) / (hf ** (4.0 / 3.0) + 1e-10)
        q = np.where(hf > MIN_DEPTH, num / denom, 0.0)
        q_cap = 0.9 * hf * np.sqrt(G * np.maximum(hf, MIN_DEPTH))
        q = np.clip(q, -q_cap, q_cap)

        # ── Per-cell volume limiter ────────────────────────────────────────────────
        # The Froude cap above bounds each EDGE to a physically plausible flow rate, but a
        # triangle with several outgoing edges (common at eave-to-ground triangles, where
        # many small roof-boundary edges converge on one small ground cell) can still have
        # its edges' combined outflow exceed the volume the triangle actually holds. Left
        # unchecked, the sender gets floored at h=0 (losing only what it had) while its
        # neighbors still receive the full, un-reduced amount each edge computed
        # independently — a real mass-creation bug (confirmed empirically: an early version
        # without this check reported 255 m^3 stored from 15.6 m^3 of rain). Fix: rescale
        # every edge leaving an over-draining triangle so total outflow never exceeds what
        # it actually has: exact volume conservation, matching what each edge's neighbor
        # receives to what its sender actually gave up.
        vol_edge = q * W_all * dt                      # signed; >0 = i loses / j gains
        out_i = np.bincount(i_all, weights=np.maximum(vol_edge, 0.0), minlength=T)
        out_j = np.bincount(j_all, weights=np.maximum(-vol_edge, 0.0), minlength=T)
        outflow_vol = out_i + out_j
        available_vol = h * area
        scale = np.minimum(1.0, available_vol / np.maximum(outflow_vol, 1e-12))
        scale_send = np.where(vol_edge >= 0, scale[i_all], scale[j_all])
        vol_edge *= scale_send

        dh = np.bincount(i_all, weights=-vol_edge / area[i_all], minlength=T)
        dh += np.bincount(j_all, weights=vol_edge / area[j_all], minlength=T)

        # ── Ground open-boundary outflow (real runoff leaving the test-area box) ──────
        hf_b = np.maximum(h[bnd_owner], 0.0)
        q_out = np.where(hf_b > MIN_DEPTH, hf_b * np.sqrt(G * np.maximum(hf_b, MIN_DEPTH)), 0.0)
        out_vol = float(np.sum(q_out * bnd_len * dt))
        cum_outflow_vol += out_vol
        dh += np.bincount(bnd_owner, weights=-q_out * bnd_len * dt / area[bnd_owner], minlength=T)

        # ── Rain + infiltration ────────────────────────────────────────────────────
        rain_ms = synthetic_rain_rate_ms(t_s, rain_duration_s, peak_mm_hr)
        inf_ms = horton_rate(t_s, f0_si, fc_si, k_si)
        Pe = np.maximum(rain_ms - inf_ms, 0.0)
        cum_rain_vol += float(rain_ms * dt * area.sum())
        cum_infil_vol += float(np.sum(np.minimum(rain_ms, inf_ms) * dt * area))
        dh += Pe * dt

        h = np.clip(h + dh, 0.0, MAX_DEPTH_CAP)
        n_capped = int((h >= MAX_DEPTH_CAP).sum())
        if n_capped:
            n_capped_total[0] += n_capped

        # ── Roof anti-ponding drain (Lance's "water shouldn't sit on raised surfaces") ──
        h_roof_below = h[roof_below_tri]
        roof_excess = np.maximum(h_roof_below - ROOF_POND_MAX_M, 0.0)
        if roof_excess.any():
            h[roof_below_tri] -= roof_excess
            excess_vol = roof_excess * area[roof_below_tri]
            h += np.bincount(roof_below_ground, weights=excess_vol, minlength=T) / area

        # ── Engineered pond outlet (real orifice discharge, not passive terrain flow) ─────
        # Q = Cd*A*sqrt(2*g*h) applied to the pond's own LUMPED mean head (a single physical
        # structure drains the whole pond, not a per-triangle process) — same category of
        # post-flux "structure" adjustment as the roof anti-ponding drain above, just modeling
        # a real engineered control instead of a LiDAR-noise-pit workaround. Removed volume is
        # real outflow LEAVING the domain (routed to a downstream storm sewer/ditch beyond this
        # mesh), so it's added to cum_outflow_vol exactly like boundary-edge outflow — same
        # physical event (water leaves the simulated area), different exit mechanism.
        if pond_outlet_area_m2 is not None:
            h_lake = h[lake_tri_idx]
            area_lake = area[lake_tri_idx]
            vol_lake = float(np.sum(h_lake * area_lake))
            h_lake_mean = vol_lake / lake_area if lake_area > 0 else 0.0
            if h_lake_mean > MIN_DEPTH:
                q_outlet = POND_OUTLET_CD * pond_outlet_area_m2 * np.sqrt(2 * G * h_lake_mean)
                vol_outlet = min(q_outlet * dt, vol_lake)   # can't drain more than exists
                if vol_outlet > 0:
                    frac = h_lake * area_lake / max(vol_lake, 1e-12)   # proportional share
                    h[lake_tri_idx] -= (vol_outlet * frac) / area_lake
                    cum_outflow_vol += vol_outlet
                    cum_pond_outlet_vol += vol_outlet

        h_max = np.maximum(h_max, h)

        if lake_mask is not None:
            lake_level_ts.append(float(np.average(h[:Tg][lake_mask], weights=area[:Tg][lake_mask])))
            lake_time_ts.append(t_s)

        if t_s - last_frame_t >= frame_interval_s:
            frames_h.append(h.astype(np.float32).copy())
            # Reconstruct a per-triangle velocity VECTOR from this step's realized edge
            # fluxes (vol_edge, already Froude- and volume-limiter-scaled — the flow that
            # actually happened this step, not the pre-cap estimate). Standard face-flux-to-
            # cell-vector reconstruction: direction = flux-weighted sum of incident edge
            # directions; speed = total unsigned flux through the cell's edges divided by a
            # depth*length scale (dimensionally m/s). See edge_dir_* setup above for why this
            # is a defensible visualization aid, not a new validated physical output.
            flux_rate = vol_edge / dt   # m^3/s, signed i->j, post all limiters this step
            vx = (np.bincount(i_all, weights=flux_rate * edge_dir_x, minlength=T)
                  + np.bincount(j_all, weights=flux_rate * edge_dir_x, minlength=T))
            vy = (np.bincount(i_all, weights=flux_rate * edge_dir_y, minlength=T)
                  + np.bincount(j_all, weights=flux_rate * edge_dir_y, minlength=T))
            vz = (np.bincount(i_all, weights=flux_rate * edge_dir_z, minlength=T)
                  + np.bincount(j_all, weights=flux_rate * edge_dir_z, minlength=T))
            w_accum = (np.bincount(i_all, weights=np.abs(flux_rate), minlength=T)
                       + np.bincount(j_all, weights=np.abs(flux_rate), minlength=T))
            speed = w_accum / (np.maximum(h, MIN_DEPTH) * np.sqrt(np.maximum(area, 1e-6)))
            dir_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) + 1e-12
            vel = np.column_stack([vx / dir_mag * speed, vy / dir_mag * speed,
                                    vz / dir_mag * speed]).astype(np.float32)
            frames_vel.append(vel)
            frame_times.append(t_s)
            last_frame_t = t_s

        if verbose and step % 200 == 0:
            n_wet = int((h > DEPTH_THR).sum())
            print(f"  t={t_s:6.1f}s  dt={dt:.4f}s  rain={rain_ms*3600*1000:6.1f}mm/hr  "
                  f"h_max={float(h.max()):.3f}m  wet_tri={n_wet:6d}/{T}  "
                  f"[{t_s/total_s*100:5.1f}% {time.time()-t0:.0f}s]")

        t_s += dt
        step += 1

    if not frame_times or frame_times[-1] < t_s - 1e-6:
        frames_h.append(h.astype(np.float32).copy())
        frames_vel.append(frames_vel[-1] if frames_vel else np.zeros((T, 3), dtype=np.float32))
        frame_times.append(t_s)

    return dict(
        h_final=h, h_max=h_max, frames_h=frames_h, frames_vel=frames_vel, frame_times=frame_times,
        n_steps=step, wall_s=time.time() - t0, n_depth_cap_hits=n_capped_total[0],
        mass_balance=dict(
            rain_vol_m3=cum_rain_vol, infil_vol_m3=cum_infil_vol,
            outflow_vol_m3=cum_outflow_vol,
            pond_outlet_vol_m3=cum_pond_outlet_vol,   # subset of outflow_vol_m3, tracked
                                                        # separately so it's visible how much
                                                        # of total outflow left via the
                                                        # engineered structure vs. the open
                                                        # boundary edge
            stored_vol_m3=float(np.sum(h * area)),
        ),
        lake_level_ts=lake_level_ts, lake_time_ts=lake_time_ts, lake_area_m2=lake_area,
    )


# ── GPU (torch/MPS) solver — same physics as run_sim(), ported for the multi-scenario ──
# ── GNN-training-data sweep (2026-07-27) ────────────────────────────────────────────────
# Added specifically because generating many training scenarios on CPU is genuinely expensive
# (measured: 1414.2s / 44,618 steps = 31.7ms/step on site3_crop's 710,228-triangle mesh) — a
# real per-step FLOPS cost over ~1M edges, unlike the earlier one-off "should we GPU-port"
# question this session, where the actual bottleneck turned out to be a CPU algorithmic bug,
# not compute. This is a literal line-by-line port of run_sim()'s physics (same formulas, same
# order of operations) to torch tensors on MPS (this machine's only GPU backend — no CUDA),
# using scatter_add_ in place of np.bincount(). MPS does NOT support float64 (PyTorch raises on
# most float64 ops on this backend), so this runs in float32 throughout — a real precision
# trade-off against the CPU version's float64, checked via mass-balance residual after every
# run rather than assumed safe. run_sim() itself is left completely untouched; this is a new,
# separate function so nothing about the existing CPU path can regress.
#
# One deliberate, explicit non-fix, flagged rather than hidden: the adaptive-CFL dt calculation
# still calls h.max().item() every step, forcing a device->host sync exactly like the "known
# GPU anti-pattern" flagged when this project first discussed GPU porting. Kept as a direct,
# same-structure port for a first honest benchmark against the real CPU number BEFORE spending
# more engineering effort restructuring it — see the module-level benchmark script this was
# tested with. If the sync overhead dominates, that will show up directly in the real measured
# ms/step, not be a background assumption.
def run_sim_gpu(mesh, dt_target, total_s, rain_duration_s, peak_mm_hr, frame_interval_s, verbose=True,
                 ground_impervious_mask=None, lake_mask=None, ground_horton=None,
                 ground_nlcd_grade=None, pond_outlet_diameter_m=None, device="mps"):
    import torch, resource
    dev = torch.device(device)
    f32 = torch.float32

    # Memory instrumentation — added 2026-07-27, explicitly for the compute/memory report
    # Lance asked about (2026-07-12 meeting notes: "what compute/memory does a full simulation
    # need"). Apple Silicon uses UNIFIED memory (no separate discrete-GPU VRAM pool the way a
    # CUDA machine would have) — torch.mps has no CUDA-style max_memory_allocated() peak
    # tracker, so peak is tracked manually here by sampling at the same cadence as the existing
    # verbose step print. Reports both the MPS allocator's own bookkeeping (current_allocated_
    # memory/driver_allocated_memory — what PyTorch itself is using on the GPU-facing
    # allocator) and total process RSS (ru_maxrss, bytes on macOS) — the two numbers together
    # are what's actually meaningful on a unified-memory machine, since "GPU memory" here isn't
    # a separate budget from "system memory" the way it would be on a discrete-GPU deployment.
    mem_peak_mps_current = 0
    mem_peak_mps_driver = 0
    mem_recommended_max = int(torch.mps.recommended_max_memory()) if hasattr(torch.mps, "recommended_max_memory") else None

    def sample_memory():
        nonlocal mem_peak_mps_current, mem_peak_mps_driver
        cur = torch.mps.current_allocated_memory()
        drv = torch.mps.driver_allocated_memory()
        mem_peak_mps_current = max(mem_peak_mps_current, cur)
        mem_peak_mps_driver = max(mem_peak_mps_driver, drv)
        return cur, drv

    def t_(a, dtype=f32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)

    T = mesh["T"]
    Tg = mesh["Tg"]
    area = t_(mesh["area"]); z = t_(mesh["z"])
    is_roof = t_(mesh["is_roof"], dtype=torch.bool)
    xy = t_(mesh["xy"])
    i_all = t_(mesh["edges"]["i"], dtype=torch.long)
    j_all = t_(mesh["edges"]["j"], dtype=torch.long)
    L_all = t_(mesh["edges"]["L"]); W_all = t_(mesh["edges"]["W"])
    L_min = float(mesh["edges"]["L"].min())

    edge_dx = xy[j_all, 0] - xy[i_all, 0]
    edge_dy = xy[j_all, 1] - xy[i_all, 1]
    edge_dz = z[j_all] - z[i_all]
    edge_dir_x, edge_dir_y, edge_dir_z = edge_dx / L_all, edge_dy / L_all, edge_dz / L_all

    edge_width_sum = (torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, W_all)
                       + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, W_all))
    has_edges = edge_width_sum > 0
    cell_scale = torch.where(has_edges, area / torch.clamp(edge_width_sum, min=1e-9),
                              torch.full_like(area, float("inf")))
    cell_scale_min = float(cell_scale[has_edges].min()) if has_edges.any() else L_min
    L_min = max(min(L_min, cell_scale_min), L_FLOOR_M)

    n_arr = torch.where(is_roof, torch.tensor(ROOF_MANNING_N, device=dev, dtype=f32),
                         torch.tensor(GROUND_MANNING_N, device=dev, dtype=f32))
    n_edge = 0.5 * (n_arr[i_all] + n_arr[j_all])

    is_impervious = is_roof.clone()
    if ground_impervious_mask is not None:
        is_impervious[:Tg] |= t_(ground_impervious_mask, dtype=torch.bool)
    if lake_mask is not None:
        is_impervious[:Tg] |= t_(lake_mask, dtype=torch.bool)

    f0_mm_hr = torch.full((T,), HORTON["f0"], dtype=f32, device=dev)
    fc_mm_hr = torch.full((T,), HORTON["fc"], dtype=f32, device=dev)
    k_hr = torch.full((T,), HORTON["k"], dtype=f32, device=dev)
    if ground_horton is not None:
        f0_mm_hr[:Tg] = t_(ground_horton["f0"])
        fc_mm_hr[:Tg] = t_(ground_horton["fc"])
        k_hr[:Tg] = t_(ground_horton["k"])

    nlcd_grade_full = torch.ones(T, dtype=f32, device=dev)
    if ground_nlcd_grade is not None:
        nlcd_grade_full[:Tg] = t_(ground_nlcd_grade)

    zero = torch.zeros(T, dtype=f32, device=dev)
    f0_si = torch.where(is_impervious, zero, f0_mm_hr / 1000 / 3600 * nlcd_grade_full)
    fc_si = torch.where(is_impervious, zero, fc_mm_hr / 1000 / 3600 * nlcd_grade_full)
    k_si = torch.where(is_impervious, zero, k_hr / 3600)

    bnd_owner = t_(mesh["ground_open_boundary"]["owner"], dtype=torch.long)
    bnd_len = t_(mesh["ground_open_boundary"]["length"])
    roof_below_tri = t_(mesh["roof_below"]["roof_tri"], dtype=torch.long)
    roof_below_ground = t_(mesh["roof_below"]["ground_tri"], dtype=torch.long)

    h = torch.zeros(T, dtype=f32, device=dev)
    q = torch.zeros(i_all.shape[0], dtype=f32, device=dev)
    h_max = torch.zeros(T, dtype=f32, device=dev)

    cum_rain_vol = 0.0
    cum_infil_vol = 0.0
    cum_outflow_vol = 0.0
    area_sum = float(area.sum().item())

    frames_h, frames_vel, frame_times = [], [], []
    lake_level_ts, lake_time_ts = [], []
    lake_mask_t = t_(lake_mask, dtype=torch.bool) if lake_mask is not None else None
    lake_area = float(area[:Tg][lake_mask_t].sum().item()) if lake_mask is not None else 0.0
    lake_area_w = area[:Tg][lake_mask_t] if lake_mask is not None else None
    lake_tri_idx = torch.where(lake_mask_t)[0] if lake_mask is not None else None
    pond_outlet_area_m2 = (np.pi * (pond_outlet_diameter_m / 2) ** 2
                           if (lake_mask is not None and pond_outlet_diameter_m) else None)
    cum_pond_outlet_vol = 0.0

    t_s = 0.0
    last_frame_t = -1e9
    step = 0
    t0 = time.time()
    n_capped_total = 0
    n_edges = i_all.shape[0]

    while t_s < total_s:
        h_max_local = float(h.max().item())   # forces a device->host sync every step — see
                                                 # the function docstring above for why this
                                                 # was kept as-is for a first honest benchmark
        dt = min(dt_target, CFL_ALPHA * L_min / (G * h_max_local) ** 0.5) if h_max_local > MIN_DEPTH else dt_target
        dt = min(dt, total_s - t_s)

        eta = z + h
        hf = torch.clamp(torch.maximum(eta[i_all], eta[j_all]) - torch.maximum(z[i_all], z[j_all]), min=0.0)
        num = q - G * hf * dt * (eta[j_all] - eta[i_all]) / L_all
        denom = 1.0 + G * dt * n_edge ** 2 * torch.abs(q) / (hf ** (4.0 / 3.0) + 1e-10)
        q = torch.where(hf > MIN_DEPTH, num / denom, torch.zeros_like(q))
        q_cap = 0.9 * hf * torch.clamp(G * torch.clamp(hf, min=MIN_DEPTH), min=0.0) ** 0.5
        q = torch.clamp(q, -q_cap, q_cap)

        vol_edge = q * W_all * dt
        out_i = torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, torch.clamp(vol_edge, min=0.0))
        out_j = torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, torch.clamp(-vol_edge, min=0.0))
        outflow_vol = out_i + out_j
        available_vol = h * area
        scale = torch.clamp(available_vol / torch.clamp(outflow_vol, min=1e-12), max=1.0)
        scale_send = torch.where(vol_edge >= 0, scale[i_all], scale[j_all])
        vol_edge = vol_edge * scale_send

        dh = torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, -vol_edge / area[i_all])
        dh = dh + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, vol_edge / area[j_all])

        hf_b = torch.clamp(h[bnd_owner], min=0.0)
        q_out = torch.where(hf_b > MIN_DEPTH, hf_b * torch.clamp(G * torch.clamp(hf_b, min=MIN_DEPTH), min=0.0) ** 0.5,
                             torch.zeros_like(hf_b))
        out_vol = float((q_out * bnd_len * dt).sum().item())
        cum_outflow_vol += out_vol
        dh = dh + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, bnd_owner, -q_out * bnd_len * dt / area[bnd_owner])

        rain_ms = synthetic_rain_rate_ms(t_s, rain_duration_s, peak_mm_hr)
        # Horton decay computed inline with torch.exp — horton_rate() (shared with the CPU
        # solver above) calls np.exp(), which cannot accept an MPS-resident tensor without an
        # explicit .cpu() copy first (implicit numpy conversion of an MPS tensor raises). Kept
        # as a separate inline computation rather than editing horton_rate() itself, so the
        # CPU path (which calls it with real numpy arrays) is completely unaffected.
        inf_ms = fc_si + (f0_si - fc_si) * torch.exp(-k_si * t_s)
        Pe = torch.clamp(rain_ms - inf_ms, min=0.0)
        cum_rain_vol += rain_ms * dt * area_sum
        # Real bug caught via benchmark_gpu_solver.py: this originally summed min(rain,inf)
        # PER-TRIANGLE RATE without the per-triangle `area` factor the CPU version has (`np.sum
        # (np.minimum(rain_ms, inf_ms) * dt * area)`), i.e. computed a volume as if every
        # triangle had area=1 m^2. Site3_crop's ground triangles average ~0.3m^2 (native
        # ~0.88m decimate=1 resolution), so this inflated infiltration ~3x — confirmed as the
        # cause of a -93.8% mass-balance residual (rain=1430m^3-scale run, -111.6m^3 residual)
        # even though the depth field h itself matched the CPU run almost exactly (h_max
        # 0.020 vs 0.0199999...), which is what pointed at the volume ACCOUNTING rather than
        # the physics itself.
        cum_infil_vol += float((torch.minimum(torch.tensor(rain_ms, device=dev, dtype=f32), inf_ms)
                                 * area).sum().item()) * dt
        dh = dh + Pe * dt

        h = torch.clamp(h + dh, 0.0, MAX_DEPTH_CAP)
        n_capped = int((h >= MAX_DEPTH_CAP).sum().item())
        if n_capped:
            n_capped_total += n_capped

        h_roof_below = h[roof_below_tri]
        roof_excess = torch.clamp(h_roof_below - ROOF_POND_MAX_M, min=0.0)
        if bool((roof_excess > 0).any().item()):
            h[roof_below_tri] = h[roof_below_tri] - roof_excess
            excess_vol = roof_excess * area[roof_below_tri]
            h = h + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, roof_below_ground, excess_vol) / area

        if pond_outlet_area_m2 is not None:
            h_lake = h[lake_tri_idx]
            area_lake = area[lake_tri_idx]
            vol_lake = float((h_lake * area_lake).sum().item())
            h_lake_mean = vol_lake / lake_area if lake_area > 0 else 0.0
            if h_lake_mean > MIN_DEPTH:
                q_outlet = POND_OUTLET_CD * pond_outlet_area_m2 * (2 * G * h_lake_mean) ** 0.5
                vol_outlet = min(q_outlet * dt, vol_lake)
                if vol_outlet > 0:
                    frac = h_lake * area_lake / max(vol_lake, 1e-12)
                    h[lake_tri_idx] = h[lake_tri_idx] - (vol_outlet * frac) / area_lake
                    cum_outflow_vol += vol_outlet
                    cum_pond_outlet_vol += vol_outlet

        h_max = torch.maximum(h_max, h)

        if lake_mask is not None:
            lake_level_ts.append(
                float((h[:Tg][lake_mask_t] * lake_area_w).sum().item() / lake_area_w.sum().item()))
            lake_time_ts.append(t_s)

        if t_s - last_frame_t >= frame_interval_s:
            frames_h.append(h.to("cpu").numpy().astype(np.float32).copy())
            flux_rate = vol_edge / dt
            vx = (torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, flux_rate * edge_dir_x)
                  + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, flux_rate * edge_dir_x))
            vy = (torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, flux_rate * edge_dir_y)
                  + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, flux_rate * edge_dir_y))
            vz = (torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, flux_rate * edge_dir_z)
                  + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, flux_rate * edge_dir_z))
            w_accum = (torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, i_all, torch.abs(flux_rate))
                       + torch.zeros(T, dtype=f32, device=dev).scatter_add_(0, j_all, torch.abs(flux_rate)))
            speed = w_accum / (torch.clamp(h, min=MIN_DEPTH) * torch.clamp(area, min=1e-6) ** 0.5)
            dir_mag = (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5 + 1e-12
            vel = torch.stack([vx / dir_mag * speed, vy / dir_mag * speed, vz / dir_mag * speed], dim=1)
            frames_vel.append(vel.to("cpu").numpy().astype(np.float32))
            frame_times.append(t_s)
            last_frame_t = t_s

        if step % 50 == 0:
            sample_memory()   # cheap (no .item() sync — these calls query the allocator's own
                                # bookkeeping, not tensor data), sampled more often than the
                                # verbose print so a short scenario still gets real peak coverage

        if verbose and step % 200 == 0:
            n_wet = int((h > DEPTH_THR).sum().item())
            cur, drv = sample_memory()
            print(f"  [gpu] t={t_s:6.1f}s  dt={dt:.4f}s  rain={rain_ms*3600*1000:6.1f}mm/hr  "
                  f"h_max={float(h.max().item()):.3f}m  wet_tri={n_wet:6d}/{T}  "
                  f"mps={cur/1e6:.0f}/{drv/1e6:.0f}MB  "
                  f"[{t_s/total_s*100:5.1f}% {time.time()-t0:.0f}s]")

        t_s += dt
        step += 1

    if not frame_times or frame_times[-1] < t_s - 1e-6:
        frames_h.append(h.to("cpu").numpy().astype(np.float32).copy())
        frames_vel.append(frames_vel[-1] if frames_vel else np.zeros((T, 3), dtype=np.float32))
        frame_times.append(t_s)

    sample_memory()   # final sample, in case the true peak landed between step%50 checks
    # ru_maxrss is bytes on macOS (Linux reports KB instead — this project only runs on macOS,
    # so no platform branch needed, but worth noting if this is ever ported).
    process_peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return dict(
        h_final=h.to("cpu").numpy(), h_max=h_max.to("cpu").numpy(),
        frames_h=frames_h, frames_vel=frames_vel, frame_times=frame_times,
        n_steps=step, wall_s=time.time() - t0, n_depth_cap_hits=n_capped_total,
        mass_balance=dict(
            rain_vol_m3=cum_rain_vol, infil_vol_m3=cum_infil_vol,
            outflow_vol_m3=cum_outflow_vol, pond_outlet_vol_m3=cum_pond_outlet_vol,
            stored_vol_m3=float((h * area).sum().item()),
        ),
        lake_level_ts=lake_level_ts, lake_time_ts=lake_time_ts, lake_area_m2=lake_area,
        memory_stats=dict(
            mps_peak_current_allocated_mb=mem_peak_mps_current / 1e6,
            mps_peak_driver_allocated_mb=mem_peak_mps_driver / 1e6,
            mps_recommended_max_mb=(mem_recommended_max / 1e6) if mem_recommended_max else None,
            process_peak_rss_mb=process_peak_rss_bytes / 1e6,
            n_triangles=T, n_edges=n_edges,
        ),
    )


# ── Physics-driven tracer particles ───────────────────────────────────────────────────

def run_flow_tracers(ground, buildings, building_polys, mesh, frames_vel, frame_times,
                      rain_duration_s, peak_mm_hr, total_s, n_tracers=2500, seed=1,
                      force_roof_seed=False):
    """Physics-driven tracer particles — the direct real-physics replacement for
    droplet_flow_test.py's fixed-step kinematic walk. Motion comes from frames_vel (this
    solver's own reconstructed per-triangle velocity — see run_sim), not an arbitrary constant
    step: fast where real flow is fast (steep roofs, actively draining ground), slow/stopped
    where it's genuinely stalled (ponding, flat impervious road). Structurally mirrors
    droplet_flow_test.py's run_droplets_fused (same vectorized-by-surface-group control flow,
    same roof/ground split, same 3-way settle/runoff/still-moving outcome) — only the motion
    rule changes. Exports in the exact same DROP binary format, so the proven dropletFlow.js
    rendering (trail + animated point, 3-way outcome color) is reused completely unchanged.

    Each particle also gets a short synthetic sky-fall lead-in prepended before its real
    advected path — the piece the depth-field-only visualization never had — with landing
    TIME staggered across the real rain hyetograph (more particles start during peak rain), so
    particles are seen falling and landing continuously through playback, not all at t=0.
    """
    rng = np.random.default_rng(seed)
    building_id = mesh["building_id"]
    n_buildings = len(buildings)
    building_offset = np.array(
        [int(np.where(building_id == bi)[0][0]) for bi in range(n_buildings)], dtype=np.int64
    ) if n_buildings else np.zeros(0, dtype=np.int64)

    frame_times_arr = np.asarray(frame_times, dtype=np.float64)
    n_frames = len(frame_times)

    # ── Seed: same convention as run_droplets_fused — random within the ground bbox, kept
    # if it lands inside the ground mesh, building assigned by point-in-footprint test. ────
    # force_roof_seed (added for a single-droplet roof-shedding demo): rejection-sample directly
    # inside a real building footprint polygon instead of the whole ground bbox, so a small
    # n_tracers (e.g. 1) is guaranteed to start on a roof, not most likely land on open ground.
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)
    if force_roof_seed and building_polys:
        bminx, bminy, bmaxx, bmaxy = building_polys[0].bounds
        for poly in building_polys[1:]:
            b = poly.bounds
            bminx, bminy = min(bminx, b[0]), min(bminy, b[1])
            bmaxx, bmaxy = max(bmaxx, b[2]), max(bmaxy, b[3])
        accepted = []
        while len(accepted) < n_tracers:
            batch = rng.uniform([bminx, bminy], [bmaxx, bmaxy], size=(max(n_tracers * 8, 64), 2))
            for pt in batch:
                if any(poly.contains(Point(pt)) for poly in building_polys):
                    accepted.append(pt)
                    if len(accepted) >= n_tracers:
                        break
        cand = np.array(accepted[:n_tracers])
    else:
        cand = rng.uniform([gxmin, gymin], [gxmax, gymax], size=(n_tracers * 2, 2))
        on_ground_init = ground.simplex_of(cand) >= 0
        cand = cand[on_ground_init][:n_tracers]
    n = len(cand)

    building_of = np.full(n, -1, dtype=np.int32)   # -1 = ground
    for bi, poly in enumerate(building_polys):
        inside = np.array([poly.contains(Point(p)) for p in cand])
        building_of[inside] = bi

    # ── Stagger landing/start times across the real rain hyetograph (rejection sampling
    # against the trapezoidal shape — cheap and exact enough for this many particles). ─────
    start_time = np.zeros(n, dtype=np.float64)
    filled = 0
    peak_rate = peak_mm_hr / 1000 / 3600
    while filled < n:
        batch = rng.uniform(0, rain_duration_s, size=n - filled)
        rate = np.array([synthetic_rain_rate_ms(t, rain_duration_s, peak_mm_hr) for t in batch])
        accept = rng.uniform(0, peak_rate, size=len(batch)) < rate
        acc_t = batch[accept]
        take = min(len(acc_t), n - filled)
        start_time[filled:filled + take] = acc_t[:take]
        filled += take

    xy = cand.copy()
    active = np.zeros(n, dtype=bool)         # currently advecting (past its start time)
    finished = np.zeros(n, dtype=bool)       # settled / left mesh / hit step cap
    settle_reason = np.full(n, "", dtype=object)
    slow_streak = np.zeros(n, dtype=np.int32)
    steps_taken = np.zeros(n, dtype=np.int32)
    paths = [[] for _ in range(n)]

    t_s = 0.0
    max_t = min(total_s, rain_duration_s + TRACER_MAX_STEPS * TRACER_SUB_DT + 5.0)

    while t_s <= max_t and not finished.all():
        frame_idx = int(np.clip(np.searchsorted(frame_times_arr, t_s, side="right") - 1,
                                 0, n_frames - 1))
        vel = frames_vel[frame_idx]

        active[(~active) & (~finished) & (start_time <= t_s)] = True
        idx_active = np.where(active & ~finished)[0]

        if len(idx_active):
            # Ground-state tracers
            gmask = np.zeros(n, dtype=bool)
            gmask[idx_active] = building_of[idx_active] < 0
            if gmask.any():
                gi = np.where(gmask)[0]
                s = ground.simplex_of(xy[gi])
                left = s < 0
                settle_reason[gi[left]] = "left_mesh"
                finished[gi[left]] = True
                active[gi[left]] = False
                stay = ~left
                if stay.any():
                    still = gi[stay]
                    v = vel[s[stay]]                      # ground local idx == global idx
                    speed = np.linalg.norm(v[:, :2], axis=1)
                    z_now = ground.z_at(xy[still], s[stay])
                    for k, p in enumerate(still):
                        paths[p].append((float(xy[p, 0]), float(xy[p, 1]), float(z_now[k])))
                    slow = speed < TRACER_SETTLE_SPEED
                    slow_streak[still[slow]] += 1
                    slow_streak[still[~slow]] = 0
                    settled_now = slow_streak[still] >= TRACER_SETTLE_PATIENCE
                    settle_reason[still[settled_now]] = "local_min"
                    finished[still[settled_now]] = True
                    active[still[settled_now]] = False
                    move = still[~settled_now]
                    xy[move] += v[~settled_now][:, :2] * TRACER_SUB_DT
                    steps_taken[move] += 1

            # Roof-state tracers, grouped by building — same eave/anti-pond handoff logic as
            # the physics itself (roof_below), triggered by leaving the roof mesh OR stalling.
            for bi, surf in enumerate(buildings):
                bmask = np.zeros(n, dtype=bool)
                bmask[idx_active] = building_of[idx_active] == bi
                if not bmask.any():
                    continue
                bi_idx = np.where(bmask)[0]
                s = surf.simplex_of(xy[bi_idx])
                on_roof = s >= 0
                gidx = np.where(on_roof, building_offset[bi] + np.maximum(s, 0), 0)

                v = np.zeros((len(bi_idx), 3))
                v[on_roof] = vel[gidx[on_roof]]
                speed = np.linalg.norm(v[:, :2], axis=1)

                z_now = np.full(len(bi_idx), np.nan)
                z_now[on_roof] = surf.z_at(xy[bi_idx][on_roof], s[on_roof])
                for k, p in enumerate(bi_idx):
                    if np.isfinite(z_now[k]):
                        paths[p].append((float(xy[p, 0]), float(xy[p, 1]), float(z_now[k])))

                slow = on_roof & (speed < TRACER_SETTLE_SPEED)
                slow_streak[bi_idx[slow]] += 1
                slow_streak[bi_idx[on_roof & ~slow]] = 0
                stuck = on_roof & (slow_streak[bi_idx] >= TRACER_SETTLE_PATIENCE)
                handoff = (~on_roof) | stuck
                if handoff.any():
                    building_of[bi_idx[handoff]] = -1
                    slow_streak[bi_idx[handoff]] = 0
                move = on_roof & ~stuck
                if move.any():
                    xy[bi_idx[move]] += v[move][:, :2] * TRACER_SUB_DT
                    steps_taken[bi_idx[move]] += 1

        capped = (active & ~finished) & (steps_taken >= TRACER_MAX_STEPS)
        settle_reason[capped] = "max_steps"
        finished[capped] = True
        t_s += TRACER_SUB_DT

    settle_reason[active & ~finished] = "max_steps"

    # ── Sky-fall lead-in, prepended to each landed particle's real advected path ──────────
    final_paths = []
    for p in range(n):
        real_path = paths[p]
        if len(real_path) < 2:
            final_paths.append(real_path)
            continue
        lx, ly, lz = real_path[0]
        fall = [(lx, ly, lz + TRACER_SKY_HEIGHT_M * (1 - k / TRACER_SKY_FALL_STEPS))
                for k in range(TRACER_SKY_FALL_STEPS)]
        final_paths.append(fall + real_path)

    n_local_min = int((settle_reason == "local_min").sum())
    n_left_mesh = int((settle_reason == "left_mesh").sum())
    n_max_steps = int((settle_reason == "max_steps").sum())
    print(f"  Flow tracers: {n} seeded  local_min={n_local_min} left_mesh={n_left_mesh} "
          f"max_steps={n_max_steps}")

    # Export in ascending start_time order — the viewer reuses dropletFlow.js's existing
    # setRainSpread() to stagger playback start by array INDEX (linear, no per-particle time
    # field in the DROP format), so sorting here is what makes that approximate the real
    # rain-weighted stagger instead of a uniform one, with zero client-side format changes and
    # no per-particle "waiting in the sky" padding bloating the file.
    order = np.argsort(start_time)
    final_paths = [final_paths[i] for i in order]
    settle_reason = settle_reason[order]

    return final_paths, settle_reason


# ── Export ──────────────────────────────────────────────────────────────────────────

def export_frames_bin(mesh, frames_h, frame_times, out_path):
    T = mesh["T"]
    with open(out_path, "wb") as fh:
        fh.write(b"MSWE")
        fh.write(struct.pack("<II", T, len(frames_h)))
        fh.write(mesh["is_roof"].astype(np.uint8).tobytes())
        fh.write(np.array(frame_times, dtype=np.float32).tobytes())
        for frame in frames_h:
            fh.write(frame.tobytes())
    mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  {os.path.basename(out_path)}: {T} triangles x {len(frames_h)} frames ({mb:.1f} MB)")


def export_tracer_paths_drop(paths, settle_reason, geo_meta, out_path):
    """Same DROP binary format droplet_flow_test.py's main() writes (b'DROP' + per-droplet
    uint32 n_points, float32[n_points*3] scene-space xyz, uint8 settle_reason_code) — so
    dropletFlow.js's existing, proven rendering can load these physics-driven tracer paths
    with zero changes, just a different URL."""
    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]
    reason_code = {"local_min": 0, "left_mesh": 1, "max_steps": 2}

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
    print(f"  {os.path.basename(out_path)}: {n_written} tracer paths written  ({kb:.0f} KB)")


def export_heightmap_bin(mesh, geo_meta, out_path, cell_m=0.25):
    """Nearest-neighbor raster of the fused mesh's surface height, in scene-space (x,y,z)
    convention matching terrain.js — lets the viewer's falling-raindrop particles detect
    where they hit the ground/roof surface via a cheap 2D lookup instead of raycasting a
    ~220k-triangle mesh every frame."""
    from scipy.spatial import cKDTree
    xy, z = mesh["xy"], mesh["z"]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    nx = int(np.ceil((xmax - xmin) / cell_m)) + 1
    ny = int(np.ceil((ymax - ymin) / cell_m)) + 1
    gx = xmin + np.arange(nx) * cell_m
    gy = ymin + np.arange(ny) * cell_m
    GX, GY = np.meshgrid(gx, gy)
    tree = cKDTree(xy)
    _, idx = tree.query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    Z = z[idx].reshape(GY.shape)

    ox, oy = geo_meta["origin_x"], geo_meta["origin_y"]
    w, h_, z_min = geo_meta["width_m"], geo_meta["height_m"], geo_meta["z_min"]
    sx0, sx1 = xmin - ox - w / 2, xmax - ox - w / 2
    sz0, sz1 = oy + h_ / 2 - ymin, oy + h_ / 2 - ymax   # y increases -> sz decreases
    sy = (Z - z_min) * VERT_EXAG

    with open(out_path, "wb") as fh:
        fh.write(b"HMAP")
        fh.write(struct.pack("<iiffff", nx, ny, sx0, sx1, sz0, sz1))
        fh.write(sy.astype(np.float32).tobytes())
    kb = os.path.getsize(out_path) / 1024
    print(f"  {os.path.basename(out_path)}: {nx}x{ny} @ {cell_m}m ({kb:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES), default="site1",
                     help="named test site (see lidar/test_sites.py)")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--radius_km", type=float, default=None)
    ap.add_argument("--peak-rain-mm-hr", type=float, default=100.0)
    ap.add_argument("--rain-duration-min", type=float, default=4.0)
    ap.add_argument("--total-min", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--frame-interval-s", type=float, default=3.0)
    ap.add_argument("--n-tracers", type=int, default=2500)
    ap.add_argument("--force-roof-seed", action="store_true",
                     help="seed all tracers directly on a building roof instead of randomly "
                          "over the whole ground bbox -- for a single-droplet (--n-tracers 1) "
                          "roof-shedding demo, where random ground-bbox seeding would almost "
                          "certainly miss the roof entirely")
    args = ap.parse_args()

    site = get_site(args.site)
    lat = args.lat if args.lat is not None else site["lat"]
    lon = args.lon if args.lon is not None else site["lon"]
    radius_km = args.radius_km if args.radius_km is not None else site["radius_km"]
    suffix = "" if args.site == "site1" else f"_{args.site}"
    pond_id3dhp = site.get("pond_id3dhp")

    # Data-source path overrides — added 2026-07-27 for site3 (see droplet_flow_test.py's
    # main() for the identical reasoning). site.get(...) is None for site1/site2, so every
    # function below falls back to its original module-level constant — no behavior change
    # for the existing sites.
    dem_cond_path = site.get("dem_cond_path")
    buildings_path = site.get("buildings_path")
    roads_path = site.get("roads_path")
    mukey_map_path = site.get("mukey_map_path")
    mukey_legend_path = site.get("mukey_legend_path")
    soil_json_path = site.get("soil_json_path")
    nlcd_path = site.get("nlcd_path")

    print("=" * 70)
    print(f"Unstructured-mesh 3D local-inertial shallow-water solver  [{args.site}: {site['label']}]")
    print("=" * 70)

    lon_min, lat_min, lon_max, lat_max = bbox_from_center(lat, lon, radius_km)

    print("\n[1/6] Building ground surface from the DEM …")
    ground = build_ground_surface(lon_min, lat_min, lon_max, lat_max, dem_cond_path=dem_cond_path,
                                   decimate=site.get("ground_decimate", 1))

    print("\n[2/6] Loading LiDAR building points + building roof meshes …")
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

    print("\n[3/6] Fusing ground + roof meshes into one edge-adjacency system "
          "(incl. eave-to-ground coupling edges) …")
    mesh = build_combined_mesh(ground, buildings)
    print(f"  {mesh['T']:,} triangles total ({mesh['Tg']:,} ground + {mesh['T']-mesh['Tg']:,} roof)  "
          f"{len(mesh['edges']['i']):,} flux edges (ground-ground + roof-roof + roof-eave-to-ground)")

    print("\n[4/6] Classifying impervious ground (OSM roads) + water body (if this site has one) …")
    ground_xy = mesh["xy"][:mesh["Tg"]]
    ground_impervious_mask = compute_ground_impervious_mask(ground_xy, roads_path=roads_path)
    lake_mask = compute_lake_mask(ground_xy, pond_id3dhp)
    print(f"  {int(ground_impervious_mask.sum())}/{mesh['Tg']} ground triangles under a road "
          f"({100*ground_impervious_mask.sum()/mesh['Tg']:.1f}%) — zero infiltration")
    if lake_mask is not None:
        lake_area_ha = float(mesh["area"][:mesh["Tg"]][lake_mask].sum()) / 1e4
        print(f"  {int(lake_mask.sum())}/{mesh['Tg']} ground triangles are the water body "
              f"({lake_area_ha:.3f} ha) — zero infiltration, level tracked over time")
    else:
        print("  no water body at this site")

    ground_horton = load_spatial_horton_points(ground_xy, mukey_map_path=mukey_map_path,
                                                mukey_legend_path=mukey_legend_path,
                                                soil_json_path=soil_json_path)
    if ground_horton is None:
        print("  Horton: soil survey files not found — falling back to uniform project-wide mean")

    already_hard = ground_impervious_mask.copy()
    if lake_mask is not None:
        already_hard |= lake_mask
    ground_nlcd_grade = load_nlcd_impervious_fraction_points(ground_xy, already_hard,
                                                              nlcd_path=nlcd_path)

    print(f"\n[5/7] Running solver: {args.peak_rain_mm_hr:.0f}mm/hr peak, "
          f"{args.rain_duration_min:.1f}min rain, {args.total_min:.1f}min total …")
    pond_outlet_d = POND_OUTLET_ORIFICE_D_M if lake_mask is not None else None
    result = run_sim(mesh, dt_target=args.dt, total_s=args.total_min * 60,
                      rain_duration_s=args.rain_duration_min * 60,
                      peak_mm_hr=args.peak_rain_mm_hr, frame_interval_s=args.frame_interval_s,
                      ground_impervious_mask=ground_impervious_mask, lake_mask=lake_mask,
                      ground_horton=ground_horton, ground_nlcd_grade=ground_nlcd_grade,
                      pond_outlet_diameter_m=pond_outlet_d)

    mb = result["mass_balance"]
    residual = mb["rain_vol_m3"] - mb["infil_vol_m3"] - mb["outflow_vol_m3"] - mb["stored_vol_m3"]
    residual_pct = 100 * residual / mb["rain_vol_m3"] if mb["rain_vol_m3"] > 0 else 0.0
    print(f"\n  Done: {result['n_steps']} steps in {result['wall_s']:.1f}s wall time  "
          f"(depth-cap safety rail triggered {result['n_depth_cap_hits']} triangle-steps)")
    print(f"  Mass balance (m^3): rain={mb['rain_vol_m3']:.3f}  infil={mb['infil_vol_m3']:.3f}  "
          f"outflow={mb['outflow_vol_m3']:.3f}  stored={mb['stored_vol_m3']:.3f}  "
          f"residual={residual:.4f} ({residual_pct:.2f}%)")
    if mb.get("pond_outlet_vol_m3", 0.0) > 0:
        print(f"  Engineered pond outlet: {mb['pond_outlet_vol_m3']:.4f} m^3 "
              f"({100*mb['pond_outlet_vol_m3']/max(mb['outflow_vol_m3'],1e-9):.1f}% of total "
              f"outflow) — {POND_OUTLET_ORIFICE_D_M*100:.0f}cm orifice, Cd={POND_OUTLET_CD}")
    h_max = result["h_max"]
    print(f"  Ground h_max: mean(wet)={h_max[~mesh['is_roof']][h_max[~mesh['is_roof']]>DEPTH_THR].mean() if (h_max[~mesh['is_roof']]>DEPTH_THR).any() else 0:.3f}m  "
          f"max={h_max[~mesh['is_roof']].max():.3f}m")
    print(f"  Roof h_max:   mean(wet)={h_max[mesh['is_roof']][h_max[mesh['is_roof']]>DEPTH_THR].mean() if (h_max[mesh['is_roof']]>DEPTH_THR).any() else 0:.3f}m  "
          f"max={h_max[mesh['is_roof']].max():.3f}m")

    lake_summary = None
    if lake_mask is not None and result["lake_level_ts"]:
        lake_level = np.array(result["lake_level_ts"])
        lake_summary = {
            "area_m2": result["lake_area_m2"],
            "level_rise_final_m": float(lake_level[-1]),
            "level_rise_peak_m": float(lake_level.max()),
            "level_rise_peak_cm": float(lake_level.max() * 100),
            "volume_added_peak_m3": float(lake_level.max() * result["lake_area_m2"]),
        }
        print(f"\n  Lake level: peak rise {lake_summary['level_rise_peak_cm']:.2f} cm "
              f"({lake_summary['volume_added_peak_m3']:.2f} m^3 over {result['lake_area_m2']:.0f} m^2), "
              f"final rise {lake_summary['level_rise_final_m']*100:.2f} cm")
        with open(os.path.join(OUT_DIR, f"lake_hydrograph{suffix}.csv"), "w") as fh:
            fh.write("time_s,level_rise_m,level_rise_cm\n")
            for t, lv in zip(result["lake_time_ts"], result["lake_level_ts"]):
                fh.write(f"{t:.2f},{lv:.5f},{lv*100:.3f}\n")
        print(f"  lake_hydrograph{suffix}.csv  ({len(result['lake_time_ts'])} rows)")

    # ── One-house sanity check (meeting note: "reduce to one house... cubic centimeter") ──
    # Reuses this same full run rather than a separate reduced-scope simulation. Picks the
    # SMALLEST-roof-area meshed building, not just building_id==0 — a raw box crop can catch
    # non-house structures too (confirmed at site2: one building in-box is a ~16,900 m^2
    # school/commercial footprint, not a house; picking by area avoids landing on it).
    house_summary = None
    n_bldg_meshed = int(mesh["building_id"].max()) + 1 if (mesh["building_id"] >= 0).any() else 0
    if n_bldg_meshed > 0:
        roof_areas_by_bldg = [float(mesh["area"][mesh["building_id"] == bi].sum())
                               for bi in range(n_bldg_meshed)]
        house_bi = int(np.argmin(roof_areas_by_bldg))
        b0_mask = mesh["building_id"] == house_bi
        b0_area = mesh["area"][b0_mask]
        vol_final_m3 = float(np.sum(result["h_final"][b0_mask] * b0_area))
        vol_peak_m3 = float(np.sum(h_max[b0_mask] * b0_area))
        house_summary = {
            "building_index": house_bi,
            "roof_area_m2": float(b0_area.sum()),
            "n_roof_triangles": int(b0_mask.sum()),
            "volume_final_cm3": vol_final_m3 * 1e6,
            "volume_peak_cm3": vol_peak_m3 * 1e6,
            "volume_peak_liters": vol_peak_m3 * 1000,
        }
        print(f"\n  Single-house check (smallest meshed building, #{house_bi}, "
              f"{house_summary['roof_area_m2']:.1f} m^2 roof): "
              f"peak standing water = {house_summary['volume_peak_cm3']:,.0f} cm^3 "
              f"({house_summary['volume_peak_liters']:.2f} L)")

    n_tracers = args.n_tracers
    print(f"\n[6/7] Advecting physics-driven flow tracers ({n_tracers} particles, real "
          f"velocity-field-driven — replaces droplet_flow_test.py's fixed-step walk"
          f"{', forced onto a roof' if args.force_roof_seed else ''}) …")
    tracer_paths, tracer_reason = run_flow_tracers(
        ground, buildings, building_polys, mesh, result["frames_vel"], result["frame_times"],
        rain_duration_s=args.rain_duration_min * 60, peak_mm_hr=args.peak_rain_mm_hr,
        total_s=args.total_min * 60, n_tracers=n_tracers, force_roof_seed=args.force_roof_seed,
    )

    print("\n[7/7] Exporting …")
    lidar_data_dir = os.path.join(PROJ_DIR, "lidar", "data")
    export_frames_bin(mesh, result["frames_h"], result["frame_times"],
                       os.path.join(OUT_DIR, f"swe_mesh_frames{suffix}.bin"))
    with open(GEO_META) as fh:
        geo_meta = json.load(fh)
    export_heightmap_bin(mesh, geo_meta,
                          os.path.join(lidar_data_dir, f"swe_surface_heightmap{suffix}.bin"))
    export_tracer_paths_drop(tracer_paths, tracer_reason, geo_meta,
                              os.path.join(OUT_DIR, f"flow_tracer_paths{suffix}.bin"))

    summary = {
        "site": args.site, "site_label": site["label"],
        "n_triangles": mesh["T"], "n_ground_triangles": mesh["Tg"],
        "n_roof_triangles": mesh["T"] - mesh["Tg"], "n_buildings": len(buildings),
        "n_edges": len(mesh["edges"]["i"]), "n_frames": len(result["frame_times"]),
        "n_steps": result["n_steps"], "wall_s": result["wall_s"],
        "peak_rain_mm_hr": args.peak_rain_mm_hr, "rain_duration_min": args.rain_duration_min,
        "total_min": args.total_min,
        "ground_h_max_m": float(h_max[~mesh["is_roof"]].max()),
        "roof_h_max_m": float(h_max[mesh["is_roof"]].max()),
        "mass_balance_m3": mb, "mass_balance_residual_pct": residual_pct,
        "n_impervious_road_triangles": int(ground_impervious_mask.sum()),
        "lake": lake_summary,
        "single_house_check": house_summary,
        "flow_tracers": {
            "n_seeded": len(tracer_paths),
            "n_local_min": int((tracer_reason == "local_min").sum()),
            "n_left_mesh": int((tracer_reason == "left_mesh").sum()),
            "n_max_steps": int((tracer_reason == "max_steps").sum()),
        },
        "test_area": {"lat": lat, "lon": lon, "radius_km": radius_km},
    }
    with open(os.path.join(OUT_DIR, f"swe_mesh_summary{suffix}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  swe_mesh_summary{suffix}.json")


if __name__ == "__main__":
    main()
