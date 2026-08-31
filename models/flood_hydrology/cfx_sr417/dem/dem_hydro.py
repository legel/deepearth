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
  dem_conditioned.tif     — DEM after stream burning + breach depression filling (what the
                            shallow-water solver reads)
  dem_flat_resolved.tif   — + fill_pits/fill_depressions/resolve_flats (what D8 and HAND
                            are actually computed on)
  flow_dir.tif            — D8 flow direction (pysheds encoding)
  flow_accum.tif          — upstream cell count (log-scaled for vis)
  stream_network.geojson  — vectorised stream cells (EPSG:4326, accum > threshold)
  hand.tif                — Height Above Nearest Drainage (metres)
  watershed.geojson       — pour-point watershed polygon (EPSG:4326)
  hydro_summary.json      — threshold used, stream extent, HAND stats

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


def apply_site(name):
    """Point every path constant at `name`'s own data tree, via the shared site registry.

    The module-level constants above are the MAIN AOI's. Any other site conditioned with them
    defaulted silently onto main-AOI inputs — which is exactly what happened to site3: it took
    the main AOI's 6 Shingle Creek flowlines, ~34 km outside its own DEM, so the burn mask came
    out empty and stream burning became a no-op nobody noticed. site_registry.py exists to stop
    that class of bug; this wires dem_hydro.py into it.

    Returns (dem_path, pour_lat, pour_lon). The pour point is the site's gauge when it has one
    (the hydrologically meaningful outlet), otherwise its box centre.
    """
    global DATA_DIR, HYDRO_DIR, HYDRO_GEO
    sys.path.insert(0, os.path.dirname(BASE_DIR))
    from site_registry import get_site

    site      = get_site(name)
    root      = site["data_root"]
    DATA_DIR  = os.path.join(root, "dem", "data")
    HYDRO_DIR = os.path.join(DATA_DIR, "hydro")
    HYDRO_GEO = os.path.join(root, "hydrography", "data")
    os.makedirs(HYDRO_DIR, exist_ok=True)

    cands = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".tif"))
    if not cands:
        sys.exit(f"No DEM found in {DATA_DIR} — run dem/dem_download.py --site {name}")
    pick = next((f for f in cands if f.endswith("_1m.tif")), cands[0])
    if len(cands) > 1 and not pick.endswith("_1m.tif"):
        print(f"  NOTE: {len(cands)} DEMs in {DATA_DIR}; using {pick}. Override with --dem.")

    lat = site.get("gauge_lat", site["lat"])
    lon = site.get("gauge_lon", site["lon"])
    print(f"Site    : {name} — {site.get('label', '')}")
    print(f"  data root  : {root}")
    print(f"  flowlines  : {os.path.join(HYDRO_GEO, '3dhp_flowlines.geojson')}")
    print(f"  pour point : ({lat}, {lon})"
          f"{'  [gauge ' + str(site['gauge_site_no']) + ']' if 'gauge_lat' in site else '  [box centre]'}")
    return os.path.join(DATA_DIR, pick), lat, lon

# Stream initiation is a CONTRIBUTING-AREA threshold, expressed in m², converted to a cell
# count at runtime from the DEM's own resolution. It used to be a bare cell count, which is
# resolution-dependent and silently wrong on any grid but the one it was tuned for: 50,000 cells
# is 38,391 m² on the main AOI's 0.876 m grid but only 766 m² on a 0.875 m grid entered as
# `--acc-threshold 1000`. site3 was conditioned that way, giving 1,719,908 stream cells (2.82 %
# of the domain, 23x the main AOI's 0.12 %) and a degenerate HAND surface with 77.2 % of cells
# below 1 m. 38,400 m² reproduces the main AOI's calibrated 50,000 cells to within 0.02 %.
DEFAULT_ACC_AREA_M2 = 38400.0  # ≈3.84 ha — resolves Shingle Creek's main channel and major
                               # tributaries while excluding micro-drainage swales.
DEFAULT_ACC_THRESHOLD = None   # None → derive from DEFAULT_ACC_AREA_M2 and the DEM's resolution
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

def _network_order(lines, tol):
    """Topological order over flowlines, using 3DHP's digitized direction.

    3DHP's `flowdirectionlabel` states vertices run downslope in digitized order, so each
    LineString's own vertex order IS the downstream direction — authoritative, and better than
    inferring direction from a DEM this flat. Lines are chained where one's last node meets
    another's first. Returns (order, node_of, ends) for a downstream sweep.
    """
    from collections import defaultdict, deque

    node, ends = {}, []
    def nid(pt):
        k = (round(pt[0] / tol), round(pt[1] / tol))
        return node.setdefault(k, len(node))
    for ln in lines:
        cs = list(ln.coords)
        ends.append((nid(cs[0]), nid(cs[-1])))

    outgoing = defaultdict(list)          # node -> lines starting there
    for i, (a, _) in enumerate(ends):
        outgoing[a].append(i)
    indeg = [0] * len(lines)
    succ  = defaultdict(list)
    for i, (_, z) in enumerate(ends):
        for j in outgoing.get(z, []):
            succ[i].append(j)
            indeg[j] += 1

    q = deque(i for i in range(len(lines)) if indeg[i] == 0)
    order = []
    while q:
        i = q.popleft(); order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    if len(order) < len(lines):           # cycle (shouldn't occur) — append the rest
        order += [i for i in range(len(lines)) if i not in set(order)]
    return order, ends


def burn_streams(z, profile, transform, dem_crs, burn_depth_m, flow_path=None,
                 mode="gradient", min_slope=None, step_m=5.0, max_carve_mult=2.0):
    """Carve the 3DHP flowlines into the DEM, enforcing a downstream gradient.

    flow_path defaults to HYDRO_GEO, which is the MAIN AOI's hydrography directory. Any site
    with its own DEM must pass its own flowlines — see apply_site(). site3 was conditioned with
    this defaulted, so it loaded the main AOI's 6 Shingle Creek lines, ~34 km outside its own
    DEM; the burn mask came out empty and stream burning silently became a no-op. The
    empty-mask check below makes that loud.

    mode="constant" is the original behaviour: subtract burn_depth_m uniformly along the
    channel. MEASURED to not work on this terrain. A constant-depth carve on ground with
    ~14 m of relief over 6.8 km is a flat-bottomed ditch: it lowers the channel but imposes no
    direction, so D8 cannot route along it. After a correct constant-depth burn of site3, the
    largest flow accumulation anywhere in the 46.78 km² domain was 0.99 km², the median
    accumulation along the burned channel itself was 5 cells, and only 10 % of burned cells
    reached the stream threshold — drainage fragmented into dozens of disconnected pockets
    instead of forming a network.

    mode="gradient" (default) additionally enforces monotonic descent downstream: each point
    is carved to at least burn_depth_m below ground AND at least min_slope x distance below the
    point upstream of it. Where the DEM already falls faster than min_slope the natural profile
    is kept, so this only intervenes on the flats where D8 has no information.

    min_slope=None (default) MEASURES the enforced gradient per stream order from the raw DEM,
    rather than assuming one. A hand-picked 1e-4 (0.1 m/km) "just enough to break ties" was
    tried first and is wrong by more than an order of magnitude: site3's network actually falls
    at 1.90e-3 m/m overall (order 1 2.76e-3, order 2 1.88e-3, order 3 5.61e-4). Because channel
    velocity goes as sqrt(S), imposing 1e-4 on the flat reaches throttled them 2.4-5x, and the
    simulated outflow peak landed 25.9 h after the rain peak against the gauge's observed 4.5 h
    — a routing failure, not a storage one (channel storage is only ~9 % of event runoff).
    Falling at the rate this channel is measured to fall elsewhere is the defensible choice.
    """
    if flow_path is None:
        flow_path = os.path.join(HYDRO_GEO, "3dhp_flowlines.geojson")
    if not os.path.exists(flow_path):
        print("  3DHP flowlines not found — skipping stream burning")
        return z

    gdf = gpd.read_file(flow_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("epsg:4326")
    gdf_proj = gdf.to_crs(dem_crs)

    rows, cols = z.shape
    cell_m = (abs(transform.a) + abs(transform.e)) / 2
    buf    = cell_m * 2

    burn_mask = rasterize(
        [(g, 1) for g in gdf_proj.geometry.buffer(buf) if g is not None],
        out_shape=(rows, cols), transform=transform, fill=0, dtype=np.uint8,
    ).astype(bool)
    n_burned = int(burn_mask.sum())
    if n_burned == 0:
        fb = gdf_proj.total_bounds
        raise SystemExit(
            f"\n  BURN MASK IS EMPTY — {len(gdf)} flowlines rasterised to 0 cells.\n"
            f"    flowlines : {flow_path}\n"
            f"    their extent ({dem_crs}) : {fb}\n"
            f"    DEM extent  ({dem_crs}) : {rasterio.transform.array_bounds(rows, cols, transform)}\n"
            f"  These do not overlap. Pass the flowlines matching this DEM via --flowlines "
            f"(or --site).\n"
            f"  (Continuing would produce a dem_burned.tif identical to the raw DEM.)")

    if mode == "constant":
        z_b = z.copy()
        z_b[burn_mask & np.isfinite(z)] -= burn_depth_m
        print(f"  Stream burned (constant {burn_depth_m} m): {n_burned:,} cells  "
              f"(flowlines={len(gdf)}, buffer={buf:.1f} m)")
        return z_b

    # ── gradient-enforced carve ──────────────────────────────────────────────
    from shapely.geometry import LineString

    def _natural_gradient(sel_idx):
        """Measured fall per unit length over the given reaches, from the raw DEM."""
        inv_t = ~transform
        D = L = 0.0
        for i in sel_idx:
            ln = gdf_proj.geometry.iloc[i]
            if ln is None or ln.geom_type != "LineString" or ln.length <= 0:
                continue
            zs = []
            for dd in np.linspace(0.0, ln.length, 50):
                pt = ln.interpolate(float(dd))
                c, r = inv_t * (pt.x, pt.y)
                r, c = int(r), int(c)
                if 0 <= r < rows and 0 <= c < cols and np.isfinite(z[r, c]):
                    zs.append(float(z[r, c]))
            if len(zs) > 2:
                D += zs[0] - zs[-1]
                L += ln.length
        return (D / L) if (L > 0 and D > 0) else None

    keep  = [i for i, g in enumerate(gdf_proj.geometry)
             if g is not None and g.geom_type == "LineString"]
    lines = [gdf_proj.geometry.iloc[i] for i in keep]

    # Carve depth scales with stream order. A 1st-order headwater swale is not as deep as the
    # 3rd-order main stem, and carving every tributary to the full depth turns 22 km of minor
    # channel into storage the runoff has to fill before any of it can leave: measured, a
    # uniform 1.5 m floor delayed the outflow peak to 25.9 h after the rain peak against the
    # gauge's observed 4.5 h. Depth ∝ order is the coarse form of the standard hydraulic-
    # geometry result that channel depth grows with drainage area (Leopold & Maddock 1953);
    # 3DHP carries `streamorder` per reach, so this uses measured order rather than a guess.
    if "streamorder" in gdf_proj.columns and gdf_proj["streamorder"].notna().all():
        so     = gdf_proj["streamorder"].to_numpy(float)
        so_max = float(np.nanmax(so))
        depths = [burn_depth_m * float(so[i]) / so_max for i in keep]
        by_ord = {int(o): burn_depth_m * o / so_max for o in sorted(set(so))}
        print(f"    depth by stream order: "
              + ", ".join(f"{k}->{v:.2f} m" for k, v in sorted(by_ord.items())))
    else:
        depths = [burn_depth_m] * len(lines)
        print(f"    no streamorder attribute — uniform {burn_depth_m} m carve")

    # Enforced gradient, measured per stream order unless the caller pinned one.
    FLOOR = 1e-4
    if min_slope is None:
        if "streamorder" in gdf_proj.columns and gdf_proj["streamorder"].notna().all():
            so_all = gdf_proj["streamorder"].to_numpy(float)
            grad_by_ord = {}
            for o in sorted(set(so_all)):
                gmeas = _natural_gradient([i for i in range(len(gdf_proj)) if so_all[i] == o])
                grad_by_ord[o] = max(gmeas, FLOOR) if gmeas else FLOOR
            slopes = [grad_by_ord[so_all[i]] for i in keep]
            print("    enforced gradient, measured per order: "
                  + ", ".join(f"{int(k)}->{v:.2e}" for k, v in sorted(grad_by_ord.items())))
        else:
            gmeas = _natural_gradient(range(len(gdf_proj)))
            slopes = [max(gmeas, FLOOR) if gmeas else FLOOR] * len(lines)
            print(f"    enforced gradient, measured network-wide: {slopes[0]:.2e} m/m")
    else:
        slopes = [min_slope] * len(lines)
        print(f"    enforced gradient: {min_slope:.2e} m/m (explicit)")

    order, ends = _network_order(lines, tol=cell_m)

    inv = ~transform
    def dem_at(pt):
        c, r = inv * (pt[0], pt[1])
        r, c = int(r), int(c)
        if 0 <= r < rows and 0 <= c < cols and np.isfinite(z[r, c]):
            return float(z[r, c])
        return np.nan

    node_z   = {}                 # node id -> carved elevation already assigned
    pieces   = []                 # (sub-segment geometry, carved elevation)
    for li in order:
        ln = lines[li]
        a, zz = ends[li]
        depth_li = depths[li]
        slope_li = slopes[li]
        # densify so the carved profile steps smoothly rather than per-whole-line
        n_steps = max(2, int(np.ceil(ln.length / step_m)) + 1)
        ds      = np.linspace(0.0, ln.length, n_steps)
        pts     = [ln.interpolate(float(d)) for d in ds]

        z_prev = node_z.get(a, np.nan)
        if not np.isfinite(z_prev):
            g0 = dem_at((pts[0].x, pts[0].y))
            z_prev = (g0 - depth_li) if np.isfinite(g0) else 0.0
        node_z[a] = min(node_z.get(a, np.inf), z_prev)

        carved = [z_prev]
        for k in range(1, len(pts)):
            seg   = ds[k] - ds[k - 1]
            gnd   = dem_at((pts[k].x, pts[k].y))
            floor = carved[-1] - slope_li * seg      # must fall going downstream
            target = min(gnd - depth_li, floor) if np.isfinite(gnd) else floor
            # Clamp to a maximum depth below ground. Without this the profile RATCHETS: the
            # moment the slope floor dips under the ground-following line it wins at every
            # subsequent step, because `floor` is computed from the already-lowered previous
            # point. Measured, that left 92.6 % of the channel slope-limited rather than
            # tracking terrain, and with a realistic gradient it plunged the bed to a median
            # 3.13 m / max 8.82 m below ground. The clamp lets the profile recover to
            # ground-following wherever terrain resumes falling; residual barriers it
            # reintroduces are exactly what the breaching step downstream of here is for.
            if np.isfinite(gnd):
                target = max(target, gnd - max_carve_mult * depth_li)
            carved.append(target)
            pieces.append((LineString([(pts[k - 1].x, pts[k - 1].y),
                                       (pts[k].x, pts[k].y)]).buffer(buf),
                           float(min(carved[-2], carved[-1]))))
        # the downstream node keeps the lowest elevation reaching it
        node_z[zz] = min(node_z.get(zz, np.inf), carved[-1])

    # rasterise deepest-last so overlaps resolve to the minimum (rasterize overwrites in order)
    pieces.sort(key=lambda t: -t[1])
    carved_grid = rasterize(pieces, out_shape=(rows, cols), transform=transform,
                            fill=np.nan, dtype="float32")

    z_b = z.copy()
    sel = burn_mask & np.isfinite(carved_grid) & np.isfinite(z)
    z_b[sel] = np.minimum(z[sel], carved_grid[sel])

    drop = (z - z_b)[sel]
    print(f"  Stream burned (gradient-enforced): {n_burned:,} cells, "
          f"{len(pieces):,} sub-segments  (flowlines={len(gdf)}, buffer={buf:.1f} m)")
    print(f"    carve depth: min {drop.min():.2f} m  median {np.median(drop):.2f} m  "
          f"max {drop.max():.2f} m   (max-order floor {burn_depth_m} m)")
    return z_b


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

    # Persist the surface D8/HAND are actually computed on. Previously this existed only in
    # memory: dem_conditioned.tif (breach-only) was saved and the solver reads it, while D8,
    # the stream network and HAND all ran on this further-conditioned array. Measured on the
    # main AOI the difference is small but real — 12.71 % of cells changed, max raise 0.020 m
    # (resolve_flats' epsilon gradient across flats), 1,970 m³ added — so the two were never
    # reproducible from the same file.
    save_tif(np.array(inflated, dtype=np.float32), profile,
             os.path.join(HYDRO_DIR, "dem_flat_resolved.tif"))
    # flow_dir.tif is listed in this module's own docstring as an output but was never
    # actually written by any run — the array existed only in memory.
    save_tif(np.array(fdir, dtype=np.float32), profile,
             os.path.join(HYDRO_DIR, "flow_dir.tif"))
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

def delineate_watershed(grid, fdir, accum, center_lat, center_lon, dem_crs, acc_threshold,
                        raster_transform, snap_radius_m=250.0):
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

    # Snap to the LARGEST channel within a search radius, not the nearest stream cell.
    # argmin(distance) takes whichever stream cell happens to be closest, which on a dense
    # network is routinely a minor tributary head sitting just above the threshold: at site3
    # that returned a catchment of 0.14 km² against a documented 33.15 km², because the
    # nearest cell carried 181,492 cells of accumulation while a channel 200 m away carried
    # 622,601. Snapping on maximum accumulation within a tolerance is the standard method
    # (ArcGIS "Snap Pour Point", pysheds `snap_to_mask`) precisely to avoid this.
    rad = max(1, int(round(snap_radius_m / ((abs(t.a) + abs(t.e)) / 2))))
    r0i, c0i = int(round(row0)), int(round(col0))
    r1, r2 = max(0, r0i - rad), min(acc_arr.shape[0], r0i + rad + 1)
    c1, c2 = max(0, c0i - rad), min(acc_arr.shape[1], c0i + rad + 1)
    sub_acc  = acc_arr[r1:r2, c1:c2]
    rr, cc   = np.mgrid[r1:r2, c1:c2]
    in_rad   = np.hypot(rr - row0, cc - col0) <= rad
    cand     = in_rad & (sub_acc > acc_threshold)
    if not cand.any():                      # nothing above threshold nearby — take the best cell
        cand = in_rad
    flat     = int(np.argmax(np.where(cand, sub_acc, -np.inf)))
    snap_row = int(rr.ravel()[flat])
    snap_col = int(cc.ravel()[flat])

    # Report what the old nearest-cell rule would have picked, so the difference is visible.
    sr, sc = np.where(stream_mask)
    nearest = int(np.argmin(np.hypot(sr - row0, sc - col0)))
    n_acc = acc_arr[int(sr[nearest]), int(sc[nearest])]
    m_acc = acc_arr[snap_row, snap_col]
    if n_acc < m_acc:
        print(f"  Snap: chose max-accumulation cell ({m_acc:,.0f} cells) over the merely-"
              f"nearest stream cell ({n_acc:,.0f} cells) within {snap_radius_m:.0f} m")
    snap_dist_cells = float(np.hypot(snap_row - row0, snap_col - col0))

    # Use rasterio's reliable xy() to convert grid index → projected coordinate
    snap_x, snap_y = rio_xy(raster_transform, snap_row, snap_col)

    print(f"  Pour point snapped: grid ({snap_row},{snap_col})  "
          f"acc={acc_arr[snap_row, snap_col]:,.0f} cells  "
          f"dist={snap_dist_cells:.1f} cells from the requested pour point")
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
        return 0, 0.0

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(geometry=polys, crs=dem_crs)
    gdf_4326 = gdf.to_crs("epsg:4326")
    dissolved = gdf_4326.dissolve()
    dissolved.to_file(out_path, driver="GeoJSON")
    # These are polygons vectorised from a raster mask, so .length is a PERIMETER, not a
    # channel centreline — the previous `total_km` computed it in degrees, called itself "not
    # meaningful" in its own comment, was never used, and raised a geographic-CRS warning on
    # every run. Report wetted area instead, which is exact and actually derivable from a mask.
    area_m2 = float(gdf.geometry.area.sum())
    print(f"  stream_network.geojson: {len(gdf_4326)} segments, {area_m2/1e4:.1f} ha")
    return len(gdf_4326), area_m2


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
        burn_depth=DEFAULT_BURN_DEPTH, center_lat=28.36687, center_lon=-81.43299,
        flowlines=None, acc_area_m2=DEFAULT_ACC_AREA_M2, burn_mode="gradient"):

    print(f"\nHydrological DEM Processing — {os.path.basename(dem_path)}")
    print("=" * 60)

    if not os.path.exists(dem_path):
        sys.exit(f"DEM not found: {dem_path}  Run: python3 dem/dem_download.py --resolution 1")

    z, profile, transform, res_x, res_y, dem_crs = load_dem(dem_path)
    print(f"DEM     : {z.shape[0]}×{z.shape[1]} px @ {res_x:.2f}m  CRS: {dem_crs}")

    cell_area = res_x * res_y
    if acc_threshold is None:
        acc_threshold = max(1, int(round(acc_area_m2 / cell_area)))
        print(f"  acc_threshold={acc_threshold:,} cells  "
              f"(= {acc_area_m2:,.0f} m² / {cell_area:.4f} m² per cell)")
    else:
        print(f"  acc_threshold={acc_threshold:,} cells  "
              f"(= {acc_threshold * cell_area:,.0f} m² contributing area) — explicit override")
    print(f"  burn_depth={burn_depth} m")
    print(f"Elevation: {np.nanmin(z):.2f}–{np.nanmax(z):.2f} m NAVD88")

    # ── Step 1: stream burning ───────────────────────────────────────────────
    print("\n[1/6] Stream burning (3DHP Shingle Creek) …")
    z_burned = burn_streams(z, profile, transform, dem_crs, burn_depth, flowlines,
                            mode=burn_mode)
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
    n_seg, stream_area_m2 = stream_to_geojson(
        stream_mask_np, profile, dem_crs, os.path.join(HYDRO_DIR, "stream_network.geojson"))

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
        "burn_mode":         burn_mode,
        "acc_threshold":     acc_threshold,
        "acc_area_m2":       float(acc_threshold * res_x * res_y),
        "stream_cells":      n_stream,
        "stream_area_m2":    stream_area_m2,
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
    summary["site_data_root"] = os.path.dirname(os.path.dirname(DATA_DIR))
    summary_path = os.path.join(HYDRO_DIR, "hydro_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone.  Output → {HYDRO_DIR}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",          default=None,               type=str,
                        help="Resolve the DEM, output dir, flowlines and pour point from "
                             "site_registry.py (e.g. --site site3). Preferred over hand-typed "
                             "paths — see apply_site().")
    parser.add_argument("--dem",           default=None,               type=str)
    parser.add_argument("--acc-threshold", default=DEFAULT_ACC_THRESHOLD, type=int,
                        help="Explicit stream-initiation threshold in CELLS. Resolution-dependent "
                             "— prefer --acc-area-m2 unless reproducing an old run.")
    parser.add_argument("--acc-area-m2",   default=DEFAULT_ACC_AREA_M2, type=float,
                        help="Stream-initiation contributing area in m² (default 38400 ≈ 3.84 ha). "
                             "Converted to cells using the DEM's own resolution.")
    parser.add_argument("--burn-depth",    default=DEFAULT_BURN_DEPTH, type=float,
                        help="Minimum metres to lower the DEM along channel centrelines")
    parser.add_argument("--burn-mode",     default="gradient",
                        choices=["gradient", "constant"],
                        help="gradient (default) enforces monotonic downstream descent; "
                             "constant reproduces the original uniform carve, which measurably "
                             "does not route on this terrain — see burn_streams()")
    parser.add_argument("--flowlines",     default=None,               type=str,
                        help="3DHP flowlines GeoJSON to burn. Defaults to the MAIN AOI's — "
                             "any other site must pass its own (see burn_streams docstring).")
    parser.add_argument("--lat",           default=28.36687,           type=float)
    parser.add_argument("--lon",           default=-81.43299,          type=float)
    args = parser.parse_args()

    dem, lat, lon = args.dem, args.lat, args.lon
    if args.site:
        site_dem, lat, lon = apply_site(args.site)
        dem = args.dem or site_dem
        if args.lat != 28.36687 or args.lon != -81.43299:
            sys.exit("  Refusing to run: --site sets the pour point; don't also pass --lat/--lon.")
    dem = dem or DEFAULT_DEM

    run(dem, args.acc_threshold, args.burn_depth, lat, lon, args.flowlines,
        args.acc_area_m2, args.burn_mode)
