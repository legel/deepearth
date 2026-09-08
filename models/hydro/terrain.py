"""Hydrological conditioning: stream burn -> depression breach -> D8 -> accumulation -> HAND.

The solver reads the conditioned DEM, so everything here is upstream of every result.

Two choices are load-bearing and were each reached by measurement. Depressions are BREACHED,
not filled: breaching carves least-cost drainage paths and leaves the surrounding surface
alone. And the stream burn enforces a downstream GRADIENT rather than a constant depth -- a
flat-bottomed ditch on ground with 14 m of relief over 6.8 km lowers the channel but imposes
no direction, so D8 cannot route along it. Measured after a correct constant-depth burn of
site3: the largest flow accumulation anywhere in 46.78 km2 was 0.99 km2 and drainage fragmented
into disconnected pockets.

Worth knowing before trusting any stream network from here: two defensible breaching algorithms
(richdem and WhiteboxTools) agree on flow direction for 87.9 % of cells but produce stream
networks agreeing at only IoU 0.29, because on terrain this flat D8 turns on sub-millimetre
differences. The delineated network carries far more uncertainty than it is usually given.
"""

import contextlib
import io
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize, shapes

from sites import SiteConfig

DEFAULT_ACC_AREA_M2 = 38400.0
"""Contributing area at which a cell becomes a stream, about 3.84 ha.

An AREA, converted to a cell count at runtime. As a bare cell count it does not survive a
change of resolution: 1,000 cells means 25,000 m2 at 5 m and 774 m2 at 0.88 m, and carrying the
main AOI's tuned count over to site3's native grid produced 1.7 million stream cells (2.82 % of
the domain) and a degenerate HAND surface where 77 % of cells sat within 1 m of "drainage".
"""

BURN_DEPTH_M = 1.5
"""Carve depth [m] for the highest stream order present; lower orders scale down with order.

Carving every tributary to full depth turns 22 km of minor channel into storage that runoff
must fill before any can leave -- measured, a uniform 1.5 m floor pushed the outflow peak to
25.9 h after the rain peak against the gauge's observed 4.5 h. Depth growing with order is the
coarse form of the standard hydraulic-geometry result (Leopold & Maddock 1953), and 3DHP
carries `streamorder` per reach, so this uses measured order rather than a guess.
"""

MIN_GRADIENT = 1e-4
"""Floor on the enforced downstream gradient [m/m], used only where none can be measured.

Not a target. A hand-picked 1e-4 "just enough to break ties" was tried and is wrong by more
than an order of magnitude: site3's network actually falls at 1.90e-3 overall. Because channel
velocity goes as sqrt(S), imposing 1e-4 on flat reaches throttled them 2.4-5x.
"""


def _network_order(lines: List, tol: float) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Topologically order flowlines from headwater to outlet.

    3DHP's `flowdirectionlabel` states vertices run downslope in digitised order, so each
    LineString's own vertex order is the downstream direction -- authoritative, and better than
    inferring direction from a DEM this flat.
    """
    from collections import defaultdict, deque

    node: Dict[Tuple[int, int], int] = {}

    def nid(pt: Tuple[float, float]) -> int:
        """Stable id for a vertex, snapped to the cell size so shared nodes collapse."""
        return node.setdefault((round(pt[0] / tol), round(pt[1] / tol)), len(node))

    ends = [(nid(list(ln.coords)[0]), nid(list(ln.coords)[-1])) for ln in lines]
    outgoing = defaultdict(list)
    for i, (a, _) in enumerate(ends):
        outgoing[a].append(i)

    indeg = [0] * len(lines)
    succ = defaultdict(list)
    for i, (_, b) in enumerate(ends):
        for j in outgoing.get(b, []):
            succ[i].append(j)
            indeg[j] += 1

    queue = deque(i for i in range(len(lines)) if indeg[i] == 0)
    order = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    order += [i for i in range(len(lines)) if i not in set(order)]  # cycles, should not occur
    return order, ends


@contextlib.contextmanager
def _quiet():
    """Silence richdem, which is noisy at two different levels.

    Its C++ core writes a progress bar straight to the process file descriptors -- measured at
    117 bytes to fd 1 and **14,769 to fd 2** on one call, so redirecting stdout alone silences
    almost none of it. Its Python wrapper separately `print()`s a geotransform warning, and that
    goes through `sys.stdout`'s buffer, which flushes AFTER an fd-level guard has been restored.
    Catching only one level leaves the other leaking, so both are held here.
    """
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        for fd in saved:
            os.close(fd)


def burn_streams(z: np.ndarray, transform: Affine, dem_crs: CRS, flowlines_path: Path,
                 burn_depth_m: float = BURN_DEPTH_M, step_m: float = 5.0,
                 max_carve_mult: float = 2.0) -> np.ndarray:
    """Carve mapped flowlines into the DEM, enforcing monotonic descent downstream.

    Each sampled point is carved to at least `burn_depth_m` below ground AND at least
    `slope * distance` below the point upstream of it, so where the DEM already falls faster
    than the enforced gradient the natural profile is kept and this only intervenes on flats.

    Args:
        z: Raw elevation [m].
        transform: Affine transform of `z`.
        dem_crs: CRS of `z`.
        flowlines_path: Mapped hydrography for THIS site. Passing another site's is the
            failure this asserts against -- site3 was once conditioned with the main AOI's six
            Shingle Creek lines from 34 km away, the burn mask came out empty, and stream
            burning silently became a no-op that nobody noticed.
        burn_depth_m: Carve depth for the highest stream order present.
        step_m: Sampling interval along each line.
        max_carve_mult: Cap on carve depth as a multiple of `burn_depth_m`.

    Returns:
        The burned DEM.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    gdf = gpd.read_file(flowlines_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("epsg:4326")
    gdf = gdf.to_crs(dem_crs)

    rows, cols = z.shape
    cell_m = (abs(transform.a) + abs(transform.e)) / 2
    buf = cell_m * 2

    burn_mask = rasterize([(g, 1) for g in gdf.geometry.buffer(buf) if g is not None],
                          out_shape=(rows, cols), transform=transform,
                          fill=0, dtype=np.uint8).astype(bool)
    assert burn_mask.any(), (
        f"{flowlines_path} rasterises to an empty mask over this DEM -- the flowlines do not "
        f"overlap the domain. Flowline bounds {tuple(gdf.total_bounds)}, DEM bounds "
        f"{rasterio.transform.array_bounds(rows, cols, transform)}")

    inv = ~transform

    def ground_at(x: float, y: float) -> float:
        """Raw ground elevation [m] at a projected point, NaN outside the raster."""
        c, r = inv * (x, y)
        r, c = int(r), int(c)
        inside = 0 <= r < rows and 0 <= c < cols
        return float(z[r, c]) if (inside and np.isfinite(z[r, c])) else np.nan

    def natural_gradient(idx: Iterable[int]) -> Optional[float]:
        """Fall per unit length over the given reaches, measured from the raw DEM."""
        drop = length = 0.0
        for i in idx:
            ln = gdf.geometry.iloc[i]
            if ln is None or ln.geom_type != "LineString" or ln.length <= 0:
                continue
            zs = [v for v in (ground_at(p.x, p.y) for p in
                              (ln.interpolate(float(d)) for d in np.linspace(0.0, ln.length, 50)))
                  if np.isfinite(v)]
            if len(zs) > 2:
                drop += zs[0] - zs[-1]
                length += ln.length
        return (drop / length) if (length > 0 and drop > 0) else None

    keep = [i for i, g in enumerate(gdf.geometry) if g is not None and g.geom_type == "LineString"]
    lines = [gdf.geometry.iloc[i] for i in keep]
    assert lines, f"{flowlines_path} contains no LineString geometries"

    has_order = "streamorder" in gdf.columns and gdf["streamorder"].notna().all()
    if has_order:
        so = gdf["streamorder"].to_numpy(float)
        so_max = float(np.nanmax(so))
        depths = [burn_depth_m * float(so[i]) / so_max for i in keep]
        by_order = {o: max(natural_gradient([i for i in range(len(gdf)) if so[i] == o]) or 0.0,
                           MIN_GRADIENT) for o in sorted(set(so))}
        slopes = [by_order[so[i]] for i in keep]
    else:
        depths = [burn_depth_m] * len(lines)
        measured = natural_gradient(range(len(gdf)))
        slopes = [max(measured or 0.0, MIN_GRADIENT)] * len(lines)

    order, ends = _network_order(lines, tol=cell_m)
    node_z: Dict[int, float] = {}
    pieces = []

    for li in order:
        ln, (a, b) = lines[li], ends[li]
        depth, slope = depths[li], slopes[li]
        n = max(2, int(np.ceil(ln.length / step_m)) + 1)
        ds = np.linspace(0.0, ln.length, n)
        pts = [ln.interpolate(float(d)) for d in ds]

        z_prev = node_z.get(a, np.nan)
        if not np.isfinite(z_prev):
            g0 = ground_at(pts[0].x, pts[0].y)
            z_prev = (g0 - depth) if np.isfinite(g0) else 0.0
        node_z[a] = min(node_z.get(a, np.inf), z_prev)

        carved = [z_prev]
        for k in range(1, len(pts)):
            gnd = ground_at(pts[k].x, pts[k].y)
            floor = carved[-1] - slope * (ds[k] - ds[k - 1])
            target = min(gnd - depth, floor) if np.isfinite(gnd) else floor
            if np.isfinite(gnd):
                target = max(target, gnd - max_carve_mult * depth)
            carved.append(target)
            pieces.append((LineString([(pts[k - 1].x, pts[k - 1].y),
                                       (pts[k].x, pts[k].y)]).buffer(buf),
                           float(min(carved[-2], carved[-1]))))
        node_z[b] = min(node_z.get(b, np.inf), carved[-1])

    pieces.sort(key=lambda t: -t[1])  # deepest last, so confluences take the lowest value
    carved_grid = rasterize(pieces, out_shape=(rows, cols), transform=transform,
                            fill=np.nan, dtype="float32")

    burned = z.copy()
    sel = burn_mask & np.isfinite(carved_grid) & np.isfinite(z)
    burned[sel] = np.minimum(z[sel], carved_grid[sel])
    return burned


def breach_depressions(z: np.ndarray, transform: Affine) -> np.ndarray:
    """Carve least-cost drainage through depressions with richdem.

    Breaching, not filling. richdem is not interchangeable here: WhiteboxTools' equivalent
    agrees on 87.9 % of flow directions but yields a stream network at IoU 0.29, which would
    silently rewrite every watershed and HAND surface downstream.
    """
    import richdem as rd

    nodata = -9999.0
    filled = z.copy()
    filled[~np.isfinite(filled)] = nodata
    with warnings.catch_warnings(), _quiet():
        warnings.simplefilter("ignore", category=UserWarning)
        arr = rd.rdarray(filled, no_data=nodata, geotransform=(
            transform.c, transform.a, 0, transform.f, 0, transform.e))
        rd.BreachDepressions(arr, in_place=True)
        out = np.array(arr, dtype=np.float32)
    out[out == nodata] = np.nan
    out[~np.isfinite(z)] = np.nan
    return out


def flow_and_hand(conditioned_path: Path, acc_threshold_cells: float) -> Tuple:
    """D8 flow direction, accumulation and HAND from a conditioned DEM on disk.

    Returns:
        (grid, fdir, accumulation, HAND, stream mask).
    """
    from pysheds.grid import Grid

    grid = Grid.from_raster(str(conditioned_path))
    dem = grid.read_raster(str(conditioned_path))
    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = grid.flowdir(inflated)
    accum = grid.accumulation(fdir)
    streams = accum > acc_threshold_cells
    hand = grid.compute_hand(fdir, inflated, streams)
    return grid, fdir, accum, hand, streams


def delineate(grid: object, fdir: object, accum: object, lat: float, lon: float, dem_crs: CRS,
              transform: Affine, acc_threshold_cells: float,
              snap_radius_m: float = 250.0) -> Optional[np.ndarray]:
    """Watershed upstream of a pour point, snapped to the highest-accumulation cell nearby.

    Snapping matters: an unsnapped pour point lands on a hillslope beside the channel and
    delineates a few cells.
    """
    from pyproj import Transformer

    x, y = Transformer.from_crs("epsg:4326", dem_crs, always_xy=True).transform(lon, lat)
    acc = np.array(accum, dtype=np.float64)
    if not (acc > acc_threshold_cells).any():
        return None

    col0 = (x - transform.c) / transform.a
    row0 = (y - transform.f) / transform.e
    rad = max(1, int(round(snap_radius_m / ((abs(transform.a) + abs(transform.e)) / 2))))
    r0, c0 = int(round(row0)), int(round(col0))
    r1, r2 = max(0, r0 - rad), min(acc.shape[0], r0 + rad + 1)
    c1, c2 = max(0, c0 - rad), min(acc.shape[1], c0 + rad + 1)

    rr, cc = np.mgrid[r1:r2, c1:c2]
    within = np.hypot(rr - row0, cc - col0) <= rad
    candidates = within & (acc[r1:r2, c1:c2] > acc_threshold_cells)
    if not candidates.any():
        candidates = within
    flat = int(np.argmax(np.where(candidates, acc[r1:r2, c1:c2], -np.inf)))
    return grid.catchment(x=int(cc.ravel()[flat]), y=int(rr.ravel()[flat]),
                          fdir=fdir, xytype="index")


def mask_to_geojson(mask: np.ndarray, transform: Affine, dem_crs: CRS, out_path: Path) -> int:
    """Vectorise a boolean raster mask to GeoJSON in WGS84. Returns the feature count."""
    import geopandas as gpd
    from shapely.geometry import shape

    geoms = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask.astype(bool),
                                         transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=dem_crs).to_crs("epsg:4326")
    gdf.to_file(out_path, driver="GeoJSON")
    return len(gdf)


def condition(site: SiteConfig, acc_area_m2: float = DEFAULT_ACC_AREA_M2) -> Dict[str, float]:
    """Run the full conditioning chain for a site and write every product to its data root.

    Args:
        site: Site to condition.
        acc_area_m2: Contributing area at which a cell becomes a stream.

    Returns:
        Summary statistics worth recording alongside the outputs.
    """
    assert site.dem.exists(), (
        f"{site.dem} missing; run `python3 cli.py fetch --site {site.name}`")
    assert site.flowlines.exists(), f"{site.flowlines} missing; run the fetch stage first"

    with rasterio.open(site.dem) as src:
        z = src.read(1).astype(np.float32)
        profile, transform, crs = src.profile.copy(), src.transform, src.crs
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
    profile.update(dtype="float32", nodata=np.nan, count=1)

    cell_m2 = abs(transform.a) * abs(transform.e)
    acc_cells = acc_area_m2 / cell_m2

    burned = burn_streams(z, transform, crs, site.flowlines)
    _write(burned, profile, site.dem_burned)
    conditioned = breach_depressions(burned, transform)
    _write(conditioned, profile, site.dem_conditioned)

    grid, fdir, accum, hand, streams = flow_and_hand(site.dem_conditioned, acc_cells)
    _write(np.array(hand, dtype=np.float32), profile, site.hand)
    _write(np.log10(np.array(accum, dtype=np.float32) + 1.0), profile, site.flow_accum)
    n_streams = mask_to_geojson(np.array(streams), transform, crs, site.streams)

    # Delineate at the GAUGE where the site has one, not the box centre. The gauge is the
    # hydrologically meaningful outlet; the centre is an arbitrary point that on this terrain
    # sits on a hillslope and returns a fraction of the real catchment (0.60 km2 against the
    # 11.65 km2 the pour point gives). Only a run caught this.
    pour_lat, pour_lon = ((site.gauge.lat, site.gauge.lon) if site.gauge is not None
                          else (site.lat, site.lon))
    catchment = delineate(grid, fdir, accum, pour_lat, pour_lon, crs, transform, acc_cells)
    catchment_km2 = 0.0
    if catchment is not None:
        catch = np.array(catchment, dtype=np.uint8).astype(bool)
        catchment_km2 = float(catch.sum()) * cell_m2 / 1e6
        mask_to_geojson(catch, transform, crs, site.watershed)

    hand_arr = np.array(hand, dtype=np.float32)
    return {
        "cell_size_m": float(abs(transform.a)),
        "acc_threshold_cells": float(acc_cells),
        "burn_mean_drop_m": float(np.nanmean(z - burned)),
        "stream_cells": int(np.array(streams).sum()),
        "stream_features": n_streams,
        "hand_mean_m": float(np.nanmean(hand_arr)),
        "hand_lt_1m_pct": float(np.nanmean(hand_arr < 1.0) * 100.0),
        "catchment_km2": catchment_km2,
        "pour_point": "gauge" if site.gauge is not None else "box centre",
    }


def _write(arr: np.ndarray, profile: Dict, path: Path) -> None:
    """Write a single-band float32 GeoTIFF."""
    prof = dict(profile, dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
