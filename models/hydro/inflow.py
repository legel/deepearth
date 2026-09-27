"""Prescribed inflow across the domain edges, and its estimate from a coarse DEM.

The parcel is simulated at sub-metre resolution; the watershed above it enters as a boundary
discharge per edge cell, estimated from D8 accumulation on a coarser DEM and scaled by rain.
"""

import heapq
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EDGES = ("west", "east", "north", "south")


@dataclass
class Inflow:
    """Unit discharge [m^2/s] entering across each edge, per edge cell, sampled in time.

    Args:
        times_s: Sample times [s] since storm start, increasing.
        west: [n, rows] entering eastward across the west edge.
        east: [n, rows] entering westward across the east edge.
        north: [n, cols] entering southward across the north edge.
        south: [n, cols] entering northward across the south edge.
        rim_index: Flat indices of the cells on the rim of a domain that is not the grid's
            rectangle, such as a disc; None when inflow enters across the edges only.
        rim: [n, len(rim_index)] depth rate [m/s] added to each rim cell.
    """

    times_s: np.ndarray
    west: np.ndarray
    east: np.ndarray
    north: np.ndarray
    south: np.ndarray
    rim_index: Optional[np.ndarray] = None
    rim: Optional[np.ndarray] = None

    @classmethod
    def uniform(cls, shape: Tuple[int, int], q: Dict[str, float], t_end_s: float) -> "Inflow":
        """Constant discharge on the named edges for the whole run."""
        rows, cols = shape
        n = {e: q.get(e, 0.0) for e in EDGES}
        return cls(
            times_s=np.array([0.0, t_end_s]),
            west=np.full((2, rows), n["west"]), east=np.full((2, rows), n["east"]),
            north=np.full((2, cols), n["north"]), south=np.full((2, cols), n["south"]),
        )

    def at(self, t_s: float) -> Dict[str, np.ndarray]:
        """Linear interpolation in time, held at the end values outside the sampled range."""
        i = int(np.clip(np.searchsorted(self.times_s, t_s, side="right"), 1, len(self.times_s) - 1))
        t0, t1 = self.times_s[i - 1], self.times_s[i]
        w = float(np.clip((t_s - t0) / (t1 - t0), 0.0, 1.0)) if t1 > t0 else 0.0
        names = EDGES + (("rim",) if self.rim is not None else ())
        return {e: (1.0 - w) * getattr(self, e)[i - 1] + w * getattr(self, e)[i] for e in names}


OFFSETS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])


def fill_and_route(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Priority-flood depression filling with D8 receivers.

    Args:
        z: Elevation [m], finite everywhere.

    Returns:
        (filled elevation, flat receiver index per cell with -1 at outlets, flood order).
    """
    rows, cols = z.shape
    zf = z.ravel().astype(np.float64)
    filled = np.full(zf.shape, np.inf)
    recv = np.full(zf.shape, -1, dtype=np.int64)
    order = np.full(zf.shape, -1, dtype=np.int64)
    seen = np.zeros(zf.shape, dtype=bool)
    edge = np.zeros(z.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    heap = [(zf[c], int(c)) for c in np.flatnonzero(edge)]
    heapq.heapify(heap)
    filled[edge.ravel()] = zf[edge.ravel()]
    seen[edge.ravel()] = True
    n = 0
    while heap:
        level, c = heapq.heappop(heap)
        order[c] = n
        n += 1
        r0, c0 = divmod(c, cols)
        for dr, dc in OFFSETS:
            r1, c1 = r0 + dr, c0 + dc
            if r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols:
                continue
            nb = r1 * cols + c1
            if seen[nb]:
                continue
            seen[nb] = True
            filled[nb] = max(zf[nb], level)
            recv[nb] = c
            heapq.heappush(heap, (filled[nb], nb))
    filled = filled.reshape(z.shape)
    recv = _steepest(filled, recv.reshape(z.shape))
    return filled, recv, order.reshape(z.shape)


def _steepest(filled: np.ndarray, recv: np.ndarray) -> np.ndarray:
    """Steepest descent on the filled surface where one exists, the flood parent elsewhere."""
    rows, cols = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    best = np.zeros(filled.shape)
    out = recv.copy()
    for dr, dc in OFFSETS:
        drop = (filled - pad[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]) / np.hypot(dr, dc)
        better = drop > best
        best = np.where(better, drop, best)
        rr, cc = np.nonzero(better)
        out[rr, cc] = (rr + dr) * cols + (cc + dc)
    return out


def accumulate(filled: np.ndarray, recv: np.ndarray, order: np.ndarray, dx: float) -> np.ndarray:
    """Contributing area [m^2] per cell, its own cell included."""
    acc = np.full(filled.size, dx * dx)
    rf = recv.ravel()
    for c in np.lexsort((-order.ravel(), -filled.ravel())):
        if rf[c] >= 0:
            acc[rf[c]] += acc[c]
    return acc.reshape(filled.shape)


def _fine_slices(n_coarse: int, n_fine: int) -> np.ndarray:
    """First fine index of each coarse cell, plus the end, for `n_coarse` cells over `n_fine`.

    The fine grid need not be an integer multiple of the coarse one: the disc at 0.2 m is
    1124 cells against 225 coarse cells of 1 m, so four of them carry an extra fine cell.
    """
    return np.round(np.arange(n_coarse + 1) * n_fine / n_coarse).astype(np.int64)


def watershed_inflow(z_coarse: np.ndarray, dx_coarse: float, window: Tuple[int, int, int, int],
                     shape: Tuple[int, int], rain: Sequence[float], dt_s: float,
                     runoff: float = 1.0) -> Inflow:
    """Edge inflow for a fine window of a coarse DEM from D8 accumulation scaled by rain.

    Every coarse cell outside the window whose D8 receiver lies inside it delivers
    `area * runoff * rain` [m^3/s] across the edge it crosses, spread as a unit discharge over
    the fine edge cells of the receiving coarse cell. A diagonal crossing is charged to the
    edge the receiver's own row or column selects.

    Args:
        z_coarse: Coarse elevation [m], finite everywhere.
        dx_coarse: Coarse cell size [m].
        window: (row0, row1, col0, col1) of the fine domain in coarse cells, end-exclusive.
        shape: (rows, cols) of the fine grid.
        rain: Rainfall rate [m/s] per `dt_s` interval.
        dt_s: Rain interval [s].
        runoff: Fraction of upstream rain that reaches the edge.

    Returns:
        An `Inflow` on the fine grid, sampled at the rain intervals.
    """
    r0, r1, c0, c1 = window
    n_rows, n_cols = shape
    cols = z_coarse.shape[1]
    filled, recv, order = fill_and_route(z_coarse)
    area = accumulate(filled, recv, order, dx_coarse)
    row_edge, col_edge = _fine_slices(r1 - r0, n_rows), _fine_slices(c1 - c0, n_cols)
    dy, dxf = dx_coarse * (r1 - r0) / n_rows, dx_coarse * (c1 - c0) / n_cols
    per_edge = {"west": np.zeros(n_rows), "east": np.zeros(n_rows),
                "north": np.zeros(n_cols), "south": np.zeros(n_cols)}
    inside = np.zeros(z_coarse.shape, dtype=bool)
    inside[r0:r1, c0:c1] = True
    for c in np.flatnonzero(~inside.ravel() & (recv.ravel() >= 0)):
        rc, cc = divmod(int(c), cols)
        rr, rc_ = divmod(int(recv.ravel()[c]), cols)
        if not inside[rr, rc_]:
            continue
        volumetric = area[rc, cc] * runoff
        if rc < r0 or rc >= r1:
            lo, hi = col_edge[rc_ - c0], col_edge[rc_ - c0 + 1]
            per_edge["north" if rc < r0 else "south"][lo:hi] += volumetric / ((hi - lo) * dxf)
        else:
            lo, hi = row_edge[rr - r0], row_edge[rr - r0 + 1]
            per_edge["west" if cc < c0 else "east"][lo:hi] += volumetric / ((hi - lo) * dy)
    p = np.asarray(rain, dtype=np.float64)[:, None]
    return Inflow(times_s=np.arange(len(rain)) * dt_s,
                  **{e: p * per_edge[e][None, :] for e in EDGES})


def cap_unit_discharge(q: np.ndarray, max_depth_m: float, g: float = 9.81,
                       froude: float = 0.9) -> np.ndarray:
    """Limit unit discharge per edge cell, spreading the excess along the same edge.

    D8 delivers a whole upstream catchment through the one coarse cell its flow path crosses,
    and a prescribed unit discharge forces whatever depth conveys it: at the Campanile that is
    2.7 m against 0.13 m from rainfall alone. Real inflow arrives across a width, and at this
    site largely in culverts. Capping at the discharge a stated depth can carry keeps the
    volume and the edge, and drops the slot.

    Args:
        q: Unit discharge [m^2/s], [n_times, n_cells] along one edge.
        max_depth_m: Depth whose Froude-capped conveyance sets the ceiling.
        g: Gravity [m/s^2].
        froude: Froude number the ceiling is taken at.

    Returns:
        Capped unit discharge with the same row sums, as far as the ceiling allows.
    """
    ceiling = froude * max_depth_m * np.sqrt(g * max_depth_m)
    out = np.array(q, dtype=np.float64, copy=True)
    for _ in range(64):
        excess = np.clip(out - ceiling, 0.0, None).sum(axis=1)
        if not (excess > 1e-15).any():
            break
        out = np.minimum(out, ceiling)
        room = out < ceiling
        n_room = room.sum(axis=1)
        share = np.divide(excess, np.maximum(n_room, 1), out=np.zeros_like(excess), where=n_room > 0)
        out = out + room * share[:, None]
    return np.minimum(out, ceiling)


def rim_cells(domain: np.ndarray) -> np.ndarray:
    """(rows, cols) cells of `domain` with a 4-neighbour outside it or on the grid's border."""
    p = np.pad(domain, 1, constant_values=False)
    return domain & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])


def rim_inflow(z_coarse: np.ndarray, t_coarse: object, domain: np.ndarray, wet: np.ndarray, t_fine: object,
               rain: Sequence[float], dt_s: float, runoff: float = 1.0,
               max_depth_m: Optional[float] = None) -> Inflow:
    """Inflow across the rim of a domain that is not the grid's rectangle, from D8 on a coarse DEM.

    Every coarse cell outside the domain whose D8 receiver lies inside it delivers
    `area * runoff * rain` [m^3/s], shared evenly by the rim cells inside that receiver, or given
    to the rim cell nearest the receiver when it holds none. With `max_depth_m`, the unit
    discharge along the rim, taken in order of azimuth about the rim's centroid with each cell
    one cell wide, is capped as `cap_unit_discharge` caps an edge.

    Args:
        z_coarse: Coarse elevation [m], finite everywhere.
        t_coarse: Affine transform of `z_coarse`.
        domain: (rows, cols) fine cells inside the domain, holes included.
        wet: (rows, cols) fine cells that can hold water; rim cells outside it receive nothing.
        t_fine: Affine transform of the fine grid, same projected frame as `t_coarse`.
        rain: Rainfall rate [m/s] per `dt_s` interval.
        dt_s: Rain interval [s].
        runoff: Fraction of upstream rain that reaches the rim.
        max_depth_m: Depth whose conveyance caps the rim's unit discharge, or None.

    Returns:
        An `Inflow` whose edges carry nothing and whose rim carries the delivery.
    """
    from scipy.spatial import cKDTree

    dx, ncols = abs(t_fine.a), z_coarse.shape[1]
    filled, recv, order = fill_and_route(z_coarse)
    area = accumulate(filled, recv, order, abs(t_coarse.a)).ravel()
    rc, cc = np.indices(z_coarse.shape)
    fx = np.floor((t_coarse.c + (cc + 0.5) * t_coarse.a - t_fine.c) / t_fine.a).astype(np.int64)
    fy = np.floor((t_coarse.f + (rc + 0.5) * t_coarse.e - t_fine.f) / t_fine.e).astype(np.int64)
    on = (fx >= 0) & (fx < domain.shape[1]) & (fy >= 0) & (fy < domain.shape[0])
    inside = np.zeros(z_coarse.shape, bool)
    inside[on] = domain[fy[on], fx[on]]
    inside = inside.ravel()
    rim = np.flatnonzero(rim_cells(domain) & wet)
    ry, rx = np.divmod(rim, domain.shape[1])
    x, y = t_fine.c + (rx + 0.5) * t_fine.a, t_fine.f + (ry + 0.5) * t_fine.e
    owner = (np.floor((y - t_coarse.f) / t_coarse.e).astype(np.int64) * ncols
             + np.floor((x - t_coarse.c) / t_coarse.a).astype(np.int64))
    rf = recv.ravel()
    src = np.flatnonzero(~inside & (rf >= 0))
    src = src[inside[rf[src]]]
    ids, inv, counts = np.unique(owner, return_inverse=True, return_counts=True)
    pos = np.minimum(np.searchsorted(ids, rf[src]), len(ids) - 1)
    held = ids[pos] == rf[src]
    per_id = np.zeros(len(ids))
    np.add.at(per_id, pos[held], area[src[held]] * runoff)
    share = per_id[inv] / counts[inv]
    lost = rf[src[~held]]
    mx = t_coarse.c + (lost % ncols + 0.5) * t_coarse.a
    my = t_coarse.f + (lost // ncols + 0.5) * t_coarse.e
    _, near = cKDTree(np.c_[x, y]).query(np.c_[mx, my])
    np.add.at(share, near, area[src[~held]] * runoff)
    ring = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
    q = np.asarray(rain, dtype=np.float64)[:, None] * share[ring][None, :] / dx
    if max_depth_m is not None:
        q = cap_unit_discharge(q, max_depth_m)
    n, (rows, cols) = len(rain), domain.shape
    return Inflow(times_s=np.arange(n) * dt_s, west=np.zeros((n, rows)), east=np.zeros((n, rows)),
                  north=np.zeros((n, cols)), south=np.zeros((n, cols)), rim_index=rim[ring], rim=q / dx)
