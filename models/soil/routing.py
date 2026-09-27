"""Water over and under the bare-earth terrain, compiled with numba.

Run-on (surface). Once per site, Priority-Flood with epsilon (Barnes, Lehman and Mulla 2014) gives every cell a way out
and each depression its storage (filled minus bare, mm). The multiple-flow-direction weights of Quinn et al. (1991)
split a cell's excess among its lower neighbors in proportion to slope times contour length. In each rain hour, one
pass runs in descending filled elevation: every cell infiltrates what it can, fills its depression and passes the rest
on. Water leaving the grid's edge is runoff. Mass is conserved exactly.

Lateral flow (subsurface). The same weights on the unfilled terrain, with each cell's weighted gradient tan(beta), a
water table parallel to the surface. A buried hollow collects the water above it, where the surface run-on fills and
spills.
"""

import heapq
from typing import NamedTuple

import numba
import numpy as np

EPS = 1e-4                      # m: the smallest drop Priority-Flood adds across a flat


@numba.njit(cache=True)
def _flood(z, valid, eps):
    ny, nx = z.shape
    zf = z.copy()
    seen = ~valid
    heap = [(0.0, 0, 0)]
    heap.pop()
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            edge = i == 0 or j == 0 or i == ny - 1 or j == nx - 1
            if not edge:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if not valid[i + di, j + dj]:
                            edge = True
            if edge:
                heapq.heappush(heap, (zf[i, j], i, j))
                seen[i, j] = True
    while len(heap):
        h, i, j = heapq.heappop(heap)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di == 0 and dj == 0) or a < 0 or b < 0 or a >= ny or b >= nx or seen[a, b]:
                    continue
                seen[a, b] = True
                if zf[a, b] <= h:
                    zf[a, b] = h + eps
                heapq.heappush(heap, (zf[a, b], a, b))
    return zf


def fill(z: np.ndarray, valid: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Priority-Flood: with eps > 0 every valid cell drains to the grid's edge or an invalid cell; with eps = 0 the
    plain fill, whose excess over the bare earth is depression storage."""
    return _flood(z.astype(np.float64), valid.astype(np.bool_), float(eps))


@numba.njit(cache=True)
def _mfd(zf, valid, dx):
    """Quinn et al. (1991): weight to each lower neighbor = drop / distance x contour length (0.5 dx straight,
    0.354 dx diagonal), normalized."""
    ny, nx = zf.shape
    recv = -np.ones((ny * nx, 8), np.int64)
    wts = np.zeros((ny * nx, 8), np.float64)
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            c = i * nx + j
            tot = 0.0
            k = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if a < 0 or b < 0 or a >= ny or b >= nx or not valid[a, b]:
                        k += 1
                        continue
                    diag = di != 0 and dj != 0
                    dist = dx * (1.4142135623730951 if diag else 1.0)
                    drop = zf[i, j] - zf[a, b]
                    if drop > 0:
                        w = (drop / dist) * dx * (0.354 if diag else 0.5)
                        recv[c, k] = a * nx + b
                        wts[c, k] = w
                        tot += w
                    k += 1
            if tot > 0:
                for m in range(8):
                    wts[c, m] /= tot
    return recv, wts


class Network(NamedTuple):
    """The run-on network in processing order (high to low): cell `order[p]` passes its excess to the positions
    `rcv[ptr[p]:ptr[p+1]]` (always later, downslope) with weights `w`; `room` is each cell's depression storage, mm."""
    order: np.ndarray       # int64 cell ids
    ptr: np.ndarray         # int64 [m + 1]
    rcv: np.ndarray         # int32 positions
    w: np.ndarray           # float32
    room: np.ndarray        # float64 per cell id


@numba.njit(cache=True)
def _csr(recv, wts, order, n):
    pos = -np.ones(n, np.int64)
    for p in range(order.shape[0]):
        pos[order[p]] = p
    m = order.shape[0]
    ptr = np.zeros(m + 1, np.int64)
    for p in range(m):
        c = order[p]
        k = 0
        for j in range(8):
            if recv[c, j] >= 0:
                k += 1
        ptr[p + 1] = ptr[p] + k
    rcv = np.empty(ptr[m], np.int32)
    w = np.empty(ptr[m], np.float32)
    for p in range(m):
        c = order[p]
        e = ptr[p]
        for j in range(8):
            r = recv[c, j]
            if r >= 0:
                rcv[e] = pos[r]
                w[e] = wts[c, j]
                e += 1
    return ptr, rcv, w


def network(z: np.ndarray, valid: np.ndarray, dx: float) -> Network:
    """Once per site: the filled terrain's flow network in processing order, and depression storage."""
    zf = fill(z, valid)
    recv, wts = _mfd(zf, valid.astype(np.bool_), float(dx))
    flat = np.where(valid.ravel(), zf.ravel(), -np.inf)
    order = np.argsort(-flat, kind="stable")[: int(valid.sum())].astype(np.int64)
    ptr, rcv, w = _csr(recv, wts, order, zf.size)
    room = np.where(valid, np.maximum(fill(z, valid, 0.0) - z, 0.0) * 1000.0, 0.0).ravel()
    return Network(order, ptr, rcv, w, room)


@numba.njit(cache=True, fastmath=True)
def _cascade(ptr, rcv, w, sup, cap, room, perv):
    m = sup.shape[0]
    inflow = np.zeros(m, np.float32)
    infil = np.zeros(m, np.float32)
    pond = np.zeros(m, np.float32)
    lost = np.zeros(m, np.float32)
    for p in range(m):
        avail = sup[p] + inflow[p]
        if avail <= 0:
            continue
        f = min(avail * perv[p], cap[p])
        infil[p] = f
        avail -= f
        q = min(avail, room[p])
        pond[p] = q
        avail -= q
        if avail <= 0:
            continue
        a, b = ptr[p], ptr[p + 1]
        if a == b:
            lost[p] = avail                      # an outlet: the grid's edge or a building's wall
            continue
        given = 0.0
        for e in range(a, b - 1):
            x = avail * w[e]
            inflow[rcv[e]] += x
            given += x
        inflow[rcv[b - 1]] += avail - given      # the last receiver takes the remainder: exact books
    return infil, pond, inflow, lost


def cascade(net: Network, supply, cap, perv):
    """One hour's run-on. supply: water arriving from above at each cell (throughfall plus standing water, mm); cap:
    what each cell can still take in this hour (mm); perv: the cell's pervious fraction, the only part water soaks
    through (a cell sheds its impervious share of whatever reaches it). Depression storage is `net.room`.
    Returns (infiltrated, standing, run-on received, water leaving the grid), mm per cell id."""
    o = net.order
    pos = lambda a: np.asarray(a, np.float32)[o]                                          # noqa: E731
    got = _cascade(net.ptr, net.rcv, net.w, pos(supply), pos(cap), net.room[o].astype(np.float32), pos(perv))
    out = []
    for a in got:
        full = np.zeros(len(supply))
        full[o] = a
        out.append(full)
    return tuple(out)


@numba.njit(cache=True)
def _downslope(z, valid, dx):
    """Per cell over the unfilled terrain: its MFD receivers and weights (as `_mfd`) and its weight-averaged gradient
    tan(beta) toward them; a pit or a flat has none (its water stays)."""
    ny, nx = z.shape
    src = np.empty(ny * nx * 8, np.int64)
    dst = np.empty(ny * nx * 8, np.int64)
    wts = np.empty(ny * nx * 8, np.float64)
    e = 0
    tanb = np.zeros(ny * nx, np.float64)
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            c = i * nx + j
            tot = 0.0
            recv = np.empty(8, np.int64)
            w8 = np.empty(8, np.float64)
            g8 = np.empty(8, np.float64)
            m = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if a < 0 or b < 0 or a >= ny or b >= nx or not valid[a, b]:
                        continue
                    diag = di != 0 and dj != 0
                    dist = dx * (1.4142135623730951 if diag else 1.0)
                    drop = z[i, j] - z[a, b]
                    if drop > 0:
                        w = (drop / dist) * dx * (0.354 if diag else 0.5)
                        recv[m] = a * nx + b
                        w8[m] = w
                        g8[m] = drop / dist
                        tot += w
                        m += 1
            if tot > 0:
                g = 0.0
                for k in range(m):
                    src[e] = c
                    dst[e] = recv[k]
                    wts[e] = w8[k] / tot
                    e += 1
                    g += w8[k] / tot * g8[k]
                tanb[c] = g
    return src[:e].copy(), dst[:e].copy(), wts[:e].copy(), tanb


class Lateral(NamedTuple):
    """Subsurface downslope routing: cell `src[e]` passes weight `w[e]` of its lateral outflow to `dst[e]`; `tanb` is
    its hydraulic gradient (the land surface's)."""
    src: np.ndarray
    dst: np.ndarray
    w: np.ndarray
    tanb: np.ndarray
    dx_m: float


LATERAL_SMOOTH_M = 2.0
"""The subsurface gradient's scale: a Gaussian of this sigma over the terrain before its slopes are taken. A water table
follows the landform, not a 0.5 m survey's micro-relief, whose few centimeters between neighbors read as slopes of 10 to
20 % and turn the root zone into cell-scale speckle."""


def smoothed(z: np.ndarray, valid: np.ndarray, dx: float, sigma_m: float = LATERAL_SMOOTH_M) -> np.ndarray:
    """z under a Gaussian of `sigma_m`, normalized over valid cells only."""
    from scipy import ndimage
    if sigma_m <= 0:
        return z
    s = sigma_m / float(dx)
    num = ndimage.gaussian_filter(np.where(valid, np.nan_to_num(z, nan=0.0), 0.0), s, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(np.float64), s, mode="nearest")
    return np.where(valid & (den > 1e-6), num / np.maximum(den, 1e-6), z)


def lateral(z: np.ndarray, valid: np.ndarray, dx: float, sigma_m: float = LATERAL_SMOOTH_M) -> Lateral:
    """Once per site: the multiple-flow-direction graph for subsurface flow over the unfilled, smoothed terrain."""
    zs = smoothed(np.nan_to_num(z, nan=0.0), valid.astype(bool), dx, sigma_m)
    src, dst, w, tanb = _downslope(zs, valid.astype(np.bool_), float(dx))
    return Lateral(src, dst, w.astype(np.float32), tanb.astype(np.float32), float(dx))
