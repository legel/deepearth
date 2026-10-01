"""The unit solve's wind at survey points in 3D: every return read from the solve by one operator.

A point is placed at its own z less the two frames' vertical offset (`datum_offset`, measured under the points), so it
keeps its place against the solids made from the same survey (`place(..., z=None)` stands it at its height over the
solver's terrain instead). It is kept `CLEARANCE_M` off every solid face it stands on or beside (ground, roof,
wall) and read trilinearly over the fluid cell centres around it. Canopy is porous in the solve, so a crown return
higher than the clearance is read where it is.

Each of the four columns around a point is read at the same height above its own solid top when that top is within
a cell and a half of the point's column's top (a terrain step or a pitched roof), so the solver's staircase of 1 m
cubes draws no contour lines. A column that rises past that (a wall) is read at the point's own height, and not at
all where it is solid there.

A return inside a solid column (a facade inside a footprint rounded to the cell, a roof under its rounded top) moves
to the nearer of its column's top and the nearest column open at its height, up to `SEARCH_CELLS` away. How far that
search went is recorded per point (`search_m`).

    placed = place(scene, x, y, h)          # once per scene
    uvw = sample(velocity, placed)          # once per heading: (3, N), NaN where no fluid centre is in reach
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CLEARANCE_M = 2.0
"""How far off a solid face a point is read: a plant's height over the ground or a roof, and as far out from a wall.
At 1 m cells it lies between the second and third fluid centres, resolved by the solve with no log law."""
SEARCH_CELLS = 2
"""How many cells out a return inside a solid column looks for a column open at its height."""
FOLLOW_CELLS = 1.5
"""A neighbour column whose top is within this many cells of the point's column's top is the same surface stepped
(terrain, a pitched roof) and is read at the same height above its own top."""
CHUNK = 1_000_000
POINT_FIELDS = ("x", "y", "z", "h")
"""The points file: float32 [N, 4], scene metres x, y, the point's own z (used only for the DTM difference) and its
height above the bare earth."""
SEARCH_STEP_M = 0.1
"""`search_u8` codes: tenths of a metre, 254 at most; 255 is a point with no fluid centre in reach."""
UNRESOLVED = 255


def write_points(path: Path, x, y, z, h) -> Path:
    np.stack([np.asarray(a, np.float32) for a in (x, y, z, h)], 1).astype("<f4").tofile(path)
    return Path(path)


def read_points(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.fromfile(path, "<f4").reshape(-1, len(POINT_FIELDS))
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def write_normals(path: Path, normals) -> Path:
    np.asarray(normals, np.float32).reshape(-1, 3).astype("<f4").tofile(path)
    return Path(path)


def normals_arg(value: str, n: int):
    """`levels --points-normals`: "geometric" (the default), "none" (the vertical and horizontal clearances alone), or a
    file of survey normals."""
    if value in (None, "none"):
        return None
    return "geometric" if value == "geometric" else read_normals(Path(value), n)


def read_normals(path: Path, n: int) -> np.ndarray:
    """[N, 3] surface normals (`--points-normals`), NaN where a point has none."""
    a = np.fromfile(path, "<f4").reshape(-1, 3)
    assert len(a) == n, f"{path}: {len(a)} normals for {n} points"
    return a


def column_tops(scene) -> Tuple[np.ndarray, np.ndarray]:
    """(first fluid level index, its face's height above the floor [m]) per column: the solid top the solve sees."""
    g = scene.grid
    fluid = ~np.asarray(scene.solid, bool)
    kf = np.where(fluid.any(axis=0), fluid.argmax(axis=0), g.nz)
    return kf, g.zf[np.minimum(kf, g.nz)]


def terrain(scene) -> np.ndarray:
    """(ny, nx) bare earth above the floor [m]."""
    return np.zeros(scene.z0.shape) if scene.terrain is None else np.asarray(scene.terrain, float)


def bilinear(a: np.ndarray, scene, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A per-column raster at scene points, bilinear between column centres (held flat past the outer centres)."""
    g, (ox, oy, _) = scene.grid, scene.origin
    fx, fy = (x - ox) / g.dx - 0.5, (y - oy) / g.dx - 0.5
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, max(g.nx - 2, 0))
    j0 = np.clip(np.floor(fy).astype(np.int64), 0, max(g.ny - 2, 0))
    tx, ty = np.clip(fx - i0, 0.0, 1.0), np.clip(fy - j0, 0.0, 1.0)
    i1, j1 = np.minimum(i0 + 1, g.nx - 1), np.minimum(j0 + 1, g.ny - 1)
    return ((1 - ty) * ((1 - tx) * a[j0, i0] + tx * a[j0, i1]) + ty * ((1 - tx) * a[j1, i0] + tx * a[j1, i1]))


def _col(scene, x, y):
    g, (ox, oy, _) = scene.grid, scene.origin
    return (np.clip(np.floor((y - oy) / g.dx).astype(np.int64), 0, g.ny - 1),
            np.clip(np.floor((x - ox) / g.dx).astype(np.int64), 0, g.nx - 1))


def _in_grid(scene, x, y):
    g, (ox, oy, _) = scene.grid, scene.origin
    return (x >= ox) & (x < ox + g.nx * g.dx) & (y >= oy) & (y < oy + g.ny * g.dx)


def _square(scene, jj, ii):
    g, (ox, oy, _) = scene.grid, scene.origin
    return ox + ii * g.dx, oy + jj * g.dx


def _nearest(scene, T, x, y, z, reach: int, want_open: bool, above=None):
    """For each point, the nearest column within `reach` cells that is open (T <= z) or solid (T > z) at height z:
    (distance from the point to that column's square, the nearest point of the square x, y, the square's centre x, y);
    distance inf where none. With `above`, a solid column counts only where its top also stands over it (a wall, not a
    step of the same surface)."""
    g = scene.grid
    j, i = _col(scene, x, y)
    n = len(x)
    best = np.full(n, np.inf)
    qx, qy, cx, cy = (np.zeros(n) for _ in range(4))
    for dj in range(-reach, reach + 1):
        for di in range(-reach, reach + 1):
            jj, ii = j + dj, i + di
            ok = (jj >= 0) & (jj < g.ny) & (ii >= 0) & (ii < g.nx)
            jc, ic = np.clip(jj, 0, g.ny - 1), np.clip(ii, 0, g.nx - 1)
            t = T[jc, ic]
            hit = ok & ((t <= z + 1e-6) if want_open else (t > z + 1e-6))
            if above is not None and not want_open:
                hit &= t > above
            x0, y0 = _square(scene, jc, ic)
            px, py = np.clip(x, x0, x0 + g.dx), np.clip(y, y0, y0 + g.dx)
            d = np.hypot(x - px, y - py)
            take = hit & (d < best)
            best = np.where(take, d, best)
            qx, qy = np.where(take, px, qx), np.where(take, py, qy)
            cx, cy = np.where(take, x0 + 0.5 * g.dx, cx), np.where(take, y0 + 0.5 * g.dx, cy)
    return best, qx, qy, cx, cy


NORMAL_SIGMA_M = 1.0
"""The scale the solids' outward normal is smoothed over: an edge's normal turns from one face's to the other's over
about twice this, so a read 2 m off a roof and one 2 m off its wall meet round the edge. The survey's own per-point
normals are noisier: offset along them, the third site's within-class jumps rose from 0.56 to 2.1 % of edges."""


class GeometricNormals:
    """The outward normal of the solve's own solids, smoothed: minus the gradient of the solid occupancy blurred by a
    Gaussian of NORMAL_SIGMA_M, read trilinearly at any position (x, y scene metres, z above the floor). Near nothing
    solid it is zero and a point keeps its place."""

    def __init__(self, scene, sigma_m: float = NORMAL_SIGMA_M):
        from scipy.ndimage import gaussian_filter
        g = scene.grid
        dz = float(np.median(g.dz[:max(1, min(len(g.dz), 8))]))           # the ground band's cell
        occ = gaussian_filter(np.asarray(scene.solid, np.float32), sigma=(sigma_m / dz, sigma_m / g.dx, sigma_m / g.dx),
                              mode="nearest")
        self.g = [np.gradient(occ, g.dx, axis=2).astype(np.float32), np.gradient(occ, g.dx, axis=1).astype(np.float32),
                  np.gradient(occ, g.zc, axis=0).astype(np.float32)]
        self.scene = scene

    def __call__(self, x, y, z) -> np.ndarray:
        g, (ox, oy, _) = self.scene.grid, self.scene.origin
        fx, fy = (np.asarray(x) - ox) / g.dx - 0.5, (np.asarray(y) - oy) / g.dx - 0.5
        fz = np.interp(np.asarray(z), g.zc, np.arange(g.nz, dtype=float))
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, max(g.nx - 2, 0))
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, max(g.ny - 2, 0))
        k0 = np.clip(np.floor(fz).astype(np.int64), 0, max(g.nz - 2, 0))
        tx, ty, tz = (np.clip(f - f0, 0.0, 1.0) for f, f0 in ((fx, i0), (fy, j0), (fz, k0)))
        out = np.zeros((len(fx), 3))
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    w = (tz if dk else 1 - tz) * (ty if dj else 1 - ty) * (tx if di else 1 - tx)
                    k, j, i = np.minimum(k0 + dk, g.nz - 1), np.minimum(j0 + dj, g.ny - 1), np.minimum(i0 + di, g.nx - 1)
                    for c in range(3):
                        out[:, c] += w * self.g[c][k, j, i]
        return -out


def _clearance(scene, T, x, y, z, reach: int, kf=None) -> np.ndarray:
    """How far a position stands off the solids: its height over its column's solid top, or the horizontal distance to
    the nearest wall at its height if that is less; negative inside a solid, -inf off the grid. A wall is a column
    whose top stands more than FOLLOW_CELLS over this column's: a step of a slope in 1 m cubes is the same surface."""
    g = scene.grid
    j, i = _col(scene, x, y)
    s = z - T[j, i]
    above = None if kf is None else T[j, i] + FOLLOW_CELLS * g.dz[np.minimum(kf[j, i], g.nz - 1)]
    d = _nearest(scene, T, x, y, z, reach, want_open=False, above=above)[0]
    return np.where(_in_grid(scene, x, y), np.where(s < 0, s, np.minimum(s, d)), -np.inf)


@dataclass
class Placed:
    """Where each point is read and with which weights: `idx` (N, 8) flat cell indices into (nz, ny, nx), `w` (N, 8)
    their weights (all zero where no fluid centre is in reach), and per point the search's distance (0 where the
    point was not inside a solid column), whether the vertical clearance raised it, how far the horizontal clearance
    moved it, the height above the floor it is read at, and how far its surface normal carried it off its surface."""

    idx: np.ndarray
    w: np.ndarray
    search_m: np.ndarray
    raised: np.ndarray
    pushed_m: np.ndarray
    z_eval: np.ndarray
    ground_m: np.ndarray
    offset_m: np.ndarray = None

    @property
    def resolved(self) -> np.ndarray:
        return self.w.sum(axis=1) > 0

    def search_u8(self) -> np.ndarray:
        q = np.clip(np.ceil(np.nan_to_num(self.search_m) / SEARCH_STEP_M - 1e-6), 0, UNRESOLVED - 1).astype(np.uint8)
        q[~self.resolved] = UNRESOLVED
        return q


def _place_chunk(scene, kf, T, G, x, y, h, clearance, search_cells, z_floor=None, normals=None):
    g = scene.grid
    x, y = x.astype(np.float64).copy(), y.astype(np.float64).copy()
    h = np.nan_to_num(np.asarray(h, np.float64), nan=0.0)
    n = len(x)
    inside = _in_grid(scene, x, y)
    ground = bilinear(G, scene, x, y)
    z = ground + h if z_floor is None else np.nan_to_num(np.asarray(z_floor, np.float64), nan=0.0)
    search = np.zeros(n)
    # 1. a return inside a solid column: its column's top, or the nearest column open at its height
    j, i = _col(scene, x, y)
    ins = np.nonzero(inside & (z < T[j, i] - 1e-6))[0]
    if len(ins):
        up = T[j[ins], i[ins]] - z[ins]
        side, qx, qy, cx, cy = _nearest(scene, T, x[ins], y[ins], z[ins], search_cells, want_open=True)
        go = side < up
        nudge = 1e-3 * g.dx                           # across the face shared with the open square, into it
        ex, ey = qx - cx, qy - cy
        on_x = np.abs(ex) >= np.abs(ey)
        x[ins[go]] = (qx - np.where(on_x, nudge * np.sign(ex), 0.0))[go]
        y[ins[go]] = (qy - np.where(on_x, 0.0, nudge * np.sign(ey)))[go]
        z[ins[~go]] = T[j[ins[~go]], i[ins[~go]]]
        search[ins] = np.minimum(up, side)
    # 1b. a surface return with a normal leaves its surface along it, to `clearance` off the solid: the normal turns
    #     continuously round an edge, so a roof's reads and a wall's meet there instead of standing 2 m apart
    offset = np.zeros(n)
    anchor = np.zeros(n)
    if normals is not None:
        nrm = normals(x, y, z) if callable(normals) else np.asarray(normals, np.float64)
        ln = np.linalg.norm(nrm, axis=1)
        ok = inside & np.isfinite(ln) & (ln > (1e-4 if callable(normals) else 0.5))   # a field: its direction
        nrm = np.where(ok[:, None], nrm / np.maximum(ln, 1e-12)[:, None], 0.0)
        reach = int(math.ceil(clearance / g.dx)) + 1
        c0 = _clearance(scene, T, x, y, z, reach, kf)
        ok &= c0 < clearance - 1e-9
        if ok.any():
            sel = np.nonzero(ok)[0]
            px, py, pz, pn = x[sel], y[sel], z[sel], nrm[sel]
            cp = _clearance(scene, T, px + clearance * pn[:, 0], py + clearance * pn[:, 1], pz + clearance * pn[:, 2],
                            reach)
            cm = _clearance(scene, T, px - clearance * pn[:, 0], py - clearance * pn[:, 1], pz - clearance * pn[:, 2],
                            reach)
            sgn = np.where(cp >= cm, 1.0, -1.0)[:, None]          # the side facing the air: the fit's sign is arbitrary
            d = (clearance - np.clip(c0[sel], 0.0, None))[:, None]
            q = np.stack([px, py, pz], 1) + sgn * d * pn
            jq, iq = _col(scene, q[:, 0], q[:, 1])
            land = _in_grid(scene, q[:, 0], q[:, 1]) & (q[:, 2] >= T[jq, iq] - 1e-6)   # into a solid: the rules below
            sel = sel[land]
            jo, io = _col(scene, x[sel], y[sel])
            anchor[sel] = T[jo, io]                        # the column of the surface it leaves
            x[sel], y[sel], z[sel] = q[land, 0], q[land, 1], q[land, 2]
            offset[sel] = d[land, 0]
    off = offset > 0
    # 2. the vertical clearance: at least `clearance` above the column's solid top
    j, i = _col(scene, x, y)
    s = z - T[j, i]
    raised = inside & ~off & (s < clearance)
    z_e = np.where(off, z, T[j, i] + np.maximum(s, clearance))   # an offset read is already clear along its normal
    # 3. the horizontal clearance: `clearance` out from any column solid at that height, twice for corners
    pushed = np.zeros(n)
    reach = int(math.ceil(clearance / g.dx)) + 1
    for _ in range(2):
        d, qx, qy, cx, cy = _nearest(scene, T, x, y, z_e, reach, want_open=False)
        need = inside & ~off & (d < clearance - 1e-9)
        if not need.any():
            break
        vx, vy = x - qx, y - qy
        flat = np.hypot(vx, vy) < 1e-9               # on the wall's face: its outward normal, the axis it lies off
        ex, ey = x - cx, y - cy
        on_x = np.abs(ex) >= np.abs(ey)
        vx = np.where(flat, np.where(on_x, np.sign(ex), 0.0), vx)
        vy = np.where(flat, np.where(on_x, 0.0, np.sign(ey)), vy)
        nv = np.hypot(vx, vy)
        need &= nv > 1e-9
        nx_, ny_ = qx + clearance * vx / np.maximum(nv, 1e-12), qy + clearance * vy / np.maximum(nv, 1e-12)
        jn, in_ = _col(scene, nx_, ny_)
        ok = need & _in_grid(scene, nx_, ny_) & (T[jn, in_] <= z_e + 1e-6)
        pushed += np.where(ok, np.hypot(nx_ - x, ny_ - y), 0.0)
        x, y = np.where(ok, nx_, x), np.where(ok, ny_, y)
    # 4. the four columns around the point, each read at the point's height above its own top where it is the same
    #    surface stepped, else at the point's own height; vertically between that column's fluid centres
    j, i = _col(scene, x, y)
    t_own = np.where(off, anchor, T[j, i])            # an offset read keeps the surface it left as its datum
    s_e = z_e - t_own
    tol = FOLLOW_CELLS * g.dz[np.minimum(kf[j, i], g.nz - 1)]
    fx, fy = (x - scene.origin[0]) / g.dx - 0.5, (y - scene.origin[1]) / g.dx - 0.5
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, max(g.nx - 2, 0))
    j0 = np.clip(np.floor(fy).astype(np.int64), 0, max(g.ny - 2, 0))
    tx, ty = np.clip(fx - i0, 0.0, 1.0), np.clip(fy - j0, 0.0, 1.0)
    zc = g.zc
    solid = np.asarray(scene.solid, bool).reshape(-1)
    idx = np.zeros((n, 8), np.int64)
    w = np.zeros((n, 8))
    for c, (dj, di) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        jj, ii = np.minimum(j0 + dj, g.ny - 1), np.minimum(i0 + di, g.nx - 1)
        wh = (ty if dj else 1 - ty) * (tx if di else 1 - tx)
        tc = T[jj, ii]
        zq = np.where(np.abs(tc - t_own) <= tol, tc + s_e, z_e)
        open_ = zq >= tc - 1e-6
        kfrac = np.interp(zq, zc, np.arange(g.nz, dtype=float))
        k0 = np.clip(np.floor(kfrac).astype(np.int64), 0, max(g.nz - 2, 0))
        t = np.clip(kfrac - k0, 0.0, 1.0)
        low = k0 < kf[jj, ii]                         # under the column's first fluid centre: that centre
        k0, t = np.where(low, np.minimum(kf[jj, ii], g.nz - 1), k0), np.where(low, 0.0, t)
        k1 = np.minimum(k0 + 1, g.nz - 1)
        a = (k0 * g.ny + jj) * g.nx + ii
        b = (k1 * g.ny + jj) * g.nx + ii
        wa = np.where(open_ & inside & ~solid[a], wh * (1 - t), 0.0)
        wb = np.where(open_ & inside & ~solid[b], wh * t, 0.0)
        idx[:, 2 * c], idx[:, 2 * c + 1], w[:, 2 * c], w[:, 2 * c + 1] = a, b, wa, wb
    tot = w.sum(axis=1, keepdims=True)
    w = np.where(tot > 1e-12, w / np.maximum(tot, 1e-12), 0.0)
    search[~inside] = np.nan
    return idx, w.astype(np.float32), search, raised, pushed, z_e, ground, offset


def datum_offset(scene, x, y, z, h) -> float:
    """The median of the survey's bare earth under each point (z - h) less the solver's (its floor plus its terrain):
    the two frames' vertical offset, which `place` removes before it reads a point at its own z."""
    d = (np.asarray(z, np.float64) - np.asarray(h, np.float64)) - (scene.origin[2] + bilinear(terrain(scene), scene,
                                                                                               np.asarray(x, np.float64),
                                                                                               np.asarray(y, np.float64)))
    return float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0


def place(scene, x, y, h, clearance: float = CLEARANCE_M, search_cells: int = SEARCH_CELLS,
          chunk: int = CHUNK, z=None, normals=None) -> Placed:
    """Where every point is read in `scene` (module docstring). x, y scene metres; h height above the bare earth.
    With `z` (scene metres, `datum_offset` removed) a point stands at its own elevation instead of at its height over
    the solver's terrain: where the two DTMs disagree (under a building, a bowl, a bridge) a return then keeps its
    place against the solids the solve saw, which were made from the same survey. With `normals` ([N, 3], NaN where
    none) a return within `clearance` of a solid leaves it along its own surface normal."""
    kf, T = column_tops(scene)
    G = terrain(scene)
    zf = None if z is None else np.asarray(z, np.float64) - scene.origin[2]
    if isinstance(normals, str):                      # "geometric": the solids' own smoothed outward normal
        normals = GeometricNormals(scene)
    nm = normals if normals is None or callable(normals) else np.asarray(normals, np.float64)
    parts = [_place_chunk(scene, kf, T, G, np.asarray(x[s:s + chunk]), np.asarray(y[s:s + chunk]),
                          np.asarray(h[s:s + chunk]), clearance, search_cells,
                          None if zf is None else zf[s:s + chunk],
                          nm if nm is None or callable(nm) else nm[s:s + chunk])
             for s in range(0, len(x), chunk)]
    if not parts:
        e = np.zeros(0)
        return Placed(np.zeros((0, 8), np.int64), np.zeros((0, 8), np.float32), e, e.astype(bool), e, e, e, e)
    cat = [np.concatenate([p[k] for p in parts]) for k in range(8)]
    idx = cat[0].astype(np.int32) if scene.grid.nz * scene.grid.ny * scene.grid.nx < 2 ** 31 else cat[0]
    return Placed(idx, cat[1], cat[2].astype(np.float32), cat[3], cat[4].astype(np.float32),
                  cat[5].astype(np.float32), cat[6].astype(np.float32), cat[7].astype(np.float32))


def sample(velocity: np.ndarray, placed: Placed, chunk: int = CHUNK) -> np.ndarray:
    """(3, N) float32 (u, v, w) at the placed points from a cell-centred (3, nz, ny, nx) field; NaN where unresolved."""
    n = len(placed.w)
    out = np.full((3, n), np.nan, np.float32)
    ok = placed.resolved
    for c in range(3):
        f = np.ascontiguousarray(velocity[c], dtype=np.float32).reshape(-1)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            v = (f[placed.idx[s:e]] * placed.w[s:e]).sum(axis=1)
            out[c, s:e] = np.where(ok[s:e], v, np.nan)
    return out


def _pct(a: np.ndarray, qs: Sequence[float]) -> Optional[List[float]]:
    a = a[np.isfinite(a)]
    return [round(float(v), 3) for v in np.percentile(a, qs)] if a.size else None


def receipt(placed: Placed, z_scene: np.ndarray, h: np.ndarray, scene, cell_m: float) -> Dict[str, object]:
    """What the placement did, for the product's sidecar and receipt: how many points needed the fluid search
    and how far it went, how many the clearance raised or moved, and the two bare earths' difference (the survey's,
    z - h, against the solver's over the same point: its median is the datums' offset, its spread the DTMs')."""
    n = max(len(placed.w), 1)
    res = placed.resolved
    s = placed.search_m
    moved = s > 0
    diff = (np.asarray(z_scene, np.float64) - np.asarray(h, np.float64)) - (scene.origin[2] + placed.ground_m)
    med = float(np.nanmedian(diff)) if np.isfinite(diff).any() else None
    return {"points": int(len(placed.w)), "resolved": int(res.sum()), "unresolved": int((~res).sum()),
            "unresolved_share": round(float((~res).sum()) / n, 6),
            "clearance_m": CLEARANCE_M, "search_cells": SEARCH_CELLS,
            "search": {"share": round(float(moved.sum()) / n, 6),
                       "beyond_one_cell_share": round(float((s > cell_m + 1e-6).sum()) / n, 6),
                       "m_p50_p99_max": _pct(s[moved], (50, 99, 100)),
                       "rule": "a return inside a solid column takes the nearer of its column's top and the nearest "
                               f"column open at its height within {SEARCH_CELLS} cells"},
            "raised_share": round(float(placed.raised.sum()) / n, 6),
            "pushed_share": round(float((placed.pushed_m > 0).sum()) / n, 6),
            "normal_offset_share": None if placed.offset_m is None else round(float((placed.offset_m > 0).sum()) / n, 6),
            "dtm_difference_m": {"median": None if med is None else round(med, 3),
                                 "abs_dev_p50_p99": _pct(np.abs(diff - med), (50, 99)) if med is not None else None,
                                 "definition": "the survey's bare earth under each point (z - h) minus the solver's "
                                               "(its floor plus its terrain there): the median is the frames' datum "
                                               "offset, the spread about it the two DTMs' disagreement"}}


class Writer:
    """`points_basis_f16.bin` [headings, N, 3] float16 (u, v, w per unit forcing, heading-major), streamed to disk,
    with `points_search_u8.bin` and `points_basis.json`."""

    NAME, SEARCH, META = "points_basis_f16.bin", "points_search_u8.bin", "points_basis.json"

    def __init__(self, out: Path, headings: Sequence[float], n: int):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.headings = [float(h) for h in headings]
        self.n = n
        self.mm = np.memmap(self.out / self.NAME, mode="w+", dtype="<f2", shape=(len(self.headings), max(n, 1), 3))
        for k in range(len(self.headings)):          # no value until a solve writes one: a missing block reads NaN
            self.mm[k] = np.nan
        self.done = np.zeros(len(self.headings), bool)
        self.missing: Dict[str, int] = {}

    def put(self, heading: float, uvw: np.ndarray, index: Optional[np.ndarray] = None) -> None:
        k = self.headings.index(float(heading))
        if index is None:
            self.mm[k] = np.asarray(uvw, np.float32).T.astype("<f2")
        else:
            self.mm[k, index] = np.asarray(uvw, np.float32).T.astype("<f2")
        self.done[k] = True

    def close(self, placed_receipt: Dict[str, object], search_u8: np.ndarray, extra: Dict[str, object]) -> Dict:
        self.mm.flush()
        del self.mm
        np.asarray(search_u8, np.uint8).tofile(self.out / self.SEARCH)
        meta = {"v": 1, "file": self.NAME, "dtype": "float16", "shape": [len(self.headings), self.n, 3],
                "layout": "heading-major: [heading k][point][u, v, w], m/s per 1 m/s unit forcing",
                "headings_deg": self.headings, "headings_done": int(self.done.sum()),
                "speed": "the norm of (u, v, w): the vertical component is part of what a plant feels by a wall or a roof",
                "search_file": self.SEARCH, "search_codes": f"tenths of a metre the fluid search went, 254 at most; "
                                                            f"{UNRESOLVED} no fluid centre in reach",
                "method": "models/wind points.py: each point at its own z less the frames' offset (placement datum) or at "
                          "its height over the solver's terrain (bare_earth); clearance "
                          f"{CLEARANCE_M:g} m off every solid face; trilinear over fluid centres, each neighbour column "
                          "read at the same height above its own top where it is the same surface stepped",
                **placed_receipt, **extra}
        (self.out / self.META).write_text(json.dumps(meta, indent=1, default=float))
        return meta


def core_points(scene, x: np.ndarray, y: np.ndarray, core: Tuple[int, int, int, int]) -> np.ndarray:
    """Indices of the points whose column lies in a unit's core (r0, r1, c0, c1): every column is in exactly one."""
    j, i = _col(scene, x, y)
    r0, r1, c0, c1 = core
    return np.nonzero(_in_grid(scene, x, y) & (j >= r0) & (j < r1) & (i >= c0) & (i < c1))[0]
