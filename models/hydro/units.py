"""Geographic units of a storm run (WS18 M3): a coarse pass over the whole domain records the discharge
across every unit's edges, and each unit runs at full resolution over its core and an overlap with that
discharge entering across its edges; the units' cores join into the whole.

The local-inertial solver takes a prescribed discharge per edge cell, sampled in time (`inflow.Inflow`);
a unit's edges are lines inside the whole domain, where the coarse pass knows the depth and velocity.
Water leaving a unit leaves across its free edges, as it leaves the whole domain; the overlap keeps a
unit's core away from its edges, so the core sees the neighbouring ground it would see in one domain.

    record = SeamRecorder(transform, shape, xs, ys)        # the coarse pass's frame sink
    ... simulate(..., sink=record) ...
    flow = unit_inflow(record.seams(), window, fine_transform, fine_shape)   # one unit's Inflow
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from inflow import Inflow


@dataclass
class Seams:
    """The coarse pass's discharge across lines of the domain, sampled in time.

    Attributes:
        times_s: Sample times [s] since storm start, increasing.
        xs: Scene x [m] of each vertical line; ys: scene y [m] of each horizontal line.
        row_y: Scene y [m] of the coarse rows the vertical lines are sampled along.
        col_x: Scene x [m] of the coarse columns the horizontal lines are sampled along.
        qx: [len(xs), n, rows] unit discharge eastward [m^2/s] across each vertical line.
        qy: [len(ys), n, cols] unit discharge southward [m^2/s] across each horizontal line: the solver's
            v and its face discharge qy are positive southward, rows running north to south
            (`solver.FIELDS`).
    """

    times_s: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    row_y: np.ndarray
    col_x: np.ndarray
    qx: np.ndarray
    qy: np.ndarray

    def save(self, path) -> None:
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path) -> "Seams":
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})


def _bracket(centres: np.ndarray, at: float) -> Tuple[int, float]:
    """The first of the two cells whose centres (evenly spaced, either way) bracket `at`, and the second's
    weight, held at the ends."""
    if len(centres) < 2:
        return 0, 0.0
    p = (at - centres[0]) / (centres[1] - centres[0])
    i = int(np.clip(np.floor(p), 0, len(centres) - 2))
    return i, float(np.clip(p - i, 0.0, 1.0))


class SeamRecorder:
    """A frame sink for the coarse pass (`solver.simulate(sink=...)`): at every frame, the unit discharge
    h*u across each vertical line and h*v (southward) across each horizontal line, interpolated linearly
    between the two coarse cells whose centres bracket the line, dry and unmodelled cells carrying none.

    Args:
        transform: Affine transform of the coarse grid (north-up: e < 0).
        shape: (rows, cols) of the coarse grid.
        xs, ys: Scene x of the vertical lines and y of the horizontal lines [m].
        chain: Another sink to hand every frame on to, or None.
    """

    def __init__(self, transform: object, shape: Tuple[int, int], xs: Sequence[float], ys: Sequence[float],
                 chain=None):
        rows, cols = shape
        self.t, self.chain = transform, chain
        self.col_x = transform.c + (np.arange(cols) + 0.5) * transform.a
        self.row_y = transform.f + (np.arange(rows) + 0.5) * transform.e
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        self.cols = [_bracket(self.col_x, x) for x in self.xs]
        self.rows = [_bracket(self.row_y, y) for y in self.ys]
        self.times: List[float] = []
        self.qx: List[np.ndarray] = []
        self.qy: List[np.ndarray] = []

    def __call__(self, t_s: float, frame: np.ndarray) -> None:
        h, u, v = (np.nan_to_num(a.astype(np.float64)) for a in frame[:3])
        qx, qy = h * u, h * v
        last_c, last_r = h.shape[1] - 1, h.shape[0] - 1
        self.times.append(float(t_s))
        self.qx.append(np.stack([(1.0 - w) * qx[:, c] + w * qx[:, min(c + 1, last_c)] for c, w in self.cols])
                       if self.cols else np.zeros((0, h.shape[0])))
        self.qy.append(np.stack([(1.0 - w) * qy[r, :] + w * qy[min(r + 1, last_r), :] for r, w in self.rows])
                       if self.rows else np.zeros((0, h.shape[1])))
        if self.chain is not None:
            self.chain(t_s, frame)

    def seams(self) -> Seams:
        qx = np.stack(self.qx, axis=1) if self.qx else np.zeros((len(self.xs), 0, len(self.row_y)))
        qy = np.stack(self.qy, axis=1) if self.qy else np.zeros((len(self.ys), 0, len(self.col_x)))
        return Seams(np.asarray(self.times), self.xs, self.ys, self.row_y, self.col_x, qx, qy)


def _along(q: np.ndarray, at: np.ndarray, to: np.ndarray) -> np.ndarray:
    """[n, len(at)] samples along a line, interpolated linearly onto the positions `to` (held at the ends)."""
    order = np.argsort(at)
    return np.stack([np.interp(to, at[order], row[order]) for row in q]) if len(q) else np.zeros((0, len(to)))


def unit_inflow(seams: Seams, bounds_m: Tuple[float, float, float, float], transform: object,
                shape: Tuple[int, int], tol_m: float = 1e-6) -> Inflow:
    """The discharge entering one unit across its four edges, from the coarse pass.

    An edge that lies on a recorded line takes that line's discharge where it points into the unit
    (eastward across the west edge, westward across the east edge, southward across the north edge,
    northward across the south edge); an edge on no line (the whole domain's own boundary) takes none.
    The recorded qy is southward, as the solver's is, so the north edge takes +qy and the south edge -qy.

    Args:
        seams: The coarse pass's record.
        bounds_m: (x0, x1, y0, y1) of the unit's window in scene metres, its overlap included.
        transform: Affine transform of the unit's fine grid (north-up).
        shape: (rows, cols) of the unit's fine grid.
    """
    rows, cols = shape
    y = transform.f + (np.arange(rows) + 0.5) * transform.e
    x = transform.c + (np.arange(cols) + 0.5) * transform.a
    x0, x1, y0, y1 = bounds_m
    n = len(seams.times_s)

    def line(values: np.ndarray, where: float, lines: np.ndarray) -> np.ndarray:
        hit = np.nonzero(np.abs(lines - where) <= tol_m)[0]
        return values[hit[0]] if len(hit) else None

    def edge(q, at, to, sign) -> np.ndarray:
        if q is None:
            return np.zeros((n, len(to)))
        return np.clip(sign * _along(q, at, to), 0.0, None)

    return Inflow(times_s=np.asarray(seams.times_s, float),
                  west=edge(line(seams.qx, x0, seams.xs), seams.row_y, y, +1.0),
                  east=edge(line(seams.qx, x1, seams.xs), seams.row_y, y, -1.0),
                  north=edge(line(seams.qy, y1, seams.ys), seams.col_x, x, +1.0),
                  south=edge(line(seams.qy, y0, seams.ys), seams.col_x, x, -1.0))


def unit_windows(n_rows: int, n_cols: int, a: int, b: int, overlap: int) -> List[Dict[str, Tuple[int, int, int, int]]]:
    """The a x b units of an n_rows x n_cols grid, row-major: each unit's core (r0, r1, c0, c1) and its
    window, the core grown by `overlap` cells a side within the grid."""
    rs = [round(i * n_rows / a) for i in range(a + 1)]
    cs = [round(j * n_cols / b) for j in range(b + 1)]
    out = []
    for i in range(a):
        for j in range(b):
            core = (rs[i], rs[i + 1], cs[j], cs[j + 1])
            win = (max(core[0] - overlap, 0), min(core[1] + overlap, n_rows),
                   max(core[2] - overlap, 0), min(core[3] + overlap, n_cols))
            out.append({"core": core, "window": win})
    return out


def join_cores(units: Sequence[Tuple[Dict[str, Tuple[int, int, int, int]], np.ndarray]],
               shape: Tuple[int, int]) -> np.ndarray:
    """A whole-grid field [..., rows, cols] from each unit's field over its window, taking only its core."""
    first = units[0][1]
    out = np.full(first.shape[:-2] + shape, np.nan, dtype=first.dtype)
    seen = np.zeros(shape, np.int32)
    for spec, field in units:
        r0, r1, c0, c1 = spec["core"]
        w0, _, v0, _ = spec["window"]
        out[..., r0:r1, c0:c1] = field[..., r0 - w0:r1 - w0, c0 - v0:c1 - v0]
        seen[r0:r1, c0:c1] += 1
    assert (seen == 1).all(), "the cores do not cover the grid once"
    return out


def crop_surface(surface, window: Tuple[int, int, int, int]):
    """The solver's surface over one unit's window: every per-cell field cropped, scalars kept."""
    import dataclasses

    r0, r1, c0, c1 = window
    cut = lambda a: a[r0:r1, c0:c1].copy() if isinstance(a, np.ndarray) and a.ndim == 2 else a  # noqa: E731
    return dataclasses.replace(surface, **{f.name: cut(getattr(surface, f.name)) for f in dataclasses.fields(surface)})


def crop_rim(flow: Inflow, shape: Tuple[int, int], window: Tuple[int, int, int, int]):
    """The rim delivery of a whole-domain `Inflow` (the fetch disc's own rim) that falls inside one unit's
    window, re-indexed to the window; (flat indices, [n, k] rates), or (None, None) when it holds none."""
    if flow is None or flow.rim_index is None:
        return None, None
    r0, r1, c0, c1 = window
    rr, cc = np.unravel_index(flow.rim_index, shape)
    inside = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    if not inside.any():
        return None, None
    return (rr[inside] - r0) * (c1 - c0) + (cc[inside] - c0), flow.rim[:, inside]


def combine(edges: Inflow, rim_index, rim, rim_times_s) -> Inflow:
    """One `Inflow` with a unit's edge discharge and the rim delivery inside it, on the union of their
    sample times (each held at its ends, linear between its samples)."""
    if rim_index is None:
        return edges
    t = np.union1d(edges.times_s, rim_times_s)
    at = lambda times, a: np.stack([np.interp(t, times, a[:, j]) for j in range(a.shape[1])], axis=1) \
        if a.shape[1] else np.zeros((len(t), 0))  # noqa: E731
    return Inflow(times_s=t, west=at(edges.times_s, edges.west), east=at(edges.times_s, edges.east),
                  north=at(edges.times_s, edges.north), south=at(edges.times_s, edges.south),
                  rim_index=rim_index, rim=at(np.asarray(rim_times_s, float), rim))


def split(centre: Tuple[float, float], reach_m: float, a: int, b: int,
          overlap_m: float) -> List[Dict[str, Tuple[float, float, float, float]]]:
    """The a x b geographic units of the fetch disc's square, in metres of the grid's frame, row-major from
    the north-west: each unit's core (x0, x1, y0, y1), the square cut evenly, and its window, the core
    grown by `overlap_m` a side within the square. Every pass computes them from the same four numbers,
    so a unit run finds its edges among the coarse pass's recorded lines exactly."""
    cx, cy = centre
    xs = [cx - reach_m + 2.0 * reach_m * j / b for j in range(b + 1)]
    ys = [cy + reach_m - 2.0 * reach_m * i / a for i in range(a + 1)]
    lo_x, hi_x, lo_y, hi_y = cx - reach_m, cx + reach_m, cy - reach_m, cy + reach_m
    out = []
    for i in range(a):
        for j in range(b):
            core = (xs[j], xs[j + 1], ys[i + 1], ys[i])
            win = (max(core[0] - overlap_m, lo_x), min(core[1] + overlap_m, hi_x),
                   max(core[2] - overlap_m, lo_y), min(core[3] + overlap_m, hi_y))
            out.append({"core": core, "window": win})
    return out


OUTER_M = 1e9
"""Where the outermost cores and windows of a box split end [m]: past every grid, so `cells` clips them to
each pass's own grid, whose snapped edges differ by a cell or two between the coarse pass and the fine."""


def split_box(fetch: Tuple[float, float, float, float], a: int, b: int,
              overlap_m: float) -> List[Dict[str, Tuple[float, float, float, float]]]:
    """`split` for a site that follows its ordered polygon: its fetch box (x0, y0, x1, y1) cut evenly, the same
    interior lines at every pass's cell. The outer edges reach every grid's own edge (OUTER_M), so the cores
    cover each pass's grid once; a window edge past the box reaches it too, as `split` clamps to the square."""
    x0, y0, x1, y1 = fetch
    xs = [x0 + (x1 - x0) * j / b for j in range(b + 1)]
    ys = [y1 - (y1 - y0) * i / a for i in range(a + 1)]
    xs[0], xs[-1], ys[0], ys[-1] = -OUTER_M, OUTER_M, OUTER_M, -OUTER_M
    lo = lambda v, edge: v if v > edge else -OUTER_M  # noqa: E731
    hi = lambda v, edge: v if v < edge else OUTER_M  # noqa: E731
    out = []
    for i in range(a):
        for j in range(b):
            core = (xs[j], xs[j + 1], ys[i + 1], ys[i])
            win = (lo(core[0] - overlap_m, x0), hi(core[1] + overlap_m, x1),
                   lo(core[2] - overlap_m, y0), hi(core[3] + overlap_m, y1))
            out.append({"core": core, "window": win})
    return out


def seam_lines_box(units: Sequence[Dict[str, Tuple[float, float, float, float]]],
                   fetch: Tuple[float, float, float, float]) -> Tuple[List[float], List[float]]:
    """`seam_lines` for `split_box`: every window edge inside the fetch box, each once."""
    xs = sorted({e for u in units for e in u["window"][:2] if fetch[0] < e < fetch[2]})
    ys = sorted({e for u in units for e in u["window"][2:] if fetch[1] < e < fetch[3]})
    return xs, ys


def seam_lines(units: Sequence[Dict[str, Tuple[float, float, float, float]]], centre: Tuple[float, float],
               reach_m: float) -> Tuple[List[float], List[float]]:
    """Every window edge inside the square: the lines the coarse pass records (x of the vertical, y of
    the horizontal), each once."""
    cx, cy = centre
    xs = sorted({e for u in units for e in u["window"][:2] if cx - reach_m < e < cx + reach_m})
    ys = sorted({e for u in units for e in u["window"][2:] if cy - reach_m < e < cy + reach_m})
    return xs, ys


def cells(transform: object, shape: Tuple[int, int], bounds_m: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """(r0, r1, c0, c1): the cells of a north-up grid whose centres lie in [x0, x1) x [y0, y1)."""
    x0, x1, y0, y1 = bounds_m
    dx, dy = transform.a, -transform.e
    c = lambda x: int(np.clip(np.ceil((x - transform.c) / dx - 0.5 - 1e-9), 0, shape[1]))  # noqa: E731
    r = lambda y: int(np.clip(np.floor((transform.f - y) / dy - 0.5 + 1e-9) + 1, 0, shape[0]))  # noqa: E731
    return r(y1), r(y0), c(x0), c(x1)
