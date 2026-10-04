"""The wind solver's finite volumes on an adaptive layout (`amr.Layout`): the leaves are the cells, and every face is the
union of the dense grid's faces between two leaves.

Discretization at a face between a leaf and its neighbor of another size: the two-point flux, conductance = open area /
distance between the two centers along the axis, the same flux leaving one cell and entering the other (conservative)
and the same conductance in both rows (symmetric): Losasso, Gibou and Fedkiw 2004 (ACM Trans Graph 23: 457), sections
4.1 to 4.3, the octree pressure solve whose truncation error is O(1) at a coarse-fine face but whose solution converges,
measured here by the method of manufactured solutions (`tests/test_amr.py`). On faces between two single grid cells
every quantity is the dense solver's own (`solver.Model`), so a layout of single cells reproduces it to round-off.

Velocities are the dense solver's: normal components on faces (the divergence-free field), cell values the mean of each
side's faces, area-weighted where a side has several. The projection is D A G with D the adjoint of G, so the corrected
faces are divergence-free to the solve's tolerance on any layout.

Grid axes are (z, y, x) = (0, 1, 2); the velocity components are (u, v, w) = (x, y, z), so axis a carries component
2 - a.
"""

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import amr

Tensor = torch.Tensor

SIDES = ("west", "east", "south", "north", "floor", "top")
SIDE_AXIS = {"west": 2, "east": 2, "south": 1, "north": 1, "floor": 0, "top": 0}
SIDE_HIGH = {"west": False, "east": True, "south": False, "north": True, "floor": False, "top": True}


def comp(axis: int) -> int:
    """The velocity component normal to faces across grid axis `axis`."""
    return 2 - axis


@dataclass
class Faces:
    """Interior faces of one axis: low leaf `a`, high leaf `b` (along +axis), sorted by (a, b)."""

    axis: int
    a: Tensor
    b: Tensor
    area: Tensor          # geometric [m^2]
    area_open: Tensor     # open [m^2]: the dense faces' open shares summed, 0 on a face to or between solids
    dist: Tensor          # [m] between the centers (the dense solver's own between single cells)
    single: Tensor        # both leaves single grid cells

    @property
    def n(self) -> int:
        return int(self.a.numel())

    @property
    def open(self) -> Tensor:
        return self.area_open > 0


@dataclass
class Bound:
    """Boundary faces on one side of the domain: one per leaf touching it."""

    side: str
    cell: Tensor
    area: Tensor
    area_open: Tensor
    dist: Tensor          # center to the boundary [m]
    values: Dict[str, Tensor] = field(default_factory=dict)   # prescribed values, area-weighted over the dense faces

    @property
    def n(self) -> int:
        return int(self.cell.numel())

    @property
    def axis(self) -> int:
        return SIDE_AXIS[self.side]

    @property
    def high(self) -> bool:
        return SIDE_HIGH[self.side]


def _plane(owner: Tensor, side: str) -> Tuple[Tensor, Tensor]:
    """(owners, (k, j, i) of each dense cell) on a side's boundary plane, kept cells only."""
    ax = SIDE_AXIS[side]
    idx = owner.shape[ax] - 1 if SIDE_HIGH[side] else 0
    o = owner.select(ax, idx)
    pos2 = (o >= 0).nonzero()
    pos = torch.zeros(pos2.shape[0], 3, dtype=torch.long, device=owner.device)
    others = [d for d in range(3) if d != ax]
    pos[:, others[0]], pos[:, others[1]] = pos2[:, 0], pos2[:, 1]
    pos[:, ax] = idx
    return o[pos2[:, 0], pos2[:, 1]].long(), pos


class Graph:
    """An adaptive layout with the geometry the solver reads, from a dense model of the same scene (`solver.Model`,
    float64, its geometry intact: built with `dense_model`)."""

    def __init__(self, model, lay: "amr.Layout", device=None):
        dev = torch.device(device or model.sink.device)
        f64 = dict(dtype=torch.float64, device=dev)
        self.device, self.lay = dev, lay
        self.n = n = lay.n
        nz, ny, nx = lay.shape
        self.shape = lay.shape
        self.dx = dx = float(model.dx)
        own = torch.as_tensor(lay.owner, device=dev)
        self.level = torch.as_tensor(lay.level, device=dev)
        an = torch.as_tensor(lay.anchor, device=dev).long()
        s = torch.as_tensor(lay.size, device=dev)
        self.anchor, self.size = an, s
        zf = model.zf.to(**f64)
        self.top_z, self.zf = float(zf[-1]), zf
        self.zc = 0.5 * (zf[an[:, 0]] + zf[an[:, 0] + s])
        self.hz = zf[an[:, 0] + s] - zf[an[:, 0]]
        self.hx = s.double() * dx
        self.yc = (an[:, 1].double() + s.double() / 2) * dx
        self.xc = (an[:, 2].double() + s.double() / 2) * dx
        self.single = self.level == 0
        self.anchor_flat = (an[:, 0] * ny + an[:, 1]) * nx + an[:, 2]
        flat = own.reshape(-1)
        keep = flat >= 0
        self._cells_of = flat[keep].int()              # each kept dense cell's leaf (int32: half the bytes)
        self._dense_kept = keep.nonzero()[:, 0].int()  # and its flat index
        self.favor = bool(model.favor)
        vol_d = (model.vol.expand(nz, ny, nx)).to(**f64)
        self.vol = self.cell_sum(vol_d)
        self.vol_open = self.cell_sum(model.vol_open.to(**f64)) if self.favor else self.vol.clone()
        self.solid = self.cell_copy(model.solid.to(dev).double(), mean=False) > 0.5
        self.solid &= self.single
        self.phantom = torch.as_tensor(getattr(lay, "phantom", np.zeros(n, bool)), device=dev)
        self.solid |= self.phantom                     # a tile's empty slot: no volume, no face, nothing moves
        self.faces: List[Faces] = []
        fs = amr.face_sets(lay.owner, dev)
        center = (self.zc, self.yc, self.xc)
        for f in fs:
            k, j, i = f.pos[:, 0], f.pos[:, 1], f.pos[:, 2]
            if f.axis == 2:
                op = model.open_x[k, j, i].double()
                ao = (model.ax_open[k, j, i + 1] if self.favor else model.area_x[k, 0, 0]).to(**f64)
                area = (model.area_x[k, 0, 0]).to(**f64)
                d = torch.full_like(area, dx)
            elif f.axis == 1:
                op = model.open_y[k, j, i].double()
                ao = (model.ay_open[k, j + 1, i] if self.favor else model.area_x[k, 0, 0]).to(**f64)
                area = (model.area_x[k, 0, 0]).to(**f64)
                d = torch.full_like(area, dx)
            else:
                op = model.open_z[k, j, i].double()
                area = torch.full(k.shape, float(model.area_z), **f64)
                ao = area
                d = (model.dzf[k, j, i] if self.favor else model.dzc[k]).to(**f64)
            single = self.single[f.a] & self.single[f.b]
            area_s = f.sum(area)
            ao_s = f.sum(op * ao)
            d_s = f.sum(d)                                  # one dense face where both are single cells
            dist = torch.where(single, d_s, (center[f.axis][f.b] - center[f.axis][f.a]).abs())
            self.faces.append(Faces(f.axis, f.a.int(), f.b.int(), area_s, ao_s, dist, single))
        self._face_sets = fs
        self.bounds: Dict[str, Bound] = {}
        for side in SIDES:
            cells, pos = _plane(own, side)
            key, inv = torch.unique(cells, return_inverse=True)
            m = key.numel()
            ssum = lambda v: torch.zeros(m, dtype=v.dtype, device=dev).index_add_(0, inv, v)  # noqa: E731
            k, j, i = pos[:, 0], pos[:, 1], pos[:, 2]
            fluid = ~model.solid[k, j, i]
            if side in ("west", "east"):
                area = model.area_x[k, 0, 0].to(**f64)
                ao = (model.ax_open[k, j, 0 if side == "west" else -1] if self.favor else area).to(**f64)
                ao = ao * fluid
                half = self.hx[key] / 2
                d = torch.where(self.single[key], torch.full_like(half, dx / 2), half)
            elif side in ("south", "north"):
                area = model.area_x[k, 0, 0].to(**f64)
                ao = (model.ay_open[k, 0 if side == "south" else -1, i] if self.favor else area).to(**f64)
                ao = ao * fluid
                half = self.hx[key] / 2
                d = torch.where(self.single[key], torch.full_like(half, dx / 2), half)
            elif side == "top":
                area = torch.full(k.shape, float(model.area_z), **f64)
                ao = area * fluid
                d = torch.where(self.single[key], torch.full_like(self.zc[key], model.d_top), self.top_z - self.zc[key])
            else:
                area = torch.full(k.shape, float(model.area_z), **f64)
                ao = torch.zeros_like(area)
                d = self.hz[key] / 2
            b = Bound(side, key, ssum(area), ssum(ao), d)
            b._inv, b._pos, b._w = inv, pos, area            # the dense faces it sums, for its prescribed values
            self.bounds[side] = b
        self.colors = self._colors(self.anchor, self.size)

    # ── sums and copies from the dense grid ──────────────────────────────────────────────────────────────────────────

    def cell_sum(self, dense: Tensor) -> Tensor:
        """Per leaf, the sum of a dense (nz, ny, nx) quantity over its cells."""
        v = dense.reshape(-1)[self._dense_kept]
        return torch.zeros(self.n, dtype=v.dtype, device=v.device).index_add_(0, self._cells_of, v)

    def cell_copy(self, dense: Tensor, mean: bool = True, weight: Optional[Tensor] = None) -> Tensor:
        """Per leaf, a dense quantity: a single cell's own value exactly, a coarse leaf's (volume-)weighted mean."""
        own = dense.reshape(-1)[self.anchor_flat]
        if not mean:
            return own
        w = (weight if weight is not None else torch.ones((), dtype=dense.dtype, device=dense.device)).expand_as(dense)
        avg = self.cell_sum(dense * w) / self.cell_sum(w.contiguous()).clamp(min=1e-300)
        return torch.where(self.single, own, avg)

    def bound_values(self, side: str, dense_face_values: Tensor) -> Tensor:
        """A side's prescribed values (..., per dense boundary face, in `_pos` order) summed area-weighted onto its
        boundary faces: a single cell's own value exactly."""
        b = self.bounds[side]
        lead = dense_face_values.shape[:-1]
        v = dense_face_values.reshape(-1, dense_face_values.shape[-1])
        out = torch.zeros(v.shape[0], b.n, dtype=v.dtype, device=v.device).index_add_(1, b._inv, v * b._w.to(v.dtype))
        wsum = torch.zeros(b.n, dtype=v.dtype, device=v.device).index_add_(0, b._inv, b._w.to(v.dtype))
        avg = out / wsum
        first = torch.zeros(b.n, dtype=torch.long, device=v.device)
        first[b._inv] = torch.arange(b._inv.numel(), device=v.device)   # one dense face of each (exact for a single)
        avg = torch.where(self.single[b.cell], v[:, first], avg)
        return avg.reshape(*lead, b.n)

    @staticmethod
    def _colors(anchor: Tensor, size: Tensor) -> Tensor:
        """Four colors, a proper coloring of the face graph under 2:1 balance: a leaf's parity in units of its own
        size, and its level's parity. Leaves of one size alternate as the red-black sweep's cells do; neighbors of two
        sizes differ in level parity."""
        s = size
        par = ((anchor[:, 0] // s + anchor[:, 1] // s + anchor[:, 2] // s) % 2)
        lev = torch.log2(s.double()).round().long() % 2
        return (par + 2 * lev).to(torch.int8)

    # ── the projection ───────────────────────────────────────────────────────────────────────────────────────────────

    def divergence(self, uf: List[Tensor], ub: Dict[str, Tensor]) -> Tensor:
        """Net outward volume flux of every leaf [m^3/s] from face velocities `uf` (one tensor per axis, (..., F_a)) and
        boundary face velocities `ub` (per side)."""
        lead = uf[0].shape[:-1]
        out = torch.zeros(*lead, self.n, dtype=uf[0].dtype, device=uf[0].device)
        for f, u in zip(self.faces, uf):
            q = f.area_open.to(u.dtype) * u
            out.index_add_(-1, f.a, q)
            out.index_add_(-1, f.b, -q)
        for side, b in self.bounds.items():
            if side not in ub:
                continue
            q = b.area_open.to(ub[side].dtype) * ub[side]
            out.index_add_(-1, b.cell, q if b.high else -q)
        return out

    def projection_conductances(self) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """Unit conductances: open area over distance on open faces, the open top a boundary at the multiplier's zero;
        the sides carry none (the profile is prescribed there)."""
        c = [f.area_open / f.dist * f.open for f in self.faces]
        top = self.bounds["top"]
        return c, {"top": top.area_open / top.dist}

    def face_gradient(self, lam: Tensor) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """The face velocities a multiplier adds: its gradient on open faces, and across the open top to its zero."""
        out = []
        for f in self.faces:
            g = (lam[..., f.b] - lam[..., f.a]) / f.dist.to(lam.dtype)
            out.append(torch.where(f.open, g, torch.zeros_like(g)))
        top = self.bounds["top"]
        gt = -lam[..., top.cell] / top.dist.to(lam.dtype)
        return out, {"top": torch.where(top.area_open > 0, gt, torch.zeros_like(gt))}

    def side_mean(self, values: Tensor, axis: int, high: bool, bvals: Optional[Dict[str, Tensor]] = None,
                  weight: str = "area") -> Tensor:
        """Per leaf, the area-weighted mean over the faces on one of its sides of a per-face quantity (`values`, (...,
        F_axis)); boundary faces on that side take `bvals[side]`."""
        f = self.faces[axis]
        w = f.area.to(values.dtype)
        idx = f.a if high else f.b                  # the leaf whose high (low) side the face is on
        num = torch.zeros(*values.shape[:-1], self.n, dtype=values.dtype, device=values.device)
        den = torch.zeros(self.n, dtype=values.dtype, device=values.device)
        num.index_add_(-1, idx, values * w)
        den.index_add_(0, idx, w)
        side = [s for s in SIDES if SIDE_AXIS[s] == axis and SIDE_HIGH[s] == high][0]
        b = self.bounds[side]
        if bvals is not None and side in bvals:
            bw = b.area.to(values.dtype)
            num.index_add_(-1, b.cell, bvals[side] * bw)
        den.index_add_(0, b.cell, b.area.to(values.dtype))
        return num / den.clamp(min=1e-300)


# ── operators: L x = diag x - sum over faces of the neighbor terms, assembled as CSR ─────────────────────────────────


class Pattern:
    """The sparsity of the face graph on one level: CSR rows and columns, and where each face's two off-diagonal
    entries and each leaf's diagonal land in the value array."""

    def __init__(self, n: int, a: Tensor, b: Tensor):
        dev = a.device
        self.n = n
        a, b = a.long(), b.long()
        rows = torch.cat([a, b, torch.arange(n, device=dev)])
        cols = torch.cat([b, a, torch.arange(n, device=dev)])
        order = torch.argsort(rows * n + cols)
        self.perm = order.int()                              # value array = cat([ab, ba, diag])[perm]
        rows = rows[order]
        self.col = cols[order].to(torch.int32)
        crow = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        crow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
        self.crow = crow.to(torch.int32)

    def matrix(self, ab: Tensor, ba: Tensor, diag: Tensor) -> Tensor:
        vals = torch.cat([ab, ba, diag])[self.perm]
        with warnings.catch_warnings():                 # torch calls its CSR support beta; the product is cuSPARSE's
            warnings.simplefilter("ignore", UserWarning)
            return torch.sparse_csr_tensor(self.crow, self.col, vals, size=(self.n, self.n),
                                           check_invariants=False)


@dataclass
class GraphOp:
    """M x = ident x + sum_f c_f (x - x_nb) + sum_f F_f x_upwind(f) over the faces of one level (all axes in one list:
    a, b, c, F), with boundary faces (cell, c, F, high): `symmetric` when no flux is given. The convective part is the
    dense `poisson.Convective`'s first-order upwinding by the face volume fluxes."""

    level: "MGLevel"
    ident: Tensor
    c: Tensor
    cb: Tensor
    fl: Optional[Tensor] = None
    flb: Optional[Tensor] = None

    def __post_init__(self):
        lv = self.level
        dt = self.c.dtype
        pos = lambda f: f.clamp(min=0.0)       # noqa: E731
        neg = lambda f: (-f).clamp(min=0.0)    # noqa: E731
        diag = self.ident.clone()
        diag.index_add_(0, lv.a, self.c)
        diag.index_add_(0, lv.b, self.c)
        diag.index_add_(0, lv.bcell, self.cb)
        if self.fl is not None:
            diag.index_add_(0, lv.a, pos(self.fl))
            diag.index_add_(0, lv.b, neg(self.fl))
            out_b = torch.where(lv.bhigh, pos(self.flb), neg(self.flb))
            diag.index_add_(0, lv.bcell, out_b)
            ab, ba = -(self.c + neg(self.fl)), -(self.c + pos(self.fl))
        else:
            ab = ba = -self.c
        self.diag = diag
        self._ab, self._ba = ab, ba
        self._A = None
        self._csr = (lv, dt)

    @property
    def symmetric(self) -> bool:
        return getattr(self, "_sym", self.fl is None)

    def offdiag(self) -> Tuple[Tensor, Tensor]:
        """Each face's two matrix entries: row a column b, and row b column a."""
        return self._ab, self._ba

    @property
    def A(self) -> Tensor:
        """The CSR matrix, assembled when first read: a level the tile kernels hold never reads it."""
        if self._A is None:
            lv, dt = self._csr
            self._A = lv.get_pattern().matrix(self._ab.to(dt), self._ba.to(dt), self.diag)
        return self._A

    def apply(self, x: Tensor) -> Tensor:
        """(C, n) -> (C, n)."""
        return (self.A @ x.T.contiguous()).T

    def to(self, dtype) -> "GraphOp":
        if dtype == self.c.dtype:
            return self
        cv = lambda t: None if t is None else t.to(dtype)  # noqa: E731
        return GraphOp(self.level, cv(self.ident), cv(self.c), cv(self.cb), cv(self.fl), cv(self.flb))

    def coarsen(self, nxt: "MGLevel") -> "GraphOp":
        """The next level's operator: conductances rediscretized (sum of c_f d_f / D over the faces a coarse face
        unites), fluxes and identity summed, as `poisson.Operator.coarsen` does on the dense grid."""
        lv = self.level
        m = lv.to_next
        ident = torch.zeros(nxt.n, dtype=self.ident.dtype, device=self.ident.device).index_add_(0, m.agg, self.ident)
        keep, fk, sk = _kept(m)
        c = torch.zeros(nxt.nf, dtype=self.c.dtype, device=self.c.device).index_add_(
            0, fk, self.c[keep] * sk.to(self.c.dtype))
        cb = torch.zeros(nxt.nb, dtype=self.c.dtype, device=self.c.device).index_add_(
            0, m.bface, self.cb * m.bscale.to(self.c.dtype))
        fl = flb = None
        if self.fl is not None:
            fl = torch.zeros(nxt.nf, dtype=self.fl.dtype, device=self.fl.device).index_add_(0, fk, self.fl[keep])
            flb = torch.zeros(nxt.nb, dtype=self.fl.dtype, device=self.fl.device).index_add_(0, m.bface, self.flb)
        return GraphOp(nxt, ident, c, cb, fl, flb)


@dataclass
class ToNext:
    agg: Tensor          # (n,) each leaf's aggregate on the next level
    face: Tensor         # (nf,) each face's coarse face, -1 inside an aggregate
    scale: Tensor        # (nf,) d_f / D_coarse
    bface: Tensor        # (nb,) each boundary face's coarse boundary face
    bscale: Tensor


@dataclass
class MGLevel:
    """One level of the hierarchy: its leaves (anchor, size), faces (a, b, axis, center distance) and boundary faces
    (cell, side), its CSR pattern and colors."""

    n: int
    anchor: Tensor
    size: Tensor
    center: Tuple[Tensor, Tensor, Tensor]
    a: Tensor
    b: Tensor
    axis: Tensor
    dist: Tensor
    bcell: Tensor
    bside: Tensor
    bhigh: Tensor
    bdist: Tensor
    pattern: Pattern = None
    colors: Tensor = None
    to_next: Optional[ToNext] = None

    def get_pattern(self) -> Pattern:
        """The CSR pattern, made when first read (a level the tile kernels hold never reads it)."""
        if self.pattern is None:
            self.pattern = Pattern(self.n, self.a, self.b)
        return self.pattern

    @property
    def nf(self) -> int:
        return int(self.a.numel())

    @property
    def nb(self) -> int:
        return int(self.bcell.numel())


def _level0(gr: Graph) -> MGLevel:
    a = torch.cat([f.a for f in gr.faces])
    b = torch.cat([f.b for f in gr.faces])
    axis = torch.cat([torch.full((f.n,), f.axis, dtype=torch.int8, device=gr.device) for f in gr.faces])
    dist = torch.cat([f.dist for f in gr.faces])
    bcell = torch.cat([gr.bounds[s].cell for s in SIDES])
    bside = torch.cat([torch.full((gr.bounds[s].n,), i, dtype=torch.int8, device=gr.device) for i, s in enumerate(SIDES)])
    bhigh = torch.cat([torch.full((gr.bounds[s].n,), SIDE_HIGH[s], dtype=torch.bool, device=gr.device) for s in SIDES])
    bdist = torch.cat([gr.bounds[s].dist for s in SIDES])
    vol = gr.hx * gr.hx * gr.hz         # geometric: an aggregate of phantom slots still has a center
    lv = MGLevel(gr.n, gr.anchor, gr.size, (gr.zc, gr.yc, gr.xc), a, b, axis, dist, bcell, bside, bhigh, bdist)
    lv.vol = vol
    lv.colors = gr.colors
    return lv


def _coarser(lv: MGLevel, m: int, nz: int, top_z: float, zf: Tensor) -> Optional[MGLevel]:
    """The level whose leaves are the blocks of 2^m grid cells (or the leaf itself where it is coarser)."""
    dev = lv.anchor.device
    S = torch.clamp(lv.size, min=1 << m)
    blk = (lv.anchor // S[:, None]) * S[:, None]
    key = ((torch.log2(S.double()).round().long() * 4096 + blk[:, 0]) * 4096 + blk[:, 1]) * 4096 + blk[:, 2]
    uniq, agg = torch.unique(key, return_inverse=True)
    n = uniq.numel()
    if n == lv.n:
        return None
    anchor = torch.zeros(n, 3, dtype=torch.long, device=dev)
    anchor[agg] = blk
    size = torch.zeros(n, dtype=torch.long, device=dev)
    size[agg] = S
    vol = torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, agg, lv.vol)
    center = tuple(torch.zeros(n, dtype=torch.float64, device=dev).index_add_(0, agg, c * lv.vol) / vol
                   for c in lv.center)
    A, B = agg[lv.a], agg[lv.b]
    inner = A == B
    fkey = (lv.axis.long() * n + A) * n + B
    fkey = torch.where(inner, torch.full_like(fkey, -1), fkey)
    fu, finv = torch.unique(fkey[~inner], return_inverse=True)
    face = torch.full((lv.nf,), -1, dtype=torch.long, device=dev)
    face[~inner] = finv
    ax = (fu // (n * n)).to(torch.int8)
    ca, cb = (fu // n) % n, fu % n
    dist = torch.zeros(fu.numel(), dtype=torch.float64, device=dev)
    for a_ in range(3):
        sel = ax == a_
        dist[sel] = (center[a_][cb[sel]] - center[a_][ca[sel]]).abs()
    scale = torch.zeros(lv.nf, dtype=torch.float64, device=dev)
    scale[~inner] = lv.dist[~inner] / dist[finv]
    bkey = lv.bside.long() * n + agg[lv.bcell]
    bu, binv = torch.unique(bkey, return_inverse=True)
    bside = (bu // n).to(torch.int8)
    bcell = bu % n
    bhigh = torch.zeros(bu.numel(), dtype=torch.bool, device=dev)
    bhigh[binv] = lv.bhigh
    bdist = torch.zeros(bu.numel(), dtype=torch.float64, device=dev)
    for i, s in enumerate(SIDES):
        sel = bside == i
        a_ = SIDE_AXIS[s]
        if a_ == 0:
            top = s == "top"
            zlo = zf[anchor[bcell[sel], 0]]
            bdist[sel] = (top_z - center[0][bcell[sel]]) if top else (center[0][bcell[sel]] - zlo)
        else:
            bdist[sel] = size[bcell[sel]].double() * 0.5 * float(lv.dx)
    bscale = lv.bdist / bdist[binv]
    lv.to_next = ToNext(agg.int(), face.int(), scale.float(), binv.int(), bscale.float())
    nxt = MGLevel(n, anchor, size, center, ca, cb, ax, dist, bcell, bside, bhigh, bdist)
    nxt.vol, nxt.dx = vol, lv.dx
    nxt.colors = Graph._colors(anchor, size)
    return nxt


class Hierarchy:
    """The multigrid levels of a layout: the leaves, then blocks of 2, 4, 8, ... grid cells, each a coarsening of the
    last (the octree is the hierarchy), until a level holds `min_cells` or fewer."""

    def __init__(self, gr: Graph, min_cells: int = 512, max_levels: int = 15):
        lv = _level0(gr)
        lv.dx = gr.dx
        self.levels = [lv]
        nz = gr.shape[0]
        zf = gr.zf
        m = 0
        while self.levels[-1].n > min_cells and len(self.levels) < max_levels:
            m += 1
            nxt = _coarser(self.levels[-1], m, nz, gr.top_z, zf)
            if nxt is None:
                if (1 << m) > max(gr.shape):
                    break
                continue
            self.levels.append(nxt)


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=-1)


def _norm(a: Tensor) -> Tensor:
    return _dot(a, a).sqrt()


def cg(op, b: Tensor, x: Tensor, tol: float, max_iter: int, precond=None, flexible: bool = False):
    """`poisson.cg` on (C, n) vectors."""
    view = (-1, 1)
    bnorm = _norm(b)
    r = b - op.apply(x)
    rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
    if bool((rel <= tol).all()):
        return x, 0, float(rel.max())
    z = precond(r) if precond else r
    p, rz = z.clone(), _dot(r, z)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = op.apply(p)
        pAp = _dot(p, Ap)
        alpha = torch.where(pAp > 0, rz / pAp, torch.zeros_like(pAp))
        x = x + alpha.view(view) * p
        r = r - alpha.view(view) * Ap
        rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
        if bool((rel <= tol).all()):
            break
        z_old = z
        z = precond(r) if precond else r
        rz_new = _dot(r, z)
        num = rz_new - _dot(r, z_old) if flexible else rz_new
        beta = torch.where(rz > 0, num / rz, torch.zeros_like(rz))
        p, rz = z + beta.view(view) * p, rz_new
    return x, it, float(rel.max())


def bicgstab(op, b: Tensor, x: Tensor, tol: float, max_iter: int, precond=None):
    """`poisson.bicgstab` on (C, n) vectors."""
    view = (-1, 1)
    M = precond or (lambda v: v)  # noqa: E731
    bnorm = _norm(b)
    r = b - op.apply(x)
    rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
    if bool((rel <= tol).all()):
        return x, 0, float(rel.max())
    r0, p, v = r.clone(), torch.zeros_like(r), torch.zeros_like(r)
    one = torch.ones_like(bnorm)
    rho, alpha, omega = one, one, one
    safe = lambda n, d: torch.where(d != 0, n / torch.where(d != 0, d, one), torch.zeros_like(n))  # noqa: E731
    it = 0
    for it in range(1, max_iter + 1):
        rho_new = _dot(r0, r)
        beta = safe(rho_new, rho) * safe(alpha, omega)
        p = r + beta.view(view) * (p - omega.view(view) * v)
        ph = M(p)
        v = op.apply(ph)
        alpha = safe(rho_new, _dot(r0, v))
        s = r - alpha.view(view) * v
        rel = torch.where(bnorm > 0, _norm(s) / bnorm, torch.zeros_like(bnorm))
        if bool((rel <= tol).all()):
            x = x + alpha.view(view) * ph
            break
        x = x + alpha.view(view) * ph
        sh = M(s)
        t = op.apply(sh)
        omega = safe(_dot(t, s), _dot(t, t))
        x = x + omega.view(view) * sh
        r = s - omega.view(view) * t
        rho = rho_new
        rel = torch.where(bnorm > 0, _norm(r) / bnorm, torch.zeros_like(bnorm))
        if bool((rel <= tol).all()):
            break
    return x, it, float(rel.max())


class MG:
    """A V-cycle over the hierarchy for one operator, as `poisson.Solver`: Gauss-Seidel over the four colors before
    and after (reversed after, so the cycle is symmetric), piecewise-constant prolongation (aggregation), the sum as
    restriction, and a Krylov solve on the coarsest level."""

    def __init__(self, hier: Hierarchy, op: GraphOp, sweeps: int = 2, coarse_iter: int = 20,
                 precond_dtype: Optional[torch.dtype] = None, plan=None, levels: Optional[int] = None):
        self.hier, self.sweeps, self.coarse_iter, self.plan = hier, sweeps, coarse_iter, plan
        self.op, self.dtype = op, op.c.dtype
        self.precond_dtype = precond_dtype or self.dtype
        level = op.to(self.precond_dtype)
        self.ops = [level]
        # `levels` truncates the hierarchy: its last level is then smoothed, not solved (a preconditioner for a loose
        # tolerance, where the coarse levels cost more than they save)
        use = hier.levels[:levels] if levels else hier.levels
        self.smooth_last = levels is not None and levels < len(hier.levels)
        for lv in use[1:]:
            level = level.coarsen(lv)
            self.ops.append(level)
        self.safe = [torch.where(o.diag > 0, o.diag, torch.ones_like(o.diag)) for o in self.ops]
        self.colors = [lv.colors for lv in use]
        self.present, self.masks = hier.color_masks(len(use))
        self.coarse_inv = None if self.smooth_last else _coarse_inverse(self.ops[-1])
        self.csr = [None] * len(self.ops)          # the coarse levels by the row kernel on a GPU (amr_csr)
        if CSR_LEVELS and len(self.ops) > 1:
            import amr_csr
            if amr_csr.available(self.ops[1]):
                self.csr[1:] = [amr_csr.CsrOp(o) for o in self.ops[1:]]
        # level 0 by the tile kernels on a tiled layout (amr_tiles), the Krylov operator too: the same numbers
        self.tile0 = self.krylov_op = None
        if plan is not None:
            first = self.ops[0]
            self.tile0, ktile = _tile_ops(plan, first, op)
            self.ops[0] = _Tiled(first, self.tile0)
            self.krylov_op = self.ops[0] if ktile is None else _Tiled(op, ktile)
            if _one_group(first):
                self.safe[0] = self.safe[0].reshape(-1)
            for o in {id(first): first, id(op): op}.values():     # the tiles hold them now: free the face arrays
                _release_faces(o)

    def refresh_level0(self, op) -> None:
        """The finest level (its smoother and the Krylov operator) on a new operator of the same pattern, the coarse
        levels kept from the one the multigrid was built on: a preconditioner whose coarse correction lags, applied to
        the current operator, so the solve's residual and its fixed point are the current operator's."""
        first = op.to(self.precond_dtype)
        self.op, self.dtype = op, op.c.dtype
        safe = torch.where(first.diag > 0, first.diag, torch.ones_like(first.diag))
        if self.plan is None:
            self.ops[0], self.krylov_op, self.safe[0] = first, None, safe
            return
        self.safe[0] = safe.reshape(-1) if _one_group(first) else safe
        self.tile0, ktile = _tile_ops(self.plan, first, op)
        self.ops[0] = _Tiled(first, self.tile0)
        self.krylov_op = self.ops[0] if ktile is None else _Tiled(op, ktile)
        for o in {id(first): first, id(op): op}.values():
            _release_faces(o)

    def _smooth(self, i: int, b: Tensor, x: Tensor, reverse: bool) -> Tensor:
        op, safe, masks = self.ops[i], self.safe[i], self.masks[i]
        order = masks[::-1] if reverse else masks
        colors = self.present[i][::-1] if reverse else self.present[i]
        for _ in range(self.sweeps):
            if i == 0 and self.tile0 is not None:
                for c in colors:
                    x = self.tile0.relax(safe, self.colors[0], c, b, x)
                continue
            if self.csr[i] is not None:          # a coarse level: one row-kernel launch a color (amr_csr)
                for c in colors:
                    x = self.csr[i].relax(safe, self.colors[i], c, b, x)
                continue
            for m in order:
                x = x + torch.where(m, _over(b - op.apply(x), safe), torch.zeros_like(x))
        return x

    def vcycle(self, b: Tensor, i: int = 0) -> Tensor:
        op = self.ops[i]
        if i == len(self.ops) - 1:
            if self.smooth_last:
                return self._smooth(i, b, self._smooth(i, b, torch.zeros_like(b), False), True)
            if self.coarse_inv is not None:          # the coarsest level solved exactly: one product, no host sync
                G, n = self.coarse_inv.shape[0], self.coarse_inv.shape[-1]
                bg = b.reshape(G, -1, n)
                return torch.matmul(bg, self.coarse_inv.transpose(-1, -2)).reshape(b.shape)
            krylov = cg if op.symmetric else bicgstab
            return krylov(op, b, torch.zeros_like(b), 1e-8, self.coarse_iter)[0]
        agg = self.hier.levels[i].to_next.agg
        x = self._smooth(i, b, torch.zeros_like(b), False)
        r = b - (self.csr[i] or op).apply(x)
        rc = torch.zeros(*r.shape[:-1], self.ops[i + 1].diag.shape[-1], dtype=r.dtype, device=r.device)
        rc.index_add_(-1, agg, r)
        e = self.vcycle(rc, i + 1)[..., agg]
        if SCALED_CORRECTION:          # the piecewise-constant correction scaled to minimize the residual it leaves
            Ae = op.apply(e)           # (Braess 1995; Notay 2010 section 3): aggregation alone under-corrects
            alpha = (r * Ae).sum(-1, keepdim=True) / (Ae * Ae).sum(-1, keepdim=True).clamp(min=1e-300)
            e = e * alpha.clamp(0.0, 3.0)
        x = x + e
        return self._smooth(i, b, x, True)

    def precondition(self, r: Tensor) -> Tensor:
        return self.vcycle(r.to(self.precond_dtype)).to(self.dtype)

    def solve(self, b: Tensor, x0: Optional[Tensor] = None, tol: float = 1e-6, max_iter: int = 200):
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        mixed = self.precond_dtype != self.dtype
        pre = self.precondition if mixed else self.vcycle
        op = self.krylov_op or self.op
        if not self.op.symmetric:
            return bicgstab(op, b, x, tol, max_iter, precond=pre)
        return cg(op, b, x, tol, max_iter, precond=pre, flexible=mixed)


def _one_group(op) -> bool:
    """A grouped operator of a single group: every channel reads it, as a plain operator's do."""
    return isinstance(op, GroupOp) and op.diag.shape[0] == 1


class _OneGroup:
    """A `GroupOp` of one group in a plain operator's shape (`amr_tiles.TileOp`'s): the tiles hold one coefficient
    set for every channel and its rest is one CSR for the row kernel, where the grouped tiles gather and scatter."""

    def __init__(self, op: "GroupOp"):
        self.group, self.diag, self.symmetric = op, op.diag[0], op.symmetric

    def offdiag(self) -> Tuple[Tensor, Tensor]:
        ab, ba = self.group.offdiag()
        return ab[0], ba[0]


def _tile_ops(plan, first, op):
    """(the finest level's tile op on `first`, the Krylov operator's on `op` or None when it is `first`)."""
    import amr_tiles
    grouped = isinstance(first, GroupOp) and not _one_group(first)
    kind = amr_tiles.GroupTileOp if grouped else amr_tiles.TileOp
    shape = (lambda o: o) if grouped or not isinstance(first, GroupOp) else _OneGroup  # noqa: E731
    return kind(plan, shape(first)), (None if first is op else kind(plan, shape(op)))


class _Tiled:
    """A GraphOp whose apply runs the tile kernels: what the V-cycle and the Krylov iteration read."""

    def __init__(self, op: GraphOp, tile_op):
        self.op, self.tile = op, tile_op
        self.diag = op.diag

    @property
    def symmetric(self) -> bool:
        return self.op.symmetric

    def apply(self, x: Tensor) -> Tensor:
        return self.tile.apply(x)

    def coarsen(self, nxt):
        return self.op.coarsen(nxt)


def dense_model(scene, profile, direction_deg: float, cfg, dtype: torch.dtype = torch.float64):
    """A dense `solver.Model` with its geometry intact (no working copy, nothing freed) and no projection hierarchy, in
    `dtype` (float64 for the single-cell tests' exactness; float32 halves the build's peak): what a Graph is built from."""
    import copy
    import solver
    c = copy.copy(cfg)
    c.work_dtype, c.precond_dtype, c.march_dtype = None, None, None
    c.dtype = dtype
    keep = solver.Poisson
    solver.Poisson = lambda op, **kw: None
    try:
        m = solver.Model(scene, profile, direction_deg, c)
    finally:
        solver.Poisson = keep
    return m


# ── per-precision geometry, neighbor means, gradients and the coarse-fine correction ───────────────────────────


class Geo:
    """The layout's geometry in one precision, with the per-side averaging weights."""

    def __init__(self, gr: "Graph", dtype: torch.dtype):
        t = lambda v: v.to(dtype)  # noqa: E731
        self.dtype = dtype
        self.n = gr.n
        self.a = [f.a for f in gr.faces]
        self.b = [f.b for f in gr.faces]
        self.ao = [t(f.area_open) for f in gr.faces]
        self.dist = [t(f.dist) for f in gr.faces]
        self.open = [f.open for f in gr.faces]
        self.bcell = {s: gr.bounds[s].cell for s in SIDES}
        self.bao = {s: t(gr.bounds[s].area_open) for s in SIDES}
        self.bdist = {s: t(gr.bounds[s].dist) for s in SIDES}
        self.bopen = {s: gr.bounds[s].area_open > 0 for s in SIDES}
        self.vol = t(gr.vol)
        self.vol_open = t(gr.vol_open)
        # side weights: per axis, the low (high) side of each leaf: its interior faces' share of the side's area, and
        # the boundary face's where the side is the domain's
        self.w_low, self.w_high, self.wb = [], [], {}
        for ax, f in enumerate(gr.faces):
            lo = [s for s in SIDES if SIDE_AXIS[s] == ax and not SIDE_HIGH[s]][0]
            hi = [s for s in SIDES if SIDE_AXIS[s] == ax and SIDE_HIGH[s]][0]
            den_lo = torch.zeros(gr.n, dtype=torch.float64, device=gr.device).index_add_(0, f.b, f.area)
            den_lo.index_add_(0, gr.bounds[lo].cell, gr.bounds[lo].area)
            den_hi = torch.zeros(gr.n, dtype=torch.float64, device=gr.device).index_add_(0, f.a, f.area)
            den_hi.index_add_(0, gr.bounds[hi].cell, gr.bounds[hi].area)
            self.w_low.append(t(f.area / den_lo[f.b]))
            self.w_high.append(t(f.area / den_hi[f.a]))
            self.wb[lo] = t(gr.bounds[lo].area / den_lo[gr.bounds[lo].cell])
            self.wb[hi] = t(gr.bounds[hi].area / den_hi[gr.bounds[hi].cell])

    def lo_side(self, ax: int) -> str:
        return [s for s in SIDES if SIDE_AXIS[s] == ax and not SIDE_HIGH[s]][0]

    def hi_side(self, ax: int) -> str:
        return [s for s in SIDES if SIDE_AXIS[s] == ax and SIDE_HIGH[s]][0]

    def low_mean(self, ax: int, face_vals: Tensor, bval: Optional[Tensor]) -> Tensor:
        """Per leaf, its low side's area-weighted mean of a face quantity (boundary faces: `bval`)."""
        out = torch.zeros(*face_vals.shape[:-1], self.n, dtype=face_vals.dtype, device=face_vals.device)
        out.index_add_(-1, self.b[ax], self.w_low[ax] * face_vals)
        if bval is not None:
            s = self.lo_side(ax)
            out.index_add_(-1, self.bcell[s], self.wb[s] * bval)
        return out

    def high_mean(self, ax: int, face_vals: Tensor, bval: Optional[Tensor]) -> Tensor:
        out = torch.zeros(*face_vals.shape[:-1], self.n, dtype=face_vals.dtype, device=face_vals.device)
        out.index_add_(-1, self.a[ax], self.w_high[ax] * face_vals)
        if bval is not None:
            s = self.hi_side(ax)
            out.index_add_(-1, self.bcell[s], self.wb[s] * bval)
        return out



class Stencil:
    """What every term that reads a neighbor needs, fixed by the layout: each leaf's neighbor means along each axis
    (area-weighted over a side's faces, the interior ones renormalized), torch.gradient's coefficients on the uneven
    spacing that leaves (one-sided at the domain's edge, a dropped solid read as a zero at the next grid cell, as the
    dense grid's solid cell is), and the coarse-fine correction.

    The correction: on a face between a leaf and a coarser one, the coarse value is carried to the finer leaf's
    transverse position by the coarse leaf's own transverse gradient, q_C + sum_t (dq/dt)_C (x_t,fine - x_t,C), so the
    difference across the face is taken along one line, as Popinet 2003 (J Comput Phys 190: 572, section 3.3, the
    gradient at a fine/coarse face) and Martin and Colella 2000 (J Comput Phys 163: 271, coarse-fine interpolation)
    take it. It is linear in q with fixed weights: one sparse matrix per axis (`K`), built once. The two-point flux
    without it carries an O(1) error in the face gradient (Losasso, Gibou and Fedkiw 2004), measured at 28 % by MMS."""

    def __init__(self, gr: Graph, geo: "Geo"):
        self.gr, self.geo = gr, geo
        dev, f64 = gr.device, torch.float64
        center = (gr.zc, gr.yc, gr.xc)
        zf = gr.zf
        self.nb_pos, self.nb_kind, self.wn_low, self.wn_high, self.has_lo, self.has_hi = [], [], [], [], [], []
        for ax in range(3):
            f = gr.faces[ax]
            sl = torch.zeros(gr.n, dtype=f64, device=dev).index_add_(0, f.b, geo.w_low[ax].double())
            sh = torch.zeros(gr.n, dtype=f64, device=dev).index_add_(0, f.a, geo.w_high[ax].double())
            wnl = geo.w_low[ax].double() / sl[f.b]
            wnh = geo.w_high[ax].double() / sh[f.a]
            has_lo, has_hi = sl > 0, sh > 0
            edge_lo = torch.zeros_like(has_lo)
            edge_lo[gr.bounds[geo.lo_side(ax)].cell] = True
            edge_hi = torch.zeros_like(has_hi)
            edge_hi[gr.bounds[geo.hi_side(ax)].cell] = True
            pos_lo = torch.zeros(gr.n, dtype=f64, device=dev).index_add_(0, f.b, wnl * center[ax][f.a])
            pos_hi = torch.zeros(gr.n, dtype=f64, device=dev).index_add_(0, f.a, wnh * center[ax][f.b])
            an = gr.anchor[:, ax]
            if ax == 0:
                top = zf.numel() - 1
                miss_lo = 0.5 * (zf[(an - 1).clamp(min=0)] + zf[an])
                miss_hi = 0.5 * (zf[(an + gr.size).clamp(max=top)] + zf[(an + gr.size + 1).clamp(max=top)])
            else:
                miss_lo = (an.double() - 0.5) * gr.dx
                miss_hi = (an.double() + gr.size.double() + 0.5) * gr.dx
            pos_lo = torch.where(has_lo, pos_lo, miss_lo)
            pos_hi = torch.where(has_hi, pos_hi, miss_hi)
            kind = torch.zeros(gr.n, dtype=torch.int8, device=dev)
            kind[edge_lo & ~has_lo] = 1
            kind[edge_hi & ~has_hi] = 2
            self.nb_pos.append((pos_lo, pos_hi))
            self.nb_kind.append(kind)
            self.wn_low.append(wnl)
            self.wn_high.append(wnh)
            self.has_lo.append(has_lo)
            self.has_hi.append(has_hi)
        # torch.gradient's coefficients, a q_lo + b q + c q_hi
        self.coef = []
        for ax in range(3):
            pl, ph = self.nb_pos[ax]
            hl, hr = center[ax] - pl, ph - center[ax]
            a = -hr / (hl * (hl + hr))
            b = (hr - hl) / (hl * hr)
            c = hl / (hr * (hl + hr))
            kind = self.nb_kind[ax]
            lo_edge, hi_edge = kind == 1, kind == 2
            a = torch.where(lo_edge, torch.zeros_like(a), torch.where(hi_edge, -1.0 / hl, a))
            b = torch.where(lo_edge, -1.0 / hr, torch.where(hi_edge, 1.0 / hl, b))
            c = torch.where(lo_edge, 1.0 / hr, torch.where(hi_edge, torch.zeros_like(c), c))
            self.coef.append((a, b, c))
        self._build_correction()
        self._cache = {}

    def _neighbor_entries(self, t: int, cells: Tensor, high: bool) -> Tuple[Tensor, Tensor, Tensor]:
        """For each entry of `cells`, every neighbor across its low (high) side along `t`: (which entry, neighbor
        leaf, normalized weight), by a ragged expansion over the faces sorted by the leaf they bound."""
        gr = self.gr
        f = gr.faces[t]
        own, nbr, w = (f.a, f.b, self.wn_high[t]) if high else (f.b, f.a, self.wn_low[t])
        order = torch.argsort(own, stable=True)
        count = torch.bincount(own, minlength=gr.n)
        start = torch.cumsum(count, 0) - count
        rep = count[cells]
        total = int(rep.sum())
        entry = torch.repeat_interleave(torch.arange(cells.numel(), device=gr.device), rep)
        offs = torch.arange(total, device=gr.device) - torch.repeat_interleave(torch.cumsum(rep, 0) - rep, rep)
        fid = order[start[cells[entry]] + offs]
        return entry, nbr[fid], w[fid]

    def _build_correction(self) -> None:
        """K per axis as COO triplets (row: the coarse-fine face, column: a leaf), assembled directly: the coarse leaf's
        transverse gradient, torch.gradient's a q_lo + b q + c q_hi with q_lo and q_hi its sides' neighbor means,
        times the finer leaf's transverse offset."""
        gr = self.gr
        center = (gr.zc, gr.yc, gr.xc)
        self.cf_idx, self.cf_coarse_a, self.cf_sign, self.K = [], [], [], []
        for ax in range(3):
            f = gr.faces[ax]
            cf = gr.size[f.a] != gr.size[f.b]
            idx = cf.nonzero()[:, 0]
            a, b = f.a[idx], f.b[idx]
            coarse_a = gr.size[a] > gr.size[b]
            C = torch.where(coarse_a, a, b)
            Fn = torch.where(coarse_a, b, a)
            m = idx.numel()
            rows, cols, vals = [], [], []
            for t in range(3):
                if t == ax or m == 0:
                    continue
                d = center[t][Fn] - center[t][C]
                ca, cb_, cc = self.coef[t]
                rows.append(torch.arange(m, device=gr.device))
                cols.append(C)
                vals.append(d * cb_[C])
                for high, coef in ((False, ca), (True, cc)):
                    e, nb, w = self._neighbor_entries(t, C, high)
                    rows.append(e)
                    cols.append(nb)
                    vals.append(d[e] * coef[C[e]] * w)
            self.cf_idx.append(idx)
            self.cf_coarse_a.append(coarse_a)
            self.cf_sign.append(torch.where(coarse_a, -1.0, 1.0).to(torch.float64))
            if m == 0:
                self.K.append(None)
                continue
            self.K.append((torch.cat([r.long() for r in rows]), torch.cat([c.long() for c in cols]), torch.cat(vals), m))

    def cast(self, dtype):
        """The weights, coefficients and correction matrices in `dtype` (cached)."""
        if dtype not in self._cache:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                K = [None if k is None else _csr(k[0], k[1], k[2].to(dtype), (k[3], self.gr.n)) for k in self.K]
            if CSR_LEVELS and self.gr.device.type == "cuda":     # the row kernel: no cuSPARSE workspace
                import amr_csr
                if amr_csr.triton is not None:
                    K = [None if k is None else amr_csr.Rows(k) for k in K]
            self._cache[dtype] = {
                "wn_low": [w.to(dtype) for w in self.wn_low], "wn_high": [w.to(dtype) for w in self.wn_high],
                "coef": [tuple(x.to(dtype) for x in abc) for abc in self.coef], "K": K,
                "sign": [s.to(dtype) for s in self.cf_sign]}
        return self._cache[dtype]

    def delta(self, q: Tensor) -> List[Optional[Tensor]]:
        """Per axis, (..., F_axis) the correction to the coarse value on each coarse-fine face, zero elsewhere, for q
        (..., n) (any leading dimensions: headings, components)."""
        c = self.cast(q.dtype)
        lead, n = q.shape[:-1], q.shape[-1]
        qf = q.reshape(-1, n)
        out = []
        for ax in range(3):
            K = c["K"][ax]
            if K is None:
                out.append(None)
                continue
            d = torch.zeros(*lead, self.gr.faces[ax].n, dtype=q.dtype, device=q.device)
            kq = K.apply(qf) if hasattr(K, "apply") else (K @ qf.T.contiguous()).T
            d[..., self.cf_idx[ax]] = kq.reshape(*lead, -1)
            out.append(d)
        return out

    def face_values(self, ax: int, q: Tensor, delta: Optional[List[Optional[Tensor]]]) -> Tuple[Tensor, Tensor]:
        """q (..., n) at the two leaves of each face of `ax`, the coarser one corrected to the finer one's line."""
        f = self.gr.faces[ax]
        qa, qb = q[..., f.a], q[..., f.b]
        if delta is not None and delta[ax] is not None:
            idx, ca = self.cf_idx[ax], self.cf_coarse_a[ax]
            d = delta[ax][..., idx]
            qa = qa.index_add(-1, idx, torch.where(ca, d, torch.zeros_like(d)))
            qb = qb.index_add(-1, idx, torch.where(ca, torch.zeros_like(d), d))
        return qa, qb

    def neighbor_values(self, ax: int, q: Tensor, delta=None) -> Tuple[Tensor, Tensor]:
        """Each leaf's low and high neighbor means along `ax` (zero for a dropped solid), q (..., n)."""
        c = self.cast(q.dtype)
        f = self.gr.faces[ax]
        qa, qb = self.face_values(ax, q, delta)
        lo = torch.zeros_like(q).index_add_(-1, f.b, c["wn_low"][ax] * qa)
        hi = torch.zeros_like(q).index_add_(-1, f.a, c["wn_high"][ax] * qb)
        return lo, hi

    def grad(self, q: Tensor, ax: int, delta=None) -> Tensor:
        """d q / d(z, y, x)[ax] at the leaves, as torch.gradient on the dense grid."""
        a, b, c = self.cast(q.dtype)["coef"][ax]
        lo, hi = self.neighbor_values(ax, q, delta)
        return a * lo + b * q + c * hi

    def projection_matrix(self, op: "GraphOp", base: bool = True) -> Optional[Tensor]:
        """The projection's operator with the corrected face gradient, -D G, as CSR: the two-point `op` (with `base`)
        plus, on every coarse-fine face, its open area over distance times the correction, out of one leaf and into
        the other. None when there is nothing to return."""
        gr = self.gr
        dt = op.diag.dtype
        rows, cols, vals = [], [], []
        if base:
            A = op.A.to_sparse_coo().coalesce()
            rows.append(A.indices()[0])
            cols.append(A.indices()[1])
            vals.append(A.values())
        for ax in range(3):
            K = self.K[ax]
            if K is None:
                continue
            kr, kc, kv, m = K
            f = gr.faces[ax]
            idx = self.cf_idx[ax]
            w = (f.area_open[idx] / f.dist[idx] * self.cf_sign[ax])[kr]
            rows += [f.a[idx][kr], f.b[idx][kr]]
            cols += [kc, kc]
            vals += [(-w * kv).to(dt), (w * kv).to(dt)]
        if not rows:
            return None
        return _csr(torch.cat([r.long() for r in rows]), torch.cat([c.long() for c in cols]), torch.cat(vals),
                    (gr.n, gr.n))



class MatOp:
    """A fixed sparse matrix as an operator for the Krylov iterations."""

    symmetric = False

    def __init__(self, A: Tensor):
        self.A = A

    def apply(self, x: Tensor) -> Tensor:
        return (self.A @ x.T.contiguous()).T


class CorrOp:
    """An operator plus a sparse correction: the corrected projection as its two-point operator (on the tile kernels
    where the layout is tiled) and the coarse-fine part (`Stencil.projection_matrix(base=False)`) apart."""

    symmetric = False

    def __init__(self, base, E: Tensor):
        self.base, self.E = base, E

    def apply(self, x: Tensor) -> Tensor:
        if hasattr(self.E, "apply"):                 # the row kernel (amr_csr.Rows)
            return self.base.apply(x) + self.E.apply(x)
        return self.base.apply(x) + (self.E @ x.T.contiguous()).T


def _csr(rows: Tensor, cols: Tensor, vals: Tensor, shape) -> Tensor:
    """COO triplets (duplicates summed) as a CSR matrix."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coo = torch.sparse_coo_tensor(torch.stack([rows.long(), cols.long()]), vals, shape, check_invariants=False).coalesce()
        return coo.to_sparse_csr()


def _compact(self: Stencil) -> None:
    """Keep the float32 weights, coefficients and neighbor centers alone (a float32 march reads nothing else); the
    float64 correction matrices stay, for the float64 final projection."""
    c32 = self.cast(torch.float32)
    c64 = self.cast(torch.float64)
    self._cache[torch.float64] = {"K": c64["K"], "sign": c64["sign"], "wn_low": None, "wn_high": None, "coef": None}
    self.nb_pos = [tuple(p.float() for p in pair) for pair in self.nb_pos]
    self.wn_low = self.wn_high = self.coef = None
    del c32


Stencil.compact = _compact


def _color_masks(self: Hierarchy, n_levels: int):
    """(colors present, their masks) for the first `n_levels` levels, made once (four host reads a level otherwise,
    at every multigrid built: two a step)."""
    if not hasattr(self, "_masks"):
        self._present, self._masks = [], []
        for lv in self.levels:
            pres = [c for c in range(4) if bool((lv.colors == c).any())]
            self._present.append(pres)
            self._masks.append([lv.colors == c for c in pres])
    return self._present[:n_levels], self._masks[:n_levels]


Hierarchy.color_masks = _color_masks


class GroupOp:
    """G operators on one level's faces at once (one heading's momentum or k each), for vectors (G R, n): channel ch is
    read by operator ch // R (a heading's three velocity components share its operator). The same terms as `GraphOp`,
    every coefficient with a leading group dimension; products by gather and scatter (one pattern, G value sets)."""

    def __init__(self, level: "MGLevel", ident: Tensor, c: Tensor, cb: Tensor, fl: Optional[Tensor] = None,
                 flb: Optional[Tensor] = None, R: int = 1):
        self.level, self.ident, self.c, self.cb, self.fl, self.flb, self.R = level, ident, c, cb, fl, flb, R
        lv = level
        pos = lambda f: f.clamp(min=0.0)       # noqa: E731
        neg = lambda f: (-f).clamp(min=0.0)    # noqa: E731
        diag = ident.clone()
        diag.index_add_(-1, lv.a, c)
        diag.index_add_(-1, lv.b, c)
        diag.index_add_(-1, lv.bcell, cb)
        if fl is not None:
            diag.index_add_(-1, lv.a, pos(fl))
            diag.index_add_(-1, lv.b, neg(fl))
            diag.index_add_(-1, lv.bcell, torch.where(lv.bhigh, pos(flb), neg(flb)))
            ab, ba = -(c + neg(fl)), -(c + pos(fl))
        else:
            ab = ba = -c
        self.diag, self._ab, self._ba = diag, ab, ba

    @property
    def symmetric(self) -> bool:
        return getattr(self, "_sym", self.fl is None)

    def offdiag(self) -> Tuple[Tensor, Tensor]:
        return self._ab, self._ba

    def apply(self, x: Tensor) -> Tensor:
        G, n = self.diag.shape
        lv = self.level
        xg = x.reshape(G, -1, n)
        out = self.diag[:, None, :] * xg
        out.index_add_(-1, lv.a, self._ab[:, None, :] * xg[..., lv.b])
        out.index_add_(-1, lv.b, self._ba[:, None, :] * xg[..., lv.a])
        return out.reshape(x.shape)

    def to(self, dtype) -> "GroupOp":
        if dtype == self.c.dtype:
            return self
        cv = lambda t: None if t is None else t.to(dtype)  # noqa: E731
        return GroupOp(self.level, cv(self.ident), cv(self.c), cv(self.cb), cv(self.fl), cv(self.flb), self.R)

    def coarsen(self, nxt: "MGLevel") -> "GroupOp":
        """As `GraphOp.coarsen`, group by group."""
        m = self.level.to_next
        G = self.c.shape[0]
        z = lambda k, t: torch.zeros(G, k, dtype=t.dtype, device=t.device)  # noqa: E731
        ident = z(nxt.n, self.ident).index_add_(-1, m.agg, self.ident)
        keep, fk, sk = _kept(m)
        c = z(nxt.nf, self.c).index_add_(-1, fk, self.c[:, keep] * sk.to(self.c.dtype))
        cb = z(nxt.nb, self.c).index_add_(-1, m.bface, self.cb * m.bscale.to(self.c.dtype))
        fl = flb = None
        if self.fl is not None:
            fl = z(nxt.nf, self.fl).index_add_(-1, fk, self.fl[:, keep])
            flb = z(nxt.nb, self.fl).index_add_(-1, m.bface, self.flb)
        return GroupOp(nxt, ident, c, cb, fl, flb, self.R)


def _kept(m: "ToNext") -> Tuple[Tensor, Tensor, Tensor]:
    """The faces between aggregates (as indices, in order), their coarse faces and scales, found once a hierarchy:
    a boolean mask asked the host for its count at every coarsening, every step."""
    got = getattr(m, "_kept", None)
    if got is None:
        idx = (m.face >= 0).nonzero()[:, 0]
        got = m._kept = (idx, m.face[idx], m.scale[idx])
    return got


def _over(v: Tensor, safe: Tensor) -> Tensor:
    """v (C, n) over the safe diagonal: one (n,) for every channel, or (G, n), channel ch taking group ch // R."""
    if safe.dim() == 1:
        return v / safe
    G, n = safe.shape
    return (v.reshape(G, -1, n) / safe[:, None, :]).reshape(v.shape)


CSR_LEVELS = os.environ.get("AMR_CSR", "1") == "1"
"""The multigrid's coarse levels smoothed and multiplied by the row kernel (amr_csr) on a GPU."""

SCALED_CORRECTION = False
"""Scale each level's coarse correction by the factor that minimizes the residual it leaves (one more product a
level): aggregation's piecewise-constant prolongation under-corrects smooth error by about half."""


def _release_faces(op) -> None:
    """Drop a level-0 operator's face arrays once the tile kernels hold its coefficients (and the coarser levels are
    built from it): only its diagonal and its kind are read again. A (G, F) set of them is the momentum's largest
    transient."""
    op._sym = op.symmetric
    op.c = op.cb = op.fl = op.flb = op._ab = op._ba = op.ident = None
    if hasattr(op, "_A"):
        op._A = None


COARSE_DIRECT = int(os.environ.get("AMR_COARSE_DIRECT", "4096"))
"""The coarsest level is inverted once a multigrid is built when it holds at most this many cells: its solve is then one
product (no Krylov loop, whose every iteration asked the host whether to stop, twenty a V-cycle)."""


def _coarse_inverse(op) -> Optional[Tensor]:
    """(G, n, n) the inverse of the coarsest operator (a row with no coupling, an aggregate of phantom slots, holds its
    unknown at zero), or None when the level is too large to hold densely."""
    n = op.diag.shape[-1]
    if n > COARSE_DIRECT:
        return None
    lv = op.level
    G = op.diag.shape[0] if op.diag.dim() > 1 else 1
    diag = op.diag.reshape(G, n).double()
    ab, ba = op.offdiag()
    ab, ba = ab.reshape(G, -1).double(), ba.reshape(G, -1).double()
    A = torch.zeros(G, n, n, dtype=torch.float64, device=diag.device)
    a, b = lv.a.long(), lv.b.long()
    for g in range(G):                  # accumulated: a pair of aggregates may share more than one face entry
        A[g].index_put_((a, b), ab[g], accumulate=True)
        A[g].index_put_((b, a), ba[g], accumulate=True)
    idx = torch.arange(n, device=diag.device)
    A[:, idx, idx] += torch.where(diag > 0, diag, torch.ones_like(diag))
    if op.symmetric:                    # a sealed pocket makes a block singular: the pseudo-inverse, as CG would
        return torch.linalg.pinv(A, hermitian=True).to(op.diag.dtype)
    return torch.linalg.inv_ex(A).inverse.to(op.diag.dtype)
