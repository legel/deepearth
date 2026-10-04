"""The production wind solve (`solver.Model` under `SolverConfig.fast`: finite volumes, k-l, the pressure drive, the
canopy inflow, settling windows, the float32 march and a float64 final projection) on an adaptive layout.

Every term is the dense solver's, written over the leaves' faces (`amr_graph`): on a layout of single cells it is the
dense solve to round-off (`tests/test_amr.py`). Where a face joins two leaves of different size:
  - fluxes (projection, viscous, convective) are conservative two-point fluxes over the face's open area and the
    distance between the centers (Losasso, Gibou and Fedkiw 2004);
  - a cell's value on a side with several faces is the area-weighted mean of the faces' (or neighbors'), the MUSCL
    reconstruction's far value and the strain's neighbor likewise;
  - the eddy viscosity of a horizontal face is the mean of its two cells', as the dense solver's.
A coarse leaf never touches a solid (amr.required), so the wall stress, the cut ground and the log-law k act on single
cells only, exactly as the dense solver's. The pseudo-time step is the dense one, so the fixed point is the same
discrete balance on the new cells.

The delivered field is put back on the dense grid for the published levels: a single cell is itself, a coarse leaf its
value plus its limited gradient times the offset (conservative linear prolongation, as Basilisk's refine_linear with
the generalized minmod limiter, theta 1.3: van Leer 1979, J Comput Phys 32: 101; Popinet 2003, section 3.2)."""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import amr
import amr_fused
import amr_graph as ag
from amr_graph import SIDES, SIDE_AXIS, SIDE_HIGH, Geo, MatOp, comp
from physics import NU_AIR
from solver import (KL_BETA_D, KL_BETA_P, KL_C_MU, KL_SIGMA_K, KL_WALL, Model, _kl_length, _logmean, _muscl,
                    tail_bound)

Tensor = torch.Tensor
LATERAL = ("west", "east", "south", "north")
THETA = 1.3
"""The generalized minmod limiter's parameter for the output prolongation (Basilisk's minmod2)."""


def _side_pos(side: str, pos: Tensor) -> Tuple[Tensor, Tensor]:
    """The two in-plane indices of a boundary plane's dense faces, in the order the dense solver's side arrays use."""
    k, j, i = pos[:, 0], pos[:, 1], pos[:, 2]
    if side in ("west", "east"):
        return k, j
    if side in ("south", "north"):
        return k, i
    return j, i


class AmrModel:
    """A scene and its forcing on an adaptive layout, ready to run under production's numerics, for one heading or a
    batch of them. Headings share every piece of geometry (the layout, the faces, the multigrid, the tile plan, the
    projection's operator); each carries its own state with a leading heading dimension H: velocities (H, 3, n), faces
    (H, F), k and the multiplier (H, n), its own momentum and k operators (`amr_graph.GroupOp`). The projection solves
    all H at once on its one operator. A heading's inflow is the first heading's turned: the same equilibrium column,
    its vector along the heading (`solver.Model._equilibrium_inflow` builds it from the column's speed and the unit
    wind vector alone)."""

    def __init__(self, scene, profile, direction_deg, cfg, lay: "amr.Layout", tiles: Optional[bool] = None,
                 local_dt: bool = False):
        assert cfg.scheme == "fv" and cfg.closure == "k-l" and cfg.drive == "pressure" and cfg.lateral == "profile"
        assert not cfg.anderson, "Anderson acceleration is not carried on adaptive cells"
        assert cfg.inflow == "canopy", "the batch turns the canopy column's inflow from heading to heading"
        self.cfg = cfg
        t0 = time.time()
        dirs = [float(direction_deg)] if np.isscalar(direction_deg) else [float(d) for d in direction_deg]
        self.directions, self.H = dirs, len(dirs)
        # the dense geometry in float32 under a float32 march (the build's peak halved), float64 otherwise (exactness)
        dm = ag.dense_model(scene, profile, dirs[0], cfg,
                            torch.float32 if cfg.march_dtype == torch.float32 else torch.float64)
        dev = dm.sink.device
        _build_peak("dense model", dev)
        self.dev = dev
        if tiles is None:                           # the tile kernels where a GPU and Triton are
            import amr_tiles
            tiles = dev.type == "cuda" and amr_tiles.triton is not None
        if tiles and not hasattr(lay, "tile"):
            lay = amr.tiled(lay)
        gr = ag.Graph(dm, lay, dev)
        self.gr, self.lay = gr, lay
        _build_peak("graph", dev)
        f64 = torch.float64
        self.favor = gr.favor
        vol_d = dm.vol.expand(dm.nz, dm.ny, dm.nx)
        vol_od = dm.vol_open if self.favor else vol_d
        self.sink = gr.cell_copy(dm.sink, weight=vol_od)
        self.wall = gr.cell_copy(dm.wall)
        self.height = gr.cell_copy(dm.height, weight=vol_d)
        self.wall_friction = gr.cell_copy(dm.wall_friction)
        self.wall_mask = self.wall_friction > 0
        self.solid = gr.solid
        self.k0 = gr.cell_copy(dm.k, weight=vol_d)
        self.k0[self.gr.phantom] = 0.0
        # the unit wind vector of each heading, and the first heading's fields as speeds along its own
        from forcing import wind_vector
        e = torch.tensor([wind_vector(1.0, d) for d in dirs], dtype=f64, device=dev)        # (H, 2)
        e0 = e[0]

        def turn(v: Tensor) -> Tensor:
            """The first heading's vectors (3, ...) for every heading (H, 3, ...): the same speed along each heading's
            own unit vector, the vertical as it is; the first heading exactly its own."""
            speed = v[0] * e0[0] + v[1] * e0[1]
            out = torch.empty((self.H,) + tuple(v.shape), dtype=v.dtype, device=v.device)
            shape = (self.H,) + (1,) * (v.dim() - 1)
            out[:, 0] = e[:, 0].reshape(shape) * speed
            out[:, 1] = e[:, 1].reshape(shape) * speed
            out[:, 2] = v[2]
            out[0] = v
            return out
        u0 = torch.stack([gr.cell_copy(dm.initial[c], weight=vol_d) for c in range(3)])
        u0[:, self.solid] = 0.0
        self.u_init = turn(u0)                                                                  # (H, 3, n)
        self.dt, self.u_top, self.dx = float(dm.dt), float(dm.u_top), float(dm.dx)
        self.k_floor = float(dm.k_floor)
        b0 = dm.body_vec.to(f64)
        self.body_vec = (b0[0] * e0[0] + b0[1] * e0[1]) * torch.cat([e, torch.zeros(self.H, 1, dtype=f64, device=dev)], 1)
        self.column = dm.column
        # prescribed values on the sides (velocity and k) and the top (velocity), each heading's
        self.bval: Dict[str, Tensor] = {}
        self.kval: Dict[str, Tensor] = {}
        for side in LATERAL + ("top",):
            b = gr.bounds[side]
            p, q = _side_pos(side, b._pos)
            self.bval[side] = turn(gr.bound_values(side, dm.bc[side][:, p, q].to(f64)))       # (H, 3, nb)
            if side != "top":
                self.kval[side] = gr.bound_values(side, dm.k_bc[side][p, q].to(f64))
        # the k-l length on every z-face: the dense solver's own between single cells, else from the leaves' heights
        fz, fs = gr.faces[0], gr._face_sets[0]
        hc = dm.canopy_height.to(f64)
        cnt = fs.sum(torch.ones(fs.pos.shape[0], dtype=f64, device=dev))
        hc_face = fs.sum(hc[fs.pos[:, 1], fs.pos[:, 2]]) / cnt
        ell_single = fs.sum(dm.ell_z[fs.pos[:, 0] + 1, fs.pos[:, 1], fs.pos[:, 2]].to(f64))
        ell_coarse = _kl_length(_logmean(self.height[fz.a], self.height[fz.b]), hc_face)
        self.ell_int = torch.where(fz.single, ell_single, ell_coarse)
        del fs, cnt, hc_face, ell_single, ell_coarse
        gr._face_sets = None                       # the build's grid-face index arrays (32 B a face): read no more
        self.ell_b = {}
        for side in ("floor", "top"):
            b = gr.bounds[side]
            kk = 0 if side == "floor" else -1
            single = gr.single[b.cell]
            ell_s = torch.zeros(b.n, dtype=f64, device=dev)
            ell_s[b._inv] = dm.ell_z[kk, b._pos[:, 1], b._pos[:, 2]].to(f64)
            hcb = torch.zeros(b.n, dtype=f64, device=dev).index_add_(0, b._inv, hc[b._pos[:, 1], b._pos[:, 2]])
            hcb = hcb / torch.zeros(b.n, dtype=f64, device=dev).index_add_(0, b._inv, torch.ones_like(b._inv, dtype=f64))
            h = self.height[b.cell]
            hh = h if side == "floor" else _logmean(h, h + b.dist)
            self.ell_b[side] = torch.where(single, ell_s, _kl_length(hh, hcb))
        ell_inv_single = gr.cell_copy(dm.ell_inv, mean=False)
        # dense coordinates for the output's vorticity
        self.grid_coords = (dm.zc.to(f64), dm.yc.to(f64), dm.xc.to(f64))
        self.dense_shape = (dm.nz, dm.ny, dm.nx)
        self.dense_solid = dm.solid.clone()
        self.d_top = float(dm.d_top)
        dm.release()                               # its working copy is itself: a cycle only the collector frees,
        del dm                                     # and the dense geometry (3 GiB at 23 M cells) stayed on the card
        f32 = cfg.march_dtype == torch.float32
        if f32:                                    # the dense-grid maps the output alone reads wait in host memory
            gr._cells_of, gr._dense_kept = gr._cells_of.cpu(), gr._dense_kept.cpu()
        import gc
        gc.collect()
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        _build_peak("fields, dense model released", dev)
        # the multigrid levels and the tile plan read the graph alone: built once the dense model is gone
        self.hier = ag.Hierarchy(gr)
        self.lv0 = self.hier.levels[0]
        self.plan = None
        if tiles:
            import amr_tiles
            self.plan = amr_tiles.TilePlan(lay, self.lv0)
        _build_peak("hierarchy, tile plan", dev)
        # the float64 geometry, the stencil and the float32 geometry built once the dense model is gone, so their
        # transients do not stand on the dense geometry (the build's peak)
        self.geo64 = Geo(gr, f64)
        g = self.geo64
        inv_lo = g.low_mean(0, 1.0 / self.ell_int, 1.0 / self.ell_b["floor"])
        inv_hi = g.high_mean(0, 1.0 / self.ell_int, 1.0 / self.ell_b["top"])
        self.ell_inv = torch.where(gr.single, ell_inv_single, 0.5 * (inv_lo + inv_hi))
        del ell_inv_single, inv_lo, inv_hi
        _build_peak("float64 geometry", dev)
        # the gradient's neighbor positions: the mean of each side's neighbor centers, or a missing (solid) cell's
        self.st = ag.Stencil(gr, self.geo64)
        self._coef = {}
        # levels the settling watches
        self._bits = self._level_bits()
        _build_peak("stencil", dev)
        # local pseudo-time steps (steady-state practice: each cell marches at its own CFL, dt times its size in grid
        # cells): the delta form's rows divided by that factor, so only the identity term changes and the fixed point,
        # where the increment vanishes, is the same discrete balance (Pulliam and Zingg 2014, Fundamental Algorithms in
        # Computational Fluid Dynamics, section 6.5; Blazek 2015, Computational Fluid Dynamics, section 9.4). Measured
        # unstable here (Harvard, 2026-10-04): an option, off.
        self.local_dt = gr.size.double() if local_dt else None
        if cfg.march_dtype == torch.float32:       # a float32 march: keep the float32 geometry alone (the float64
            self.st.compact()                      # weights come back for the final projection)
            self.geo64.w_low = self.geo64.w_high = self.geo64.wb = None
            # the per-leaf fields the march reads in float32 alone: cast once, the float64 masters dropped (the start
            # fields are cast as the march starts); the dense-grid maps the output alone reads wait in host memory
            for name in ("wall", "sink", "ell_inv", "wall_friction"):
                self._cast(name, torch.float32)
                setattr(self, name, None)
            self.k0, self.u_init = self.k0.float(), self.u_init.float()
            gr.anchor = gr.anchor.int()
            self.lv0.anchor = gr.anchor
        self.geo32 = Geo(gr, torch.float32)       # after the stencil's float64 weights are dropped
        _build_peak("end of build", dev)
        self.build_s = time.time() - t0
        self.k = None
        self._nu = None
        import os
        self.profile, self.prof, self._t_mark = os.environ.get("AMR_PROFILE") == "1", {}, 0.0

    # ── fields: (H, 3, n) cells, per axis (H, F) faces, per side (H, nb) boundary faces ─────────────────────────────

    def faces(self, geo: Geo, u: Tensor):
        uf = []
        for ax in range(3):
            c = comp(ax)
            v = 0.5 * (u[:, c][..., geo.a[ax]] + u[:, c][..., geo.b[ax]])
            uf.append(torch.where(geo.open[ax], v, torch.zeros_like(v)))
        ub = {}
        zero = lambda s: torch.zeros(u.shape[0], geo.bao[s].numel(), dtype=u.dtype, device=u.device)  # noqa: E731
        for s in LATERAL:
            c = comp(SIDE_AXIS[s])
            ub[s] = torch.where(geo.bopen[s], self.bval[s][:, c].to(u.dtype), zero(s))
        ub["top"] = torch.where(geo.bopen["top"], u[:, 2][..., geo.bcell["top"]], zero("top"))
        ub["floor"] = zero("floor")
        return uf, ub

    def cells(self, geo: Geo, uf, ub) -> Tensor:
        H = uf[0].shape[0]
        u = torch.empty(H, 3, geo.n, dtype=uf[0].dtype, device=uf[0].device)
        for ax in range(3):
            lo = geo.low_mean(ax, uf[ax], ub[geo.lo_side(ax)])
            hi = geo.high_mean(ax, uf[ax], ub[geo.hi_side(ax)])
            u[:, comp(ax)] = 0.5 * (lo + hi)
        return u

    def divergence(self, geo: Geo, uf, ub) -> Tensor:
        H = uf[0].shape[0]
        out = torch.zeros(H, geo.n, dtype=uf[0].dtype, device=uf[0].device)
        for ax in range(3):
            q = geo.ao[ax] * uf[ax]
            out.index_add_(-1, geo.a[ax], q)
            out.index_add_(-1, geo.b[ax], -q)
        for s in SIDES:
            q = geo.bao[s] * ub[s]
            out.index_add_(-1, geo.bcell[s], q if SIDE_HIGH[s] else -q)
        return out

    def face_gradient(self, geo: Geo, lam: Tensor):
        """The face velocities a multiplier (H, n) adds: its gradient on open faces, corrected on coarse-fine faces
        (`amr_graph.Stencil`), and across the open top to its zero."""
        gf = []
        delta = self.st.delta(lam)
        for ax in range(3):
            la, lb = self.st.face_values(ax, lam, delta)
            g = (lb - la) / geo.dist[ax]
            gf.append(torch.where(geo.open[ax], g, torch.zeros_like(g)))
        gb = {s: torch.zeros(lam.shape[0], geo.bao[s].numel(), dtype=lam.dtype, device=lam.device) for s in SIDES}
        gt = -lam[..., geo.bcell["top"]] / geo.bdist["top"]
        gb["top"] = torch.where(geo.bopen["top"], gt, torch.zeros_like(gt))
        return gf, gb

    def projection_op(self, dtype) -> "ag.GraphOp":
        """The two-point projection operator, one for every heading: the multigrid's (its preconditioner)."""
        gr = self.gr
        c = torch.cat([torch.where(f.open, f.area_open / f.dist, torch.zeros_like(f.dist)) for f in gr.faces])
        cb = torch.cat([gr.bounds[s].area_open / gr.bounds[s].dist if s == "top"
                        else torch.zeros(gr.bounds[s].n, dtype=torch.float64, device=gr.device) for s in SIDES])
        return ag.GraphOp(self.lv0, torch.zeros(gr.n, dtype=torch.float64, device=gr.device), c, cb).to(dtype)

    def projection_correction(self, dtype) -> Optional[Tensor]:
        """The coarse-fine part of -D G (`amr_graph.Stencil.projection_matrix`), CSR in `dtype`; None without one.
        Built once in float64: every precision is cast from it, and a float32 march keeps the float64 matrix in host
        memory until the final projection (`_correction_to_host`), where rebuilding it was the run's peak."""
        key = ("projection", dtype)
        if key not in self._coef:
            host = getattr(self, "_e64_host", None)
            if host is not None:
                E = host.to(self.dev)
            else:
                E = self.st.projection_matrix(self.projection_op(torch.float64), base=False)
            if E is not None:
                if dtype != torch.float64:
                    self._e64_device = E                # kept while it may go to the host for the march
                E = E.to(dtype)
                if ag.CSR_LEVELS and E.is_cuda:
                    import amr_csr
                    E = amr_csr.Rows(E) if amr_csr.triton is not None else E
            self._coef[key] = E
        return self._coef[key]

    def _correction_to_host(self) -> None:
        """The float64 correction (made by the first `projection_correction`) to host memory for the march."""
        E = getattr(self, "_e64_device", None)
        if E is not None:
            self._e64_host = E.cpu()
            self._e64_device = None

    def project(self, geo: Geo, mg: "ag.MG", uf, ub, tol: float, max_iter: int):
        """The minimal correction making each heading's faces divergence-free under the corrected gradient: BiCGSTAB
        on the two-point operator plus its coarse-fine part, every heading a channel of one solve, the two-point
        multigrid as preconditioner."""
        self._mark(None)
        div = self.divergence(geo, uf, ub)
        E = self.projection_correction(div.dtype)
        if E is None:                                           # no coarse-fine face: the symmetric operator is exact
            lam, it, rel = mg.solve(div, tol=tol, max_iter=max_iter)
        else:
            A = ag.CorrOp(mg.krylov_op or mg.op, E)
            pre = mg.precondition if mg.precond_dtype != mg.dtype else mg.vcycle
            lam, it, rel = ag.bicgstab(A, div, torch.zeros_like(div), tol, max_iter, precond=pre)
        self._mark("p.solve")
        gf, gb = self.face_gradient(geo, lam)
        uf = [u + g for u, g in zip(uf, gf)]
        ub = {s: ub[s] + gb[s] for s in SIDES}
        self._mark("p.faces")
        return uf, ub, lam, it, rel

    def add_increment(self, geo: Geo, uf, ub, du: Tensor):
        out = []
        for ax in range(3):
            c = comp(ax)
            inc = 0.5 * (du[:, c][..., geo.a[ax]] + du[:, c][..., geo.b[ax]])
            out.append(uf[ax] + torch.where(geo.open[ax], inc, torch.zeros_like(inc)))
        ub = dict(ub)
        ub["top"] = ub["top"] + torch.where(geo.bopen["top"], du[:, 2][..., geo.bcell["top"]], torch.zeros_like(ub["top"]))
        return out, ub

    def pressure_cells(self, geo: Geo, pi: Tensor) -> Tensor:
        gf, gb = self.face_gradient(geo, pi)
        return self.cells(geo, gf, gb)

    # ── closure ──────────────────────────────────────────────────────────────────────────────────────────────────────

    def strain_rest(self, geo: Geo, u: Tensor, deltas=None) -> Tensor:
        """`Model.strain_rest` on the leaves, (H, n); `deltas` each component's coarse-fine corrections."""
        deltas = deltas or [self.st.delta(u[:, c]) for c in range(3)]
        if FUSED and amr_fused.available(u):
            return amr_fused.strain(self.st, u, deltas)
        g = lambda c, d: self.st.grad(u[:, c], d, deltas[c])  # noqa: E731
        out = 2 * (g(0, 2) ** 2 + g(1, 1) ** 2 + g(2, 0) ** 2)
        out = out + (g(0, 1) + g(1, 2)) ** 2
        wx = g(2, 2)
        out = out + wx ** 2 + 2 * g(0, 0) * wx
        wy = g(2, 1)
        return out + wy ** 2 + 2 * g(1, 0) * wy

    def viscosity_z(self, geo: Geo, u: Tensor, k: Tensor, deltas=None):
        """The eddy viscosity on the z-faces (interior, floor, top) and |S|^2 there, each (H, ...), as
        `Model.viscosity_z` under k-l."""
        deltas = deltas or [self.st.delta(u[:, c]) for c in range(3)]
        rest = self.strain_rest(geo, u, deltas)
        a, b = geo.a[0], geo.b[0]
        fa = [self.st.face_values(0, u[:, c], deltas[c]) for c in range(2)]
        shear = torch.sqrt((fa[0][1] - fa[0][0]) ** 2 + (fa[1][1] - fa[1][0]) ** 2) / geo.dist[0]
        s_int = torch.clamp(shear ** 2 + 0.5 * (rest[..., a] + rest[..., b]), min=0).sqrt()
        tc = geo.bcell["top"]
        above = self.bval["top"][:, :2].to(u.dtype)
        shear_top = (above - u[:, :2][..., tc]).norm(dim=1) / geo.bdist["top"]
        s_top = torch.clamp(shear_top ** 2 + rest[..., tc], min=0).sqrt()
        fc = geo.bcell["floor"]
        s_floor = geo.high_mean(0, s_int, s_top)[..., fc]      # the floor face takes the face above its cell's
        s2 = (s_int ** 2, s_floor ** 2, torch.zeros_like(s_top))   # no stress through a pressure-driven top
        kf_int = (0.5 * (k[..., a] + k[..., b])).clamp(min=0)
        kf_floor, kf_top = k[..., fc].clamp(min=0), k[..., tc].clamp(min=0)
        c = KL_C_MU ** 0.25
        ell = self._ell(u.dtype)
        nu = (c * ell[0] * kf_int.sqrt() + NU_AIR, c * ell[1] * kf_floor.sqrt() + NU_AIR, c * ell[2] * kf_top.sqrt() + NU_AIR)
        return nu, s2

    def _ell(self, dtype):
        key = ("ell", dtype)
        if key not in self._coef:
            self._coef[key] = (self.ell_int.to(dtype), self.ell_b["floor"].to(dtype), self.ell_b["top"].to(dtype))
        return self._coef[key]

    def nu_cells(self, geo: Geo, nu) -> Tensor:
        lo = geo.low_mean(0, nu[0], nu[1])
        hi = geo.high_mean(0, nu[0], nu[2])
        return 0.5 * (lo + hi)

    def conductances(self, geo: Geo, nu, sigma: float = 1.0, top_nu: Optional[Tensor] = None):
        """(c on the level's faces, c on its boundary faces), each (H, ...): nu / sigma times open area over distance;
        the sides are held (Dirichlet, half a cell), the top by `top_nu` (None: none), the floor never."""
        nu_c = self.nu_cells(geo, nu)
        c = []
        for ax in range(3):
            nuf = nu[0] if ax == 0 else 0.5 * (nu_c[..., geo.a[ax]] + nu_c[..., geo.b[ax]])
            c.append(torch.where(geo.open[ax], nuf / sigma * geo.ao[ax] / geo.dist[ax], torch.zeros_like(nuf)))
        cb = {}
        H = nu_c.shape[0]
        for s in SIDES:
            zero = torch.zeros(H, geo.bao[s].numel(), dtype=nu_c.dtype, device=nu_c.device)
            if s in LATERAL:
                cb[s] = torch.where(geo.bopen[s], nu_c[..., geo.bcell[s]] / sigma * geo.bao[s] / geo.bdist[s], zero)
            elif s == "top" and top_nu is not None:
                cb[s] = torch.where(geo.bopen[s], top_nu / sigma * geo.bao[s] / geo.bdist[s], zero)
            else:
                cb[s] = zero
        return c, cb

    def apply_diffusion(self, geo: Geo, c, cb, q: Tensor, delta=None) -> Tensor:
        """sum_f c_f (q - q_nb) for one quantity (H, n), the difference across a coarse-fine face corrected
        (`delta`), with the boundary faces' c q (their prescribed values enter the right-hand side)."""
        if FUSED and amr_fused.available(q):
            return amr_fused.diffusion(self, geo, c, cb, q, delta)
        out = torch.zeros_like(q)
        for ax in range(3):
            qa, qb = self.st.face_values(ax, q, delta)
            fl = c[ax] * (qa - qb)
            out.index_add_(-1, geo.a[ax], fl)
            out.index_add_(-1, geo.b[ax], -fl)
        for s in SIDES:
            out.index_add_(-1, geo.bcell[s], cb[s] * q[..., geo.bcell[s]])
        return out

    # ── convection ───────────────────────────────────────────────────────────────────────────────────────────────────

    def convection(self, geo: Geo, q: Tensor, F, Fb, ghosts: Dict[str, Tensor], delta=None) -> Tensor:
        """Net outflow sum_f F_f q_f [per leaf] for one quantity (H, n), q_f the upwind MUSCL value (`solver._muscl`,
        van Leer limited) from the two leaves' values (the coarser corrected to the finer's line on a coarse-fine
        face), the far value the area-weighted mean of the upwind cell's own upwind side; `ghosts` (H, nb) the
        boundary values (also the far value beyond them, as the dense halo's replicate pad)."""
        if FUSED and self.cfg.limiter == "vanleer" and amr_fused.available(q):
            return amr_fused.convection(self, geo, q, F, Fb, ghosts, delta)
        lim = self.cfg.limiter
        out = torch.zeros_like(q)
        for ax in range(3):
            a, b = geo.a[ax], geo.b[ax]
            lo_s, hi_s = geo.lo_side(ax), geo.hi_side(ax)
            far_lo = geo.low_mean(ax, q[..., a], ghosts[lo_s])
            far_hi = geo.high_mean(ax, q[..., b], ghosts[hi_s])
            f = F[ax]
            qa, qb = self.st.face_values(ax, q, delta)
            face = torch.where(f > 0, _muscl(far_lo[..., a], qa, qb, lim), _muscl(far_hi[..., b], qb, qa, lim))
            fl = f * face
            out.index_add_(-1, a, fl)
            out.index_add_(-1, b, -fl)
            for s, high in ((lo_s, False), (hi_s, True)):
                cell, gv, fb = geo.bcell[s], ghosts[s], Fb[s]
                qc = q[..., cell]
                if high:
                    face = torch.where(fb > 0, _muscl(far_lo[..., cell], qc, gv, lim), gv)
                    out.index_add_(-1, cell, fb * face)
                else:
                    face = torch.where(fb > 0, gv, _muscl(far_hi[..., cell], qc, gv, lim))
                    out.index_add_(-1, cell, -fb * face)
        return out

    def _fluxes(self, geo: Geo, uf, ub):
        return [geo.ao[ax] * uf[ax].to(geo.dtype) for ax in range(3)], {s: geo.bao[s] * ub[s].to(geo.dtype) for s in SIDES}

    def _vghosts(self, dtype, c: int) -> Dict[str, Tensor]:
        key = ("vg", dtype, c)
        if key not in self._coef:
            g = {s: self.bval[s][:, c].to(dtype) for s in LATERAL}
            g["top"] = self.bval["top"][:, c].to(dtype)
            g["floor"] = torch.zeros(self.H, self.gr.bounds["floor"].n, dtype=dtype, device=self.dev)
            self._coef[key] = g
        return self._coef[key]

    # ── steps ────────────────────────────────────────────────────────────────────────────────────────────────────────

    def _relaxed_nu(self, geo: Geo, u: Tensor, deltas=None):
        nu, s2 = self.viscosity_z(geo, u, self.k, deltas)
        a = self.cfg.relax_nu
        if a < 1.0 and self._nu is not None:
            nu = tuple(a * n + (1.0 - a) * o for n, o in zip(nu, self._nu))
        self._nu, self._s2 = nu, s2
        return nu

    def _level_op(self, ident: Tensor, c, cb, F=None, Fb=None, R: int = 1) -> "ag.GroupOp":
        """Each heading's operator on the leaves (`amr_graph.GroupOp`), the faces of all axes and the boundary faces of
        all sides in the hierarchy's order."""
        cc = torch.cat(c, -1)
        cbb = torch.cat([cb[s] for s in SIDES], -1)
        fl = None if F is None else torch.cat(F, -1)
        flb = None if Fb is None else torch.cat([Fb[s] for s in SIDES], -1)
        return ag.GroupOp(self.lv0, ident, cc, cbb, fl, flb, R)

    def fv_increment(self, geo: Geo, uf, ub, u: Tensor, pi: Tensor) -> Tuple[Tensor, int]:
        dt, H = self.dt, self.H
        F, Fb = self._fluxes(geo, uf, ub)
        u = u.to(geo.dtype)
        self._mark(None)
        deltas = [self.st.delta(u[:, c]) for c in range(3)]       # each component's coarse-fine corrections
        self._mark("m.deltas")
        nu = self._relaxed_nu(geo, u, deltas)
        self._mark("m.viscosity")
        c, cb = self.conductances(geo, nu, top_nu=None)          # pressure drive: no stress through the top
        vol = geo.vol_open
        wall, sink = self._cast("wall", geo.dtype), self._cast("sink", geo.dtype)
        drag_vol = geo.vol * wall + vol * sink
        speed = u.norm(dim=1)                                     # (H, n)
        residual = torch.zeros_like(u)
        for s in LATERAL:
            cell = geo.bcell[s]
            residual.index_add_(-1, cell, cb[s][:, None, :] * self.bval[s].to(geo.dtype))
        for comp_ in range(3):
            residual[:, comp_] -= self.apply_diffusion(geo, c, cb, u[:, comp_], deltas[comp_])
            residual[:, comp_] -= self.convection(geo, u[:, comp_], F, Fb, self._vghosts(geo.dtype, comp_), deltas[comp_])
        self._mark("m.diffusion_convection")
        residual -= (drag_vol * speed)[:, None] * u
        open_ = (~self.solid).to(geo.dtype)
        residual += vol * self.body_vec.to(geo.dtype)[:, :, None] * open_
        ident = (vol if self.local_dt is None else vol / self._cast("local_dt", geo.dtype)) + dt * drag_vol * speed
        M = self._level_op(ident, [x * dt for x in c], {s: v * dt for s, v in cb.items()},
                           [f * dt for f in F], {s: v * dt for s, v in Fb.items()}, R=3)
        rhs = residual * dt + vol * self.pressure_cells(geo, pi)
        self.last_residual = (rhs.reshape(H, -1).norm(dim=1) / dt)
        self._mark("m.rhs_operator")
        mg = self._step_mg("momentum", M)
        self._mark("m.mg_build")
        # every heading's three components in one Krylov solve (each channel its own scalars, `amr_graph.bicgstab`,
        # channel 3 h + c read by heading h's operator)
        if MOMENTUM_SPLIT and H == 1:              # a component at a time: a third of the Krylov vectors on the card
            parts = [mg.solve(rhs[0, c:c + 1], tol=self.cfg.tol_momentum, max_iter=self.cfg.max_iter) for c in range(3)]
            du, it = torch.cat([p[0] for p in parts]), max(p[1] for p in parts)
        else:
            du, it, _ = mg.solve(rhs.reshape(3 * H, -1), tol=self.cfg.tol_momentum, max_iter=self.cfg.max_iter)
        du = du.reshape(H, 3, -1)
        self._mark("m.solve")
        du.masked_fill_(self.solid, 0.0)      # indexing by the mask asked the host for its count
        return du, it

    def _step_mg(self, key: str, M) -> "ag.MG":
        """The momentum's (or k's) multigrid on this step's operator M: built anew every MG_EVERY-th step and, between,
        the last one with its finest level on M (`MG.refresh_level0`): the residual and the fixed point are always M's,
        only the coarse correction lags."""
        mgs, count = self.__dict__.setdefault("_mgs", {}), self.__dict__.setdefault("_mg_n", {})
        n = count.get(key, 0)
        count[key] = n + 1
        held = mgs.get(key) if MG_EVERY > 1 else None
        if held is not None and n % MG_EVERY:
            held.refresh_level0(M)
            return held
        mgs.pop(key, None)                          # the last one goes before the next is built
        mg = ag.MG(self.hier, M, plan=self.plan, levels=MOMENTUM_LEVELS, sweeps=MOMENTUM_SWEEPS)
        if MG_EVERY > 1:
            mgs[key] = mg
        return mg

    def _cast(self, name: str, dtype=torch.float32) -> Tensor:
        key = (name, dtype)
        if key not in self._coef:
            self._coef[key] = getattr(self, name).to(dtype)
        return self._coef[key]

    def k_step(self, geo: Geo, uf, ub, u: Tensor) -> int:
        self._mark(None)
        k, dt = self.k, self.dt
        vol = geo.vol_open
        F, Fb = self._fluxes(geo, uf, ub)
        nu = self._nu
        c, cb = self.conductances(geo, nu, sigma=KL_SIGMA_K, top_nu=None)
        inflow = torch.zeros_like(k)
        for s in LATERAL:
            inflow.index_add_(-1, geo.bcell[s], cb[s] * self.kval[s].to(geo.dtype))
        speed = u.to(geo.dtype).norm(dim=1)
        s2 = self._s2
        prod_lo = geo.low_mean(0, (nu[0] - NU_AIR) * s2[0], (nu[1] - NU_AIR) * s2[1])
        prod_hi = geo.high_mean(0, (nu[0] - NU_AIR) * s2[0], (nu[2] - NU_AIR) * s2[2])
        production = 0.5 * (prod_lo + prod_hi)
        rate = KL_C_MU ** 0.75 * k.clamp(min=0).sqrt() * self._cast("ell_inv", geo.dtype)
        canopy = self._cast("sink", geo.dtype) * speed
        k_wall = self._cast("wall_friction", geo.dtype) * speed ** 2 / math.sqrt(KL_C_MU)
        source = (production + KL_BETA_P * canopy * speed ** 2 - (rate + KL_BETA_D * canopy) * k
                  + torch.where(self.wall_mask, (KL_WALL / dt) * (k_wall - k), torch.zeros_like(k)))
        ghosts = {s: self.kval[s].to(geo.dtype).expand(self.H, -1).contiguous() for s in LATERAL}
        ghosts["floor"] = k[..., geo.bcell["floor"]]
        ghosts["top"] = k[..., geo.bcell["top"]]
        dk = self.st.delta(k)
        residual = (inflow - self.apply_diffusion(geo, c, cb, k, dk) - self.convection(geo, k, F, Fb, ghosts, dk)
                    + vol * source)
        one = 1.0 if self.local_dt is None else 1.0 / self._cast("local_dt", geo.dtype)
        ident = vol * (one + dt * (1.5 * rate + KL_BETA_D * canopy) + KL_WALL * self.wall_mask.to(geo.dtype))
        M = self._level_op(ident, [x * dt for x in c], {s: v * dt for s, v in cb.items()},
                           [f * dt for f in F], {s: v * dt for s, v in Fb.items()}, R=1)
        self._mark("k.terms")
        mg = self._step_mg("k", M)
        self._mark("k.mg_build")
        dk, it, _ = mg.solve(residual * dt, tol=self.cfg.tol_momentum, max_iter=self.cfg.max_iter)
        self._mark("k.solve")
        k = (k + dk).clamp(min=self.k_floor)
        k.masked_fill_(self.solid, 0.0)
        self.k = k
        return it

    # ── settling ─────────────────────────────────────────────────────────────────────────────────────────────────────

    def _level_bits(self) -> Tensor:
        lo, hi = Model.NEAR_GROUND_M
        fluid, h = ~self.solid, self.height
        half = torch.clamp(0.5 * self.gr.hz, min=0.5)
        bits = (fluid & (h >= lo) & (h <= hi)).to(torch.uint8)
        for j, level in enumerate(Model.LEVELS_M):
            bits |= (fluid & ((h - level).abs() <= half)).to(torch.uint8) << (j + 1)
        return bits

    def near_ground(self, u: Tensor) -> Tensor:
        """One heading's (3, n) horizontal speed at the near-ground cells, as one vector."""
        return u[:2, (self._bits & 1).bool()].norm(dim=0)

    def level_stats(self, u: Tensor) -> dict:
        out = {}
        for j, h in enumerate(Model.LEVELS_M):
            m = ((self._bits >> (j + 1)) & 1).bool()
            s = u[:2, m].norm(dim=0)
            if s.numel() == 0:
                continue
            s = s[::max(1, -(-s.numel() // (1 << 24)))]
            q = torch.quantile(s, torch.tensor([0.5, 0.95], dtype=s.dtype, device=s.device))
            out[f"{h:g}"] = [float(q[0]), float(q[1])]
        return out

    # ── run ──────────────────────────────────────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """March every heading together until each has settled (its own settling windows and rule, as production's);
        a settled heading's delivered field is its last window's mean, projected, kept while the rest march on. Each
        is then projected once more in float64."""
        cfg, t0, H = self.cfg, time.time(), self.H
        if self.profile and self.dev.type == "cuda":
            self.prof["persistent at start GiB"] = torch.cuda.memory_allocated() / 2 ** 30
            print("  CENSUS (MiB, count, kind) " + str(cuda_census()), flush=True)
        md = torch.float32 if cfg.march_dtype == torch.float32 else torch.float64   # production marches in float32
        geo = self.geo32 if md == torch.float32 else self.geo64
        op32 = self.projection_op(md)
        self.projection_correction(md)             # cached: built from the float64 geometry before it is parked
        self._correction_to_host()                 # its float64 master waits on the host for the final projection
        parked = PARK and md == torch.float32 and self.dev.type == "cuda"
        if parked:                                 # the float64 geometry waits on the host while the march runs
            _park(self, True)
        mg32 = ag.MG(self.hier, op32, plan=self.plan)
        self.k = self.k0.to(md).expand(H, -1).clone()
        u = self.u_init.to(md)
        uf, ub = self.faces(geo, u)
        uf, ub, _, it, rel = self.project(geo, mg32, uf, ub, cfg.tol, cfg.max_iter)
        u = self.cells(geo, uf, ub)
        pi = torch.zeros(H, self.gr.n, dtype=md, device=self.dev)
        iterations, its_m, its_k, change = [it], [], [], []
        settle = [[] for _ in range(H)]
        hist = [{} for _ in range(H)]
        ref, acc, calm = [None] * H, None, [0] * H
        done_at = [None] * H                       # the step each heading settled at
        kept_f, kept_b = [None] * H, [None] * H     # each settled heading's delivered faces
        last = cfg.max_steps if cfg.settle_tol is not None and cfg.max_steps else cfg.steps
        res0 = None
        timing = {"momentum": 0.0, "projection": 0.0, "k": 0.0}
        # the phase times wait for the GPU (a host sync each, the queue drained three times a step): profiling only
        timed = self.dev.type == "cuda" and (self.profile or TIMED)
        sync = (lambda: torch.cuda.synchronize()) if timed else (lambda: None)  # noqa: E731
        step = -1
        for step in range(last):
            prev = u
            if self.profile and step in (60, 63) and self.dev.type == "cuda":
                self._torch_profile(step == 60)
            sync()
            ta = time.time()
            du, im = self.fv_increment(geo, uf, ub, u, pi)
            res_norm = self.last_residual
            res0 = res_norm if step == 0 else res0
            uf, ub = self.add_increment(geo, uf, ub, du)
            sync()
            tb = time.time()
            uf, ub, lam, it, rel = self.project(geo, mg32, uf, ub, cfg.tol, cfg.max_iter)
            pi = pi + lam
            u = self.cells(geo, uf, ub)
            sync()
            tc = time.time()
            change.append((u - prev).abs().amax(dim=(1, 2)) / self.u_top)
            ik = self.k_step(geo, uf, ub, u)
            sync()
            timing["momentum"] += tb - ta
            timing["projection"] += tc - tb
            timing["k"] += time.time() - tc
            iterations.append(it)
            its_m.append(im)
            its_k.append(ik)
            if cfg.verbose and (step + 1) % 20 == 0:
                print(f"  amr step {step + 1}  change {float(change[-1].max()):.2e}  its p {it} m {im} k {ik}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            if cfg.settle_tol is None:
                continue
            acc = [f.clone() for f in uf] + [dict(ub)] if acc is None else \
                [x.add_(f) for x, f in zip(acc[:3], uf)] + [{s: acc[3][s] + ub[s] for s in SIDES}]
            if (step + 1) % cfg.settle_every:
                continue
            mean_f = [x / cfg.settle_every for x in acc[:3]]
            mean_b = {s: v / cfg.settle_every for s, v in acc[3].items()}
            mean = self.cells(geo, mean_f, mean_b)
            acc = None
            newly = []
            for h in range(H):
                if done_at[h] is not None:
                    continue
                now, levels = self.near_ground(mean[h]), self.level_stats(mean[h])
                for kk, v in levels.items():
                    for j, name in enumerate(("median", "p95")):
                        hist[h].setdefault(f"{kk}m {name}", []).append(v[j])
                if ref[h] is not None:
                    row = dict(Model.settle_change(ref[h], now), step=step + 1, levels=levels,
                               residual=float(res_norm[h] / res0[h]))
                    settle[h].append(row)
                    if cfg.settle_rule == "tail":
                        tail = max(tail_bound(x) for x in hist[h].values()) if hist[h] else float("inf")
                        calm[h] = 2 if tail <= cfg.settle_tol else 0
                    else:
                        calm[h] = calm[h] + 1 if (row["d_median"] < cfg.settle_tol and row["d_p95"] < cfg.settle_tol
                                                  and row["rms_rel"] < Model.SETTLE_RMS * cfg.settle_tol) else 0
                    if calm[h] >= 2 and step + 1 >= cfg.steps:
                        done_at[h] = step + 1
                        newly.append(h)
                ref[h] = now
            if newly:                              # the window means of the headings that settled, projected together
                idx = torch.tensor(newly, device=self.dev)
                pf, pb, _, _, _ = self.project(geo, mg32, [f[idx] for f in mean_f], {s: v[idx] for s, v in mean_b.items()},
                                               cfg.tol, cfg.max_iter)
                for j, h in enumerate(newly):
                    kept_f[h], kept_b[h] = [f[j:j + 1].clone() for f in pf], {s: v[j:j + 1].clone() for s, v in pb.items()}
            if all(d is not None for d in done_at):
                break
        settled = [d is not None for d in done_at] if cfg.settle_tol is not None else None
        for h in range(H):                         # an unsettled heading delivers where the march stopped
            if kept_f[h] is None:
                kept_f[h], kept_b[h] = [f[h:h + 1] for f in uf], {s: v[h:h + 1] for s, v in ub.items()}
        uf = [torch.cat([kept_f[h][ax] for h in range(H)]) for ax in range(3)]
        ub = {s: torch.cat([kept_b[h][s] for h in range(H)]) for s in SIDES}
        march_s = time.time() - t0
        # the final projection in float64 (a float32 V-cycle under float64 conjugate gradients, as production's), on a
        # card holding no more of the march than the delivered faces and k: its working fields, the window sums and
        # the float32 caches go first (the final projection was the run's peak)
        _build_peak("march", self.dev)
        del mg32, op32
        kept_f = kept_b = u = prev = pi = None
        acc = mean = mean_f = mean_b = du = lam = pf = pb = None
        self._nu = self._s2 = self.last_residual = None
        self.__dict__.pop("_mgs", None)
        self._coef = {k: v for k, v in self._coef.items() if not _float32(v)}
        if md == torch.float32:
            self.geo32 = None
        import gc
        gc.collect()
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
        if parked:
            _park(self, False, skip=OUTPUT_ONLY)    # the leaf centers wait: the output alone reads them
        geo64 = self.geo64
        uf = [f.double() for f in uf]
        ub = {s: v.double() for s, v in ub.items()}
        mg64 = ag.MG(self.hier, self.projection_op(torch.float64), precond_dtype=torch.float32, plan=self.plan)
        for lv in self.hier.levels:                # the coarsening is done: its index caches go before the solve
            if lv.to_next is not None:
                lv.to_next._kept = None
        tol_f = cfg.tol_final if cfg.tol_final is not None else cfg.tol
        uf, ub, _, it, rel = self.project(geo64, mg64, uf, ub, tol_f,
                                          cfg.max_iter_final if cfg.tol_final is not None else cfg.max_iter)
        iterations.append(it)
        div = self.divergence(geo64, uf, ub)
        div_max = float((div / geo64.vol_open.clamp(min=1e-12)).abs()[..., ~self.solid].max())
        flux_in = [sum(float((geo64.bao[s] * ub[s][h] * (-1 if SIDE_HIGH[s] else 1)).clamp(min=0).sum()) for s in SIDES)
                   for h in range(H)]
        flux_out = [sum(float((geo64.bao[s] * ub[s][h] * (1 if SIDE_HIGH[s] else -1)).clamp(min=0).sum()) for s in SIDES)
                    for h in range(H)]
        del mg64, div                              # the solve's operators go before the side weights are made
        if geo64.w_low is None:                    # the float64 side weights, for the field at the leaves alone
            self.geo64 = geo64 = Geo(self.gr, torch.float64)
        u = self.cells(geo64, uf, ub)
        _build_peak("final projection", self.dev)
        if parked:                                 # the leaf centers the dense output reads, after the projection
            _park(self, False)
        self._release_march()
        wall = time.time() - t0
        steps = [d if d is not None else step + 1 for d in done_at] if cfg.settle_tol is not None else [step + 1] * H
        one = H == 1
        return {"u": u[0] if one else u, "k": self.k.double()[0] if one else self.k.double(),
                "steps": steps[0] if one else steps, "settled": (settled[0] if settled else None) if one else settled,
                "settle": settle[0] if one else settle, "batch_steps": step + 1,
                "projection_iterations": iterations, "momentum_iterations": its_m, "k_iterations": its_k,
                "change": [c.tolist() for c in change], "divergence_max_1_s": div_max,
                "flux_in": flux_in[0] if one else flux_in, "flux_out": flux_out[0] if one else flux_out,
                "march_s": march_s, "wall_s": wall, "timing": timing, "leaves": self.gr.n, "headings": H}

    # ── output on the dense grid ─────────────────────────────────────────────────────────────────────────────────────

    def _release_march(self) -> None:
        """After the final projection the card keeps what the output reads (the graph, the stencil's float32 neighbor
        weights and positions, the dense maps): the hierarchy, the tiles, both geometries and the march's caches go
        before the field is put on the dense grid."""
        self.hier = self.lv0 = self.plan = None
        self._e64_host = self._e64_device = None
        self.geo32 = self.geo64 = None
        self._coef = {}
        self.st._cache.pop(torch.float64, None)
        import gc
        gc.collect()
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()

    def limited_gradient(self, q: Tensor, ax: int) -> Tensor:
        """The generalized minmod gradient (THETA) of a leaf quantity (n,) along `ax`, from its neighbors' means."""
        lo, hi = self.st.neighbor_values(ax, q.float())          # the output's slopes in float32: limited, smooth
        lo, hi = lo.double(), hi.double()
        gr = self.gr
        center = (gr.zc, gr.yc, gr.xc)[ax]
        pl, ph = self.st.nb_pos[ax]
        dl = (q - lo) / (center - pl)
        dr = (hi - q) / (ph - center)
        kind = self.st.nb_kind[ax]
        dl = torch.where(kind == 1, dr, dl)
        dr = torch.where(kind == 2, dl, dr)
        cen = 0.5 * (dl + dr)
        same = (dl * dr) > 0
        mag = torch.minimum(torch.minimum(THETA * dl.abs(), cen.abs()), THETA * dr.abs())
        return torch.where(same, torch.sign(cen) * mag, torch.zeros_like(cen))

    def to_dense(self, q: Tensor) -> Tensor:
        """A leaf quantity (..., n) on the dense grid: single cells their own, coarse leaves linear about their center
        with the limited gradient (volume averages kept), dropped cells zero. One axis's offsets at a time (each a
        dense float64 grid)."""
        gr = self.gr
        nz, ny, nx = self.dense_shape
        lead = q.shape[:-1]
        qq = q.reshape(-1, gr.n)
        out = torch.zeros(qq.shape[0], nz * ny * nx, dtype=q.dtype, device=q.device)
        cells, flat = gr._cells_of.to(q.device), gr._dense_kept.to(q.device)
        idx = (flat // (ny * nx), (flat // nx) % ny, flat % nx)
        centers = (gr.zc, gr.yc, gr.xc)
        coarse = ~gr.single[cells]
        for r in range(qq.shape[0]):
            v = qq[r][cells]
            corr = torch.zeros_like(v)
            for ax in range(3):
                off = self.grid_coords[ax][idx[ax]] - centers[ax][cells]
                corr = corr + self.limited_gradient(qq[r], ax)[cells] * off
                del off
            out[r, flat] = torch.where(coarse, v + corr, v)
        return out.reshape(*lead, nz, ny, nx)

    def dense_velocity(self, u: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """(velocity, vorticity) of one heading (3, n) on the dense grid, (3, nz, ny, nx) each, the curl as
        `Model.vorticity` takes it. Each velocity component is put on the grid and sent to host memory in turn, and
        each curl component made from the two it reads: the card holds two dense components at a time, never six."""
        v = np.empty((3,) + tuple(self.dense_shape), dtype=np.float64)
        for c in range(3):
            vc = self.to_dense(u[c:c + 1].double())[0]
            vc[self.dense_solid] = 0.0
            v[c] = vc.cpu().numpy()
            del vc
        coords = self.grid_coords
        dev = u.device

        def g(c: int, d: int) -> Tensor:
            return torch.gradient(torch.from_numpy(v[c]).to(dev), spacing=[coords[d]], dim=[d])[0]
        w = np.empty_like(v)
        w[0] = (g(2, 1) - g(1, 0)).cpu().numpy()
        w[1] = (g(0, 0) - g(2, 2)).cpu().numpy()
        w[2] = (g(1, 2) - g(0, 1)).cpu().numpy()
        _build_peak("dense output", dev)
        return v, w

    # ── profiling ────────────────────────────────────────────────────────────────────────────────────────────────────

    PROFILE = False
    """Synchronized seconds by section of the step (AMR_PROFILE=1), for the kernel work; it changes no number."""

    def _mark(self, name: Optional[str]) -> None:
        if not (self.PROFILE or getattr(self, "profile", False)):
            return
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        if name is not None:
            self.prof[name] = self.prof.get(name, 0.0) + now - self._t_mark
            if self.dev.type == "cuda":         # the section's own peak, GiB
                key = "peak " + name
                self.prof[key] = max(self.prof.get(key, 0.0), torch.cuda.max_memory_allocated() / 2 ** 30)
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._t_mark = now


DENSE_CELL_BYTES = 213.0
"""The dense solver's CUDA peak per grid cell, a whole heading (Harvard Central, 5fe8187; 206 at Fisher Museum)."""

OCTREE_BUILD_BYTES = (72.0, 591.0)
"""The octree build's peak (the graph made from the dense model): bytes per grid cell and per leaf (Fisher Museum,
20.97 M cells, 3.86 M leaves: 1.40 GiB of dense model, 3.53 GiB at the graph; wind 085a24c, L4, 2026-10-04)."""

OCTREE_LEAF_BYTES = 1115.0
"""The octree run's peak per leaf, the final projection (Fisher Museum: 4.01 GiB over 3.86 M leaves, whole run and
dense output included, against the dense solver's 4.03 GiB on the same grid; wind 085a24c, L4, 2026-10-04)."""


def octree_bytes(cells: int, leaves: int) -> float:
    """The CUDA peak a heading on adaptive cells is expected to hold: the build's or the run's, whichever is more."""
    return max(OCTREE_BUILD_BYTES[0] * cells + OCTREE_BUILD_BYTES[1] * leaves, OCTREE_LEAF_BYTES * leaves)


PARK = os.environ.get("AMR_PARK", "1") == "1"
OUTPUT_ONLY = ("zc", "yc", "xc", "hz", "hx")
"""The graph's float64 leaf centers and heights: read by the dense output, not by the final projection."""
"""Under a float32 march, the float64 geometry only the final projection reads waits in host memory (`_park`): the
march's peak drops by those bytes and no number changes."""


def _float32(v) -> bool:
    """Whether a cached coefficient (a tensor, or a tuple, list or dict of them) is a float32 one."""
    if isinstance(v, Tensor):
        return v.dtype == torch.float32
    if isinstance(v, dict):
        v = list(v.values())
    return isinstance(v, (list, tuple)) and any(isinstance(x, Tensor) and x.dtype == torch.float32 for x in v)


def _holders(m: "AmrModel") -> list:
    gr = m.gr
    return [gr, *gr.faces, *gr.bounds.values(), m.geo64]


def _park(m: "AmrModel", to_host: bool, skip: Tuple[str, ...] = ()) -> None:
    """Every float64 tensor the graph, its faces and bounds and the float64 geometry hold, to the host (`to_host`) or
    back to the device: one copy per tensor, however many of them hold it. A tensor something else also holds (the
    hierarchy's) stays where it was in that holder's hands, so a missed reader fails loudly or not at all."""
    import weakref
    memo = {}
    if to_host:
        m._parked = {}                  # host copy id -> the device tensor it came from, while anything else holds it

    def move(t):
        if not isinstance(t, Tensor) or t.dtype != torch.float64 or (t.device.type == "cpu") == to_host:
            return t
        if id(t) not in memo:
            if to_host:
                memo[id(t)] = host = t.to("cpu")
                m._parked[id(host)] = weakref.ref(t)
            else:                       # the original where another holder kept it alive: no second copy
                ref = m._parked.get(id(t))
                live = ref() if ref is not None else None
                memo[id(t)] = live if live is not None else t.to(m.dev)
        return memo[id(t)]
    for h in _holders(m):
        for k, v in list(vars(h).items()):
            if h is m.gr and k in skip:
                continue
            if isinstance(v, Tensor):
                setattr(h, k, move(v))
            elif isinstance(v, list):
                setattr(h, k, [move(x) for x in v])
            elif isinstance(v, dict):
                setattr(h, k, {kk: move(x) for kk, x in v.items()})
    if not to_host:
        m._parked = {}
    if m.dev.type == "cuda":
        torch.cuda.empty_cache()


def _build_peak(stage: str, dev) -> None:
    """The build's CUDA memory at a stage (AMR_PROFILE=1): held now and the most held since the last stage, GiB."""
    if os.environ.get("AMR_PROFILE") != "1" or dev.type != "cuda":
        return
    print(f"  BUILD {stage}: held {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB, peak "
          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
    torch.cuda.reset_peak_memory_stats()


def cuda_census(top: int = 25) -> list:
    """The live CUDA tensors by shape and type, largest first: what a run holds (profiling only)."""
    import gc
    seen, rows = set(), {}
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                if o.is_sparse or o.layout != torch.strided:
                    key, nbytes = (f"{o.layout}", tuple(o.shape), str(o.dtype)), 0
                    if o.layout == torch.sparse_csr:
                        nbytes = sum(t.untyped_storage().nbytes() for t in (o.crow_indices(), o.col_indices(), o.values()))
                else:
                    ptr = o.untyped_storage().data_ptr()
                    if ptr in seen:
                        continue
                    seen.add(ptr)
                    key, nbytes = ("strided", tuple(o.shape), str(o.dtype)), o.untyped_storage().nbytes()
                n, b = rows.get(key, (0, 0))
                rows[key] = (n + 1, b + nbytes)
        except Exception:                               # an object that is not what it says (a proxy): skip it
            continue
    out = sorted(((b, n, k) for k, (n, b) in rows.items()), reverse=True)[:top]
    return [(round(b / 2 ** 20, 1), n, f"{k[0]} {k[1]} {k[2]}") for b, n, k in out]


# ── fused face passes (torch.compile on CUDA; AMR_COMPILE=0 runs them eagerly, the same arithmetic) ───────────────────

import os as _os

COMPILE = _os.environ.get("AMR_COMPILE", "0") == "1"
"""Off: measured on Harvard (3.85 M leaves), the compiled passes ran 24.5 s a heading against 17.2 eager."""
FUSED = _os.environ.get("AMR_FUSED", "1") == "1"
"""The stencil phases (convection, diffusion, the velocity gradients) by the fused face kernels (amr_fused) on a GPU."""
TIMED = _os.environ.get("AMR_TIMED", "0") == "1"
"""The march's phase times (`timing`) synchronized with the GPU each step; otherwise they time the host's launches."""
MG_EVERY = int(_os.environ.get("AMR_MG_EVERY", "1"))
"""The momentum and k multigrids built every this many steps, their finest level refreshed between (1: every step).
Every 4 steps measured 2.7 % faster on Fisher at the same steps and errors, and held 0.2 GiB more: off."""
MOMENTUM_SPLIT = _os.environ.get("AMR_MOMENTUM_SPLIT", "1") == "1"
"""One heading's momentum solved a component at a time (each channel's Krylov scalars are its own either way): the
march's peak, its Krylov vectors, falls by a third of them (Fisher: 4.22 to 3.96 GiB in the cli, 56.6 against 56.4 s)."""
MOMENTUM_LEVELS = None
"""The momentum and k multigrids' levels: None the whole hierarchy; k smooths its last instead of solving it."""
MOMENTUM_SWEEPS = 2
"""Gauss-Seidel sweeps before and after each coarse correction in the momentum and k multigrids."""


def _vanleer_muscl(far: Tensor, up: Tensor, down: Tensor) -> Tensor:
    jump = down - up
    r = (up - far) / torch.where(jump == 0, torch.full_like(jump, 1e-30), jump)
    return up + 0.5 * ((r + r.abs()) / (1.0 + r.abs())) * jump


def _convect_axis(q, a, b, wl, wh, f, qa, qb, lo_cell, lo_w, lo_g, lo_f, hi_cell, hi_w, hi_g, hi_f, out):
    """One axis of `AmrModel.convection` (van Leer) as one fused graph: the far values, the faces' upwind MUSCL fluxes
    and their divergence, the boundary faces' too."""
    far_lo = torch.zeros_like(q).index_add(0, b, wl * q[a]).index_add(0, lo_cell, lo_w * lo_g)
    far_hi = torch.zeros_like(q).index_add(0, a, wh * q[b]).index_add(0, hi_cell, hi_w * hi_g)
    face = torch.where(f > 0, _vanleer_muscl(far_lo[a], qa, qb), _vanleer_muscl(far_hi[b], qb, qa))
    fl = f * face
    out = out.index_add(0, a, fl).index_add(0, b, -fl)
    qc = q[lo_cell]
    face_lo = torch.where(lo_f > 0, lo_g, _vanleer_muscl(far_hi[lo_cell], qc, lo_g))
    out = out.index_add(0, lo_cell, -lo_f * face_lo)
    qc = q[hi_cell]
    face_hi = torch.where(hi_f > 0, _vanleer_muscl(far_lo[hi_cell], qc, hi_g), hi_g)
    return out.index_add(0, hi_cell, hi_f * face_hi)


def _diffuse_axis(qa, qb, c, a, b, out):
    fl = c * (qa - qb)
    return out.index_add(0, a, fl).index_add(0, b, -fl)


_FUSED = {}


def fused(name: str, fn, device):
    if not (COMPILE and device.type == "cuda"):
        return fn
    if name not in _FUSED:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        _FUSED[name] = torch.compile(fn, dynamic=False)
    return _FUSED[name]


def _torch_profile(self, start: bool) -> None:
    """Three steps under torch.profiler (AMR_PROFILE=1): the kernels' own device time against the wall, the launches,
    and the costliest kernels, to tell a launch-bound step from a bandwidth-bound one."""
    if start:
        self._tp = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
        torch.cuda.synchronize()
        self._tp_t0 = time.perf_counter()
        self._tp.__enter__()
        return
    torch.cuda.synchronize()
    wall = time.perf_counter() - self._tp_t0
    self._tp.__exit__(None, None, None)
    ev = self._tp.key_averages()
    dev = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))  # noqa: E731
    kernels = [e for e in ev if dev(e) > 0]
    total = sum(dev(e) for e in kernels) / 1e6
    launches = sum(e.count for e in ev if e.key.startswith("cuda") and "Launch" in e.key)
    top = sorted(kernels, key=dev, reverse=True)[:12]
    print(f"  TORCHPROF 3 steps: wall {wall:.3f} s, device time {total:.3f} s ({total / wall:.0%} busy), "
          f"{launches} launches; top " + "; ".join(f"{e.key[:60]} {dev(e) / 1e3:.0f} ms x{e.count}" for e in top),
          flush=True)
    self._tp = None


AmrModel._torch_profile = _torch_profile
