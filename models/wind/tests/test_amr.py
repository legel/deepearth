"""Adaptive cells (amr, amr_graph): the layout is a balanced octree, its operators are the dense solver's own on a layout
of single cells, and on a refined layout the projection's discretization converges at a measured order (the method of
manufactured solutions, Roache 2002, J Fluids Eng 124: 4; Salari and Knupp 2000, SAND2000-1444).

The order is reported globally and in the band of leaves beside a coarse-fine face, where the two-point flux of
Losasso, Gibou and Fedkiw (2004) is first order in its truncation."""

import math

import numpy as np
import pytest
import torch

import amr
import amr_graph as ag
import checks
import domain
from domain import Grid
from solver import Model, SolverConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _site():
    return checks.site(checks.grid(1.0, 32, 32, 24))


def _crit(**kw):
    return amr.Criterion(**dict(dict(shell=2.0, shell_out=2.0, band=1.0, margin=0.0), **kw))


# ── the layout ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_layout_is_balanced_and_owns_every_open_cell():
    scene = _site()
    leaf = amr.balance(amr.required(scene, _crit()), amr.LMAX)
    assert amr.check_balance(leaf) == 0
    lay = amr.layout(leaf, scene.solid)
    assert (leaf >= 1).any() and (leaf == 0).any()
    assert (lay.owner[~scene.solid] >= 0).all()                    # every open cell is in a kept leaf
    for n in range(lay.n):                                         # each leaf owns exactly its block
        k, j, i = lay.anchor[n]
        s = 1 << int(lay.level[n])
        blk = lay.owner[k:k + s, j:j + s, i:i + s]
        assert (blk[blk >= 0] == n).all()
    # no coarse leaf holds a solid cell, the ground's cut cells and the cells beside a solid are single
    coarse = np.isin(lay.owner, np.nonzero(lay.level > 0)[0])
    assert not (coarse & scene.solid).any()
    assert not (coarse & (np.asarray(scene.cut_open) < 1)).any()


# ── the dense solver's own operators on single cells ─────────────────────────────────────────────────────────────


def _uniform(scene, cfg=None):
    from poisson import Solver
    cfg = cfg or SolverConfig(device=DEV)
    m = ag.dense_model(scene, checks.profile(), 250.0, cfg)
    m.poisson = Solver(m._projection_operator())
    gr = ag.Graph(m, amr.uniform_layout(scene.grid.shape), DEV)
    return m, gr


def _dense_faces(m):
    torch.manual_seed(0)
    u = torch.randn(3, m.nz, m.ny, m.nx, dtype=torch.float64, device=m.sink.device)
    u[:, m.solid] = 0.0
    return m.faces(u)


def _to_graph(gr, faces):
    """Dense face arrays as the graph's: interior faces in (k, j, i) order of their low cell, and the boundary planes."""
    ux, uy, uz = faces
    uf = [uz[1:-1].reshape(-1), uy[:, 1:-1, :].reshape(-1), ux[..., 1:-1].reshape(-1)]
    ub = {"west": ux[..., 0].reshape(-1), "east": ux[..., -1].reshape(-1), "south": uy[:, 0, :].reshape(-1),
          "north": uy[:, -1, :].reshape(-1), "top": uz[-1].reshape(-1), "floor": uz[0].reshape(-1)}
    return uf, ub


def test_single_cells_carry_the_dense_geometry():
    m, gr = _uniform(_site())
    assert [f.n for f in gr.faces] == [(m.nz - 1) * m.ny * m.nx, m.nz * (m.ny - 1) * m.nx, m.nz * m.ny * (m.nx - 1)]
    assert torch.equal(gr.vol_open, m.vol_open.reshape(-1))
    assert torch.equal(gr.faces[2].area_open, (m.open_x * m.ax_open[..., 1:-1]).reshape(-1))
    assert torch.equal(gr.faces[0].dist, m.dzf.reshape(-1))
    assert torch.equal(gr.bounds["top"].dist, torch.full_like(gr.bounds["top"].dist, m.d_top))


def test_the_divergence_is_the_dense_solver_s():
    m, gr = _uniform(_site())
    faces = _dense_faces(m)
    uf, ub = _to_graph(gr, faces)
    want = m.divergence(faces).reshape(-1)
    got = gr.divergence(uf, ub)
    assert float((got - want).abs().max()) <= 1e-12 * float(want.abs().max())


def test_the_projection_operator_and_gradient_are_the_dense_solver_s():
    m, gr = _uniform(_site())
    op_d = m._projection_operator()
    hier = ag.Hierarchy(gr, min_cells=10 ** 9)                    # the leaves alone
    c, cb = gr.projection_conductances()
    lv = hier.levels[0]
    cbv = torch.zeros(lv.nb, dtype=torch.float64, device=gr.device)
    cbv[lv.bside == ag.SIDES.index("top")] = cb["top"]
    op = ag.GraphOp(lv, torch.zeros(gr.n, dtype=torch.float64, device=gr.device), torch.cat(c), cbv)
    torch.manual_seed(1)
    p = torch.randn(1, m.nz, m.ny, m.nx, dtype=torch.float64, device=gr.device)
    want = op_d.apply(p).reshape(1, -1)
    got = op.apply(p.reshape(1, -1))
    assert float((got - want).abs().max()) <= 1e-12 * float(want.abs().max())
    gf, gb = gr.face_gradient(p.reshape(-1))
    for axis in range(3):
        dense = m.face_gradient_axis(p[0], 2 - axis)
        interior = dense.narrow(axis, 1, dense.shape[axis] - 2).reshape(-1)
        assert float((gf[axis] - interior).abs().max()) <= 1e-12 * float(interior.abs().max())
    assert float((gb["top"] - m.face_gradient_axis(p[0], 2)[-1].reshape(-1)).abs().max()) <= 1e-12


def test_a_projection_on_single_cells_is_the_dense_solver_s():
    m, gr = _uniform(_site())
    faces = _dense_faces(m)
    uf, ub = _to_graph(gr, faces)
    hier = ag.Hierarchy(gr)
    c, cb = gr.projection_conductances()
    lv = hier.levels[0]
    cbv = torch.zeros(lv.nb, dtype=torch.float64, device=gr.device)
    cbv[lv.bside == ag.SIDES.index("top")] = cb["top"]
    mg = ag.MG(hier, ag.GraphOp(lv, torch.zeros(gr.n, dtype=torch.float64, device=gr.device), torch.cat(c), cbv))
    div = gr.divergence(uf, ub)
    lam, it, rel = mg.solve(div[None], tol=1e-12, max_iter=400)
    assert rel <= 1e-12
    gf, gb = gr.face_gradient(lam[0])
    uf2 = [u + g for u, g in zip(uf, gf)]
    ub2 = dict(ub, top=ub["top"] + gb["top"])
    after = gr.divergence(uf2, ub2)
    assert float(after.abs().max()) <= 1e-9 * float(div.abs().max())
    from poisson import Solver
    lam_d, _, _ = Solver(m._projection_operator()).solve(m.divergence(faces)[None], tol=1e-12, max_iter=400)
    fluid = ~m.solid.reshape(-1)                  # a solid cell has no open face: its multiplier is anything
    assert float((lam[0] - lam_d.reshape(-1))[fluid].abs().max()) <= 1e-8 * float(lam_d.abs().max())
    corrected = tuple(f + m.face_gradient_axis(lam_d[0], a) for a, f in enumerate(faces))
    want, _ = _to_graph(gr, corrected)
    for got_a, want_a in zip(uf2, want):
        assert float((got_a - want_a).abs().max()) <= 1e-8 * float(want_a.abs().max())


# ── manufactured solutions on refined layouts ────────────────────────────────────────────────────────────────────

L = 64.0


def _box(n: int):
    """A cube of side L in n^3 cells, no solid, refined to single cells near a sphere and doubling outward at fixed
    physical distances, so each n halves every leaf of the last."""
    dx = L / n
    g = Grid.uniform(dx, n, n, n)
    scene = domain.flat(g)
    zc, yc, xc = np.meshgrid(g.zc, g.yc, g.xc, indexing="ij")
    d = np.abs(np.sqrt((xc - L / 2) ** 2 + (yc - L / 2) ** 2 + (zc - L / 3) ** 2) - L / 6)
    req = np.zeros(g.shape, np.int8)
    for edge in (4.0, 10.0, 20.0):
        req += (d > edge).astype(np.int8)
    return scene, amr.balance(req, amr.LMAX)


def _exact(x, y, z):
    return np.cos(np.pi * x / L) * np.cos(np.pi * y / L) * np.cos(np.pi * z / (2 * L))


def _cell_integral_of_minus_laplacian(gr):
    """The exact integral of -lap p over each leaf: p's cosines integrate in closed form."""
    kx = ky = np.pi / L
    kz = np.pi / (2 * L)
    lam = kx ** 2 + ky ** 2 + kz ** 2
    zc, yc, xc = (t.cpu().numpy() for t in (gr.zc, gr.yc, gr.xc))
    hz, hx = gr.hz.cpu().numpy(), gr.hx.cpu().numpy()
    ix = (np.sin(kx * (xc + hx / 2)) - np.sin(kx * (xc - hx / 2))) / kx
    iy = (np.sin(ky * (yc + hx / 2)) - np.sin(ky * (yc - hx / 2))) / ky
    iz = (np.sin(kz * (zc + hz / 2)) - np.sin(kz * (zc - hz / 2))) / kz
    return lam * ix * iy * iz


def _mms(n: int, corrected: bool = True):
    scene, leaf = _box(n)
    lay = amr.layout(leaf, scene.solid)
    m = ag.dense_model(scene, checks.profile(), 270.0, SolverConfig(device=DEV))
    gr = ag.Graph(m, lay, DEV)
    hier = ag.Hierarchy(gr)
    c, cb = gr.projection_conductances()
    lv = hier.levels[0]
    cbv = torch.zeros(lv.nb, dtype=torch.float64, device=gr.device)
    cbv[lv.bside == ag.SIDES.index("top")] = cb["top"]
    op = ag.GraphOp(lv, torch.zeros(gr.n, dtype=torch.float64, device=gr.device), torch.cat(c), cbv)
    f = torch.as_tensor(_cell_integral_of_minus_laplacian(gr), device=gr.device)
    mg = ag.MG(hier, op)
    if corrected:                                   # the corrected gradient (amr_graph.Stencil), its exact operator
        st = ag.Stencil(gr, ag.Geo(gr, torch.float64))
        A = ag.MatOp(st.projection_matrix(op))
        p, it, rel = ag.bicgstab(A, f[None], torch.zeros_like(f[None]), 1e-12, 500, precond=mg.vcycle)
    else:
        A = op
        p, it, rel = mg.solve(f[None], tol=1e-12, max_iter=500)
    exact = torch.as_tensor(_exact(gr.xc.cpu().numpy(), gr.yc.cpu().numpy(), gr.zc.cpu().numpy()), device=gr.device)
    err = (p[0] - exact).abs()
    trunc = (A.apply(exact[None])[0] - f).abs() / gr.vol             # the local truncation error, per unit volume
    # leaves beside a coarse-fine face
    band = torch.zeros(gr.n, dtype=torch.bool, device=gr.device)
    for fc in gr.faces:
        cf = gr.size[fc.a] != gr.size[fc.b]
        band[fc.a[cf]] = True
        band[fc.b[cf]] = True
    vol = gr.vol
    l2 = lambda e, w: float(torch.sqrt((e ** 2 * w).sum() / w.sum()))  # noqa: E731
    # the face gradient (what the projection adds to the velocity) against the exact derivative at the face's center
    if corrected:
        dl = st.delta(p[0])
        gf = []
        for ax, fc in enumerate(gr.faces):
            la, lb = st.face_values(ax, p[0], dl)
            gf.append((lb - la) / fc.dist)
    else:
        gf, _ = gr.face_gradient(p[0])
    gerr, gerr_cf, gw = [], [], []
    center = (gr.zc, gr.yc, gr.xc)
    k_ = (np.pi / (2 * L), np.pi / L, np.pi / L)
    for ax, fc in enumerate(gr.faces):
        fine = torch.where(gr.size[fc.a] <= gr.size[fc.b], fc.a, fc.b)        # the face lies on the smaller leaf's side
        pos = [c[fine].cpu().numpy() for c in center]
        half = (gr.hz if ax == 0 else gr.hx)[fine].cpu().numpy() / 2
        pos[ax] = np.where((fine == fc.a).cpu().numpy(), pos[ax] + half, pos[ax] - half)
        z, y, x = pos
        d = [-k_[0] * np.cos(np.pi * x / L) * np.cos(np.pi * y / L) * np.sin(np.pi * z / (2 * L)),
             -k_[1] * np.cos(np.pi * x / L) * np.sin(np.pi * y / L) * np.cos(np.pi * z / (2 * L)),
             -k_[2] * np.sin(np.pi * x / L) * np.cos(np.pi * y / L) * np.cos(np.pi * z / (2 * L))][ax]
        e = (gf[ax] - torch.as_tensor(d, device=gr.device)).abs()
        gerr.append(e)
        gerr_cf.append(e[gr.size[fc.a] != gr.size[fc.b]])
        gw.append(fc.area)
    ge, gw = torch.cat(gerr), torch.cat(gw)
    gscale = max(k_)
    return {"n": n, "leaves": gr.n, "levels": np.bincount(lay.level).tolist(), "iterations": it,
            "l2": l2(err, vol), "linf": float(err.max()), "band_l2": l2(err[band], vol[band]),
            "band_linf": float(err[band].max()), "trunc_band_linf": float(trunc[band].max()),
            "trunc_away_linf": float(trunc[~band].max()), "band_leaves": int(band.sum()),
            "grad_l2": l2(ge, gw) / gscale, "grad_linf": float(ge.max()) / gscale,
            "grad_cf_linf": float(torch.cat(gerr_cf).max()) / gscale}


@pytest.mark.parametrize("ns", [(32, 64, 128)])
def test_the_projection_converges_on_a_refined_layout(ns):
    """The corrected coarse-fine gradient converges: second order in the solution, first at least in the face
    gradient beside the coarse-fine faces; the plain two-point flux is measured beside it and does not."""
    keys = ("l2", "linf", "band_l2", "band_linf", "grad_l2", "grad_linf", "grad_cf_linf", "trunc_band_linf")
    out = {}
    for corrected in (False, True):
        rows = [_mms(n, corrected) for n in ns]
        orders = {k: [round(math.log2(a[k] / b[k]), 3) for a, b in zip(rows, rows[1:])] for k in keys}
        name = "corrected" if corrected else "two-point"
        print(f"MMS {name} " + str(rows))
        print(f"MMS {name} orders " + str(orders))
        out[name] = orders
        assert all(r["levels"][0] < r["leaves"] for r in rows)    # the layouts are refined, not uniform
    assert out["corrected"]["l2"][-1] >= 1.8
    assert out["corrected"]["grad_cf_linf"][-1] >= 0.9


# ── the march ────────────────────────────────────────────────────────────────────────────────────────────────────


def _kl(**kw):
    return SolverConfig(**dict(dict(scheme="fv", closure="k-l", drive="pressure", inflow="canopy", cfl=16.0, tol=1e-12,
                                     tol_momentum=1e-12, max_iter=600, steps=3, tol_final=1e-12, device=DEV), **kw))


def test_the_march_on_single_cells_is_the_dense_solver_s():
    """Three steps of production's terms (k-l, the pressure drive, the canopy inflow, the cut ground) in float64, every
    solve driven to round-off so the two multigrids' different paths do not show: the same velocity and k."""
    import amr_model
    from solver import solve
    scene = checks.site(checks.grid(1.0, 24, 16, 16))
    cfg = _kl()
    dense = solve(scene, checks.profile(), 250.0, cfg)
    am = amr_model.AmrModel(scene, checks.profile(), 250.0, cfg, amr.uniform_layout(scene.grid.shape))
    out = am.run()
    u = am.to_dense(out["u"]).cpu().numpy()                       # the leaves in the dense order, whatever their own
    k = am.to_dense(out["k"]).cpu().numpy()
    du = float(np.abs(u - dense.velocity).max() / np.abs(dense.velocity).max())
    dk = float(np.abs(k - dense.tke).max() / np.abs(dense.tke).max())
    print(f"MARCH single cells: velocity {du:.2e}, k {dk:.2e} of their largest")
    assert du <= 1e-7 and dk <= 1e-7


def test_a_refined_march_settles_near_the_dense_one():
    """Production's numerics on a refined layout of the synthetic site: settled, divergence-free, and its near-ground
    speeds within a few per cent of the dense solve's (the frontier on real sites was measured separately)."""
    import amr_model
    from solver import solve
    scene = checks.site(checks.grid(1.0, 48, 32, 24))
    cfg = SolverConfig(scheme="fv", closure="k-l", drive="pressure", inflow="canopy", cfl=16.0, tol=1e-3,
                       tol_momentum=0.1, settle_tol=0.002, settle_every=20, steps=40, max_steps=400, tol_final=1e-10,
                       device=DEV).fast()
    dense = solve(scene, checks.profile(), 250.0, cfg)
    leaf = amr.balance(amr.required(scene, _crit(shell=2.0, shell_out=2.0, band=1.0)), amr.LMAX)
    lay = amr.layout(leaf, scene.solid)
    am = amr_model.AmrModel(scene, checks.profile(), 250.0, cfg, lay)
    out = am.run()
    v, _ = am.dense_velocity(out["u"])
    h = scene.grid.zc[:, None, None] - np.asarray(scene.terrain)[None]
    band = (~np.asarray(scene.solid)) & (h >= 2) & (h <= 30)
    sd, sa = np.hypot(*dense.velocity[:2])[band], np.hypot(*v[:2])[band]
    med = abs(np.median(sa) / np.median(sd) - 1)
    p99 = float(np.percentile(np.abs(sa - sd), 99) / np.median(sd))
    print(f"MARCH refined: {lay.n} leaves of {scene.grid.cells}, {out['steps']} steps (dense {dense.steps}), median "
          f"{med:.4f}, cell p99 {p99:.4f}, divergence {out['divergence_max_1_s']:.1e}")
    assert out["settled"] and out["divergence_max_1_s"] <= 1e-6
    assert med <= 0.03


# ── the tile kernels ─────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(DEV != "cuda", reason="the tile kernels are Triton's, on a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_tile_kernels_are_the_csr_product(dtype):
    """On a tiled refined layout, the tile kernel's M x and one Gauss-Seidel color equal the CSR product's, for the
    projection's symmetric operator and a convective one with random fluxes."""
    import amr_tiles
    scene = checks.site(checks.grid(1.0, 48, 32, 24))
    leaf = amr.balance(amr.required(scene, _crit(shell=2.0, shell_out=2.0, band=1.0)), amr.LMAX)
    lay = amr.tiled(amr.layout(leaf, scene.solid))
    m = ag.dense_model(scene, checks.profile(), 250.0, SolverConfig(device=DEV))
    gr = ag.Graph(m, lay, DEV)
    hier = ag.Hierarchy(gr)
    lv = hier.levels[0]
    plan = amr_tiles.TilePlan(lay, lv)
    torch.manual_seed(2)
    c = torch.rand(lv.nf, dtype=torch.float64, device=gr.device) * (~gr.phantom[lv.a]).double()
    cb = torch.rand(lv.nb, dtype=torch.float64, device=gr.device)
    ident = torch.rand(gr.n, dtype=torch.float64, device=gr.device) * (~gr.phantom).double()
    fl = torch.randn(lv.nf, dtype=torch.float64, device=gr.device)
    flb = torch.randn(lv.nb, dtype=torch.float64, device=gr.device)
    for op in (ag.GraphOp(lv, ident, c, cb), ag.GraphOp(lv, ident, c, cb, fl, flb)):
        op = op.to(dtype)
        t = amr_tiles.TileOp(plan, op)
        x = torch.randn(1, gr.n, dtype=dtype, device=gr.device)
        want, got = op.apply(x), t.apply(x)
        tol = 1e-12 if dtype == torch.float64 else 1e-5
        assert float((got - want).abs().max()) <= tol * float(want.abs().max())
        safe = torch.where(op.diag > 0, op.diag, torch.ones_like(op.diag))
        b = torch.randn_like(x)
        for color in range(4):
            mask = lv.colors == color
            want = x + torch.where(mask, (b - op.apply(x)) / safe, torch.zeros_like(x))
            got = t.relax(safe, lv.colors, color, b, x)
            assert float((got - want).abs().max()) <= tol * float(want.abs().max())


def test_a_batch_of_headings_is_each_heading_alone():
    """Two headings marched together (one projection for both, each its own momentum and k operator, the inflow turned
    from the first's) give each heading's own march, float64, every solve driven to round-off."""
    import amr_model
    scene = checks.site(checks.grid(1.0, 24, 16, 16))
    leaf = amr.balance(amr.required(scene, _crit(shell=2.0, shell_out=2.0, band=1.0)), amr.LMAX)
    lay = amr.layout(leaf, scene.solid)
    cfg = _kl()
    batch = amr_model.AmrModel(scene, checks.profile(), [250.0, 290.0], cfg, lay)
    out = batch.run()
    for h, d in enumerate((250.0, 290.0)):
        one = amr_model.AmrModel(scene, checks.profile(), d, cfg, lay).run()
        du = float((out["u"][h] - one["u"]).abs().max() / one["u"].abs().max())
        dk = float((out["k"][h] - one["k"]).abs().max() / one["k"].abs().max())
        print(f"BATCH heading {d:g}: velocity {du:.2e}, k {dk:.2e} of their largest")
        assert du <= 1e-7 and dk <= 1e-7


@pytest.mark.skipif(DEV != "cuda", reason="the fused face kernels are Triton's, on a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_fused_face_kernels_are_the_eager_terms(dtype, monkeypatch):
    """On a refined layout with coarse-fine faces, the fused kernels' convection, diffusion and velocity gradients
    (amr_fused) equal the eager terms', two headings at once, random fields and fluxes, with the corrections."""
    import amr_fused
    import amr_model
    scene = checks.site(checks.grid(1.0, 48, 32, 24))
    leaf = amr.balance(amr.required(scene, _crit(shell=2.0, shell_out=2.0, band=1.0)), amr.LMAX)
    lay = amr.layout(leaf, scene.solid)
    cfg = _kl()
    am = amr_model.AmrModel(scene, checks.profile(), [250.0, 290.0], cfg, lay)
    geo = am.geo32 if dtype == torch.float32 else am.geo64
    if geo.w_low is None:
        geo = ag.Geo(am.gr, dtype)
    am.geo64 = geo if dtype == torch.float64 else am.geo64
    assert any(i.numel() for i in am.st.cf_idx), "the layout must carry coarse-fine faces"
    torch.manual_seed(3)
    H, n = 2, am.gr.n
    u = torch.randn(H, 3, n, dtype=dtype, device=am.dev)
    F = [torch.randn(H, f.numel(), dtype=dtype, device=am.dev) for f in geo.a]
    Fb = {s: torch.randn(H, geo.bcell[s].numel(), dtype=dtype, device=am.dev) for s in ag.SIDES}
    c = [torch.rand(H, f.numel(), dtype=dtype, device=am.dev) for f in geo.a]
    cb = {s: torch.rand(H, geo.bcell[s].numel(), dtype=dtype, device=am.dev) for s in ag.SIDES}
    ghosts = {s: torch.randn(H, geo.bcell[s].numel(), dtype=dtype, device=am.dev) for s in ag.SIDES}
    deltas = [am.st.delta(u[:, k]) for k in range(3)]
    tol = 1e-12 if dtype == torch.float64 else 2e-5

    def close(got, want, what):
        err = float((got - want).abs().max() / want.abs().max())
        print(f"FUSED {what} {dtype}: {err:.1e} of the largest")
        assert err <= tol, what
    fused = (am.convection(geo, u[:, 0], F, Fb, ghosts, deltas[0]), am.apply_diffusion(geo, c, cb, u[:, 1], deltas[1]),
             am.strain_rest(geo, u, deltas))
    monkeypatch.setattr(amr_model, "FUSED", False)
    eager = (am.convection(geo, u[:, 0], F, Fb, ghosts, deltas[0]), am.apply_diffusion(geo, c, cb, u[:, 1], deltas[1]),
             am.strain_rest(geo, u, deltas))
    for got, want, what in zip(fused, eager, ("convection", "diffusion", "strain")):
        close(got, want, what)
    assert amr_fused.available(u)


@pytest.mark.skipif(DEV != "cuda", reason="the row kernel is Triton's, on a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_row_kernel_is_the_eager_level_product(dtype):
    """On every coarse level of a refined layout's hierarchy, the row kernel's M x and each Gauss-Seidel color
    (amr_csr) equal the eager product's, for one symmetric operator and two convective ones grouped (G 2, R 3)."""
    import amr_csr
    scene = checks.site(checks.grid(1.0, 48, 32, 24))
    leaf = amr.balance(amr.required(scene, _crit(shell=2.0, shell_out=2.0, band=1.0)), amr.LMAX)
    lay = amr.layout(leaf, scene.solid)
    m = ag.dense_model(scene, checks.profile(), 250.0, SolverConfig(device=DEV))
    gr = ag.Graph(m, lay, DEV)
    hier = ag.Hierarchy(gr)
    lv = hier.levels[0]
    torch.manual_seed(5)
    kw = dict(dtype=torch.float64, device=gr.device)
    c, cb, ident = torch.rand(lv.nf, **kw), torch.rand(lv.nb, **kw), torch.rand(gr.n, **kw)
    ops = [ag.GraphOp(lv, ident, c, cb).to(dtype)]
    G = 2
    ops.append(ag.GroupOp(lv, torch.rand(G, gr.n, **kw), torch.rand(G, lv.nf, **kw), torch.rand(G, lv.nb, **kw),
                          torch.randn(G, lv.nf, **kw), torch.randn(G, lv.nb, **kw), R=3).to(dtype))
    tol = 1e-12 if dtype == torch.float64 else 2e-5
    checked = 0
    for op in ops:
        C = 3 if isinstance(op, ag.GraphOp) else G * 3
        for nxt in hier.levels[1:]:
            op = op.coarsen(nxt)
            k = amr_csr.CsrOp(op)
            x = torch.randn(C, nxt.n, dtype=dtype, device=gr.device)
            b = torch.randn_like(x)
            want, got = op.apply(x), k.apply(x)
            assert float((got - want).abs().max()) <= tol * float(want.abs().max())
            safe = torch.where(op.diag > 0, op.diag, torch.ones_like(op.diag))
            for color in range(4):
                mask = nxt.colors == color
                want = x + torch.where(mask, ag._over(b - op.apply(x), safe), torch.zeros_like(x))
                got = k.relax(safe, nxt.colors, color, b, x)
                assert float((got - want).abs().max()) <= tol * float(want.abs().max())
            checked += 1
    print(f"ROWS {checked} coarse levels checked, {dtype}")
    assert checked >= 4


def test_the_dense_output_is_the_whole_field_s():
    """The output put on the dense grid a component at a time (amr_model.dense_velocity) is the whole field's, put
    there at once and curled on the device: the same arithmetic, so equal to the float32 slopes' round-off (their
    neighbor sums are atomic, in no fixed order, run to run)."""
    import amr_model
    scene = checks.site(checks.grid(1.0, 24, 16, 16))
    leaf = amr.balance(amr.required(scene, _crit(shell=2.0, shell_out=2.0, band=1.0)), amr.LMAX)
    am = amr_model.AmrModel(scene, checks.profile(), 250.0, _kl(), amr.layout(leaf, scene.solid))
    u = am.run()["u"]
    v, w = am.dense_velocity(u)
    ref = am.to_dense(u.double())
    ref[:, am.dense_solid] = 0.0
    zc, yc, xc = am.grid_coords
    g = lambda c, d: torch.gradient(ref[c], spacing=[(zc, yc, xc)[d]], dim=[d])[0]  # noqa: E731
    curl = torch.stack([g(2, 1) - g(1, 0), g(0, 0) - g(2, 2), g(1, 2) - g(0, 1)])
    ref, curl = ref.cpu().numpy(), curl.cpu().numpy()
    dv, dw = np.abs(v - ref).max() / np.abs(ref).max(), np.abs(w - curl).max() / np.abs(curl).max()
    print(f"OUTPUT velocity {dv:.1e}, vorticity {dw:.1e} of their largest")
    assert dv <= 1e-6 and dw <= 1e-5


@pytest.mark.skipif(DEV != "cuda", reason="the row kernel is Triton's, on a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_row_kernel_is_cusparse_s_product(dtype):
    """A fixed CSR matrix with few rows filled (as the projection's coarse-fine correction), times two channels: the
    row kernel (amr_csr.Rows) equals cuSPARSE's product."""
    import amr_csr
    torch.manual_seed(7)
    n, nnz = 50_000, 30_000
    rows = torch.randint(0, n // 10, (nnz,), device=DEV) * 10
    cols = torch.randint(0, n, (nnz,), device=DEV)
    A = ag._csr(rows, cols, torch.randn(nnz, dtype=dtype, device=DEV), (n, n))
    x = torch.randn(2, n, dtype=dtype, device=DEV)
    import warnings
    with warnings.catch_warnings():                 # torch calls its CSR support beta
        warnings.simplefilter("ignore", UserWarning)
        want = (A @ x.T.contiguous()).T
    got = amr_csr.Rows(A).apply(x)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert float((got - want).abs().max()) <= tol * float(want.abs().max())
