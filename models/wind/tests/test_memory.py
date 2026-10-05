"""The solver's memory economies change no number they are not allowed to.

Each economy is switched off and on within the same test run, on the same scene, and the two runs compared: what frees
storage nothing reads again (`solver.LEAN`) must give the same field bit for bit. The scenes are production's: a
cut-cell site with buildings and crowns under k-l, the pressure drive, the canopy inflow and `SolverConfig.fast`, and
a cube under the mixing length, each settling in windows and ending on the final projection. On the CPU, where every
reduction is deterministic."""

import numpy as np
import pytest

import checks
import domain
import solver
from solver import Model, SolverConfig, solve


def _production(**kw) -> SolverConfig:
    kw = dict(dict(steps=20, max_steps=20), **kw)
    return SolverConfig(scheme="fv", closure="k-l", drive="pressure", inflow="canopy", cfl=16.0, tol=1e-3,
                        tol_momentum=0.1, settle_tol=0.002, settle_every=10, tol_final=1e-9,
                        **kw).fast()


def _mixing(**kw) -> SolverConfig:
    return SolverConfig(steps=10, cfl=8.0, tol=1e-6, tol_final=1e-9, settle_tol=0.01, settle_every=5, max_steps=10,
                        **kw).fast()


CASES = {
    "site k-l": (lambda: checks.site(checks.grid(1.0, 48, 32, 24)), _production),
    "cube mixing": (lambda: domain.cube(checks.grid(1.0, 32, 16, 12), 6.0, centre=(12.0, 8.0)), _mixing),
}


def _assert_identical(a, b):
    assert np.array_equal(a.velocity, b.velocity)
    for fa, fb in zip(a.faces, b.faces):
        assert np.array_equal(fa, fb)
    assert np.array_equal(a.state["pressure"], b.state["pressure"])
    assert (a.tke is None and b.tke is None) or np.array_equal(a.tke, b.tke)
    assert a.change == b.change and a.poisson_iterations == b.poisson_iterations
    assert a.settle == b.settle and a.steps == b.steps and a.settled == b.settled
    assert (a.divergence_max_1_s, a.flux_in_m3_s, a.flux_out_m3_s) == (b.divergence_max_1_s, b.flux_in_m3_s,
                                                                       b.flux_out_m3_s)


@pytest.mark.parametrize("case", sorted(CASES))
def test_freeing_what_the_run_no_longer_reads_changes_no_bit(case, monkeypatch):
    make, config = CASES[case]
    scene = make()
    monkeypatch.setattr(solver, "LEAN", False)
    kept = solve(scene, checks.profile(), 250.0, config())
    monkeypatch.setattr(solver, "LEAN", True)
    lean = solve(scene, checks.profile(), 250.0, config())
    assert lean.settle, "the settling windows ran"
    _assert_identical(kept, lean)


def test_the_lean_model_holds_no_float64_copy_of_what_its_working_copy_holds():
    scene = checks.site(checks.grid(1.0, 32, 24, 16))
    m = Model(scene, checks.profile(), 250.0, _production())
    w = m._work
    assert w is not m
    for name in Model.LEAN_MODEL:
        assert getattr(m, name, None) is None, name
    for name in Model.LEAN_WORK + ("height",):
        assert getattr(w, name, None) is None, name
    assert w.k is not None and w.ell_z is not None and w.wall is not None, "the working copy keeps what it reads"
    assert m.poisson.op.ax is None and m.poisson.levels[0].op.ax is None, "the projection's face arrays are freed"
    m.release()


def _rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-30))


@pytest.mark.parametrize("case", sorted(CASES))
def test_carrying_the_viscosity_on_the_z_faces_alone_moves_the_field_under_1e_6(case, monkeypatch):
    """The x- and y-face viscosities are linear averages of the z-faces': relaxed apart or derived from the relaxed
    z-faces they differ by rounding alone, and the delivered field by under 1e-6 of its largest value."""
    make, config = CASES[case]
    scene = make()
    monkeypatch.setattr(solver, "NU_Z_ONLY", False)
    three = solve(scene, checks.profile(), 250.0, config())
    monkeypatch.setattr(solver, "NU_Z_ONLY", True)
    one = solve(scene, checks.profile(), 250.0, config())
    got = {"velocity": _rel(one.velocity, three.velocity), "pressure": _rel(one.state["pressure"], three.state["pressure"])}
    if three.tke is not None:
        got["tke"] = _rel(one.tke, three.tke)
    print(f"NU_Z_ONLY {case}: {got}")
    assert max(got.values()) <= 1e-6, got
    assert one.steps == three.steps and one.divergence_max_1_s <= 1e-6


def _model():
    return Model(checks.site(checks.grid(1.0, 24, 20, 16)), checks.profile(), 250.0, _production())


def test_the_strain_and_the_curl_one_gradient_at_a_time_are_the_same_numbers():
    """`strain_rest` and `vorticity` once took all nine gradients at once; made one or two at a time, every bit holds."""
    import torch
    m = _model()
    w = m._work
    u = torch.randn((3,) + m.solid.shape, dtype=torch.float32, generator=torch.Generator().manual_seed(7))
    (uz, uy, ux), (vz, vy, vx), (wz, wy, wx) = [torch.gradient(u[c], spacing=[w.zc, w.yc, w.xc], dim=(0, 1, 2))
                                                for c in range(3)]
    strain = (2 * (ux ** 2 + vy ** 2 + wz ** 2) + (uy + vx) ** 2 + wx ** 2 + 2 * uz * wx + wy ** 2 + 2 * vz * wy)
    curl = torch.stack([wy - vz, uz - wx, vx - uy])
    assert torch.equal(w.strain_rest(u), strain)
    assert torch.equal(w.vorticity(u), curl)
    m.release()


def test_the_pressure_gradient_one_axis_at_a_time_is_the_same_numbers():
    import torch
    m = _model()
    pi = torch.randn(m.solid.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    for dtype in (torch.float32, torch.float64):
        assert torch.equal(m.pressure_cells(pi, dtype), m.cells(m.face_gradient(pi)).to(dtype))
    m.release()


def test_the_red_cells_are_the_even_ones():
    import torch
    from poisson import Solver
    op = Model(checks.site(checks.grid(1.0, 12, 10, 8)), checks.profile(), 250.0, SolverConfig()).poisson.op
    z, y, x = [torch.arange(n) for n in op.shape]
    assert torch.equal(Solver._parity(op), (z[:, None, None] + y[None, :, None] + x[None, None, :]) % 2 == 0)


def _levels(scene, velocity) -> dict:
    """Median and p95 of the horizontal speed on each published level (`Model.level_stats`) of a delivered field."""
    import torch
    m = Model(scene, checks.profile(), 250.0, _production())
    out = m.level_stats(torch.as_tensor(velocity))
    m.release()
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_float32_march_with_a_float64_final_projection_delivers_the_float64_march_s_field(case):
    """`SolverConfig.fast` marches in float32 (faces, multiplier, every step's projection) and projects the field it
    reaches once more in float64. Every published level's median and p95 of the speed within 1e-4 of the float64
    march's, and the delivered divergence under the 1e-6 1/s gate, cut cells open 5 % included."""
    make, config = CASES[case]
    scene = make()
    f64 = config()
    f64.march_dtype = None
    ref = solve(scene, checks.profile(), 250.0, f64)
    got = solve(scene, checks.profile(), 250.0, config())
    lv_ref, lv_got = _levels(scene, ref.velocity), _levels(scene, got.velocity)
    assert lv_ref.keys() == lv_got.keys() and lv_ref
    worst = max(abs(g / r - 1.0) for k in lv_ref for g, r in zip(lv_got[k], lv_ref[k]))
    print(f"MARCH32 {case}: levels {worst:.2e}, velocity {_rel(got.velocity, ref.velocity):.2e}, divergence "
          f"{got.divergence_max_1_s:.2e} 1/s (float64 march {ref.divergence_max_1_s:.2e}), steps {got.steps}/{ref.steps}")
    assert worst <= 1e-4, (lv_got, lv_ref)
    assert got.divergence_max_1_s <= 1e-6 and got.flux_balance <= 1e-6
    assert got.velocity.dtype == np.float64 and got.state["pressure"].dtype == np.float64
    if getattr(scene, "cut_open", None) is not None:
        assert float(np.min(scene.cut_open)) <= 0.06, "the site has cut cells open about 5 %"


def test_the_march_shares_the_float32_hierarchy_and_parks_the_float64_operator_until_the_final_projection():
    import torch
    m = Model(checks.site(checks.grid(1.0, 24, 20, 16)), checks.profile(), 250.0, _production(steps=2, max_steps=2))
    assert m._marcher() is m._work and m._work.poisson.levels is m.poisson.levels, "one hierarchy, shared"
    assert m._work.poisson.op.diag.dtype == torch.float32 and m.poisson.op.diag.dtype == torch.float64
    res = m.run()
    assert m.poisson.op.diag.device == m.solid.device, "back on the model's device after the march"
    assert res.divergence_max_1_s <= 1e-6 and res.velocity.dtype == np.float64
    m.release()
