"""Physics on synthetic scenes, asserted in physical units. No network, no site data."""

import numpy as np
import pytest
import torch

import checks
import domain
from solver import Model, SolverConfig, solve

CFG = SolverConfig(steps=60, tol=1e-6)
FAST = SolverConfig(steps=40, tol=1e-6)


@pytest.fixture(scope="module")
def cube():
    return checks.cube_wake(CFG)


@pytest.fixture(scope="module")
def flat():
    return checks.flat_profile(CFG)


def face_divergence(res, g) -> np.ndarray:
    """Net outward volume flux per cell from the staggered field, independently of the solver."""
    ux, uy, uz = res.faces
    area = (g.dx * g.dz)[:, None, None]
    return (area * (ux[..., 1:] - ux[..., :-1]) + area * (uy[:, 1:, :] - uy[:, :-1, :])
            + g.dx * g.dx * (uz[1:] - uz[:-1]))


def test_projection_alone_is_divergence_free_to_1e6(cube):
    g = checks.grid()
    res = solve(domain.cube(g, 8.0, centre=(20.0, 16.0)), checks.profile(), checks.WEST,
                SolverConfig(steps=0))
    assert res.divergence_rel <= 1e-6, res.divergence_rel
    assert res.divergence_max <= 1e-6, res.divergence_max
    u_top = checks.profile().speed(np.array([g.top]))[0]
    assert np.abs(face_divergence(res, g)).max() / (u_top * g.dx ** 2) <= 1e-6
    assert res.poisson_iterations == [res.poisson_iterations[0]] and res.poisson_iterations[0] < 40


def test_relaxed_field_is_divergence_free_to_1e6(cube):
    assert cube["divergence_rel"] <= 1e-6, cube
    assert cube["divergence_max"] <= 1e-6, cube


def test_flat_domain_recovers_the_log_profile_to_its_discretization(flat):
    """The log profile sampled at cell centers is not the discrete steady state exactly: the mixing-length flux and the
    wall law at the first cell leave a residual of 1e-4 of it, which the finite volumes resolve (the semi-Lagrangian
    passed 1e-10 only because its momentum solve accepted that residual as converged)."""
    assert flat["max_profile_error_rel"] < 1e-5, flat
    assert flat["max_crossflow_rel"] < 1e-5, flat
    assert flat["max_change_rel"] < 1e-5, flat


def test_mass_flux_in_equals_mass_flux_out_to_1e6(cube, flat):
    assert cube["flux_balance"] <= 1e-6, cube
    assert flat["flux_balance"] <= 1e-6, flat
    assert cube["flux_in_m3_s"] > 0


def test_velocity_is_zero_inside_solids(cube):
    assert cube["solid_max_speed"] == 0.0


def test_speed_through_canopy_falls_monotonically_with_lai():
    out = checks.canopy_lai(FAST)
    assert out["monotonic"], out
    assert out["exit_speed_ratio"][0] < 1.0 and out["exit_speed_ratio"][-1] < 0.6, out


def test_cube_sheds_a_wake_with_reversed_flow_and_speeds_up_over_its_roof(cube):
    assert cube["wake_min_u_over_u_h"] < -0.05, cube
    assert cube["wake_reversed_fraction"] > 0.05, cube
    assert cube["roof_max_speedup"] > 1.02, cube
    assert cube["side_max_speedup"] > 1.05, cube
    assert cube["roof_max_speedup_projection_only"] > 1.02, cube


def test_a_rotated_grid_gives_the_rotated_field():
    out = checks.rotation(FAST)
    assert out["max_speed_difference_rel"] < 1e-4, out


def test_the_wake_is_independent_of_grid_orientation_to_1_percent():
    out = checks.orientation(FAST)
    assert out["difference_rel"] < 0.01, out


def test_the_field_scales_linearly_with_inflow_speed():
    """Every steady term is homogeneous of degree two in velocity, so 2, 5 and 10 m/s give one
    field scaled; the molecular viscosity floor and the solver tolerances are the residue."""
    out = checks.linearity(FAST, g=checks.grid(1.0, 48, 24, 16), size=6.0, spacings_deg=(22.5,))
    assert out["max_deviation_from_linear_rel"] < 1e-3, out
    assert out["midway_heading_interpolation_error_rel"]["22.5"] < 1.0, out


def test_vorticity_of_solid_body_rotation_is_twice_the_rate():
    out = checks.vorticity_of_rotation()
    assert out["omega_z_over_2omega"] == pytest.approx(1.0, abs=1e-12)
    assert out["omega_z_spread"] < 1e-12 and out["omega_xy_max"] < 1e-12


def test_ridge_flow_conserves_mass_and_speeds_up_at_the_crest():
    out = checks.ridge_speedup(FAST)
    assert out["flux_balance"] <= 1e-6 and out["divergence_rel"] <= 1e-6, out
    assert out["crest_speedup"] > 1.0, out


def test_open_sides_prescribe_the_profile_only_where_the_background_enters():
    g = checks.grid(1.0, 16, 8, 8)
    easterly = SolverConfig(steps=0, lateral="open")  # flow toward -x
    m = Model(domain.flat(g), checks.profile(), 90.0, easterly)
    ux, uy, uz = m.faces(m.background())
    assert m.inflow == [False, True, False, False]
    assert torch.allclose(ux[..., -1], m.u0[0][:, None].expand(-1, g.ny))
    assert float(uz[0].abs().max()) == 0.0 and float(uy.abs().max()) < 1e-12
    assert Model(domain.flat(g), checks.profile(), 90.0, SolverConfig(steps=0)).inflow == [True] * 4


def test_a_warm_start_continues_where_the_cold_run_stopped():
    g = checks.grid(1.0, 32, 16, 12)
    scene = domain.cube(g, 6.0, centre=(12.0, 8.0))
    cold = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=30))
    start = cold.state if cold.state is not None else cold.velocity       # the finite volumes carry their pressure
    warm = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=10), initial=start)
    assert max(warm.change) < cold.change[0], (warm.change, cold.change)
    assert warm.divergence_rel <= 1e-6


def test_progress_lines_count_every_solve_and_change_no_number(capsys):
    """A caller reads `PROGRESS wind k/n` for its progress: printed only
    while `solver.PROGRESS` is set, counting this solve's steps after the solves before it, the field untouched."""
    import solver

    g = checks.grid(1.0, 32, 16, 12)
    scene = domain.cube(g, 6.0, centre=(12.0, 8.0))
    quiet = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=25))
    assert "PROGRESS" not in capsys.readouterr().out
    solver.PROGRESS = (2, 4)
    try:
        told = solve(scene, checks.profile(), checks.WEST, SolverConfig(steps=25))
    finally:
        solver.PROGRESS = None
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PROGRESS")]
    assert lines == ["PROGRESS wind 60/100", "PROGRESS wind 70/100", "PROGRESS wind 75/100"]
    assert np.array_equal(quiet.velocity, told.velocity) and quiet.change == told.change


def test_a_settling_run_stops_once_the_near_ground_speed_holds_and_says_when_it_cannot():
    """A delivered field is converged, not a fixed step count. The run steps until the
    fluid cells 2 to 30 m above their surface stop moving (median, p95, RMS), at least `steps`, at most `max_steps`."""
    g = checks.grid(1.0, 64, 32, 24)
    scene, p = domain.cube(g, 8.0, centre=(32.0, 16.0)), checks.profile()
    cfg = SolverConfig(steps=20, cfl=8.0, tol=1e-6, settle_tol=0.01, settle_every=10, max_steps=400)
    res = solve(scene, p, checks.WEST, cfg)
    assert res.settled is True and 20 <= res.steps < 400 and len(res.settle) == res.steps // 10 - 1
    last = res.settle[-1]
    assert last["d_median"] < 0.01 and last["d_p95"] < 0.01 and last["rms_rel"] < 0.01 * Model.SETTLE_RMS
    hard = solve(scene, p, checks.WEST, SolverConfig(steps=10, cfl=8.0, tol=1e-6, settle_tol=1e-12, settle_every=10,
                                                     max_steps=30))
    assert hard.settled is False and hard.steps == 30, "unsettled at its cap, and the result says so"
    plain = solve(scene, p, checks.WEST, SolverConfig(steps=15, tol=1e-6))
    assert plain.settled is None and plain.steps == 15, "without settle_tol, exactly `steps` as before"


def test_the_fast_numerics_deliver_the_same_settled_field_and_pass_the_divergence_gate():
    """`SolverConfig.fast`: float32 momentum, a float32 V-cycle under float64 conjugate gradients, warm projections.
    The settled field matches the all-float64 one to well under the product's 1 % and the delivered divergence is
    the float64 projection's."""
    g = checks.grid(1.0, 64, 32, 24)
    scene, p = domain.cube(g, 8.0, centre=(32.0, 16.0)), checks.profile()
    kw = dict(steps=20, cfl=8.0, tol=1e-6, tol_final=1e-9, settle_tol=0.01, settle_every=10, max_steps=400)
    ref = solve(scene, p, checks.WEST, SolverConfig(**kw))
    fast = solve(scene, p, checks.WEST, SolverConfig(**kw).fast())
    fluid = ~scene.solid
    rel = np.abs(fast.speed - ref.speed)[fluid].max() / ref.speed[fluid].max()
    assert fast.settled and rel < 1e-3, rel
    assert fast.divergence_max_1_s <= 1e-6 and fast.flux_balance <= 1e-6
    assert fast.velocity.dtype == ref.velocity.dtype
