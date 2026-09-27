"""The k-l closure (Katul et al. 2004): literature constants, the log law as an exact steady state, a canopy inflow that
holds whatever the fetch, a wake that stays linear in speed, and turbulence made in the canopy."""

import math

import numpy as np
import pytest
import torch

import checks
import domain
import solver
from physics import KAPPA
from solver import Model, SolverConfig, solve

KL = dict(closure="k-l", tol=1e-6)


def test_the_constants_are_the_published_ones_and_none_is_fitted():
    assert (solver.KL_C_MU, solver.KL_SIGMA_K) == (0.09, 1.0)                    # Launder and Spalding (1974)
    assert (solver.KL_BETA_P, solver.KL_BETA_D) == (1.0, 5.1)                    # Katul et al. (2004)
    assert solver.KL_D_OVER_H == 2.0 / 3.0


def test_the_length_is_kappa_h_in_the_open_and_constant_within_a_canopy():
    h = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0])
    assert np.allclose(solver._kl_length(h, 0.0), KAPPA * h)
    ell = solver._kl_length(h, 15.0)
    assert np.allclose(ell[:2], KAPPA * h[:2]), "near the ground, kappa h"
    assert np.allclose(ell[2:4], KAPPA * 5.0), "within the canopy, kappa (h_c - d) with d = 2 h_c / 3"
    assert np.allclose(ell[4:], KAPPA * (h[4:] - 10.0)), "above it, kappa (h - d), continuous at h_c"


def test_the_column_over_open_ground_is_the_log_law_with_k_at_u_star_squared_over_root_c_mu():
    g, p = checks.grid(1.0, 16, 16, 24), checks.profile()
    m = Model(domain.flat(g, checks.Z0), p, checks.WEST, SolverConfig(steps=1, inflow="canopy", **KL))
    col = m.column
    assert col["iterations"] < 5000
    want = p.speed(g.zc)
    assert np.abs(col["u"] - want).max() <= 1e-3 * want.max(), np.abs(col["u"] - want).max()
    k_log = p.u_star ** 2 / math.sqrt(solver.KL_C_MU)
    assert np.abs(col["k"] / k_log - 1).max() < 1e-3, col["k"] / k_log


def test_a_flat_domain_keeps_the_log_profile_and_its_k():
    """nu_t = C_mu^(1/4) kappa h sqrt(k) is (kappa h)^2 |dU/dh| wherever k = u*^2 / sqrt(C_mu), and production and
    dissipation are taken on the same faces, so the discrete log law and its k are an exact steady state."""
    g, p = checks.grid(), checks.profile()
    res = solve(domain.flat(g, checks.Z0), p, checks.WEST, SolverConfig(steps=40, **KL))
    want = p.speed(g.zc)[:, None, None]
    assert np.abs(res.velocity[0] - want).max() / want.max() < 1e-5
    k_log = p.u_star ** 2 / math.sqrt(solver.KL_C_MU)
    assert np.abs(res.tke / k_log - 1).max() < 1e-4, np.abs(res.tke / k_log - 1).max()   # 4.7e-5 measured


def test_a_canopy_inflow_holds_over_a_uniform_canopy_whatever_the_fetch():
    g = checks.grid(1.0, 64, 16, 24)
    s = domain.flat(g, checks.Z0)
    s.sink[g.zc < 8.0] = 0.2 * 3.0 / 8.0
    k = [i for i, z in enumerate(g.zc) if z < 8.0]
    res = solve(s, checks.profile(), checks.WEST, SolverConfig(inflow="canopy", steps=60, cfl=8.0, **KL))
    near, far = res.velocity[0, k, 8, 4], res.velocity[0, k, 8, 56]
    assert np.abs(far - near).max() <= 0.01 * near.max(), (near, far)
    kn, kf = res.tke[k, 8, 4], res.tke[k, 8, 56]
    assert np.abs(kf - kn).max() <= 0.02 * kn.max(), (kn, kf)


def test_k_is_uniform_above_the_canopy_and_falls_through_it_faster_with_the_wakes_short_circuit():
    """Above a dense canopy the stress is constant and so is k (u*^2 / sqrt(C_mu) of the column's own u*); within it k
    falls toward the ground, and the leaves' wakes, which take energy out of the cascade (beta_d), make it fall faster."""
    g, p = checks.grid(1.0, 16, 16, 30), checks.profile()
    s = domain.flat(g, checks.Z0)
    s.sink[g.zc < 12.0] = 0.2 * 4.0 / 12.0
    col = Model(s, p, checks.WEST, SolverConfig(steps=1, inflow="canopy", **KL)).column
    assert col["iterations"] < 5000
    h, k = col["h"], col["k"]
    above = k[(h > 20.0) & (h < 60.0)]
    assert above.max() / above.min() - 1 < 0.02, above
    inside = (h > 1.0) & (h < 12.0)
    assert (np.diff(k[inside]) > 0).all() and k[inside].max() < above.min(), k[inside]
    beta_d = solver.KL_BETA_D
    try:
        solver.KL_BETA_D = 0.0
        more = Model(s, p, checks.WEST, SolverConfig(steps=1, inflow="canopy", **KL)).column
    finally:
        solver.KL_BETA_D = beta_d
    assert (more["k"][inside] > k[inside]).all()


def test_the_field_scales_linearly_with_inflow_speed():
    """k goes as the speed squared, so every term of both equations is homogeneous and one heading at three speeds is
    one field scaled."""
    out = checks.linearity(SolverConfig(steps=40, **KL), g=checks.grid(1.0, 48, 24, 16), size=6.0,
                           spacings_deg=(22.5,))
    assert out["max_deviation_from_linear_rel"] < 1e-3, out


def test_a_state_carries_k_and_a_warm_start_continues_from_it():
    g, p = checks.grid(1.0, 32, 16, 16), checks.profile()
    scene = domain.cube(g, 6.0, centre=(12.0, 8.0), z0=checks.Z0)
    cold = solve(scene, p, checks.WEST, SolverConfig(steps=30, **KL))
    assert cold.state["k"].shape == g.shape and (cold.tke[~scene.solid] > 0).all()
    half = solve(scene, p, checks.WEST, SolverConfig(steps=15, **KL))
    warm = solve(scene, p, checks.WEST, SolverConfig(steps=15, **KL), initial=half.state)
    assert np.abs(warm.tke - cold.tke).max() < 1e-2 * cold.tke.max()


def test_a_pressure_driven_column_balances_its_drag_and_makes_a_trunk_space_maximum():
    """Driven by a mean pressure gradient (no stress through the top), the column's drag sums to the driving force
    over its depth, and a crown over an open trunk space lets the trunk space run faster than the crown's base, which a
    stress-driven column never does (its stress is down-gradient everywhere)."""
    g, p = checks.grid(1.0, 16, 16, 40), checks.profile()
    s = domain.flat(g, checks.Z0)
    s.sink[(g.zc > 10.0) & (g.zc < 20.0)] = 0.2 * 4.0 / 10.0
    col = Model(s, p, checks.WEST, SolverConfig(steps=1, inflow="canopy", drive="pressure", **KL)).column
    dz = np.diff(g.zf)
    u, drag = col["u"], col["drag"].copy()
    drag[0] += float(solver._wall(torch.tensor(dz[0] / 2), torch.tensor(col["z0"]), torch.tensor(dz[0])))
    assert np.sum(dz * drag * u * u) == pytest.approx(col["body"] * np.sum(dz), rel=1e-3)
    h = col["h"]
    trunk, crown_base = u[(h > 3.0) & (h < 6.0)].max(), u[(h > 10.0) & (h < 12.0)].min()
    assert trunk > crown_base, (trunk, crown_base)
    shear = Model(s, p, checks.WEST, SolverConfig(steps=1, inflow="canopy", **KL)).column["u"]
    assert (np.diff(shear) > 0).all(), "a stress-driven column rises everywhere"


def test_a_pressure_driven_canopy_inflow_holds_whatever_the_fetch():
    g = checks.grid(1.0, 64, 16, 24)
    s = domain.flat(g, checks.Z0)
    s.sink[g.zc < 8.0] = 0.2 * 3.0 / 8.0
    k = [i for i, z in enumerate(g.zc) if z < 8.0]
    cfg = SolverConfig(inflow="canopy", drive="pressure", steps=60, cfl=8.0, **KL)
    res = solve(s, checks.profile(), checks.WEST, cfg)
    near, far = res.velocity[0, k, 8, 4], res.velocity[0, k, 8, 56]
    assert np.abs(far - near).max() <= 0.01 * near.max(), (near, far)
