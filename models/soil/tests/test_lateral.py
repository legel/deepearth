"""Water moves downslope below the surface: valleys are wetter than ridges for a physical reason, and the books close."""

import numpy as np
import torch

import balance as B
import routing


def cells(n, fc=0.25, tsat=0.45, ksat=39.0, z_r=0.6):
    f = lambda v: torch.full((n,), float(v))                                            # noqa: E731
    return B.Cells(alpha=f(0.15), fc=f(fc), wp=f(0.1), theta_sat=f(tsat), theta_r=f(0.05), ksat=f(ksat), lam=f(0.3),
                   psi_f=f(90.0), z_r=f(z_r), kcb=f(1.0), kc_max=f(1.2), rew=f(9.0), f_ew=f(0.5), s_max=f(0.0),
                   no_soil=torch.zeros(n, dtype=torch.bool), perv=f(1.0), roof=torch.zeros(n, dtype=torch.bool))


def state(n, th2, lanes=1):
    z = lambda: torch.zeros(lanes, n)                                                   # noqa: E731
    return B.State(c=z(), w=z(), theta1=torch.full((lanes, n), 0.25),
                   theta2=torch.as_tensor(th2, dtype=torch.float32).reshape(lanes, n).clone(), F=z(), dry=z())


def valley(ny=20, nx=20, dx=0.5):
    """A V valley draining south: 10 % side slopes into a channel at the middle column, 2 % down the channel."""
    j = np.abs(np.arange(nx) - nx // 2) * dx * 0.10
    i = -np.arange(ny)[:, None] * dx * 0.02
    return j[None, :] + i + 10.0


def total(s, k):
    return float((s.w + 1000 * s.theta1 * B.ZE + 1000 * s.theta2 * (k.z_r - B.ZE)).sum())


def test_water_above_field_capacity_moves_into_the_valley_and_mass_is_kept():
    z = valley()
    ny, nx = z.shape
    n = ny * nx
    g = B.LateralGraph(routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0), n)
    k = cells(n)
    s = state(n, np.full(n, 0.35))                                  # a wet spring: every cell above fc
    before = total(s, k)
    for _ in range(300):
        B.lateral_step(s, k, g)
    assert abs(total(s, k) - before) < 1e-4 * before
    th = s.theta2.reshape(ny, nx)
    assert th[2:-2, nx // 2].mean() > th[2:-2, [1, -2]].mean() + 0.02
    assert 0.07 < float(g.tanb.reshape(ny, nx)[5, 2]) < 0.11, "the weighted gradient of a 10 % side slope"


def test_a_dry_slope_passes_nothing_and_a_flat_has_no_gradient():
    z = valley()
    n = z.size
    g = B.LateralGraph(routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0), n)
    out = B.lateral_step(state(n, np.full(n, 0.20)), cells(n), g)
    assert float(out["lateral_out"].abs().max()) == 0.0
    flat = routing.lateral(np.full((10, 10), 5.0), np.ones((10, 10), bool), 0.5, sigma_m=0.0)
    assert len(flat.src) == 0 and float(np.abs(flat.tanb).max()) == 0.0


def test_a_full_receiver_returns_the_rest_to_the_surface():
    z = np.array([[2.0, 1.0]])
    g = B.LateralGraph(routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0), 2)
    k = cells(2, ksat=5000.0)
    s = state(2, np.array([0.45, 0.45]))
    before = total(s, k)
    out = B.lateral_step(s, k, g)
    assert float(out["return_flow"][0, 1]) > 0 and float(s.w[0, 1]) > 0, "exfiltration where the hollow is full"
    assert abs(total(s, k) - before) < 1e-3


def test_the_flux_is_the_layer_s_own_conductivity_times_the_gradient():
    """K(theta2) tan(beta) through the layer, consistent with its vertical drainage."""
    z = np.array([[1.0, 0.95]])                                     # a 10 % slope over 0.5 m
    g = B.LateralGraph(routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0), 2)
    out = B.lateral_step(state(2, np.array([0.35, 0.25])), cells(2), g)
    se = (0.35 - 0.05) / 0.40
    want = 39.0 * se ** (3 + 2 / 0.3) * 0.1 * 1000 * 0.5 / 500.0
    assert abs(float(out["lateral_out"][0, 0]) - want) < 1e-3 * want


def test_one_dimensional_state_steps_the_same_as_one_lane():
    z = valley(8, 8)
    n = z.size
    g = B.LateralGraph(routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0), n)
    k = cells(n)
    th = np.linspace(0.26, 0.40, n)
    a, b = state(n, th), state(n, th)
    one = B.State(**{f: getattr(b, f)[0].clone() for f in ("c", "w", "theta1", "theta2", "F", "dry")})
    B.lateral_step(a, k, g)
    B.lateral_step(one, k, g)
    assert torch.allclose(a.theta2[0], one.theta2) and torch.allclose(a.w[0], one.w)


def test_micro_relief_does_not_steer_the_subsurface_but_the_landform_does():
    rng = np.random.default_rng(1)
    z = valley(40, 40) + rng.normal(0.0, 0.05, (40, 40))
    raw = routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0)
    lat = routing.lateral(z, np.ones_like(z, bool), 0.5)
    side = np.s_[8:32, 4:16]
    assert np.std(raw.tanb.reshape(40, 40)[side]) > 3 * np.std(lat.tanb.reshape(40, 40)[side])
    assert 0.05 < float(np.median(lat.tanb.reshape(40, 40)[side])) < 0.13, "the 10 % side slope survives"
