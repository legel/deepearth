"""The hourly balance: published reference ET, exact drainage, Green-Ampt's equation, and the books closing every hour."""

import numpy as np
import torch

import balance as B
import routing
import soils


def cells(n, tex=(40, 20), z_r=1.0, no_soil=None, perv=None, s_max=1.0):
    h = soils.hydraulics(*tex)
    f = lambda v: torch.full((n,), float(v), dtype=torch.float64)                       # noqa: E731
    ns = torch.zeros(n, dtype=torch.bool) if no_soil is None else torch.as_tensor(no_soil)
    return B.Cells(alpha=f(0.15), fc=f(h["fc"]), wp=f(h["wp"]), theta_sat=f(h["theta_sat"]), theta_r=f(h["theta_r"]),
                   ksat=f(h["ksat"]), lam=f(h["lam"]), psi_f=f(h["psi_f"]), z_r=f(z_r), kcb=f(0.95), kc_max=f(1.2),
                   rew=f(9.0), f_ew=f(0.2), s_max=f(s_max), no_soil=ns,
                   perv=f(1.0) if perv is None else torch.as_tensor(perv, dtype=torch.float64),
                   roof=torch.zeros(n, dtype=torch.bool))


def state(n, th=0.25):
    z = lambda: torch.zeros(n, dtype=torch.float64)                                     # noqa: E731
    return B.State(c=z(), w=z(), theta1=torch.full((n,), th, dtype=torch.float64),
                   theta2=torch.full((n,), th, dtype=torch.float64), F=z(), dry=z())


def stored(s, k):
    return s.c + s.w + 1000.0 * (s.theta1 * B.ZE + s.theta2 * (k.z_r - B.ZE))


def test_et0_matches_fao56_example_19_with_the_asce_daytime_cd():
    # FAO-56 Example 19, 14-15 h: Rn 1.749 MJ m-2 h-1, T 38 C, ea 3.445 kPa, u2 3.3 m/s, P 101.3 kPa. FAO's own Cd 0.34
    # gives 0.627 mm/h (published 0.63); ASCE-EWRI 2005's daytime 0.24 gives 0.656.
    rn = torch.tensor([1.749 / 0.0036])
    got = B.et0_hourly(rn, rn, torch.tensor([38.0]), torch.tensor([3.445]), torch.tensor([101.3]), torch.tensor([3.3]))
    assert abs(float(got) - 0.656) < 0.002


def test_drainage_is_the_exact_hour_of_the_unit_gradient_equation():
    """A wet 10 cm sandy loam layer: the closed form matches a fine explicit integration and ends above fc."""
    f = lambda v: torch.tensor([float(v)], dtype=torch.float64)                         # noqa: E731
    tr, ts, ks, lam, fc, dz = f(0.041), f(0.3866), f(38.916), f(0.378), f(0.1386), f(0.1)
    th = 0.36
    q = float(B.drain(f(th), tr, ts, ks, lam, fc, dz))
    t = th
    for _ in range(200000):
        se = (t - 0.041) / (0.3866 - 0.041)
        t -= 38.916 * se ** (3 + 2 / 0.378) / (1000 * 0.1) / 200000
    assert abs(q - 1000 * 0.1 * (th - t)) < 0.01 and th - q / 100 > 0.1386 + 0.02
    assert float(B.drain(f(0.12), tr, ts, ks, lam, fc, dz)) == 0.0, "never below field capacity"


def test_green_ampt_capacity_solves_its_equation_and_falls_toward_ksat():
    k = cells(3)
    theta = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    F0 = torch.zeros(3, dtype=torch.float64)
    first = B.green_ampt_capacity(F0, k, theta)
    later = B.green_ampt_capacity(F0 + 200.0, k, theta)
    pd = k.psi_f * (k.theta_sat - theta)
    F1 = F0 + first
    assert torch.all((F1 - F0 - k.ksat - pd * torch.log((F1 + pd) / (F0 + pd))).abs() < 1e-6)
    assert torch.all(later < first) and torch.all(later >= k.ksat - 1e-9)


def _plane(ny=6, nx=8, slope=0.05, dx=0.5):
    return np.tile(10.0 - slope * np.arange(nx) * dx, (ny, 1))


def test_the_books_close_every_hour_through_rain_and_drying():
    """storage change = rain - AET - drainage - water leaving the grid, per hour, over the whole grid."""
    z = _plane()
    z[2:4, 3:5] -= 0.1                                                                  # a depression
    n = z.size
    net = routing.network(z, np.ones_like(z, bool), 0.5)
    lat = B.LateralGraph(routing.lateral(z, np.ones_like(z, bool), 0.5, sigma_m=0.0), n)
    sealed = np.zeros(n, bool)
    sealed[:8] = True                                                                   # a paved strip uphill
    k = cells(n, no_soil=sealed, perv=(~sealed).astype(float))
    s = state(n, 0.30)
    rain = [0, 12, 30, 5, 0, 0, 0, 0, 0, 2, 0, 0] * 4
    before = float(stored(s, k).sum())
    for h, p in enumerate(rain):
        noon = 12 <= h % 24 <= 16
        rs = torch.full((n,), 600.0 if noon else 0.0, dtype=torch.float64)
        f = B.hour(s, k, float(p), net, rs, torch.full((n,), 2.0, dtype=torch.float64), 20.0, 1.5, 100.0, 330.0,
                   lateral=lat)
        after = float(stored(s, k).sum())
        out = float((f["aet"] + f["drain"] + f["lost"]).sum())
        assert abs((before + p * n - out) - after) < 1e-3 * max(1.0, p * n), h
        assert torch.all(s.theta1 <= k.theta_sat + 1e-9) and torch.all(s.theta2 <= k.theta_sat + 1e-9)
        assert torch.all(s.c >= 0) and torch.all(s.w >= -1e-9)
        before = after


def test_a_sealed_cell_sheds_its_rain_and_keeps_no_soil_water():
    z = _plane(1, 4)
    net = routing.network(z, np.ones_like(z, bool), 0.5)
    k = cells(4, no_soil=[True, False, False, False], perv=[0.0, 1.0, 1.0, 1.0], s_max=0.0)
    s = state(4, 0.2)
    r = B.rain_step(s, k, 10.0, net)
    assert float(r["infil"][0]) == 0.0 and float(r["runon"][1]) > 9.99
    f = B.local_step(s, k, torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64), 20.0, 1.5,
                     100.0, 330.0)
    assert float(f["tr"][0]) == 0.0 and float(f["es"][0]) == 0.0 and float(f["drain"][0]) == 0.0


def test_the_root_zone_is_the_depth_weighted_mean_of_the_two_layers():
    k = cells(1, z_r=0.5)
    s = state(1)
    s.theta1[:] = 0.40
    s.theta2[:] = 0.20
    assert abs(float(B.root_zone(s, k)) - (0.40 * 0.1 + 0.20 * 0.4) / 0.5) < 1e-12


def test_a_run_s_metrics_are_nan_without_soil_and_ponding_stays_everywhere():
    z = _plane(2, 4)
    n = z.size
    net = routing.network(z, np.ones_like(z, bool), 0.5)
    sealed = np.zeros(n, bool)
    sealed[0] = True
    k = cells(n, no_soil=sealed, perv=(~sealed).astype(float))
    s = state(n, 0.2)
    rain = [0.0] * 10 + [20.0, 20.0] + [0.0] * 24
    one = lambda v: (lambda h: torch.full((n,), v, dtype=torch.float64))              # noqa: E731
    y = B.run(s, k, net, len(rain), rain, one(300.0), one(2.0), lambda h: (20.0, 1.2, 100.0, 330.0))
    m = y.metrics(k)
    assert torch.isnan(m["cwd_mm"][0]) and torch.all(m["cwd_mm"][1:] >= 0) and torch.all(m["aet_mm"][1:] > 0)
    assert not torch.isnan(m["pond_peak_mm"]).any() and float(m["received_mm"][1:].sum()) > 0
