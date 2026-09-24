"""Green-Ampt with redistribution: against Green and Ampt's exact solution, and inside the flow solver."""

import numpy as np
import pytest
import torch

import infiltration as gar
from infiltration import Soil
from solver import SolverConfig, Surface, simulate

MM_HR = 1.0 / 1000.0 / 3600.0


def tensors(soil: Soil):
    g = {k: torch.as_tensor(np.asarray(getattr(soil, k), dtype=np.float64))
         for k in ("ks", "psi_f", "theta_s", "theta_r", "lam", "theta_i")}
    g["f_max"] = torch.as_tensor(np.full(soil.ks.shape, np.inf) if soil.f_max is None else soil.f_max)
    return g


def green_ampt_time(F: float, S: float, ks: float) -> float:
    """Green and Ampt (1911), ponded from t = 0: t = (F - S ln(1 + F / S)) / K_s."""
    return (F - S * np.log1p(F / S)) / ks


def run_cells(soil: Soil, supply, dt: float, bank=None, keep_ponded=False):
    """Drive `step` directly: `supply` [m/s] per step on a (1,) cell; returns infiltrated per step and the bank."""
    g = tensors(soil)
    b = torch.as_tensor(soil.bank((1,)) if bank is None else bank)
    h = torch.zeros(1, dtype=torch.float64)
    infs = []
    for q in supply:
        h = h + q * dt if not keep_ponded else torch.full((1,), 10.0, dtype=torch.float64)
        inf, b = gar.step(h, b, torch.tensor(dt, dtype=torch.float64), g, 1e-5)
        h = h - inf
        infs.append(float(inf))
    return np.array(infs), b.numpy(), float(h)


@pytest.mark.parametrize("texture,dt", [("loam", 1.0), ("loam", 60.0), ("sandy loam", 10.0), ("clay", 600.0)])
def test_ponded_infiltration_is_green_and_ampt_exactly(texture, dt):
    soil = Soil.texture(texture, theta_i=0.10, shape=(1,))
    infs, bank, _ = run_cells(soil, [0.0] * int(3 * 3600 / dt), dt, keep_ponded=True)
    S = soil.psi_f[0] * (soil.theta_s[0] - 0.10)
    F = np.cumsum(infs)
    t = dt * np.arange(1, len(F) + 1)
    t_ga = np.array([green_ampt_time(f, S, soil.ks[0]) for f in F])
    assert np.abs(t_ga - t).max() / t[-1] < 1e-9
    assert bank[0, 0] == pytest.approx(F[-1], rel=1e-12) and bank[1, 0] == pytest.approx(soil.theta_s[0])


def smith_1993_reference(soil: Soil, r: float, t_end: float, t0: float) -> float:
    """theta_0 at t_end for constant r < K_s from an independent stiff solve (scipy Radau) of Smith et al. (1993):
    F = r t, Z dtheta/dt = r - (K(theta) - K_i) - K_s G(theta_i, theta) / Z, Z = F / (theta - theta_i)."""
    from scipy.integrate import solve_ivp

    ks, ps, ts, tr, lam, ti = (float(np.asarray(getattr(soil, k)).ravel()[0])
                               for k in ("ks", "psi_f", "theta_s", "theta_r", "lam", "theta_i"))
    se = lambda th: min(max((th - tr) / (ts - tr), 0.0), 1.0)  # noqa: E731
    K = lambda th: ks * se(th) ** (3 + 2 / lam)  # noqa: E731
    c = 3 + 1 / lam
    G = lambda th: ps * (se(th) ** c - se(ti) ** c) / (1 - se(ti) ** c)  # noqa: E731

    def rhs(t, y):
        Z = r * t / max(y[0] - ti, 1e-12)
        return [(r - (K(y[0]) - K(ti)) - ks * G(y[0]) / Z) / Z]

    return float(solve_ivp(rhs, (t0, t_end), [ts], method="Radau", rtol=1e-10, atol=1e-12).y[0, -1])


def test_light_rain_all_enters_and_the_front_follows_smith_1993():
    """r < K_s never ponds; the wetted zone's content follows an independent stiff solve of Smith et al. (1993), and
    its capillary share falls as the front deepens toward gravity drainage, K(theta) - K_i -> r."""
    soil = Soil.texture("loam", theta_i=0.12, shape=(1,))
    g = tensors(soil)
    r, dt = 0.5 * soil.ks[0], 60.0
    shares = []
    for hours in (6, 48):
        infs, bank, h = run_cells(soil, [r] * hours * 60, dt)
        assert infs.sum() == pytest.approx(r * hours * 3600, rel=1e-9) and h == pytest.approx(0.0, abs=1e-12)
        assert bank[1, 0] == pytest.approx(smith_1993_reference(soil, r, hours * 3600.0, dt), abs=2e-3)
        th, F = torch.tensor(bank[1]), torch.tensor(bank[0])
        cap = g["ks"] * gar.capillary_drive(g["theta_i"], th, g["psi_f"], g["theta_s"], g["theta_r"], g["lam"]) / (
            F / (th - g["theta_i"]))
        shares.append(float(cap) / r)
    assert shares[1] < shares[0]


def test_redistribution_restores_capacity_for_the_next_storm():
    soil = Soil.texture("silt loam", theta_i=0.15, shape=(1,))
    storm = [30.0 * MM_HR] * 60
    first, bank, _ = run_cells(soil, storm, 60.0)
    drained, bank2, _ = run_cells(soil, [0.0] * 24 * 60, 60.0, bank=bank)
    assert bank2[1, 0] < bank[1, 0] - 0.05, "the wetted zone drains between storms"
    assert bank2[0, 0] == pytest.approx(bank[0, 0], rel=1e-12), "draining moves water down, it does not lose it"
    again, bank3, _ = run_cells(soil, storm, 60.0, bank=bank2)
    no_break, _, _ = run_cells(soil, storm, 60.0, bank=bank)
    assert again.sum() > no_break.sum() * 1.05, "a drained soil takes more of the same storm than a wet one"
    assert bank3[0, 0] + bank3[2, 0] == pytest.approx(first.sum() + again.sum(), rel=1e-9)


def test_sealed_and_full_soils_take_nothing_more():
    soil = Soil.texture("sand", theta_i=0.05, shape=(2,), f_max=0.004)
    soil.ks[1] = 0.0
    g = tensors(soil)
    b = torch.as_tensor(soil.bank((2,)))
    total = torch.zeros(2, dtype=torch.float64)
    for _ in range(600):
        inf, b = gar.step(torch.full((2,), 0.01, dtype=torch.float64), b, torch.tensor(10.0, dtype=torch.float64),
                          g, 1e-5)
        total += inf
    assert float(total[1]) == 0.0
    assert float(total[0]) == pytest.approx(0.004, rel=1e-12)


def test_bank_stays_physical_under_intermittent_rain():
    soil = Soil.texture("sandy clay loam", theta_i=0.10, shape=(1,))
    rng = np.random.default_rng(3)
    supply = np.where(rng.random(3000) < 0.3, rng.exponential(20.0 * MM_HR, 3000), 0.0)
    infs, bank, h = run_cells(soil, supply, 30.0)
    assert np.all(infs >= 0.0) and np.isfinite(bank).all()
    assert infs.sum() + h == pytest.approx(supply.sum() * 30.0, rel=1e-9)
    assert bank[0, 0] + bank[2, 0] == pytest.approx(infs.sum(), rel=1e-9)
    assert 0.10 <= bank[1, 0] <= soil.theta_s[0]
    assert bank[2, 0] == 0.0 or bank[1, 0] <= bank[3, 0] + 1e-12


def plane(n: int = 30, dx: float = 5.0, slope: float = 0.01) -> np.ndarray:
    rows = np.arange(n, dtype=np.float64)[:, None]
    return np.repeat(20.0 - rows * dx * slope, n, axis=1).astype(np.float32)


def solve(z, rain, soil, dtype="float64", soil_state=None):
    surface = Surface(z=z, soil=soil, soil_state=soil_state)
    cfg = SolverConfig(dx=5.0, dt_s=60.0, dtype=dtype, device="cpu", frame_interval_min=30.0)
    return simulate(surface, rain, cfg, verbose=False)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mass_closes_with_green_ampt_in_the_flow_solver(dtype):
    z = plane()
    soil = Soil.texture("loam", theta_i=0.15, shape=z.shape)
    res = solve(z, [60.0 * MM_HR] * 60 + [0.0] * 60, soil, dtype)
    m = res.mass
    out = m.infiltrated + m.abstracted + m.stored + m.outflow
    assert m.infiltrated > 0 and m.outflow > 0
    assert abs(m.supplied - out) / m.supplied < (1e-6 if dtype == "float64" else 1e-4)
    bank = res.soil_state
    assert bank.shape == (5,) + z.shape
    np.testing.assert_allclose(bank[0] + bank[2], res.cum_infil, rtol=1e-5, atol=1e-9)


def test_ponded_water_in_the_flow_solver_follows_green_and_ampt():
    z = np.full((12, 12), 10.0, dtype=np.float32)
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 200.0
    soil = Soil.texture("clay loam", theta_i=0.20, shape=z.shape)
    surface = Surface(z=z, soil=soil, initial_h=np.where(z < 50.0, 0.5, 0.0))
    res = simulate(surface, [0.0] * 120, SolverConfig(dx=5.0, dt_s=60.0, dtype="float64", device="cpu"),
                   verbose=False)
    F = float(res.cum_infil[5, 5])
    S = soil.psi_f[0, 0] * (soil.theta_s[0, 0] - 0.20)
    assert green_ampt_time(F, S, soil.ks[0, 0]) == pytest.approx(7200.0, rel=1e-6)


def test_the_bank_carries_one_storm_into_the_next():
    z = plane()
    soil = Soil.texture("silt loam", theta_i=0.15, shape=z.shape)
    storm = [40.0 * MM_HR] * 30
    first = solve(z, storm + [0.0] * 30, soil)
    fresh = solve(z, storm, soil)
    carried = solve(z, storm, soil, soil_state=first.soil_state)
    assert carried.mass.infiltrated < fresh.mass.infiltrated
    assert carried.mass.outflow + carried.mass.stored > fresh.mass.outflow + fresh.mass.stored


def test_horton_runs_unchanged_without_a_soil():
    z = plane()
    res = simulate(Surface(z=z, f0=25.0 * MM_HR, fc=10.0 * MM_HR, k=2.0 / 3600.0), [60.0 * MM_HR] * 30,
                   SolverConfig(dx=5.0, dt_s=60.0, dtype="float64", device="cpu"), verbose=False)
    assert res.soil_state is None and res.mass.infiltrated > 0
