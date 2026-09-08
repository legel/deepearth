"""Solver invariants on synthetic domains: no network, no site data, seconds to run.

Mass balance is the headline. It is what found all four of the integration defects documented
in `solver`, and it is the one property that cannot be satisfied by an accidentally
plausible-looking result.
"""

import numpy as np
import pytest

from solver import MassBalance, Probes, Result, Surface, SolverConfig, horton_rate, simulate

MM_HR = 1.0 / 1000.0 / 3600.0  # mm/hr -> m/s


def tilted_plane(nrows: int = 40, ncols: int = 40, dx: float = 5.0, slope: float = 0.002):
    """A uniformly sloping domain draining toward the south edge."""
    rows = np.arange(nrows, dtype=np.float32)[:, None]
    return np.repeat(20.0 - rows * dx * slope, ncols, axis=1).astype(np.float32)


def run(z, rain, dt_s=60.0, dx=5.0, **kw) -> Result:
    """Build a Surface and SolverConfig from loose kwargs and run, quietly."""
    surface = Surface(z=z, **{k: kw.pop(k) for k in list(kw) if k in Surface.__dataclass_fields__})
    kw.setdefault("frame_interval_min", 10.0)
    return simulate(surface, rain, SolverConfig(dx=dx, dt_s=dt_s, **kw), verbose=False)


def test_mass_balance_closes_without_infiltration():
    """Rain in must equal water stored plus water out, to a few thousandths of a percent."""
    rain = [50.0 * MM_HR] * 60 + [0.0] * 60
    res = run(tilted_plane(), rain)
    assert abs(res.mass.residual_pct) < 0.01, res.mass


def test_mass_balance_closes_with_infiltration_and_storage_cap():
    z = tilted_plane()
    rain = [80.0 * MM_HR] * 60 + [0.0] * 120
    res = run(
        z, rain,
        f0=25.0 * MM_HR, fc=10.0 * MM_HR, k=2.0 / 3600.0,
        max_deficit_m=np.full(z.shape, 0.05, dtype=np.float32),
    )
    assert abs(res.mass.residual_pct) < 0.01, res.mass
    assert res.mass.infiltrated > 0.0


def test_storage_cap_bounds_infiltration():
    """Once the profile fills, infiltration stops and further rain becomes runoff."""
    z = tilted_plane()
    cap = 0.01
    rain = [100.0 * MM_HR] * 240
    res = run(z, rain, f0=200.0 * MM_HR, fc=200.0 * MM_HR, k=0.0,
              max_deficit_m=np.full(z.shape, cap, dtype=np.float32))
    assert res.cum_infil.max() <= cap + 1e-9, res.cum_infil.max()
    assert res.cum_infil.max() == pytest.approx(cap, rel=1e-3), "cap should actually be reached"


def test_no_rain_produces_no_water():
    res = run(tilted_plane(), [0.0] * 30)
    assert res.h_max.max() == 0.0
    assert res.mass.outflow == pytest.approx(0.0, abs=1e-12)


def test_water_drains_and_does_not_accumulate_forever():
    """Given a long dry tail, water must leave rather than pond indefinitely.

    Measured as stored volume, not thresholded flooded area: on a grid this small the 0.05 m
    extent threshold quantises to a handful of cells and oscillates, so it cannot resolve the
    claim. Storage is continuous and says exactly what is being asserted. Flooded area still
    *rising* a day after rain stopped was the defect this solver was repaired for.
    """
    rain = [60.0 * MM_HR] * 60 + [0.0] * 600
    res = run(tilted_plane(), rain)
    assert res.mass.rain > 0.0
    stored_fraction = res.mass.stored / res.mass.rain
    outflow_fraction = res.mass.outflow / res.mass.rain
    assert stored_fraction < 0.10, f"{stored_fraction:.1%} of the storm still ponded at the end"
    assert outflow_fraction > 0.85, f"only {outflow_fraction:.1%} of the storm drained out"


def test_depths_stay_physical_under_an_intense_burst():
    """CFL sub-stepping must keep an extreme burst stable: no NaN, no absurd depths."""
    rain = [400.0 * MM_HR] * 120 + [0.0] * 120
    res = run(tilted_plane(), rain)
    assert np.isfinite(res.h_final).all()
    assert res.h_max.max() < 5.0, res.h_max.max()
    assert res.substep_cap_hits == 0


def test_substepping_delivers_the_whole_storm():
    """The clock defect: a nominal storm must actually arrive, not 7-11 % of it."""
    dt_s, n = 60.0, 90
    rate = 50.0 * MM_HR
    res = run(tilted_plane(), [rate] * n, dt_s=dt_s)
    expected_mm = rate * dt_s * n * 1000.0
    delivered_mm = res.mass.rain / (40 * 40 * 25.0) * 1000.0
    assert delivered_mm == pytest.approx(expected_mm, rel=1e-9)
    assert res.n_substeps > n, "an accumulating domain should need more sub-steps than intervals"


def test_final_frame_lands_on_the_end_of_the_run():
    """frames[-1] sampled before the end manufactures a mass-balance error that is not there."""
    rain = [50.0 * MM_HR] * 25
    res = run(tilted_plane(), rain, dt_s=60.0)
    assert res.frame_times_min[-1] == pytest.approx(25 * 60.0 / 60.0)
    assert np.array_equal(res.frames[-1], res.h_final)


def test_frames_are_saved_when_interval_exceeds_run_length():
    """The empty-list short circuit meant a long interval captured nothing at all."""
    res = run(tilted_plane(), [50.0 * MM_HR] * 3, dt_s=60.0, frame_interval_min=1e6)
    assert len(res.frames) >= 1
    assert np.array_equal(res.frames[-1], res.h_final)


def test_nodata_cells_never_hold_water():
    z = tilted_plane()
    z[0:5, 0:5] = np.nan
    res = run(z, [80.0 * MM_HR] * 60)
    assert res.h_final[0:5, 0:5].max() == 0.0
    assert res.cum_infil[0:5, 0:5].max() == 0.0


def test_scalar_and_uniform_manning_agree_to_float_noise():
    """Documented as non-bit-identical for a casting reason, so pin the magnitude."""
    z = tilted_plane()
    rain = [60.0 * MM_HR] * 60
    a = run(z, rain)
    b = run(z, rain, manning_n=np.full(z.shape, 0.040))
    assert np.abs(a.h_final - b.h_final).max() < 1e-6


def test_horton_rate_decays_from_f0_to_fc():
    f0, fc, k = 76.0 * MM_HR, 25.0 * MM_HR, 2.0 / 3600.0
    assert horton_rate(0.0, f0, fc, k) == pytest.approx(f0)
    assert horton_rate(1e9, f0, fc, k) == pytest.approx(fc)
    assert horton_rate(3600.0, f0, fc, k) < horton_rate(600.0, f0, fc, k)


def test_watershed_probe_reports_flux_across_its_own_boundary():
    z = tilted_plane()
    mask = np.zeros(z.shape, dtype=bool)
    mask[:20, :] = True
    res = simulate(
        Surface(z=z), [60.0 * MM_HR] * 90,
        SolverConfig(dx=5.0, dt_s=60.0), Probes(watershed_mask=mask), verbose=False,
    )
    ws = res.series["watershed_outflow_cms"]
    assert len(ws) == 90
    assert ws.max() > 0.0, "water draining south must cross the mask boundary"


def test_mass_balance_residual_is_zero_when_nothing_supplied():
    mass = MassBalance(rain=0.0, initial=0.0, infiltrated=0.0, stored=0.0, outflow=0.0)
    assert mass.residual_pct == 0.0
