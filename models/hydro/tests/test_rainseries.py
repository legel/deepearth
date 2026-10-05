"""A measured rain series as forcing: read, integrated onto the solver's steps, and conserved through a run."""

import numpy as np
import pytest

import rainseries
from solver import SolverConfig, Surface, simulate


def write(path, rows):
    path.write_text("time,rain_mm\n" + "".join(f"{t},{d}\n" for t, d in rows))
    return path


def test_the_steps_carry_exactly_the_series_total_at_any_step(tmp_path):
    rows = [("2022-09-28T00:00:00Z", 0.0), ("2022-09-28T01:00:00Z", 4.2), ("2022-09-28T02:00:00Z", 11.5),
            ("2022-09-28T03:00:00Z", 0.7)]
    t0, start, depth = rainseries.read(write(tmp_path / "p.csv", rows))
    assert t0 == "2022-09-28T00:00:00+00:00" and list(start) == [0.0, 3600.0, 7200.0, 10800.0]
    for dt in (20.0, 60.0, 700.0, 3600.0):
        r = rainseries.rates(start, depth, dt)
        assert r.sum() * dt * 1000.0 == pytest.approx(16.4, rel=1e-12), dt
        assert np.all(r >= 0.0)
    hourly = rainseries.rates(start, depth, 3600.0)
    np.testing.assert_allclose(hourly * 3.6e6, [0.0, 4.2, 11.5, 0.7], rtol=1e-12)


def test_half_hourly_rows_and_offsets_are_read_as_given(tmp_path):
    rows = [("2022-09-27T20:00:00-04:00", 1.0), ("2022-09-27T20:30:00-04:00", 3.0)]
    t0, start, depth = rainseries.read(write(tmp_path / "p.csv", rows))
    assert t0 == "2022-09-27T20:00:00-04:00" and list(start) == [0.0, 1800.0]
    r = rainseries.rates(start, depth, 600.0)
    assert len(r) == 6 and r.sum() * 600.0 * 1000.0 == pytest.approx(4.0)
    np.testing.assert_allclose(r[:3] * 1.8e6, 1.0)
    np.testing.assert_allclose(r[3:] * 1.8e6, 3.0)


def test_a_run_forced_by_a_series_closes_its_volume_against_the_series_total(tmp_path):
    rows = [(f"2022-09-28T{h:02d}:00:00Z", d) for h, d in enumerate((2.0, 18.0, 35.0, 6.0))]
    _, start, depth = rainseries.read(write(tmp_path / "p.csv", rows))
    rows_n, dx = 30, 5.0
    z = np.repeat(20.0 - np.arange(rows_n)[:, None] * dx * 0.01, rows_n, axis=1).astype(np.float32)
    rain = rainseries.rates(start, depth, 60.0)
    res = simulate(Surface(z=z, f0=10.0 / 3.6e6, fc=5.0 / 3.6e6, k=1.0 / 3600.0), rain,
                   SolverConfig(dx=dx, dt_s=60.0, dtype="float64", device="cpu"), verbose=False)
    m = res.mass
    assert m.rain == pytest.approx(depth.sum() / 1000.0 * z.size * dx * dx, rel=1e-12)
    assert abs(m.supplied - (m.infiltrated + m.abstracted + m.stored + m.outflow)) / m.supplied < 1e-6
