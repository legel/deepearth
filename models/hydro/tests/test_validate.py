"""Gauge scoring: the two headline numbers, and the guards that stop them being overstated.

Both results this project reports come out of this module, and neither had a test. The rising
limb is a timing claim that is only meaningful above the gauge's own 15-minute interval, and the
runoff coefficient is a RANGE over baseflow and window choices -- a single unreproducible 19.6 %
figure propagated through earlier write-ups for months precisely because nothing pinned the
shape of the answer.
"""

import numpy as np
import pytest

from validate import GAUGE_DT_H, Score, rising_limb_50


def triangular(peak_h=10.0, peak_q=100.0, end_h=30.0, n=601):
    """A single-peaked hydrograph rising from zero and receding to zero."""
    t = np.linspace(0.0, end_h, n)
    q = np.where(t <= peak_h, peak_q * t / peak_h,
                 peak_q * np.maximum(0.0, (end_h - t) / (end_h - peak_h)))
    return t, q


def test_rising_limb_is_the_half_peak_crossing():
    t, q = triangular(peak_h=10.0, peak_q=100.0)
    assert rising_limb_50(t, q) == pytest.approx(5.0, abs=0.05)


def test_rising_limb_interpolates_between_samples():
    """Coarse sampling must not quantise the answer onto a sample time."""
    t = np.array([0.0, 4.0, 8.0])
    q = np.array([0.0, 40.0, 100.0])  # half peak = 50, between the last two samples
    assert rising_limb_50(t, q) == pytest.approx(4.0 + (50 - 40) / (100 - 40) * 4.0)


def test_rising_limb_at_the_first_sample_returns_that_sample():
    """A series already above half peak at t0 has no crossing to interpolate."""
    t = np.array([0.0, 1.0, 2.0])
    q = np.array([90.0, 95.0, 100.0])
    assert rising_limb_50(t, q) == pytest.approx(0.0)


def test_rising_limb_on_a_flat_plateau_does_not_divide_by_zero():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.array([0.0, 50.0, 50.0, 50.0])
    assert np.isfinite(rising_limb_50(t, q))


def test_rising_limb_of_a_dead_series_is_nan():
    """No flow means no rising limb. Returning t0 here would score it as an instant response."""
    assert np.isnan(rising_limb_50(np.array([0.0, 1.0]), np.array([0.0, 0.0])))
    assert np.isnan(rising_limb_50(np.array([0.0, 1.0, 2.0]), np.array([-1.0, -2.0, -3.0])))


def test_an_unmeasurable_limb_is_reported_as_such_not_as_agreement():
    """NaN must not fall through to the 'within one sample' verdict, which reads as success."""
    s = _score(float("nan"), 31.60)
    assert not s.rising_limb_resolved
    assert "NOT MEASURABLE" in s.report()


def _score(sim_h, obs_h, runoff_obs=None):
    return Score(
        rising_limb_sim_h=sim_h, rising_limb_obs_h=obs_h, runoff_sim=0.7135,
        runoff_obs=runoff_obs or {"sim window 0-72 h, baseflow 0 cfs": 0.314,
                                  "sim window 0-72 h, baseflow 45 cfs": 0.289,
                                  "full gauge record, baseflow 0 cfs": 0.601,
                                  "0-48 h, baseflow 45 cfs": 0.196},
        rain_mm=391.7, domain_km2=46.6, window_h=(0.0, 72.0))


def test_a_difference_under_the_gauge_interval_is_not_an_accuracy_claim():
    """0.09 h was reported as a result once. It sits below what the instrument can see."""
    s = _score(31.69, 31.60)
    assert s.rising_limb_error_h < GAUGE_DT_H
    assert not s.rising_limb_resolved
    assert "BELOW the gauge's own" in s.report()


def test_a_difference_above_the_gauge_interval_is_reported_in_units_of_it():
    s = _score(31.88, 31.60)
    assert s.rising_limb_error_h == pytest.approx(0.28)
    assert s.rising_limb_resolved
    assert "1.1x the gauge sampling interval" in s.report()


def test_runoff_range_spans_only_like_for_like_windows():
    """The range must not quietly absorb the full-record or 0-48 h variants."""
    s = _score(31.88, 31.60)
    lo, hi = s.runoff_obs_range
    assert (lo, hi) == pytest.approx((0.289, 0.314))
    assert 0.601 not in (lo, hi) and 0.196 not in (lo, hi)


def test_report_states_a_ratio_range_never_a_single_number():
    s = _score(31.88, 31.60)
    text = s.report()
    assert "2.3x-2.5x observed" in text
    assert "28.9-31.4 %" in text
    for label in s.runoff_obs:
        assert label in text, f"{label} dropped from the report"


def test_the_reproduced_run_backs_the_readme_table():
    """The result table quotes a run whose every input `cli.py fetch` regenerates.

    It previously quoted a run built partly on artifacts symlinked in from another checkout, so
    no clone could have reproduced it and nothing said so.
    """
    import json
    from pathlib import Path

    r = json.loads((Path(__file__).resolve().parents[1]
                    / "docs" / "reproduction_site3_ian_25m.json").read_text())
    assert abs(r["mass_residual_pct"]) < 0.01, "mass balance must still close"
    assert r["rising_limb_error_h"] == pytest.approx(
        abs(r["rising_limb_sim_h"] - r["rising_limb_obs_h"]))
    assert r["rising_limb_error_h"] <= 2 * r["gauge_dt_h"], "timing claim is within a sample or two"
    lo, hi = r["runoff_overshoot_ratio"]
    assert 2.0 < lo < hi < 3.0, "the magnitude gap is stated as 2.3-2.5x; it is open, not closed"
    assert r["runoff_sim"] > max(r["runoff_obs_like_for_like"]), "overshoot, by definition"
