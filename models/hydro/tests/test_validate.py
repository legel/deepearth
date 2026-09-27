"""Gauge scoring: peak discharge, NSE and KGE, and the receipt that records them."""

import json
from pathlib import Path

import numpy as np
import pytest

from validate import Score, align, kge, nse, peak

ROOT = Path(__file__).resolve().parents[1]


def triangular(peak_h=10.0, peak_q=100.0, end_h=30.0, n=601):
    t = np.linspace(0.0, end_h, n)
    q = np.where(t <= peak_h, peak_q * t / peak_h,
                 peak_q * np.maximum(0.0, (end_h - t) / (end_h - peak_h)))
    return t, q


def test_nse_is_one_for_a_perfect_match_and_zero_for_the_mean():
    _, q = triangular()
    assert nse(q, q) == pytest.approx(1.0)
    assert nse(q, np.full_like(q, q.mean())) == pytest.approx(0.0)
    assert nse(q, 2.0 * q) < 0.0


def test_kge_is_one_for_a_perfect_match_and_reports_its_components():
    _, q = triangular()
    k, r, alpha, beta = kge(q, q)
    assert (k, r, alpha, beta) == pytest.approx((1.0, 1.0, 1.0, 1.0))
    k2, r2, alpha2, beta2 = kge(q, 2.0 * q)
    assert (r2, alpha2, beta2) == pytest.approx((1.0, 2.0, 2.0))
    assert k2 == pytest.approx(1.0 - np.sqrt(2.0))


def test_peak_returns_the_maximum_and_its_time():
    t, q = triangular(peak_h=10.0, peak_q=100.0)
    assert peak(t, q) == pytest.approx((100.0, 10.0))


def test_align_interpolates_the_simulation_onto_gauge_samples_inside_the_window():
    t_sim = np.array([0.0, 2.0, 4.0])
    q_sim = np.array([0.0, 20.0, 40.0])
    t_obs = np.array([-1.0, 1.0, 3.0, 5.0])
    q_obs = np.array([9.0, 10.0, 30.0, 50.0])
    t, sim, obs = align(t_sim, q_sim, t_obs, q_obs)
    assert t.tolist() == [1.0, 3.0] and sim.tolist() == [10.0, 30.0] and obs.tolist() == [10.0, 30.0]


def test_align_refuses_a_window_with_fewer_than_two_samples():
    with pytest.raises(AssertionError, match="fewer than two"):
        align(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([5.0]), np.array([1.0]))


def test_score_derived_fields_and_dict():
    s = Score(peak_sim_cms=50.0, peak_obs_cms=20.0, peak_time_sim_h=34.0, peak_time_obs_h=33.5,
              nse=0.4, kge=0.3, kge_r=0.9, kge_alpha=2.0, kge_beta=2.2, n_samples=289,
              window_h=(0.0, 72.0))
    assert s.peak_ratio == pytest.approx(2.5) and s.peak_lag_h == pytest.approx(0.5)
    d = s.as_dict()
    assert d["peak_ratio"] == pytest.approx(2.5) and d["gauge_dt_h"] == 0.25
    assert "NSE" in s.report() and "ratio 2.50" in s.report()


def test_the_validation_receipt_backs_the_readme_table():
    """The result table quotes numbers a machine wrote, from the shipped 25 m hydrograph."""
    r = json.loads((ROOT / "docs" / "validation_site3_ian_25m.json").read_text())
    s = r["score"]
    assert s["peak_ratio"] == pytest.approx(s["peak_sim_cms"] / s["peak_obs_cms"])
    assert s["peak_lag_h"] == pytest.approx(s["peak_time_sim_h"] - s["peak_time_obs_h"])
    assert s["n_samples"] > 100 and np.isfinite(s["nse"]) and np.isfinite(s["kge"])
    assert s["kge"] <= 1.0 and s["nse"] <= 1.0
    assert abs(r["mass_residual_pct"]) < 0.01
