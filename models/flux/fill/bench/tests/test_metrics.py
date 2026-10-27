"""Each score against a value known in closed form or by construction."""

import numpy as np

from metrics import (block_sums, coverage_samples, crps_gaussian, crps_laplace, crps_samples, crps_student_t,
                     gap_class, gap_runs, pit_samples, point_scores)

RNG = np.random.default_rng(7)


def test_gaussian_crps_at_the_mean_is_2phi0_minus_1_over_sqrt_pi():
    assert np.isclose(crps_gaussian(0.0, 1.0, 0.0), 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    assert np.isclose(crps_gaussian(3.0, 2.0, 3.0), 2 * crps_gaussian(0.0, 1.0, 0.0))


def test_sample_crps_converges_to_each_closed_form():
    y = np.array([-1.3, 0.0, 0.4, 2.5])
    n = 200_000
    g = RNG.normal(0.5, 1.5, size=(n, 1)) * np.ones((1, 4))
    assert np.allclose(crps_samples(g, y), crps_gaussian(0.5, 1.5, y), atol=5e-3)
    lap = RNG.laplace(0.5, 0.8, size=(n, 1)) * np.ones((1, 4))
    assert np.allclose(crps_samples(lap, y), crps_laplace(0.5, 0.8, y), atol=5e-3)
    t = (0.5 + 0.9 * RNG.standard_t(4.0, size=(n, 1))) * np.ones((1, 4))
    assert np.allclose(crps_samples(t, y), crps_student_t(0.5, 0.9, 4.0, y), atol=8e-3)


def test_fair_crps_is_unbiased_for_small_ensembles():
    y = np.zeros(1)
    small = np.array([crps_samples(RNG.normal(size=(4, 1)), y)[0] for _ in range(40_000)]).mean()
    assert abs(small - crps_gaussian(0.0, 1.0, 0.0)) < 4e-3


def test_a_point_mass_ensemble_scores_mae():
    s = np.full((10, 3), 2.0)
    assert np.allclose(crps_samples(s, np.array([2.0, 3.0, -1.0])), [0.0, 1.0, 3.0])


def test_coverage_and_pit_of_a_calibrated_ensemble():
    s = RNG.normal(size=(500, 20_000))
    y = RNG.normal(size=20_000)
    cov = coverage_samples(s, y)
    assert all(abs(cov[lv] - lv) < 0.02 for lv in cov)
    hist = np.histogram(pit_samples(s, y), bins=10, range=(0, 1))[0] / y.size
    assert np.all(np.abs(hist - 0.1) < 0.01)


def test_gap_runs_and_classes():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert gap_runs(m).tolist() == [0, 2, 2, 0, 1, 0, 0, 3, 3, 3]
    assert gap_class(np.array([0, 1, 8, 9, 64, 65, 576, 577, 10_000])).tolist() == [-1, 0, 1, 2, 2, 3, 3, 4, 4]


def test_block_sums_count_only_scored_steps():
    y = np.arange(8, dtype=float)
    scored = np.array([1, 1, 0, 0, 0, 0, 1, 0], bool)
    s, t = block_sums(np.stack([y, y + 1]), y, scored, block=4)
    assert t.tolist() == [1.0, 6.0] and s.tolist() == [[1.0, 6.0], [3.0, 7.0]]


def test_point_scores_of_a_perfect_prediction():
    y = RNG.normal(size=100)
    p = point_scores(y.copy(), y)
    assert p["mae"] == 0 and p["r2"] == 1 and np.isclose(p["pearson_r2"], 1) and p["bias"] == 0
