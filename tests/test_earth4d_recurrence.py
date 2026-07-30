"""Focused correctness tests for the exact cKDTree causal-window builder."""
import unittest

import numpy as np

from autoresearch.spacetime.experiments.recurrence import (
    build_causal_windows_kdtree,
    normalize_time_from_train,
    strict_spatiotemporal_masks,
)


def brute_force_windows(q_lat, q_lon, q_day, p_lat, p_lon, p_day, K):
    """Reference implementation: global scan, deterministic ties, strict earlier-day filter."""
    q_lat = np.asarray(q_lat)
    q_lon = np.asarray(q_lon)
    q_day = np.asarray(q_day)
    p_lat = np.asarray(p_lat)
    p_lon = np.asarray(p_lon)
    p_day = np.asarray(p_day)
    result = np.full((len(q_lat), K), -1, dtype=np.int64)
    pool_index = np.arange(len(p_lat), dtype=np.int64)
    for query_i in range(len(q_lat)):
        candidates = pool_index[p_day < q_day[query_i]]
        d2 = ((p_lat[candidates] - q_lat[query_i]) ** 2
              + (p_lon[candidates] - q_lon[query_i]) ** 2)
        spatial_order = np.lexsort((candidates, d2))
        selected = candidates[spatial_order[:K]]
        selected_d2 = d2[spatial_order[:K]]
        temporal_order = np.lexsort((selected, selected_d2, p_day[selected]))
        result[query_i, :len(selected)] = selected[temporal_order]
    return result, result >= 0


class CausalWindowKDTreeTests(unittest.TestCase):

    def assert_matches_brute_force(self, q_lat, q_lon, q_day, p_lat, p_lon, p_day, K):
        expected_idx, expected_valid = brute_force_windows(
            q_lat, q_lon, q_day, p_lat, p_lon, p_day, K
        )
        actual_idx, actual_valid = build_causal_windows_kdtree(
            q_lat, q_lon, q_day, p_lat, p_lon, p_day, K
        )
        np.testing.assert_array_equal(actual_idx, expected_idx)
        np.testing.assert_array_equal(actual_valid, expected_valid)

    def test_matches_global_brute_force(self):
        rng = np.random.default_rng(20260729)
        p_lat = rng.uniform(-70.0, 70.0, 257)
        p_lon = rng.uniform(-170.0, 170.0, 257)
        p_day = rng.integers(0, 100, 257)
        q_lat = rng.uniform(-70.0, 70.0, 31)
        q_lon = rng.uniform(-170.0, 170.0, 31)
        q_day = rng.integers(0, 100, 31)
        self.assert_matches_brute_force(
            q_lat, q_lon, q_day, p_lat, p_lon, p_day, K=7
        )

    def test_adapts_past_more_than_eight_k_noncausal_neighbours(self):
        K = 3
        # Thirty closer observations are future/simultaneous (30 > 8*K). The old fixed-width query returned
        # an entirely padded window even though four valid past observations exist slightly farther away.
        close = np.arange(1, 31, dtype=np.float64) / 1000.0
        p_lat = np.concatenate([close, np.array([1.0, 1.2, 1.4, 1.6])])
        p_lon = np.zeros_like(p_lat)
        p_day = np.concatenate([np.full(15, 11), np.full(15, 10), np.array([5, 7, 6, 8])])
        self.assert_matches_brute_force(
            q_lat=[0.0], q_lon=[0.0], q_day=[10],
            p_lat=p_lat, p_lon=p_lon, p_day=p_day, K=K
        )
        idx, valid = build_causal_windows_kdtree(
            [0.0], [0.0], [10], p_lat, p_lon, p_day, K
        )
        np.testing.assert_array_equal(idx, [[30, 32, 31]])  # past-to-present: days 5, 6, 7
        self.assertTrue(valid.all())

    def test_padding_and_no_history(self):
        self.assert_matches_brute_force(
            q_lat=[0.0, 0.0],
            q_lon=[0.0, 0.0],
            q_day=[1, 5],
            p_lat=[0.1, 0.2],
            p_lon=[0.0, 0.0],
            p_day=[2, 3],
            K=4,
        )
        idx, valid = build_causal_windows_kdtree(
            [0.0, 0.0], [0.0, 0.0], [1, 5],
            [0.1, 0.2], [0.0, 0.0], [2, 3], 4,
        )
        np.testing.assert_array_equal(idx[0], [-1, -1, -1, -1])
        np.testing.assert_array_equal(valid[0], [False, False, False, False])
        np.testing.assert_array_equal(idx[1], [0, 1, -1, -1])

    def test_simultaneous_day_is_strictly_excluded(self):
        p_lat = np.array([0.001, 0.02, 0.03, 0.04])
        p_lon = np.zeros(4)
        p_day = np.array([10, 9, 8, 11])
        self.assert_matches_brute_force(
            q_lat=[0.0], q_lon=[0.0], q_day=[10],
            p_lat=p_lat, p_lon=p_lon, p_day=p_day, K=3,
        )
        idx, valid = build_causal_windows_kdtree(
            [0.0], [0.0], [10], p_lat, p_lon, p_day, 3
        )
        np.testing.assert_array_equal(idx, [[2, 1, -1]])
        np.testing.assert_array_equal(valid, [[True, True, False]])
        self.assertNotIn(0, idx[0])  # day == query day is not causal

    def test_rejects_nonfinite_or_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            build_causal_windows_kdtree(
                [0.0], [0.0], [np.nan], [0.1], [0.1], [1.0], 1
            )
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            build_causal_windows_kdtree(
                [0.0, 1.0], [0.0], [2.0], [0.1], [0.1], [1.0], 1
            )


class TrainTimeNormalizationTests(unittest.TestCase):

    def test_fit_uses_training_period_only(self):
        normalized, origin, span = normalize_time_from_train(
            [10.0, 20.0, 30.0, 100.0],
            [True, True, True, False],
        )
        self.assertEqual(origin, 10.0)
        self.assertEqual(span, 20.0)
        np.testing.assert_allclose(normalized[:3], [0.0, 0.5, 1.0])
        self.assertGreater(normalized[3], 1.0)

    def test_fixed_horizon_reserves_predeclared_headroom(self):
        normalized, origin, span = normalize_time_from_train(
            [10.0, 20.0, 30.0, 100.0],
            [True, True, True, False],
            horizon=2.0,
        )
        self.assertEqual(origin, 10.0)
        self.assertEqual(span, 40.0)
        np.testing.assert_allclose(normalized[:3], [0.0, 0.25, 0.5])
        self.assertEqual(float(normalized[3]), 2.25)

    def test_rejects_nonfinite_or_empty_training_period(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_time_from_train([1.0, np.nan], [True, False])
        with self.assertRaisesRegex(ValueError, "training"):
            normalize_time_from_train([1.0, 2.0], [False, False])
        with self.assertRaisesRegex(ValueError, "horizon"):
            normalize_time_from_train([1.0, 2.0], [True, False], horizon=0.5)


class StrictSpatiotemporalMaskTests(unittest.TestCase):

    def test_embargoes_cross_quadrants(self):
        train, test, embargo = strict_spatiotemporal_masks(
            lat=[0.1, 0.1, 1.1, 1.1],
            lon=[0.1, 0.1, 1.1, 1.1],
            days=[1.0, 3.0, 1.0, 3.0],
            future=[False, True, False, True],
            held_place=[False, False, True, True],
            block=1.0,
        )
        np.testing.assert_array_equal(train, [True, False, False, False])
        np.testing.assert_array_equal(test, [False, False, False, True])
        np.testing.assert_array_equal(embargo, [False, True, True, False])

    def test_rejects_row_level_place_leakage_or_nonfinite_time(self):
        with self.assertRaisesRegex(ValueError, "reuses"):
            strict_spatiotemporal_masks(
                lat=[0.1, 0.2, 1.1],
                lon=[0.1, 0.2, 1.1],
                days=[1.0, 3.0, 3.0],
                future=[False, True, True],
                held_place=[False, True, True],
                block=1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            strict_spatiotemporal_masks(
                lat=[0.1, 1.1],
                lon=[0.1, 1.1],
                days=[1.0, np.nan],
                future=[False, True],
                held_place=[False, True],
                block=1.0,
            )


if __name__ == "__main__":
    unittest.main()
