"""Tests for the leakage-resistant LFMC science gate."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from autoresearch.probes.spacetime.editable_files.lib.science_gate import (
    ScienceGateError,
    assert_split_integrity,
    audit_rolling_origin_pairs,
    audit_split,
    build_rolling_origin_pairs,
    collapse_lfmc_visits,
    evaluate_rolling_pair_baselines,
    fit_baselines,
    fit_rolling_pair_baselines,
    make_split_masks,
    prepare_lfmc_frame,
    regression_metrics,
    run_gate,
)


def synthetic_lfmc() -> pd.DataFrame:
    rows = []
    sorting_id = 0
    spatial_sites = {
        "train": ("site-a", 35.0, -120.0),
        "validation": ("site-b", 36.0, -119.0),
        "test": ("site-c", 37.0, -118.0),
    }
    years = (2019, 2020, 2021, 2022)
    random_labels = ("train", "validation", "test", "train")
    for spatial_label, (site, latitude, longitude) in spatial_sites.items():
        for year, random_label in zip(years, random_labels):
            rows.append(
                {
                    "sorting_id": sorting_id,
                    "site_name": site,
                    "latitude": latitude,
                    "longitude": longitude,
                    "sampling_date": "{}-06-15".format(year),
                    "lfmc_value": 80.0 + sorting_id,
                    "species_collected": "species-{}".format(sorting_id % 2),
                    "elevation": 100.0 + sorting_id,
                    "random_split": random_label,
                    "spatial_split": spatial_label,
                }
            )
            sorting_id += 1
    return prepare_lfmc_frame(pd.DataFrame(rows))


def synthetic_rolling_lfmc() -> pd.DataFrame:
    """Four visit dates in each chronological/spatial target partition."""

    rows = []
    sorting_id = 0
    series = (
        ("train", "site-a", 35.0, -120.0, pd.Timestamp("2020-01-01")),
        (
            "validation",
            "site-b",
            36.0,
            -119.0,
            pd.Timestamp("2021-01-01"),
        ),
        ("test", "site-c", 37.0, -118.0, pd.Timestamp("2022-01-01")),
    )
    for spatial_split, site, latitude, longitude, anchor in series:
        for offset in (0, 30, 90, 180):
            replicate_values = (80.0, 100.0) if offset == 0 else (90.0 + offset,)
            for value in replicate_values:
                rows.append(
                    {
                        "sorting_id": sorting_id,
                        "site_name": site,
                        "latitude": latitude,
                        "longitude": longitude,
                        "sampling_date": anchor + pd.Timedelta(days=offset),
                        "lfmc_value": value,
                        "species_collected": "species-a",
                        "elevation": 100.0,
                        "random_split": "train",
                        "spatial_split": spatial_split,
                    }
                )
                sorting_id += 1
    return prepare_lfmc_frame(pd.DataFrame(rows))


class PrepareLFMCFrameTest(unittest.TestCase):
    def test_exact_benchmark_filter(self) -> None:
        valid = synthetic_lfmc()
        base = valid.iloc[0].drop(labels=["_source_row", "_month"]).to_dict()
        additions = []
        for value, elevation in ((-1.0, 100.0), (303.0, 100.0), (100.0, np.nan)):
            row = dict(base)
            row["sorting_id"] = 100 + len(additions)
            row["lfmc_value"] = value
            row["elevation"] = elevation
            additions.append(row)
        prepared = prepare_lfmc_frame(
            pd.concat(
                [
                    valid.drop(columns=["_source_row", "_month"]),
                    pd.DataFrame(additions),
                ],
                ignore_index=True,
            )
        )
        self.assertEqual(len(prepared), len(valid))
        self.assertTrue(prepared["lfmc_value"].between(0, 302).all())


class SplitIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic_lfmc()

    def test_temporal_split_is_strictly_chronological(self) -> None:
        masks = make_split_masks(self.frame, "temporal")
        audit = assert_split_integrity(self.frame, masks, "temporal")
        self.assertTrue(audit["row_disjoint"])
        self.assertTrue(audit["chronological"])
        self.assertLess(
            self.frame.loc[masks["train"], "sampling_date"].max(),
            self.frame.loc[masks["validation"], "sampling_date"].min(),
        )
        self.assertLess(
            self.frame.loc[masks["validation"], "sampling_date"].max(),
            self.frame.loc[masks["test"], "sampling_date"].min(),
        )

    def test_temporal_integrity_rejects_late_validation_row(self) -> None:
        masks = make_split_masks(self.frame, "temporal")
        bad_masks = {name: mask.copy() for name, mask in masks.items()}
        late_test_row = self.frame.index[masks["test"]][0]
        bad_masks["test"].loc[late_test_row] = False
        bad_masks["validation"].loc[late_test_row] = True
        with self.assertRaises(ScienceGateError):
            assert_split_integrity(self.frame, bad_masks, "temporal")

    def test_spatial_split_has_no_site_overlap(self) -> None:
        masks = make_split_masks(self.frame, "spatial")
        audit = assert_split_integrity(self.frame, masks, "spatial")
        self.assertTrue(audit["site_disjoint"])
        self.assertTrue(audit["coordinate_disjoint"])
        self.assertTrue(all(value == 0 for value in audit["site_overlaps"].values()))

    def test_spatiotemporal_split_has_both_guarantees(self) -> None:
        masks = make_split_masks(self.frame, "spatiotemporal")
        audit = assert_split_integrity(self.frame, masks, "spatiotemporal")
        self.assertTrue(audit["chronological"])
        self.assertTrue(audit["site_disjoint"])
        self.assertTrue(audit["coordinate_disjoint"])
        self.assertGreater(audit["excluded_rows"], 0)

    def test_random_audit_exposes_site_overlap(self) -> None:
        masks = make_split_masks(self.frame, "random")
        audit = audit_split(self.frame, masks, "random")
        self.assertGreater(audit["site_overlaps"]["train__test"], 0)

    def test_spatial_integrity_rejects_reused_site(self) -> None:
        bad = self.frame.copy()
        test_row = bad.index[bad["spatial_split"].eq("test")][0]
        bad.loc[test_row, "site_name"] = "site-a"
        masks = make_split_masks(bad, "spatial")
        with self.assertRaises(ScienceGateError):
            assert_split_integrity(bad, masks, "spatial")

    def test_spatial_integrity_rejects_renamed_duplicate_coordinate(self) -> None:
        bad = self.frame.copy()
        train_row = bad.index[bad["spatial_split"].eq("train")][0]
        test_row = bad.index[bad["spatial_split"].eq("test")][0]
        bad.loc[test_row, ["latitude", "longitude"]] = bad.loc[
            train_row, ["latitude", "longitude"]
        ].to_numpy()
        masks = make_split_masks(bad, "spatial")
        with self.assertRaises(ScienceGateError):
            assert_split_integrity(bad, masks, "spatial")


class TrainOnlyBaselineTest(unittest.TestCase):
    def test_predictions_ignore_validation_and_test_labels(self) -> None:
        frame = synthetic_lfmc()
        masks = make_split_masks(frame, "temporal")

        original_suite = fit_baselines(frame.loc[masks["train"]])
        original_validation = original_suite.predict(frame.loc[masks["validation"]])
        original_test = original_suite.predict(frame.loc[masks["test"]])

        changed = frame.copy()
        held_out = masks["validation"] | masks["test"]
        changed.loc[held_out, "lfmc_value"] += 10_000.0
        changed_suite = fit_baselines(changed.loc[masks["train"]])
        changed_validation = changed_suite.predict(changed.loc[masks["validation"]])
        changed_test = changed_suite.predict(changed.loc[masks["test"]])

        for name in original_validation:
            np.testing.assert_array_equal(
                original_validation[name], changed_validation[name]
            )
            np.testing.assert_array_equal(original_test[name], changed_test[name])

    def test_temporal_persistence_origin_precedes_test(self) -> None:
        frame = synthetic_lfmc()
        masks = make_split_masks(frame, "temporal")
        suite = fit_baselines(frame.loc[masks["train"]])
        self.assertLess(
            suite.forecast_origin,
            frame.loc[masks["test"], "sampling_date"].min(),
        )
        self.assertTrue((suite.persistence_date <= suite.forecast_origin).all())

    def test_metrics_drop_nonfinite_pairs(self) -> None:
        metrics = regression_metrics(
            [1.0, np.nan, 3.0, 4.0],
            [1.5, 2.0, np.inf, 3.5],
        )
        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], 0.5)


class RollingOriginAuditTest(unittest.TestCase):
    def test_visit_collapse_preserves_mean_count_and_dispersion(self) -> None:
        visits = collapse_lfmc_visits(synthetic_rolling_lfmc())
        first = visits.loc[
            visits["site_name"].eq("site-a")
            & visits["sampling_date"].eq(pd.Timestamp("2020-01-01"))
        ].iloc[0]
        self.assertEqual(first["replicate_count"], 2)
        self.assertEqual(first["lfmc_value"], 90.0)
        self.assertEqual(first["lfmc_replicate_range"], 20.0)
        self.assertAlmostEqual(
            first["lfmc_replicate_std"], np.sqrt(200.0)
        )
        self.assertEqual(visits.attrs["source_rows"], 15)
        self.assertEqual(visits.attrs["unique_visits"], 12)

    def test_pair_selection_is_same_series_strict_and_tie_more_recent(self) -> None:
        target_date = pd.Timestamp("2020-06-30")
        visits = pd.DataFrame(
            {
                "_visit_id": [0, 1, 2, 3],
                "site_name": ["site-a", "site-a", "site-a", "site-b"],
                "species_collected": ["sp", "sp", "sp", "sp"],
                "sampling_date": [
                    target_date - pd.Timedelta(days=93),
                    target_date - pd.Timedelta(days=87),
                    target_date,
                    target_date - pd.Timedelta(days=90),
                ],
            }
        )
        pairs = build_rolling_origin_pairs(visits, 90, tolerance_days=7)
        selected = pairs.loc[pairs["target_visit_id"].eq(2)].iloc[0]
        self.assertEqual(selected["origin_visit_id"], 1)
        self.assertEqual(selected["lag_days"], 87)
        self.assertEqual(selected["site_name"], "site-a")
        self.assertTrue((pairs["origin_date"] < pairs["target_date"]).all())

    def test_spatiotemporal_counts_and_dev_lock(self) -> None:
        visits = collapse_lfmc_visits(synthetic_rolling_lfmc())
        audit = audit_rolling_origin_pairs(
            visits, "spatiotemporal", open_test=False
        )
        expected_pairs = {"30": 1, "90": 2, "180": 1}
        for horizon, count in expected_pairs.items():
            record = audit["horizons"][horizon]
            for partition in ("train", "validation", "test"):
                self.assertEqual(
                    record["partitions"][partition]["pairs"], count
                )
                self.assertEqual(
                    record["partitions"][partition]["target_visits"], 4
                )
            self.assertEqual(
                set(record["causal_baselines"]["metrics"]), {"validation"}
            )
            self.assertNotIn(
                "test", record["causal_baselines"]["metrics"]
            )
        test_replicates = audit["visit_collapse"]["partitions"]["test"]
        self.assertFalse(test_replicates["lfmc_dispersion_reported"])
        self.assertNotIn("lfmc_replicate_dispersion", test_replicates)
        self.assertEqual(
            audit["evaluation_semantics"], "rolling new-site-with-history"
        )
        self.assertFalse(audit["earth4d_evaluated"])

    def test_pair_partition_is_determined_only_by_target(self) -> None:
        frame = synthetic_rolling_lfmc()
        extra = frame.iloc[0].drop(labels=["_source_row", "_month"]).to_dict()
        origin = dict(extra)
        origin.update(
            {
                "sorting_id": 1000,
                "site_name": "site-d",
                "latitude": 38.0,
                "longitude": -117.0,
                "sampling_date": pd.Timestamp("2021-12-15"),
                "spatial_split": "test",
            }
        )
        target = dict(origin)
        target["sorting_id"] = 1001
        target["sampling_date"] = pd.Timestamp("2022-01-14")
        combined = prepare_lfmc_frame(
            pd.concat(
                [
                    frame.drop(columns=["_source_row", "_month"]),
                    pd.DataFrame([origin, target]),
                ],
                ignore_index=True,
            )
        )
        visits = collapse_lfmc_visits(combined)
        masks = make_split_masks(visits, "spatiotemporal")
        pairs = build_rolling_origin_pairs(visits, 30)
        selected = pairs.loc[pairs["site_name"].eq("site-d")].iloc[0]
        origin_row = visits["_visit_id"].eq(selected["origin_visit_id"])
        target_row = visits["_visit_id"].eq(selected["target_visit_id"])
        self.assertFalse(
            any(bool((masks[name] & origin_row).any()) for name in masks)
        )
        self.assertTrue(bool((masks["test"] & target_row).any()))

    def test_held_out_target_mutation_cannot_change_fitted_predictions(self) -> None:
        visits = collapse_lfmc_visits(synthetic_rolling_lfmc())
        masks = make_split_masks(visits, "spatiotemporal")
        pairs = build_rolling_origin_pairs(visits, 30)
        train_ids = visits.loc[masks["train"], "_visit_id"]
        validation_ids = set(visits.loc[masks["validation"], "_visit_id"])
        validation_pairs = pairs.loc[
            pairs["target_visit_id"].isin(validation_ids)
        ]

        original_suite = fit_rolling_pair_baselines(
            visits, pairs, train_ids
        )
        original_origin = visits.set_index("_visit_id")["lfmc_value"].reindex(
            validation_pairs["origin_visit_id"]
        )
        original_predictions = original_suite.predict(original_origin)
        original_metrics = evaluate_rolling_pair_baselines(
            original_suite, visits, validation_pairs
        )

        changed = visits.copy()
        changed.loc[
            changed["_visit_id"].isin(validation_pairs["target_visit_id"]),
            "lfmc_value",
        ] += 50.0
        changed_suite = fit_rolling_pair_baselines(changed, pairs, train_ids)
        changed_origin = changed.set_index("_visit_id")["lfmc_value"].reindex(
            validation_pairs["origin_visit_id"]
        )
        changed_predictions = changed_suite.predict(changed_origin)
        changed_metrics = evaluate_rolling_pair_baselines(
            changed_suite, changed, validation_pairs
        )

        self.assertEqual(original_suite.mean_change, changed_suite.mean_change)
        for name in original_predictions:
            np.testing.assert_array_equal(
                original_predictions[name], changed_predictions[name]
            )
        self.assertNotEqual(
            original_metrics["observed_state_persistence"]["mae"],
            changed_metrics["observed_state_persistence"]["mae"],
        )


class GateArtifactTest(unittest.TestCase):
    def test_full_gate_is_json_serializable_and_not_an_earth4d_result(self) -> None:
        raw = synthetic_lfmc().drop(columns=["_source_row", "_month"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lfmc.csv"
            raw.to_csv(path, index=False)
            result = run_gate(path, designs=("spatiotemporal",))
        json.dumps(result)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["evidence_level"], "data-and-baseline audit")
        self.assertFalse(result["earth4d_evaluated"])
        self.assertFalse(result["dataset"]["official_hash_verified"])
        self.assertIsNone(result["dataset"]["official_url"])
        self.assertFalse(result["policy"]["test_opened"])
        self.assertNotIn(
            "test", result["designs"]["spatiotemporal"]["baselines"]
        )
        rolling = result["designs"]["spatiotemporal"]["rolling_origin"]
        for record in rolling["horizons"].values():
            self.assertNotIn(
                "test", record["causal_baselines"]["metrics"]
            )
            self.assertIn("test", record["partitions"])
            self.assertEqual(
                len(
                    record["partitions"]["test"][
                        "pair_membership_sha256"
                    ]
                ),
                64,
            )
        self.assertNotIn(
            "lfmc_replicate_dispersion",
            rolling["visit_collapse"]["partitions"]["test"],
        )
        self.assertTrue(
            result["designs"]["spatiotemporal"]["audit"]["chronological"]
        )

    def test_open_test_requires_explicit_frozen_identifier(self) -> None:
        raw = synthetic_lfmc().drop(columns=["_source_row", "_month"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lfmc.csv"
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ScienceGateError, "frozen_id"):
                run_gate(path, designs=("temporal",), open_test=True)
            result = run_gate(
                path,
                designs=("temporal",),
                open_test=True,
                frozen_id="unit-test-frozen-run",
            )
        self.assertTrue(result["policy"]["test_opened"])
        self.assertEqual(
            result["policy"]["frozen_id"], "unit-test-frozen-run"
        )
        self.assertIn("test", result["designs"]["temporal"]["baselines"])
        for record in result["designs"]["temporal"][
            "rolling_origin"
        ]["horizons"].values():
            self.assertIn("test", record["causal_baselines"]["metrics"])
        self.assertIn(
            "train",
            result["designs"]["temporal"]["audit"]["membership_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
