"""Contract tests: the probe declares its identity, the harness stops guessing.

Each test here corresponds to a real mis-record the old stdout-scraping interface produced.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_PARENT = str(Path(__file__).resolve().parents[2])
if REPO_PARENT not in sys.path:
    sys.path.insert(0, REPO_PARENT)

from autoresearch.spacetime.editable_files.harness import (
    CONTRACT_VERSION,
    ContractError,
    Primary,
    ProbeResult,
)


def result(**overrides):
    base = dict(
        capability="family_from_spacetime",
        mode="FORECAST(past->future)",
        primary=Primary("family_top1_accuracy", 0.1769),
        protocol="v2-leakfix",
        split="spatiotemporal-block",
        n_shards=12,
        seed=0,
        gains={"vs RFF": 0.0772, "vs raw": 0.0424},
        baselines={"RFF": 0.0997, "raw": 0.1345},
    )
    base.update(overrides)
    return ProbeResult(**base)


class ValidationTests(unittest.TestCase):
    def test_a_mode_that_will_not_name_itself_cannot_record(self):
        """Four probe paths printed no mode= at all, so they all read mode=None -- and None == None
        made them mutually 'like-for-like', defeating the cross-target gate."""
        with self.assertRaisesRegex(ContractError, "mode is empty"):
            result(mode="").validate()

    def test_score_without_units_is_refused(self):
        with self.assertRaisesRegex(ContractError, "primary.name is empty"):
            result(primary=Primary("", 0.5)).validate()

    def test_nan_primary_and_nan_gain_are_refused(self):
        with self.assertRaisesRegex(ContractError, "NaN"):
            result(primary=Primary("acc", float("nan"))).validate()
        with self.assertRaisesRegex(ContractError, "NaN"):
            result(gains={"vs RFF": float("nan")}).validate()

    def test_valid_result_passes_and_returns_itself(self):
        r = result()
        self.assertIs(r.validate(), r)


class IdentityTests(unittest.TestCase):
    def test_identity_covers_the_full_comparability_tuple(self):
        self.assertEqual(
            set(result().identity()),
            {"capability", "mode", "split", "n_shards", "protocol", "metric"},
        )

    def test_different_mode_is_never_comparable(self):
        """--pheno_disttarget peak_week took flowering_peak_month 0.067 -> 0.683 by measuring a
        different target that printed a matching 'acc'."""
        stored = result().identity()
        self.assertFalse(result(mode="PHENOLOGY-HELD").comparable_to(stored))

    def test_different_metric_is_never_comparable(self):
        """calibration maxed over four uncertainty signals, so a run on a different signal could beat a
        record set on max-softmax."""
        stored = result().identity()
        self.assertFalse(
            result(primary=Primary("conf_auroc_bald", 0.9)).comparable_to(stored))

    def test_different_shard_count_is_not_comparable(self):
        stored = result().identity()
        self.assertFalse(result(n_shards=8).comparable_to(stored))

    def test_missing_field_on_an_older_record_is_skipped_not_asserted(self):
        legacy = {"mode": "FORECAST(past->future)", "metric": "family_top1_accuracy"}
        self.assertTrue(result().comparable_to(legacy))

    def test_identity_digest_is_stable_and_mode_sensitive(self):
        self.assertEqual(result().identity_digest(), result().identity_digest())
        self.assertNotEqual(result().identity_digest(), result(mode="ENV").identity_digest())


class FairGainTests(unittest.TestCase):
    def test_strongest_fair_baseline_wins_not_the_flattering_one(self):
        order = ["best-ctrl", "RFF", "mlp", "GAIN", "best-coord", "raw"]
        value, label = result().fair_gain(order)
        self.assertEqual(label, "vs RFF")
        self.assertEqual(value, 0.0772)

    def test_absent_fair_baseline_is_none_not_zero(self):
        """calibration's live record carries no fair baseline at all. That must read as 'undiagnosable',
        never as a gain of zero."""
        self.assertEqual(result(gains={}).fair_gain(["RFF"]), (None, None))


class RoundTripTests(unittest.TestCase):
    def test_write_read_round_trip_preserves_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            result().write(path)
            back = ProbeResult.read(path)
            self.assertEqual(back.primary.value, 0.1769)
            self.assertEqual(back.gains["vs RFF"], 0.0772)
            self.assertEqual(back.identity(), result().identity())

    def test_foreign_contract_version_is_refused(self):
        """A second agent tree on this box runs its own harness with its own event schema. A result
        written by a different contract must not be silently reinterpreted."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            payload = result().to_dict()
            payload["contract_version"] = CONTRACT_VERSION + 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ContractError, "different contract"):
                ProbeResult.read(path)

    def test_corrupt_result_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text("{not json")
            with self.assertRaisesRegex(ContractError, "not valid JSON"):
                ProbeResult.read(path)


class RenderTests(unittest.TestCase):
    def test_render_states_identity_and_protocol(self):
        text = result().render()
        self.assertIn("capability=family_from_spacetime", text)
        self.assertIn("mode=FORECAST(past->future)", text)
        self.assertIn("protocol=v2-leakfix", text)
        self.assertIn("family_top1_accuracy = 0.176900", text)
        self.assertIn("encoder=frozen-random", text)

    def test_render_distinguishes_a_trained_encoder(self):
        """Only the trained protocol can support a claim about learned hash state; the default probe
        reads a frozen RANDOM table."""
        self.assertIn("encoder=trained", result(trained_encoder=True).render())



class DiagnosticTests(unittest.TestCase):
    """Six of the 19 probe modes measure targets that are not scorecard capabilities, and four of
    those run on raw PE only -- Earth4D is not in the comparison. They must not be forced into a
    capability slot just to fit the board."""

    def test_diagnostic_may_omit_capability_but_must_give_a_reason(self):
        d = ProbeResult(
            capability="", mode="breadth probe(occupancy)",
            primary=Primary("absR2", 0.12), protocol="v2-leakfix",
            diagnostic=True, diagnostic_reason="raw PE only; occupancy is not a scorecard capability")
        self.assertIs(d.validate(), d)
        self.assertFalse(d.records())

    def test_diagnostic_without_a_reason_is_refused(self):
        with self.assertRaisesRegex(ContractError, "WHY it cannot set a record"):
            ProbeResult(capability="", mode="breadth probe", primary=Primary("absR2", 0.1),
                        protocol="v2-leakfix", diagnostic=True).validate()

    def test_a_capability_result_records(self):
        self.assertTrue(result().records())

    def test_render_marks_a_diagnostic(self):
        text = ProbeResult(
            capability="", mode="propagator-ARCH", primary=Primary("absR2", 0.1),
            protocol="v2-leakfix", diagnostic=True,
            diagnostic_reason="raw PE only").render()
        self.assertIn("DIAGNOSTIC (cannot set a record)", text)
        self.assertIn("capability=DIAGNOSTIC", text)

if __name__ == "__main__":
    unittest.main()
