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

from autoresearch.probes.spacetime.editable_files.harness import (
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


class NoiseBarrierTests(unittest.TestCase):
    """A record must beat the standing one by more than the noise.

    Calibrated against noise actually observed in this probe: a second agent walked
    family_from_spacetime 0.1769 -> 0.19143 in seven accepted single-seed steps
    (+0.0007/+0.0008/+0.0112/+0.0005/+0.0002/+0.0006/+0.0006), and a verification run here took
    flowering_peak_month 0.0521 -> 0.052131 — a delta of +0.000031. Both were accepted because the gate
    asked only "is this float larger".
    """

    def setUp(self):
        from autoresearch.probes.spacetime.editable_files.harness import _record_gate, noise_barrier
        self.gate, self.barrier = _record_gate, noise_barrier

    def test_the_flowering_noise_record_would_now_be_refused(self):
        beats = self.gate(0.052131012, 0.0521, "v2-leakfix", "PHENOLOGY-FUTURE", "PHENOLOGY-FUTURE",
                          12, 12)[2]
        self.assertFalse(beats, "+0.000031 on one seed must not be a record")

    def test_the_overnight_walk_can_no_longer_accumulate(self):
        """Replay the seven observed steps through the gate.

        Six are refused outright. The one that passes (+0.0112, a 6.3% relative jump) is a plausible
        single-seed improvement and refusing it would mean the loop can never progress at all — but it
        lands marked provisional (n_seeds < 5), which under the evidence standard is discovery, not a
        claim. What the barrier kills is the ACCUMULATION: a refused step does not move the baseline, so
        the six noise steps can no longer stack on top of each other to manufacture a headline.
        """
        prev, accepted = 0.1769, []
        for step in (0.0007, 0.0008, 0.0112, 0.0005, 0.0002, 0.0006, 0.0006):
            beats = self.gate(prev + step, prev, "v2-leakfix", "FORECAST", "FORECAST", 12, 12)[2]
            if beats:
                accepted.append(step)
                prev = prev + step
        self.assertEqual(accepted, [0.0112], "only the 6.3% step should survive the barrier")
        self.assertLess(prev, 0.19143524765968323,
                        "the walk must not be able to reach the score it originally published")

    def test_a_real_improvement_still_passes(self):
        beats = self.gate(0.1769 * 1.10, 0.1769, "v2-leakfix", "FORECAST", "FORECAST", 12, 12)[2]
        self.assertTrue(beats, "a 10% improvement must still be recordable")

    def test_barrier_is_relative_with_an_absolute_floor(self):
        self.assertAlmostEqual(self.barrier(0.5), 0.01)        # 2% dominates
        self.assertAlmostEqual(self.barrier(0.01), 0.002)      # absolute floor dominates
        self.assertEqual(self.barrier(None), 0.0)              # nothing to beat yet

    def test_measured_spread_raises_the_barrier_above_the_floor(self):
        """With >=3 seeds the spread is real, so the run must clear its OWN noise."""
        self.assertAlmostEqual(self.barrier(0.5, seed_std=0.02, n_seeds=5), 0.04)   # 2 sigma > floor
        self.assertAlmostEqual(self.barrier(0.5, seed_std=0.001, n_seeds=5), 0.01)  # floor > 2 sigma

    def test_two_seeds_cannot_claim_a_measured_spread(self):
        """A standard deviation from two points is not a spread; fall back to the floor."""
        self.assertAlmostEqual(self.barrier(0.5, seed_std=0.02, n_seeds=2), 0.01)


class ScaleFreeDiagnosisTests(unittest.TestCase):
    """The read must mean the same thing for a 166-class and a 2,009-class target.

    It used to be `fair_gain > 0 and score < 0.20 -> ENCODER-LIMITED`, an absolute constant applied
    regardless of target difficulty. species_from_spacetime (~2,009 classes, chance ~0.0005) scoring
    0.0512 and family_from_spacetime (166 classes, chance ~0.006) scoring 0.1769 both tripped it, and
    acting on that sent four consecutive mechanism changes at a capability whose encoder was already
    contributing 84% of its score.
    """

    def setUp(self):
        from autoresearch.probes.spacetime.editable_files.harness import _bottleneck
        self.read = _bottleneck

    def test_a_hard_target_carried_by_the_encoder_reads_EARNING(self):
        """species_from_spacetime: 0.0512 with +0.0432 of it from Earth4D — 84% is not encoder-limited,
        whatever the absolute number looks like."""
        self.assertIn("EARNING", self.read(0.0432, 0.0512))
        self.assertIn("84%", self.read(0.0432, 0.0512))

    def test_an_easier_target_with_the_same_share_reads_the_same(self):
        """Scale-free: 10x the score with 10x the gain must give the identical verdict."""
        self.assertEqual(self.read(0.0432, 0.0512).split(":")[0],
                         self.read(0.432, 0.512).split(":")[0])

    def test_a_marginal_encoder_still_reads_ENCODER_LIMITED(self):
        """flowering_peak_month: +0.0087 on 0.0521 is 17% — the mechanism really is the weak part."""
        self.assertIn("ENCODER-LIMITED", self.read(0.0087, 0.0521))

    def test_no_gain_over_the_fair_baseline_reads_INPUT_LIMITED(self):
        self.assertIn("INPUT-LIMITED", self.read(-0.0072, 0.1423))

    def test_absolute_score_alone_never_decides(self):
        """A low score with a high share and a high score with a low share must NOT read the same."""
        low_score_high_share = self.read(0.09, 0.10)
        high_score_low_share = self.read(0.05, 0.90)
        self.assertIn("EARNING", low_score_high_share)
        self.assertIn("ENCODER-LIMITED", high_score_low_share)

    def test_missing_baseline_is_undiagnosable_not_zero(self):
        self.assertIn("NO-FAIR-BASELINE", self.read(None, 0.5))


class ProvenanceTests(unittest.TestCase):
    """Every published number must carry the tree that produced it and the evidence behind it.

    The evidence standard says a record from an unpushed commit is discovery-only — unenforceable while
    nothing recorded WHICH commit produced a number. Two concrete failures motivated this: a foreign
    agent's record on this board claims a `trained_rff` baseline that exists in no reachable tree, and a
    run of this loop was contaminated for an hour by an uncommitted edit to earth4d.py that nothing in
    the record would have revealed.
    """

    def setUp(self):
        from autoresearch.probes.spacetime.editable_files.harness import _code_provenance
        self.prov = _code_provenance()

    def test_provenance_reports_commit_branch_and_dirtiness(self):
        self.assertEqual(set(self.prov), {"commit", "branch", "dirty"})
        self.assertTrue(self.prov["commit"], "a run with no commit SHA cannot be reproduced")
        self.assertIsInstance(self.prov["dirty"], bool)

    def test_a_dirty_tree_is_detectable_not_silent(self):
        """A dirty tree does not block a run — it must simply be impossible to hide afterwards."""
        self.assertIn("dirty", self.prov)


class FairBaselineTests(unittest.TestCase):
    """The control must be given the same courtesy as the encoder, or the gain measures its handicap.

    The old control was `(lat/90, lon/180) @ N(0, 8)`. Across California that projection varies ~0.04
    cycles end to end, so it scored 0.008 — BELOW the raw-coordinate baseline at 0.0166. A nonlinear
    control that loses to raw coordinates is not a control.
    """

    def setUp(self):
        import numpy as np
        from autoresearch.probes.spacetime.editable_files.lib.fair_baseline import fair_rff, _rff_features
        self.np, self.fair_rff, self.feats = np, fair_rff, _rff_features
        # a regional corpus, as a fraction of the globe-normalized range
        rng = np.random.default_rng(0)
        lat = rng.uniform(32.5, 42.0, 4096)
        lon = rng.uniform(-124.4, -114.2, 4096)
        self.rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)

    def test_train_extent_normalization_actually_spans_the_range(self):
        """The old normalization left a regional corpus in ~5% of the domain; this must fill it."""
        scaled, _, _ = self.fair_rff(self.rn, 64)
        self.assertGreater(scaled.max() - scaled.min(), 1.9, "train-extent scaling must span ~[-1,1]")
        self.assertLess(abs(float(scaled.mean())), 0.1)

    def test_the_old_fixed_bandwidth_was_degenerate_on_this_corpus(self):
        """Quantify the defect: at the old sigma the features are near-constant across the dataset."""
        proj = self.rn @ (self.np.random.default_rng(0).normal(0, 8.0, (2, 32)).astype("float32"))
        cycles = float((proj.max(0) - proj.min(0)).mean()) / (2 * self.np.pi)
        self.assertLess(cycles, 0.2, "the old control varied well under one cycle across the corpus")

    def test_the_fixed_control_varies_over_many_cycles(self):
        scaled, base, sigmas = self.fair_rff(self.rn, 64)
        proj = scaled @ (base * max(sigmas))
        cycles = float((proj.max(0) - proj.min(0)).mean()) / (2 * self.np.pi)
        self.assertGreater(cycles, 1.0, "the control must be able to resolve structure at data scale")

    def test_bandwidth_is_a_sweep_not_a_constant(self):
        _, _, sigmas = self.fair_rff(self.rn, 64)
        self.assertGreater(len(sigmas), 1, "a baseline pinned to one sigma is a straw man")

    def test_extent_is_fit_on_train_rows_only(self):
        """Fitting on all rows would leak the evaluation range into the control's features."""
        mask = self.np.zeros(len(self.rn), dtype=bool); mask[: len(self.rn) // 2] = True
        scaled_train, _, _ = self.fair_rff(self.rn, 64, train_mask=mask)
        scaled_all, _, _ = self.fair_rff(self.rn, 64)
        self.assertFalse(self.np.allclose(scaled_train, scaled_all),
                         "train-fitted extent must differ from all-rows extent")
