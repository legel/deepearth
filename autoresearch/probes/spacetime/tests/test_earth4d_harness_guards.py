import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_PARENT = str(Path(__file__).resolve().parents[2])
if REPO_PARENT not in sys.path:
    sys.path.insert(0, REPO_PARENT)

from autoresearch.probes.spacetime.editable_files.harness import (
    CAPABILITIES,
    DEFAULT_PROBE_MODULE,
    EXCLUDED_CAPABILITIES,
    PROTOCOL,
    _bottleneck,
    _commit_records_if_unchanged,
    _read_records,
    _record_gate,
)
from autoresearch.main.editable_files.harness.perception_diag import _macro_accuracy, probe_genus_from_semantic
from autoresearch.probes.spacetime.editable_files.lib.recurrence import (
    TRACE_AUTH_FD_ENV,
    normalize_forecast_time,
    phenology_feature_set,
    phenology_mode,
    validate_dynamic_target_causality,
    require_recorded_entrypoint,
    trace_authorization_payload,
    ALLOW_UNRECORDED_ENV,
)


class TraceProtocolGateTests(unittest.TestCase):
    def test_unstamped_record_never_auto_rebaselines(self):
        decision = _record_gate(
            key_val=0.1,
            prev=0.9,
            prev_proto=None,
            mode="FORECAST",
            prev_mode="ENV",
            shards=8,
            prev_shards=12,
        )
        self.assertEqual(decision, (False, False, False, False, False))

    def test_known_protocol_requires_same_mode_and_shards(self):
        same = _record_gate(
            0.1, 0.9, "v1-prefix", "FORECAST", "FORECAST", 12, 12,
            "--forecast --n_shards 12", "--forecast  --n_shards 12",
        )
        wrong_mode = _record_gate(
            0.1, 0.9, "v1-prefix", "ENV", "FORECAST", 12, 12,
            "--forecast --n_shards 12", "--forecast --n_shards 12",
        )
        wrong_shards = _record_gate(
            0.1, 0.9, "v1-prefix", "FORECAST", "FORECAST", 8, 12,
            "--forecast --n_shards 8", "--forecast --n_shards 12",
        )
        self.assertEqual(same, (True, True, False, True, True))
        self.assertFalse(wrong_mode[0])
        self.assertFalse(wrong_mode[1])
        self.assertFalse(wrong_shards[0])
        self.assertFalse(wrong_shards[1])

    def test_known_protocol_requires_exact_probe_command(self):
        changed = _record_gate(
            1.0, 0.9, "v1-prefix", "FORECAST", "FORECAST", 12, 12,
            "--forecast --head_hidden 512 --n_shards 12",
            "--forecast --head_hidden 256 --n_shards 12",
        )
        self.assertFalse(changed[0])
        self.assertFalse(changed[1])

    def test_current_protocol_uses_ordinary_record_comparison(self):
        worse = _record_gate(0.1, 0.9, PROTOCOL, "ENV", "ENV", 12, 12)
        better = _record_gate(1.0, 0.9, PROTOCOL, "ENV", "ENV", 12, 12)
        self.assertFalse(worse[0])
        self.assertTrue(better[0])
        self.assertFalse(better[1])

    def test_pre_contract_bare_mode_matches_its_submoded_form(self):
        """A pre-contract record stored the mode FAMILY; the contract appended the split as a submode.

        family_from_env's record is mode "ENV" and every run since the contract declares
        "ENV(spatial-block)" -- the same measurement, verified by a no-edit control reproducing
        0.142318 against a stored 0.1423. Before this rule the capability could not be recorded at all:
        the gate refused every run as not like-for-like, which is why its dead-end ledger is full of
        RECORD WITHHELD entries whose only fault was the label.
        """
        migrated = _record_gate(0.2, 0.1423, "v2-leakfix", "ENV(spatial-block)", "ENV", 12, 12)
        self.assertTrue(migrated[0])       # is_record
        self.assertTrue(migrated[1])       # rebaseline
        self.assertTrue(migrated[3])       # mode_ok
        beats = _record_gate(0.2, 0.1423, PROTOCOL, "ENV(spatial-block)", "ENV", 12, 12)
        self.assertTrue(beats[0])
        # ...but it must not merge two submodes of one family, nor two different families.
        two_submodes = _record_gate(
            0.2, 0.1, PROTOCOL, "FORECAST(future+newplace)", "FORECAST(past->future)", 12, 12)
        self.assertFalse(two_submodes[0])
        self.assertFalse(two_submodes[3])
        other_family = _record_gate(0.2, 0.1, PROTOCOL, "ENV-DECODE(spatial-block)", "ENV", 12, 12)
        self.assertFalse(other_family[0])
        self.assertFalse(other_family[3])

    def test_capability_registry_matches_the_scorecard_contract(self):
        """Every legal --metric must be parseable, and every refusal must carry a reason.

        The 9 non-probeable capabilities used to sit in CAPABILITIES with no PRIMARY_RE entry, so they
        were legal objectives whose score fell through to a generic Earth4D regex.
        """
        self.assertEqual(len(CAPABILITIES), 7)
        self.assertFalse(set(CAPABILITIES) & set(EXCLUDED_CAPABILITIES))
        for capability in ("lfmc_from_env", "infer_clay", "flowering_auc", "family_from_vision"):
            self.assertIn(capability, EXCLUDED_CAPABILITIES)
            self.assertTrue(EXCLUDED_CAPABILITIES[capability].strip())
        # There is no longer a regex table to keep in sync: the probe declares its own metric via the
        # result contract, so a capability is parseable iff some mode declares it.

    def test_bottleneck_diagnosis_agrees_with_the_program(self):
        """A flat/negative fair-gain is INPUT-limited (DATA lever), not architecture-limited.

        The old string told the agent to "swing bigger on the architecture" on a flat gain -- the exact
        inverse of program.md -- and that string is published to Ensue as the swarm's reason-to-move.
        """
        self.assertIn("DATA lever", _bottleneck(-0.01, 0.5))
        self.assertIn("INPUT-LIMITED", _bottleneck(0.0, 0.5))
        # the lever now turns on the encoder's SHARE of the score, not an absolute cutoff
        self.assertIn("ARCHITECTURE lever", _bottleneck(0.05, 0.80))     # 6% share -> weak mechanism
        self.assertIn("ENCODER-LIMITED", _bottleneck(0.05, 0.80))
        self.assertIn("EARNING", _bottleneck(0.05, 0.10))                # 50% share -> carrying it
        self.assertIn("NO-FAIR-BASELINE", _bottleneck(None, 0.5))

    def test_records_commit_is_atomic_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.json"
            path.write_text(json.dumps({"version": 1}))
            snapshot, records = _read_records(path)
            records["version"] = 2
            self.assertTrue(
                _commit_records_if_unchanged(snapshot, records, path)
            )
            stale_snapshot = snapshot
            path.write_text(json.dumps({"version": 3}))
            self.assertFalse(
                _commit_records_if_unchanged(
                    stale_snapshot,
                    {"version": 4},
                    path,
                )
            )
            self.assertEqual(json.loads(path.read_text()), {"version": 3})


class SatelliteTimeHeadroomTests(unittest.TestCase):
    def test_forecast_caller_reserves_headroom(self):
        days = np.array([0.0, 10.0, 15.0, 20.0])
        test = np.array([False, False, True, True])
        normalized, origin, span = normalize_forecast_time(days, test)
        self.assertEqual(origin, 0.0)
        self.assertEqual(span, 20.0)
        self.assertAlmostEqual(float(normalized[~test].max()), 0.5)
        self.assertLessEqual(float(normalized.max()), 1.0)


class PerceptionDiagnosticTests(unittest.TestCase):
    def test_macro_accuracy_weights_classes_equally(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        self.assertEqual(_macro_accuracy(y_true, y_pred), 0.5)
        self.assertNotEqual(_macro_accuracy(y_true, y_pred), float((y_true == y_pred).mean()))

    def test_five_fold_filter_excludes_four_sample_class(self):
        rng = np.random.default_rng(7)
        emb = rng.normal(size=(14, 6)).astype(np.float32)
        bino = np.array(
            [f"GenusA species{i}" for i in range(5)]
            + [f"GenusB species{i}" for i in range(5)]
            + [f"GenusC species{i}" for i in range(4)]
        )
        result = probe_genus_from_semantic(emb, bino, min_species=4, seed=3)
        self.assertEqual(result["n_species"], 10)
        self.assertEqual(result["n_genera"], 2)
        self.assertEqual(result["cv_folds"], 5)


class PhenologyWorkflowTests(unittest.TestCase):
    def test_split_label_is_exact(self):
        self.assertEqual(phenology_mode(), "PHENOLOGY-FUTURE")
        self.assertEqual(phenology_mode(forecast_spatial=True), "PHENOLOGY-FUTURE-HELD")
        self.assertEqual(phenology_mode(pheno_spatial=True), "PHENOLOGY-HELD")

    def test_earth4d_automatically_gets_raw_and_rff_controls(self):
        self.assertEqual(phenology_feature_set("e4d"), ("e4d", "raw", "rff"))
        self.assertEqual(phenology_feature_set("e4d", nofair=True), ("e4d",))


class CausalTargetQuarantineTests(unittest.TestCase):
    def test_label_leaky_paths_fail_closed(self):
        cases = (
            {"ar_rollout": True},
            {"ar_cond_lead": True},
            {"abundance": True, "lead": 30},
            {"abund_prop_arch": True, "lead": 30},
            {"breadth_target": "community_activity", "lead": 30},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "causality audit quarantine"):
                    validate_dynamic_target_causality(**kwargs)

    def test_observed_time_nowcasts_remain_available(self):
        validate_dynamic_target_causality(abundance=True, lead=0)
        validate_dynamic_target_causality(abund_prop_arch=True, lead=0)
        validate_dynamic_target_causality(breadth_target="occupancy", lead=0)

    def test_unrecorded_run_is_refused_by_default(self):
        """A probe whose result reaches no ledger is a run the swarm cannot learn from."""
        with self.assertRaisesRegex(ValueError, "would never be recorded"):
            require_recorded_entrypoint("probe.py", False)
        require_recorded_entrypoint("probe.py", True)

    def test_deliberately_unrecorded_run_is_allowed_explicitly(self):
        """Parity checks and smoke tests opt out by name -- but must say so."""
        with patch.dict(os.environ, {ALLOW_UNRECORDED_ENV: "1"}, clear=False):
            require_recorded_entrypoint("probe.py")

    def test_plain_environment_flag_cannot_forge_trace_authorization(self):
        with patch.dict(os.environ, {"EARTH4D_TRACE_AUTHORIZED": "1"}, clear=False):
            with self.assertRaisesRegex(ValueError, "would never be recorded"):
                require_recorded_entrypoint("probe.py")

    def test_inherited_one_shot_pipe_authorizes_exact_trace_child(self):
        argv = ["--forecast", "--n_shards", "12"]
        read_fd, write_fd = os.pipe()
        os.write(
            write_fd,
            trace_authorization_payload(DEFAULT_PROBE_MODULE, argv),
        )
        os.close(write_fd)
        with patch.dict(
            os.environ,
            {TRACE_AUTH_FD_ENV: str(read_fd)},
            clear=False,
        ):
            require_recorded_entrypoint(
                "probe.py",
                module=DEFAULT_PROBE_MODULE,
                argv=argv,
            )
            self.assertNotIn(TRACE_AUTH_FD_ENV, os.environ)
        with self.assertRaisesRegex(ValueError, "would never be recorded"):
            require_recorded_entrypoint(
                "probe.py",
                module=DEFAULT_PROBE_MODULE,
                argv=argv,
            )

    def test_trace_pipe_is_bound_to_exact_argv(self):
        authorized_argv = ["--forecast", "--n_shards", "12"]
        read_fd, write_fd = os.pipe()
        os.write(
            write_fd,
            trace_authorization_payload(
                DEFAULT_PROBE_MODULE,
                authorized_argv,
            ),
        )
        os.close(write_fd)
        with patch.dict(
            os.environ,
            {TRACE_AUTH_FD_ENV: str(read_fd)},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "would never be recorded",
            ):
                require_recorded_entrypoint(
                    "probe.py",
                    module=DEFAULT_PROBE_MODULE,
                    argv=authorized_argv + ["--train_encoder"],
                )
            self.assertNotIn(TRACE_AUTH_FD_ENV, os.environ)


class LegacyEntryPointWiringTests(unittest.TestCase):
    @staticmethod
    def run_module(module, *args, env_extra=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_PARENT + (
            os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH")
            else ""
        )
        env.update(env_extra or {})
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

    def test_canonical_probe_rejects_unrecorded_direct_call(self):
        result = self.run_module(
            "deepearth.autoresearch.probes.spacetime.editable_files.probe"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would never be recorded", result.stdout)

    def test_probe_clis_are_reachable_not_retired(self):
        """These CLIs were hard-retired by a research directive, which bricked five entrypoints --
        including calib_probe, the only path that can reproduce the live calibration record.

        They must now be reachable; the only gate is that a run be recordable.
        """
        modules = (
            "editable_files.probe",
            "editable_files.lib.calib_probe",
        )
        for name in modules:
            with self.subTest(module=name):
                result = self.run_module(
                    f"deepearth.autoresearch.probes.spacetime.{name}",
                    "--help",
                    env_extra={ALLOW_UNRECORDED_ENV: "1"},
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("retired", result.stdout)

    def test_lfmc_science_gate_remains_runnable(self):
        result = self.run_module(
            "deepearth.autoresearch.probes.spacetime.editable_files.lib.science_gate",
            "--help",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Globe-LFMC", result.stdout)


if __name__ == "__main__":
    unittest.main()
