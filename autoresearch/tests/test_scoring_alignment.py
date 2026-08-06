"""Regression tests for probe-to-champion scientific alignment."""
from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import torch

from deepearth.autoresearch.main.harness import champion_report, hooks, score
from deepearth.autoresearch.main.harness.evaluate import BENCHMARK_PROTOCOL
from deepearth.autoresearch.scoring.definitions import capability_to_benchmark, noise_barrier
from deepearth.autoresearch.scoring import graduation
from deepearth.autoresearch.scoring.graduation import _receipt_mismatch, compare_fusion


class _Ones(torch.nn.Module):
    def forward(self, x):
        return torch.ones((*x.shape[:-1], 4), device=x.device, dtype=x.dtype)


class _Relative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 4)

    def forward(self, query, neighbors):
        return torch.ones((*neighbors.shape[:2], 4), device=neighbors.device)


class _Model:
    d_model = 4

    def __init__(self):
        self.absolute_proj_s = _Ones()
        self.absolute_proj_t = _Ones()
        self.neighbors = type("Neighbors", (), {"space_time": _Relative()})()


class ScoringAlignmentTests(unittest.TestCase):
    def test_promotion_margin_is_one_and_half_percent_with_absolute_floor(self):
        self.assertAlmostEqual(noise_barrier(0.2), 0.003)
        self.assertAlmostEqual(noise_barrier(0.1), 0.002)

    def test_graduation_mapping_is_unique_and_mask_aligned(self):
        mapping = capability_to_benchmark()
        self.assertEqual(mapping["family_from_phylo"], "B64_family_phylo_masked_imputation")
        self.assertEqual(mapping["myco_from_species"], "B65_myco_phylo_masked_imputation_f1")
        self.assertEqual(mapping["community_from_species"], "B66_community_phylo_masked_recall")
        self.assertEqual(mapping["pollinator_transfer"], "B67_pollinator_phylo_masked_recall")
        self.assertEqual(mapping["species_from_env"], "B1_species_from_env_top10")

    def test_gain_objective_requires_the_complete_vector(self):
        values = {name: 0.1 for name in score.ST_GAIN}
        self.assertAlmostEqual(score.gain_scalar(values, score.ST_GAIN), 0.1)
        values.pop(score.ST_GAIN[-1])
        self.assertIsNone(score.gain_scalar(values, score.ST_GAIN))

    def test_earth4d_ablation_removes_absolute_and_relative_channels(self):
        model = _Model()
        x = torch.zeros(2, 4)
        query = torch.zeros(2, 4)
        neighbors = torch.zeros(2, 3, 4)
        self.assertTrue(model.absolute_proj_s(x).bool().all())
        self.assertTrue(model.neighbors.space_time(query, neighbors).bool().all())
        with hooks.ablate_earth4d(model):
            self.assertFalse(model.absolute_proj_s(x).bool().any())
            self.assertFalse(model.absolute_proj_t(x).bool().any())
            self.assertFalse(model.neighbors.space_time(query, neighbors).bool().any())
        self.assertTrue(model.absolute_proj_s(x).bool().all())
        self.assertTrue(model.neighbors.space_time(query, neighbors).bool().all())

    def test_run_log_carries_the_benchmark_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.log"
            path.write_text(
                "training_seed:     1337\n"
                "trained 5000 steps in 600s\n"
                f"BENCHMARK PROTOCOL: {BENCHMARK_PROTOCOL}\n"
                "  B64_family_phylo_masked_imputation 0.250\n"
                "peak_vram_mb:     12345.6\n"
                'RUN RECEIPT: {"schema":"fusion-run-v1","source":{"dirty":false}}\n'
            )
            parsed = champion_report.parse_run(str(path))
        self.assertEqual(parsed["benchmark_protocol"], BENCHMARK_PROTOCOL)
        self.assertEqual(parsed["training_seed"], 1337)
        self.assertEqual(parsed["steps"], 5000)
        self.assertEqual(parsed["peak_vram_mb"], 12345.6)
        self.assertEqual(parsed["scores"]["B64_family_phylo_masked_imputation"], 0.25)
        self.assertEqual(parsed["receipt"]["schema"], "fusion-run-v1")

    def test_frozen_control_requires_exact_pairing_identity(self):
        shared = {
            "schema": "fusion-run-v1",
            "judge": {"protocol": "p", "evaluate_sha256": "e", "definitions_sha256": "d"},
            "data": {"identity": "i", "prepared_sha256": "a"},
            "training": {"seed": 1337, "steps": 8000, "time_budget_s": 600, "batch": 512,
                         "precision": "bf16"},
            "runtime": {"torch": "2.7", "cuda": "12.6", "gpu": "RTX"},
        }
        control = {"receipt": {**shared, "source": {"tree": "base", "parent_tree": "old", "dirty": False}}}
        candidate = {"receipt": {**shared, "source": {"tree": "candidate", "parent_tree": "base", "dirty": False}}}
        self.assertEqual(_receipt_mismatch(control, candidate), [])
        candidate["receipt"]["judge"] = {**shared["judge"], "evaluate_sha256": "changed"}
        self.assertIn("judge.evaluate_sha256", _receipt_mismatch(control, candidate))

    def test_two_seed_control_can_be_frozen_once(self):
        receipt = {
            "schema": "fusion-run-v1",
            "source": {"tree": "base", "parent_tree": "old", "dirty": False},
            "config": {"sha256": "config", "effective_sha256": "effective"},
            "judge": {"protocol": BENCHMARK_PROTOCOL, "evaluate_sha256": "e",
                      "definitions_sha256": "d"},
            "data": {"identity": "i", "prepared_sha256": "a"},
            "training": {"seed": 1337, "steps": 8000, "time_budget_s": 600, "batch": 512,
                         "precision": "bf16"},
            "runner": {"train_sha256": "t", "hooks_sha256": "h"},
            "runtime": {"torch": "2.7", "cuda": "12.6", "gpu": "RTX"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = []
            for seed in (1337, 1338):
                current = json.loads(json.dumps(receipt))
                current["training"]["seed"] = seed
                current["config"]["effective_sha256"] = f"effective-{seed}"
                path = root / f"control-{seed}.log"
                path.write_text(
                    f"training_seed:     {seed}\nBENCHMARK PROTOCOL: {BENCHMARK_PROTOCOL}\n"
                    "  B64_family_phylo_masked_imputation 0.250\n"
                    f"RUN RECEIPT: {json.dumps(current)}\n"
                )
                logs.append(str(path))
            prior = graduation.CONTROLS
            graduation.CONTROLS = root / "controls.json"
            try:
                graduation.register_control("base", logs)
                stored = json.loads(graduation.CONTROLS.read_text())
            finally:
                graduation.CONTROLS = prior
        self.assertEqual([run["training_seed"] for run in stored["base"]["runs"]], [1337, 1338])

    def test_propagation_uses_paired_fusion_runs(self):
        control = {"scores": {"B64_family_phylo_masked_imputation": 0.20, "B2_other": 0.60},
                   "harmonic": 0.30, "arithmetic": 0.40}
        candidate = {"scores": {"B64_family_phylo_masked_imputation": 0.23, "B2_other": 0.61},
                     "harmonic": 0.31, "arithmetic": 0.41}
        result = compare_fusion(control, candidate, "B64_family_phylo_masked_imputation", 0.06)
        self.assertAlmostEqual(result["bench_delta"], 0.03)
        self.assertAlmostEqual(result["propagation_ratio"], 0.5)
        self.assertTrue(result["transferred"])
        self.assertTrue(result["fusion_breakthrough"])

    def test_propagation_rejects_suite_drift(self):
        control = {"scores": {"B1": 0.2}, "harmonic": 0.2, "arithmetic": 0.2}
        candidate = {"scores": {"B2": 0.3}, "harmonic": 0.3, "arithmetic": 0.3}
        with self.assertRaisesRegex(ValueError, "benchmark suites differ"):
            compare_fusion(control, candidate, "B1", 0.1)


if __name__ == "__main__":
    unittest.main()
