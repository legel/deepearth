"""Regression tests for probe-to-champion scientific alignment."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from deepearth.autoresearch.main.harness import champion_report, hooks, score
from deepearth.autoresearch.main.harness.evaluate import BENCHMARK_PROTOCOL
from deepearth.autoresearch.scoring.definitions import capability_to_benchmark


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
                f"BENCHMARK PROTOCOL: {BENCHMARK_PROTOCOL}\n"
                "  B64_family_phylo_masked_imputation 0.250\n"
            )
            parsed = champion_report.parse_run(str(path))
        self.assertEqual(parsed["benchmark_protocol"], BENCHMARK_PROTOCOL)
        self.assertEqual(parsed["scores"]["B64_family_phylo_masked_imputation"], 0.25)


if __name__ == "__main__":
    unittest.main()
