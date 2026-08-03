"""Focused regression tests for the biological loop's measurement boundary."""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import torch

from deepearth.autoresearch.probes.biological.editable_files.lib import training
from deepearth.autoresearch.probes.biological.harness import board
from deepearth.autoresearch.scoring.contract import Primary, ProbeResult


AUTORESEARCH = Path(__file__).resolve().parents[1]
BIO_PROBE = AUTORESEARCH / "probes" / "biological" / "harness" / "probe.py"
BIO_TRAIT_PROBE = AUTORESEARCH / "probes" / "biological" / "harness" / "traitprobe.py"


class _DummySpeciesGraph(torch.nn.Module):
    """Small differentiable stand-in that exposes the SpeciesGraph interface."""

    def __init__(self, n_species, d_model, *, species_text, **_):
        super().__init__()
        self.register_buffer("base", species_text[:, :d_model].float().clone())
        self.offset = torch.nn.Parameter(torch.randn(d_model))

    def _seed(self):
        return self.base + self.offset

    def forward(self, mask=None):
        out = self._seed()
        if mask is not None:
            out = torch.where(mask.unsqueeze(-1), self.offset.expand_as(out), out)
        return out


class BiologicalContractTests(unittest.TestCase):
    def test_recordable_entrypoints_declare_trained_encoder(self):
        for path in (BIO_PROBE, BIO_TRAIT_PROBE):
            tree = ast.parse(path.read_text())
            sink_calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_set_result_sink"
            ]
            self.assertEqual(len(sink_calls), 1, path)
            config = sink_calls[0].args[4]
            self.assertIsInstance(config, ast.Dict, path)
            values = {
                key.value: value
                for key, value in zip(config.keys, config.values)
                if isinstance(key, ast.Constant)
            }
            trained = values.get("train_encoder")
            self.assertIsInstance(trained, ast.Constant, path)
            self.assertIs(trained.value, True, path)

    def test_family_record_uses_masked_imputation(self):
        tree = ast.parse(BIO_PROBE.read_text())
        declarations = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "declare"
        ]
        family = []
        for call in declarations:
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            capability = keywords.get("capability")
            if isinstance(capability, ast.IfExp):
                values = (capability.body, capability.orelse)
                if any(isinstance(value, ast.Constant) and value.value == "family_from_phylo"
                       for value in values):
                    family.append(keywords)
        self.assertEqual(len(family), 1)
        primary = family[0]["value"]
        self.assertIsInstance(primary, ast.Name)
        self.assertEqual(primary.id, "impute_acc")

    def test_trait_board_rows_use_paired_imputation_means(self):
        tree = ast.parse(BIO_TRAIT_PROBE.read_text())
        found = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "declare"):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            capability = keywords.get("capability")
            if isinstance(capability, ast.Constant):
                found[capability.value] = keywords
        for capability in ("community_from_species", "myco_from_species"):
            self.assertIn(capability, found)
            primary = found[capability]["value"]
            self.assertIsInstance(primary, ast.Name)
            self.assertEqual(primary.id, "primary_m")
            extras = {key for key in found[capability] if key not in {
                "capability", "mode", "metric", "value", "split", "gains", "baselines",
                "diagnostic", "diagnostic_reason",
            }}
            self.assertIn("primary_seed_values", extras)

    def test_interaction_fit_restarts_both_graphs_and_head(self):
        plant_seed = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        poll_seed = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
        plant_rows = torch.tensor([0, 1, 2])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        train_plants = torch.tensor([True, True, False])
        test_plants = ~train_plants
        train_pollinators = torch.tensor([True, True])
        test_pollinators = ~train_pollinators

        original = training.SpeciesGraph
        training.SpeciesGraph = _DummySpeciesGraph
        try:
            args = (plant_seed, poll_seed, {}, torch.arange(3), torch.eye(2),
                    plant_rows, targets, train_plants, test_plants,
                    train_pollinators, test_pollinators, "cpu", 4, 2, 0.5, 1e-2)
            first = training.train_interaction_graphs(*args, run_seed=17)
            second = training.train_interaction_graphs(*args, run_seed=17)
        finally:
            training.SpeciesGraph = original

        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        for left, right in zip(first[:2], second[:2]):
            for key, value in left.state_dict().items():
                self.assertTrue(torch.equal(value, right.state_dict()[key]), key)
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertEqual(first[3], second[3])
        self.assertTrue(torch.equal(first[4], second[4]))
        self.assertTrue(torch.equal(first[5], second[5]))

    def test_operator_inert_uses_movement_not_score_delta(self):
        active = board._bottleneck(-0.1, primary=0.4, seed_score=0.8, refined_seed_norm=0.2)
        inert = board._bottleneck(-0.1, primary=0.4, seed_score=0.8, refined_seed_norm=0.0)
        self.assertNotIn("OPERATOR-INERT", active)
        self.assertIn("OPERATOR-INERT", inert)

    def test_board_aggregates_matched_seed_results(self):
        common = dict(capability="family_from_phylo", mode="MASK-RECON", protocol=board.PROTOCOL,
                      split="species-random", gains={"vs null-tree": 0.1},
                      baselines={"seed": 0.4, "null-tree/best": 0.5}, loop="biological")
        first = ProbeResult(primary=Primary("family_nn_accuracy", 0.6), seed=0, **common)
        second = ProbeResult(primary=Primary("family_nn_accuracy", 0.8), seed=1,
                             **{**common, "gains": {"vs null-tree": 0.2},
                                "baselines": {"seed": 0.5, "null-tree/best": 0.6}})
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "seed0.json", Path(tmp) / "seed1.json"]
            first.write(paths[0]); second.write(paths[1])
            result = board._aggregate_results(paths)
        self.assertAlmostEqual(result.primary.value, 0.7)
        self.assertAlmostEqual(result.gains["vs null-tree"], 0.15)
        self.assertEqual(result.extras["primary_seed_values"], [0.6, 0.8])


if __name__ == "__main__":
    unittest.main()
