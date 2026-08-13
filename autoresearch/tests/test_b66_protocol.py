"""Protocol-v6 correctness checks for conditional phylogenetic community transfer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

root = Path(__file__).resolve().parents[2]
if "deepearth" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "deepearth", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["deepearth"] = package
    spec.loader.exec_module(package)
from deepearth.autoresearch.main.harness.evaluate import (  # noqa: E402
    BENCHMARKS,
    BENCHMARK_PROTOCOL,
    _conditional_community_auc,
    _paired_calls,
    _tie_aware_binary_auc,
)
from deepearth.autoresearch.scoring.objective import capability_suite  # noqa: E402


def test_binary_auc_gives_ties_half_credit():
    target = np.array([True, True, False, False])
    assert _tie_aware_binary_auc([0, 0, 0, 0], target) == pytest.approx(0.5)
    assert _tie_aware_binary_auc([2, 1, 0, 0], target) == pytest.approx(1.0)
    assert _tie_aware_binary_auc([0, 0, 2, 1], target) == pytest.approx(0.0)


def test_conditional_auc_cancels_shared_context_per_query():
    no_identity = torch.tensor([[9.0, -4.0, 3.0, 1.0],
                                [-2.0, 8.0, 0.0, 5.0]])
    increment = torch.tensor([[0.4, 0.1, 0.4, -0.2],
                              [0.0, 0.2, -0.1, 0.2]])
    target = torch.tensor([[True, False, True, False],
                           [False, True, False, True]])
    total, count = _conditional_community_auc(no_identity + increment, no_identity, target)
    expected = sum(_tie_aware_binary_auc(row, truth)
                   for row, truth in zip(increment.numpy(), target.numpy()))

    shared_context = torch.tensor([[100.0, -50.0, 7.0, 11.0],
                                   [3.0, 90.0, -20.0, 4.0]])
    shifted_total, shifted_count = _conditional_community_auc(
        no_identity + increment + shared_context,
        no_identity + shared_context,
        target,
    )
    assert count == shifted_count == 2
    assert total == pytest.approx(expected), "queries are scored separately before averaging"
    assert total / count == pytest.approx(shifted_total / shifted_count)


def test_paired_arms_share_rng_and_consume_one_call():
    torch.manual_seed(7)
    first, second = _paired_calls(lambda: torch.rand(5), lambda: torch.rand(5), "cpu")
    following = torch.rand(5)

    torch.manual_seed(7)
    expected_first = torch.rand(5)
    expected_following = torch.rand(5)
    assert torch.equal(first, second)
    assert torch.equal(first, expected_first)
    assert torch.equal(following, expected_following)


def test_v6_suite_promotes_conditional_auc_not_contextual_recall():
    active = "B66_community_phylo_conditional_auc"
    diagnostic = "B66_contextual_masked_community_recall"
    assert BENCHMARK_PROTOCOL == "v6-canonical-family-identity"
    assert active in BENCHMARKS and diagnostic in BENCHMARKS
    assert active in capability_suite({active: 0.5, diagnostic: 0.9})
    assert diagnostic not in capability_suite({active: 0.5, diagnostic: 0.9})
