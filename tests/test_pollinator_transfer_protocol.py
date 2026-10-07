import itertools

import numpy as np
import pytest
import torch

from deepearth.autoresearch.data import California
from deepearth.autoresearch.evaluate import (
    BENCHMARK_PROTOCOL,
    arithmetic_net,
    format_benchmarks,
    net_score,
    normalized_ndcg_at_k,
)


def test_interaction_holdout_is_deterministic_and_keeps_labelled_relatives():
    names = np.array([f"Genus species{i}" for i in range(12)])
    valid = np.ones(12, dtype=bool)
    groups = np.repeat(np.arange(3), 4)

    first = California._interaction_holdout(names, valid, groups)
    second = California._interaction_holdout(names, valid, groups)

    assert np.array_equal(first, second)
    for group in np.unique(groups):
        members = groups == group
        assert first[members].any()
        assert (~first[members]).any()


def test_held_interactions_are_removed_from_supervision_and_lookup():
    source = object.__new__(California)
    source.binomial = np.array(["A one", "A two", "B one", "B two"])
    source.class_group = torch.tensor([0, 0, 1, 1])
    source.poll_valid = torch.ones(4, dtype=torch.bool)
    source.poll_idx = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]])
    source.poll_frq = torch.tensor([[0.7, 0.3]] * 4)

    source._set_pollinator_transfer_holdout()

    held = source.poll_transfer_holdout
    assert held.sum() == 2
    assert not source.poll_train_valid[held].any()
    assert not source.poll_train_frq[held].any()
    assert not source.poll_train_idx[held].any()
    assert torch.equal(source.poll_train_frq[~held], source.poll_frq[~held])


def test_normalized_ndcg_has_perfect_one_and_uniform_null_zero():
    target_idx = torch.tensor([[0, 2]])
    relevance = torch.tensor([[0.7, 0.3]])
    perfect = torch.tensor([[4.0, 1.0, 3.0, 0.0]])
    assert normalized_ndcg_at_k(perfect, target_idx, relevance, k=2).item() == pytest.approx(1.0)

    scores = []
    for order in itertools.permutations(range(4)):
        logits = torch.empty(1, 4)
        for rank, item in enumerate(order):
            logits[0, item] = 4 - rank
        scores.append(normalized_ndcg_at_k(logits, target_idx, relevance, k=2).item())
    assert np.mean(scores) == pytest.approx(0.0, abs=1e-6)


def test_headlines_exclude_mechanism_gains_and_legacy_b55():
    raw = {
        "B1_species_from_env_top10": 0.2,
        "B2_species_from_photo_top1": 0.8,
        "B55_pollinator_phylo_transfer_recall": 0.04,
        "B56_family_phylo_graph_gain": 0.1,
    }

    assert net_score(raw) == pytest.approx(0.32)
    assert arithmetic_net(raw) == pytest.approx(0.5)
    rendered = format_benchmarks(raw)
    assert f"BENCHMARK PROTOCOL: {BENCHMARK_PROTOCOL}" in rendered
    assert "QUARANTINED" in rendered
    assert "MECHANISM DIAGNOSTICS" in rendered
