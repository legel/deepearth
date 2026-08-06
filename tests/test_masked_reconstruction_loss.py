import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_SPEC = importlib.util.spec_from_file_location(
    "phylogenomic", Path(__file__).parents[1] / "encoders" / "biological" / "phylogenomic.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
SpeciesGraph = _MODULE.SpeciesGraph


def _graph(n=6, d=8):
    torch.manual_seed(0)
    text = torch.randn(n, 5)
    return SpeciesGraph(n, d, phylo_distance=SpeciesGraph.distance_from_embedding(text),
                        n_layers=0, species_text=text)


def test_mse_matches_the_inlined_expression():
    graph = _graph()
    target = torch.randn(6, 8)
    mask = torch.tensor([True, False, True, False, False, True])

    expected = F.mse_loss(graph(mask=mask)[mask], target[mask])
    assert torch.allclose(graph.masked_reconstruction_loss(mask, target, metric="mse"), expected)


def test_cosine_scores_one_minus_similarity():
    graph = _graph()
    target = torch.randn(6, 8)
    mask = torch.tensor([True, True, False, False, False, False])

    expected = (1.0 - F.cosine_similarity(graph(mask=mask)[mask], target[mask], dim=-1)).mean()
    assert torch.allclose(graph.masked_reconstruction_loss(mask, target), expected)


def test_caller_may_supply_its_own_reconstruction():
    graph = _graph()
    target = torch.randn(6, 8)
    mask = torch.tensor([True, False, True, False, False, False])
    reconstructed = graph(mask=mask)

    assert torch.allclose(
        graph.masked_reconstruction_loss(mask, target, metric="mse", reconstructed=reconstructed),
        graph.masked_reconstruction_loss(mask, target, metric="mse"),
    )


def test_empty_mask_returns_a_zero_that_still_carries_gradient():
    graph = _graph()
    target = torch.randn(6, 8)
    mask = torch.zeros(6, dtype=torch.bool)

    loss = graph.masked_reconstruction_loss(mask, target)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()


def test_unknown_metric_is_rejected():
    graph = _graph()
    mask = torch.tensor([True, False, False, False, False, False])

    with pytest.raises(ValueError):
        graph.masked_reconstruction_loss(mask, torch.randn(6, 8), metric="l1")
