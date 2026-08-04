import importlib.util
from pathlib import Path

import torch


_SPEC = importlib.util.spec_from_file_location(
    "phylogenomic", Path(__file__).parents[1] / "encoders" / "biological" / "phylogenomic.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
SpeciesGraph = _MODULE.SpeciesGraph


def test_text_prior_uses_only_shared_probe_parameters():
    text = torch.randn(5, 7)
    distance = SpeciesGraph.distance_from_embedding(text)
    graph = SpeciesGraph(5, 8, phylo_distance=distance, n_layers=0, species_text=text)

    expected = graph.probe(text)
    assert graph.free is None
    assert torch.equal(graph._seed(), expected)
    assert not any(name == "free" for name, _ in graph.named_parameters())


def test_missing_text_prior_keeps_discriminative_species_table():
    distance = torch.ones(5, 5) - torch.eye(5)
    graph = SpeciesGraph(5, 8, phylo_distance=distance, n_layers=0)

    assert graph.free is not None
    assert torch.equal(graph._seed(), graph.free)
    assert dict(graph.named_parameters())["free"] is graph.free
