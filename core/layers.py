"""Small neural-network building blocks shared by the core model."""
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def preserve_rng():
    state = torch.random.get_rng_state()
    yield
    torch.random.set_rng_state(state)


def mlp(input_dim, output_dim, hidden_dim=None, *, normalize=True):
    layers = [nn.LayerNorm(input_dim)] if normalize else []
    if hidden_dim:
        layers += [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        input_dim = hidden_dim
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


def per_name(names, factory):
    return nn.ParameterDict({
        name: nn.Parameter(factory(name)) for name in names
    })


def consume_rng(*_unused) -> None:
    """Advance the published initialization stream without retaining dead weights."""
