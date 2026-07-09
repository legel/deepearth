"""Coordinate fields turn a query's neighbors into context tokens.

Every observation sits in coordinate subspaces, and DeepEarth treats them alike. Space and time form one
subspace: the offset from a query to each neighbor is encoded by Earth4D in relative mode, so structure learned at
one place and time applies everywhere. A vector manifold, such as an evolutionary-position vector per observation,
forms another: a neighbor's position within that space is encoded directly.

The neighbors produce one token per (neighbor, subspace): a space-time token carries a neighbor's offset from the
query, a biological token carries a neighbor's own evolutionary position. Each token also carries projections of
the neighbor's observed features and a marker naming its subspace, so the latents attend to each relation
separately. A subspace only ever reads coordinates that are known at inference, so a query's own position in a
predicted subspace stays out of the context.
"""
from __future__ import annotations
import math
from typing import Dict, Sequence
import torch
import torch.nn as nn

from deepearth.encoders.spacetime.earth4d import Earth4D


class SpaceTimeField(nn.Module):
    """Encodes each neighbor's space-time offset from the query. Coordinates are (latitude, longitude, elevation,
    time); the offset is converted to metres and time, then encoded by Earth4D in relative mode."""

    def __init__(self, d_model: int, window: Sequence[float], levels: int = 24, reference_latitude_deg: float = 0.0,
                 finest: Sequence[float] = (0.1, 0.1, 1.0, 0.042), log2_hashmap_size: int = 22):
        super().__init__()
        # This field only calls ``encode_relative``, so ask Earth4D not to allocate the absolute projections. The
        # relative encoder carries the high-frequency local structure: a wide window (metres, days) down to a very
        # fine finest resolution, packed across many levels.
        self.earth4d = Earth4D(verbose=False, enable_relative=True, enable_absolute=False,
                               relative_window=tuple(window), relative_finest=tuple(finest),
                               relative_levels=levels, relative_log2_hashmap_size=log2_hashmap_size)
        self.proj = nn.Sequential(nn.Linear(self.earth4d.relative_output_dim, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.m_per_deg = 111_320.0
        self.m_per_deg_lon = 111_320.0 * math.cos(math.radians(reference_latitude_deg))

    def forward(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor) -> torch.Tensor:
        delta = neighbor_coords - query_coords.unsqueeze(1)
        offset = torch.stack([delta[..., 0] * self.m_per_deg, delta[..., 1] * self.m_per_deg_lon,
                              delta[..., 2], delta[..., 3]], dim=-1)
        return self.proj(self.earth4d.encode_relative(offset))


class ManifoldField(nn.Module):
    """Encodes each neighbor's own position within a vector subspace, such as an evolutionary manifold."""

    def __init__(self, d_model: int, dim: int, hidden: int = 256):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, neighbor_positions: torch.Tensor) -> torch.Tensor:
        return self.encode(neighbor_positions)


class NeighborContext(nn.Module):
    """Emit one token per (neighbor, subspace), each carrying that subspace's encoding, the neighbor's feature
    projections, and a marker naming the subspace.

    Args:
        d_model: token width.
        space_time: keyword args for :class:`SpaceTimeField` (``window``, ``levels``, ...).
        manifolds: ``{name: dim}`` for each vector subspace (e.g. ``{"biological": 2048}``).
        feature_dims: ``{name: dim}`` for each per-neighbor feature to project in.
    """

    def __init__(self, d_model: int, space_time: dict, manifolds: Dict[str, int] | None = None,
                 feature_dims: Dict[str, int] | None = None):
        super().__init__()
        self.d_model = d_model
        self.space_time = SpaceTimeField(d_model, **space_time)
        self.manifolds = nn.ModuleDict({name: ManifoldField(d_model, dim) for name, dim in (manifolds or {}).items()})
        self.features = nn.ModuleDict({name: nn.Linear(dim, d_model) for name, dim in (feature_dims or {}).items()})
        self.field_marker = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(d_model) * 0.02) for name in ["space_time", *(manifolds or {})]})

    def forward(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                manifold_positions: Dict[str, torch.Tensor] | None = None,
                neighbor_features: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        B, K = neighbor_coords.shape[0], neighbor_coords.shape[1]
        features = query_coords.new_zeros(B, K, self.d_model)
        for name, val in (neighbor_features or {}).items():
            features = features + self.features[name](val)
        tokens = [self.space_time(query_coords, neighbor_coords) + features + self.field_marker["space_time"]]
        for name, field in self.manifolds.items():
            tokens.append(field(manifold_positions[name]) + features + self.field_marker[name])
        return torch.cat(tokens, dim=1)

