"""Earth4D-addressed world state and typed scientific fibers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from deepearth.encoders.spacetime.earth4d import ECEF_NORM_FACTOR, to_ecef
from deepearth.encoders.spacetime.hashencoder.hashgrid import HashEncoder

LENSES = ("abiotic", "visual", "biological", "ecological")
LENS_INDEX = {name: index for index, name in enumerate(LENSES)}


def hash_field(levels, features, log2_size, resolution, scale):
    return HashEncoder(
        input_dim=3,
        num_levels=levels,
        level_dim=features,
        base_resolution=resolution,
        per_level_scale=scale,
        log2_hashmap_size=log2_size,
    )


def project_fields(encoders, coords, axes, levels, features):
    lead = coords.shape[:-1]
    return torch.cat([
        encoder(coords[..., dims].contiguous(), size=1).reshape(
            *lead, levels, features
        )
        for encoder, dims in zip(encoders, axes)
    ], -1)


def signal_lens(name: str, kind: str | None = None) -> str:
    if name in {"climate", "soil", "clay", "topo", "hydro", "water", "soil_drainage"}:
        return "abiotic"
    if name in {"vision_dino", "naip_rgb", "naip_ir", "alphaearth"}:
        return "visual"
    if name in {"identity", "phylo", "vision_bio"} or kind == "categorical":
        return "biological"
    return "ecological"


class FieldProjection(nn.Module):
    """Named boundary used by the canonical Earth4D ablation."""

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(source_dim, target_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NestedFieldProjection(nn.Module):
    """Keep the proven field intact while a new field earns influence."""

    def __init__(self, base: nn.Module, residual: nn.Module, base_dim: int, levels: int):
        super().__init__()
        self.base = base
        self.residual = residual
        self.base_dim = base_dim
        self.gate = nn.Parameter(torch.full((levels,), -3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base, residual = x.split((self.base_dim, x.shape[-1] - self.base_dim), -1)
        gate = torch.sigmoid(self.gate).view(*([1] * (x.dim() - 2)), -1, 1)
        return self.base(base) + gate * self.residual(residual)


class WorldMesh(nn.Module):
    """Compact persistent state addressed at several space-time resolutions."""

    def __init__(self, d_model: int, levels: int, log2_size: int, features: int = 2):
        super().__init__()
        self.levels = levels
        self.features = features
        self.spatial = hash_field(levels, features, log2_size, 16, 1.7)
        self.temporal = nn.ModuleList([
            hash_field(
                levels, features, log2_size,
                (16, 16, 4), (1.7, 1.7, 1.5),
            )
            for _ in range(3)
        ])
        self.spatial_projection = FieldProjection(features, d_model)
        self.temporal_projection = FieldProjection(3 * features, d_model)

        residual_rng = torch.random.get_rng_state()
        self.spatial_residual = hash_field(levels, 4, log2_size, 16, 1.7)
        self.temporal_residual = nn.ModuleList([
            hash_field(
                levels, 4, log2_size,
                (16, 16, 4), (1.7, 1.7, 1.5),
            )
            for _ in range(3)
        ])
        residual_s = FieldProjection(4, d_model)
        residual_t = FieldProjection(12, d_model)
        self.spatial_projection = NestedFieldProjection(
            self.spatial_projection, residual_s, features, levels
        )
        self.temporal_projection = NestedFieldProjection(
            self.temporal_projection, residual_t, 3 * features, levels
        )
        torch.random.set_rng_state(residual_rng)

    @staticmethod
    def coordinates(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, z = to_ecef(coords[..., 0], coords[..., 1], coords[..., 2])
        xyz = torch.stack((x, y, z), -1) / ECEF_NORM_FACTOR
        xyzt = torch.cat((xyz, coords[..., 3:4] * 2.0 - 1.0), -1)
        return xyz.clamp(-0.999, 0.999), xyzt.clamp(-0.999, 0.999)

    def raw(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lead = coords.shape[:-1]
        xyz, xyzt = self.coordinates(coords)
        spatial = self.spatial(xyz.contiguous(), size=1.0).reshape(*lead, self.levels, self.features)
        spatial_residual = self.spatial_residual(
            xyz.contiguous(), size=1.0
        ).reshape(*lead, self.levels, 4)
        projections = ((0, 1, 3), (1, 2, 3), (0, 2, 3))
        temporal = project_fields(
            self.temporal, xyzt, projections, self.levels, self.features
        )
        temporal_residual = project_fields(
            self.temporal_residual, xyzt, projections, self.levels, 4
        )
        return torch.cat((spatial, spatial_residual), -1), torch.cat(
            (temporal, temporal_residual), -1
        )


class MeshSpaceTimeField(nn.Module):
    """Transferable metric offsets between a query and its neighboring mesh cells."""

    def __init__(self, d_model: int, levels: int, log2_size: int):
        super().__init__()
        self.levels = levels
        self.hash = nn.ModuleList([
            hash_field(
                levels, 2, log2_size, (8, 8, 4), (1.8, 1.8, 1.5)
            )
            for _ in range(4)
        ])
        self.project = FieldProjection(8, d_model)
        self.register_buffer("window", torch.tensor((8000.0, 8000.0, 300.0, 130.0)))

    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        delta = neighbors - query.unsqueeze(1)
        north = delta[..., 0] * 111_320.0
        east = delta[..., 1] * (111_320.0 * math.cos(math.radians(37.0)))
        offset = torch.stack((north, east, delta[..., 2], delta[..., 3]), -1)
        norm = (offset / self.window).clamp(-0.999, 0.999)
        projections = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
        return self.project(project_fields(
            self.hash, norm, projections, self.levels, 2
        ))


class MeshNeighborhood(nn.Module):
    """Holder keeps the fixed evaluator's relative-Earth4D ablation meaningful."""

    def __init__(self, d_model: int, levels: int, log2_size: int):
        super().__init__()
        self.space_time = MeshSpaceTimeField(d_model, levels, log2_size)


class FiberAdapter(nn.Module):
    """Translate one measurement system into the common mesh-state language."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())
