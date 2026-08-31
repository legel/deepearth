"""Editable mesh thesis: situated signals write state; fusion reads only state.

Run through the fixed evaluator:
    python mesh_research/evaluate.py --cache /path/to/deepcal --device cuda

`data.py` and the canonical evaluator are fixed. This is the only research-editable
file: architecture, writes, fusion, objectives, optimization, and training live here.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load as load_data
from deepearth.autoresearch.main.editable_files.encoders.earth4d import ECEF_NORM_FACTOR, to_ecef
from deepearth.autoresearch.main.editable_files.encoders.hashencoder.hashgrid import HashEncoder
from deepearth.autoresearch.main.editable_files.encoders.phylogenomic import SpeciesGraph


@dataclass(frozen=True)
class Variable:
    name: str
    kind: str
    dim: int = 0
    num_classes: int = 0
    reconstruct: bool = True


@dataclass(frozen=True)
class Experiment:
    """Starting hypothesis, not a closed menu of allowed changes.

    The research loop may replace this structure, add state, or rewrite the model
    and training procedure. It is kept here only so one experiment is legible.
    """

    seed: int = int(os.environ.get("MESH_SEED", "1337"))
    steps: int = int(os.environ.get("MESH_STEPS", "1000"))
    batch: int = 256
    width: int = int(os.environ.get("MESH_WIDTH", "192"))
    levels: int = int(os.environ.get("MESH_LEVELS", "12"))
    hash_log2: int = int(os.environ.get("MESH_HASH_LOG2", "14"))
    latents: int = int(os.environ.get("MESH_LATENTS", "16"))
    layers: int = int(os.environ.get("MESH_LAYERS", "2"))
    hide_probability: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    reader_steps: int = int(os.environ.get("MESH_READER_STEPS", "100"))
    graph_learning_rate_scale: float = float(os.environ.get("MESH_GRAPH_LR_SCALE", "0.02"))
    init_checkpoint: str = os.environ.get("MESH_INIT_CHECKPOINT", "")
    reader_only: bool = os.environ.get("MESH_READER_ONLY", "0") == "1"


EXPERIMENT = Experiment()
ACTIVATION_CHECKPOINTING = os.environ.get(
    "MESH_ACTIVATION_CHECKPOINT", "1"
) == "1"
TRAIN_BFLOAT16 = os.environ.get("MESH_BFLOAT16", "0") == "1"

LENSES = ("abiotic", "visual", "biological", "ecological")
LENS_INDEX = {name: index for index, name in enumerate(LENSES)}
READER_PARAMETERS = (
    "latents", "read.", "read_norm.", "blocks.",
    "fiber_query", "fiber_read", "fiber_fuse", "fiber_fusion_gate",
    "sparse_fusion_gate", "decode_query", "decoders.", "community_metric.",
    "species_graph.",
    "poll_head.", "pollinator_reader_query", "pollinator_reader.",
    "pollinator_reader_norm.", "pollinator_reader_output_norm.",
    "pollinator_reader_gate", "pollinator_reader_cell_key",
    "pollinator_reader_level_key", "pollinator_reader_lens_key",
    "poll_transfer_head.", "pollinator_transfer_router.",
    "identity_detail_query", "identity_detail_reader.",
    "identity_detail_norm.", "identity_detail_output_norm.",
    "identity_detail_gate", "identity_detail_cell_key",
    "identity_detail_level_key", "identity_detail_lens_key",
    "lfmc_head.", "myco_head.", "species_myco_head.", "myco_relation_gate",
    "flower_head.",
    "mesh_read_query.", "mesh_read_gate.", "mesh_scale_read_gate.",
    "mesh_scale_attention_gate.",
    "task_mesh_reader.", "task_mesh_reader_gate.", "task_mesh_reader_norm.",
    "task_mesh_reader_output_norm.", "scale_mesh_reader.",
    "scale_mesh_reader_mix.", "scale_mesh_reader_router.",
    "deep_mesh_reader.", "deep_mesh_reader_gate.",
    "deep_mesh_reader_output_norm.",
    "mesh_prior_read_gate.", "mesh_prior_information_gate.",
    "mesh_task_norm.", "mesh_scale_task_norm.", "mesh_prior_task_norm.",
    "mesh_condition_gate.", "mesh_condition_norm.",
    "mesh_cell_key", "mesh_level_key", "mesh_lens_key",
    "species_niche_key", "species_niche_adapter.",
    "position_species_",
    "specialist_meshes.", "specialist_pair_mix", "specialist_fusion.",
    "specialist_fusion_gate",
    "specialist_aggregate_norm.", "specialist_output_norm.", "specialist_type",
    "raw_residual_read.", "raw_residual_gate", "raw_residual_norm.",
    "raw_residual_output_norm.", "specialist_decode_query",
    "specialist_reconstruct.",
    "relation_meshes.", "relation_pair_mix.", "relation_readers.",
    "relation_reader_norms.",
    "relation_output_norms.", "relation_query.", "relation_gate.",
    "segment_denoisers.", "segment_type.", "segment_gate.",
    "segment_fusion.", "segment_fusion_norm.",
    "segment_output_norm.", "segment_task_gate.",
)
EXPANSION_PARAMETERS = (
    "deep_mesh_reader.", "deep_mesh_reader_gate.",
    "deep_mesh_reader_output_norm.",
)
SPECIES_LENS_PARAMETERS = (
    "species_lens_reader.", "species_lens_reader_norm."
)
LFMC_LENS_PARAMETERS = (
    "lfmc_lens_reader.", "lfmc_lens_reader_norm.", "lfmc_lens_head."
)
IDENTITY_DETAIL_PARAMETERS = ("identity_detail_",)
RELATION_PARAMETERS = ("species_myco_head.", "myco_relation_gate")
CALIBRATION_PARAMETERS = ("pollinator_log_temperature",)
POSITION_PARAMETERS = ("position_species_",)


def signal_lens(name: str, kind: str | None = None) -> str:
    if name in {"climate", "worldclim", "soil", "clay", "topo", "hydro", "water", "soil_drainage"}:
        return "abiotic"
    if name in {"vision_dino", "naip_rgb", "naip_ir", "alphaearth"}:
        return "visual"
    if name in {"identity", "phylo", "vision_bio"} or kind == "categorical":
        return "biological"
    return "ecological"


class Projection(nn.Module):
    """Named boundary used by the canonical Earth4D ablation."""

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(source_dim, target_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NestedProjection(nn.Module):
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
        self.spatial = HashEncoder(
            input_dim=3,
            num_levels=levels,
            level_dim=features,
            base_resolution=16,
            per_level_scale=1.7,
            log2_hashmap_size=log2_size,
        )
        self.temporal = nn.ModuleList([
            HashEncoder(
                input_dim=3,
                num_levels=levels,
                level_dim=features,
                base_resolution=(16, 16, 4),
                per_level_scale=(1.7, 1.7, 1.5),
                log2_hashmap_size=log2_size,
            )
            for _ in range(3)
        ])
        self.spatial_projection = Projection(features, d_model)
        self.temporal_projection = Projection(3 * features, d_model)

        residual_rng = torch.random.get_rng_state()
        self.spatial_residual = HashEncoder(
            input_dim=3,
            num_levels=levels,
            level_dim=4,
            base_resolution=16,
            per_level_scale=1.7,
            log2_hashmap_size=log2_size,
        )
        self.temporal_residual = nn.ModuleList([
            HashEncoder(
                input_dim=3,
                num_levels=levels,
                level_dim=4,
                base_resolution=(16, 16, 4),
                per_level_scale=(1.7, 1.7, 1.5),
                log2_hashmap_size=log2_size,
            )
            for _ in range(3)
        ])
        residual_s = Projection(4, d_model)
        residual_t = Projection(12, d_model)
        self.spatial_projection = NestedProjection(
            self.spatial_projection, residual_s, features, levels
        )
        self.temporal_projection = NestedProjection(
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
        temporal = torch.cat([
            encoder(xyzt[..., axes].contiguous(), size=1.0).reshape(
                *lead, self.levels, self.features)
            for encoder, axes in zip(self.temporal, projections)
        ], -1)
        temporal_residual = torch.cat([
            encoder(xyzt[..., axes].contiguous(), size=1.0).reshape(
                *lead, self.levels, 4
            )
            for encoder, axes in zip(self.temporal_residual, projections)
        ], -1)
        return torch.cat((spatial, spatial_residual), -1), torch.cat(
            (temporal, temporal_residual), -1
        )


class RelativeField(nn.Module):
    """Transferable metric offsets between a query and its neighboring mesh cells."""

    def __init__(self, d_model: int, levels: int, log2_size: int):
        super().__init__()
        self.levels = levels
        self.hash = nn.ModuleList([
            HashEncoder(
                input_dim=3,
                num_levels=levels,
                level_dim=2,
                base_resolution=(8, 8, 4),
                per_level_scale=(1.8, 1.8, 1.5),
                log2_hashmap_size=log2_size,
            )
            for _ in range(4)
        ])
        self.project = Projection(8, d_model)
        self.register_buffer("window", torch.tensor((8000.0, 8000.0, 300.0, 130.0)))

    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        delta = neighbors - query.unsqueeze(1)
        north = delta[..., 0] * 111_320.0
        east = delta[..., 1] * (111_320.0 * math.cos(math.radians(37.0)))
        offset = torch.stack((north, east, delta[..., 2], delta[..., 3]), -1)
        norm = (offset / self.window).clamp(-0.999, 0.999)
        projections = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
        raw = torch.cat([
            encoder(norm[..., axes].contiguous(), size=1.0).reshape(
                *norm.shape[:-1], self.levels, 2)
            for encoder, axes in zip(self.hash, projections)
        ], -1)
        return self.project(raw)


class Neighborhood(nn.Module):
    """Holder keeps the fixed evaluator's relative-Earth4D ablation meaningful."""

    def __init__(self, d_model: int, levels: int, log2_size: int):
        super().__init__()
        self.space_time = RelativeField(d_model, levels, log2_size)


class SignalAdapter(nn.Module):
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


class CrossFiberReaderBlock(nn.Module):
    """Refine one scientific query against routed mesh fibers."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        update = self.attention(
            self.query_norm(query).unsqueeze(1),
            self.token_norm(keys),
            self.token_norm(values),
            need_weights=False,
        )[0].squeeze(1)
        query = query + update
        return query + self.mlp(self.mlp_norm(query))


class SpecialistMesh(nn.Module):
    """One independently parameterized graph and fusion stream over Earth4D cells."""

    def __init__(self, d_model: int, n_heads: int, n_latents: int = 4):
        super().__init__()
        self.cell_message = nn.Linear(d_model, d_model, bias=False)
        self.coarse_message = nn.Linear(d_model, d_model, bias=False)
        self.fine_message = nn.Linear(d_model, d_model, bias=False)
        self.graph_norm = nn.LayerNorm(d_model)
        self.graph_gate = nn.Parameter(torch.tensor(0.1))
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.read_norm = nn.LayerNorm(d_model)
        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fuse = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = state[:, :1]
        neighbors = state[:, 1:]
        neighbor_mean = neighbors.mean(1, keepdim=True) if neighbors.shape[1] else query
        cell_context = torch.cat((neighbor_mean, query.expand(-1, neighbors.shape[1], -1, -1)), 1)
        coarse = torch.cat((torch.zeros_like(state[:, :, :1]), state[:, :, :-1]), 2)
        fine = torch.cat((state[:, :, 1:], torch.zeros_like(state[:, :, :1])), 2)
        message = self.cell_message(cell_context) \
                  + self.coarse_message(coarse) + self.fine_message(fine)
        state = state + torch.tanh(self.graph_gate) * self.graph_norm(message)
        tokens = self.read_norm(state.flatten(1, 2))
        latent = self.latents.unsqueeze(0).expand(state.shape[0], -1, -1)
        latent = latent + self.read(latent, tokens, tokens, need_weights=False)[0]
        return state, self.fuse(latent)


class SegmentDenoiser(nn.Module):
    """Retrieve and clean one query-local view without rewriting source state."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        levels: int,
        *,
        token_drop: float,
        cell_drop: float,
        level_drop: float,
        jitter: float,
        top_k: int = 8,
    ):
        super().__init__()
        self.levels = levels
        self.top_k = top_k
        self.token_drop = token_drop
        self.cell_drop = cell_drop
        self.level_drop = level_drop
        self.jitter = jitter
        self.token_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cell_key = nn.Parameter(torch.zeros(2, d_model))
        self.level_key = nn.Parameter(torch.zeros(levels, d_model))
        self.latents = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, 4 * d_model,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            for _ in range(2)
        ])
        self.output = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def _keep_mask(
        self, index: torch.Tensor, cells: int
    ) -> torch.Tensor:
        batch, selected = index.shape
        device = index.device
        token = torch.rand(batch, selected, device=device) >= self.token_drop
        cell = torch.rand(batch, cells, device=device) >= self.cell_drop
        level = torch.rand(batch, self.levels, device=device) >= self.level_drop
        keep = token \
               & cell.gather(1, index.div(self.levels, rounding_mode="floor")) \
               & level.gather(1, index.remainder(self.levels))
        keep[:, 0] = True
        return keep

    def forward(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch, cells, levels, width = state.shape
        tokens = self.token_norm(state.flatten(1, 2))
        cell_type = (torch.arange(cells, device=state.device) > 0).long()
        address = (
            self.cell_key[cell_type].view(1, cells, 1, width)
            + self.level_key.view(1, 1, levels, width)
        ).flatten(1, 2)
        keys = tokens + address
        score = torch.einsum(
            "bkd,bd->bk", keys, self.query_norm(query)
        ) / math.sqrt(width)
        _, index = score.topk(min(self.top_k, score.shape[-1]), dim=-1)
        gather = index[..., None].expand(-1, -1, width)
        selected_keys = keys.gather(1, gather)
        selected_values = tokens.gather(1, gather)
        keep = None
        if self.training:
            keep = self._keep_mask(index, cells)
            selected_values = selected_values + self.jitter * torch.randn_like(
                selected_values
            )
        latent = query.unsqueeze(1) + self.latents.unsqueeze(0)
        latent = latent + self.read(
            latent, selected_keys, selected_values,
            key_padding_mask=None if keep is None else ~keep,
            need_weights=False,
        )[0]
        increment = self.jitter / math.sqrt(len(self.blocks))
        for block in self.blocks:
            latent = block(latent)
            if self.training:
                latent = latent + increment * torch.randn_like(latent)
        return self.output(
            query.unsqueeze(1), latent, latent, need_weights=False
        )[0].squeeze(1)


class MeshModel(nn.Module):
    """All scientific evidence must enter fusion through a mesh-state update."""

    def __init__(
        self,
        variables: Sequence[Variable],
        always_dims: Dict[str, int],
        source,
        *,
        d_model: int = 128,
        levels: int = 12,
        log2_size: int = 14,
        n_latents: int = 16,
        n_layers: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.variables = list(variables)
        self.names = [v.name for v in variables]
        self.d_model = d_model
        self.levels = levels
        self.species_variable = "identity"
        has_worldclim = "worldclim" in source.extra
        self.always_names = (*always_dims, *(("worldclim",) if has_worldclim else ()))
        self._ablate_species = False

        self.mesh = WorldMesh(d_model, levels, log2_size)
        self.absolute_proj_s = self.mesh.spatial_projection
        self.absolute_proj_t = self.mesh.temporal_projection
        self.neighbors = Neighborhood(d_model, levels, max(10, log2_size - 2))

        self.adapters = nn.ModuleDict()
        self.category_inputs = nn.ModuleDict()
        for v in variables:
            if v.kind == "continuous":
                self.adapters[v.name] = SignalAdapter(v.dim, d_model)
            elif v.name != self.species_variable:
                self.category_inputs[v.name] = nn.Embedding(v.num_classes, d_model)
        for name, dim in always_dims.items():
            self.adapters[name] = SignalAdapter(dim, d_model)
        if has_worldclim:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260824)
                self.adapters["worldclim"] = SignalAdapter(
                    int(source.extra["worldclim"][2]), d_model
                )

        graph_args = dict(n_species=source.n_classes, d_model=d_model, n_layers=2, n_heads=4)
        if source.lca_tree is not None:
            graph_args.update(operator="latent-clade", tree=source.lca_tree,
                              tip_row=source.lca_tip_row, species_text=source.species_text)
        else:
            distance = SpeciesGraph.distance_from_embedding(source.phylo)
            graph_args.update(operator="ou-attention", phylo_distance=distance,
                              top_k=min(128, source.n_classes), species_text=source.species_text)
        self.species_graph = SpeciesGraph(**graph_args)
        self._refined_species = None
        self.species_niche_key = nn.Parameter(
            torch.zeros(source.n_classes, d_model)
        )
        niche_rng = torch.random.get_rng_state()
        self.species_niche_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.species_niche_adapter[-1].weight)
        nn.init.zeros_(self.species_niche_adapter[-1].bias)
        torch.random.set_rng_state(niche_rng)
        position_devices = list(range(torch.cuda.device_count())) \
                           if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=position_devices):
            torch.manual_seed(20260831)
            self.position_species_level = nn.Parameter(torch.zeros(levels))
            self.position_species_adapter = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
            nn.init.zeros_(self.position_species_adapter[-1].weight)
            self.position_species_rgb_adapter = SignalAdapter(
                int(source.naip_patch_dim), d_model
            )
            self.position_species_ir_adapter = SignalAdapter(
                int(source.naip_patch_dim), d_model
            ) if int(source.naip_patch_views) > 1 else None
            self.position_species_local = nn.Sequential(
                nn.Conv2d(
                    d_model, d_model, 3, padding=1,
                    groups=d_model, bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, 1, bias=False),
            )
            nn.init.zeros_(self.position_species_local[-1].weight)
            self.position_species_scale_type = nn.Parameter(
                torch.randn(3, d_model) * 0.02
            )
            self.position_species_patch_norm = nn.LayerNorm(d_model)
            self.position_species_patch_attention = nn.MultiheadAttention(
                d_model, 4, batch_first=True
            )
            self.position_species_patch_output = nn.LayerNorm(d_model)
            self.position_species_gate = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            "position_species_patch_tokens",
            source.naip_patch_tokens,
            persistent=False,
        )
        patch_coords = getattr(source, "naip_patch_coords", None)
        if patch_coords is not None:
            self.register_buffer(
                "position_species_patch_coords",
                patch_coords,
                persistent=False,
            )
        else:
            self.position_species_patch_coords = None
        fine_axis = torch.linspace(83.125, -83.125, 8)
        fine_north, fine_east = torch.meshgrid(
            fine_axis, -fine_axis, indexing="ij"
        )
        mid_axis = torch.tensor((71.25, 23.75, -23.75, -71.25))
        mid_north, mid_east = torch.meshgrid(
            mid_axis, -mid_axis, indexing="ij"
        )
        self.register_buffer(
            "position_species_patch_offsets",
            torch.stack((fine_north.flatten(), fine_east.flatten()), -1),
            persistent=False,
        )
        self.register_buffer(
            "position_species_mid_offsets",
            torch.stack((mid_north.flatten(), mid_east.flatten()), -1),
            persistent=False,
        )
        lens_rng = torch.random.get_rng_state()
        self.species_lens_reader_norm = nn.LayerNorm(d_model)
        self.species_lens_reader = nn.MultiheadAttention(
            d_model, 4, batch_first=True
        )
        nn.init.zeros_(self.species_lens_reader.out_proj.weight)
        nn.init.zeros_(self.species_lens_reader.out_proj.bias)
        torch.random.set_rng_state(lens_rng)
        self.register_buffer("species_family", source.class_group)
        self.family_count = len(source.group_names)
        self.environment_names = tuple(
            name for name in ("climate", "soil", "naip_rgb", "naip_ir", "clay", "topo", "chm", "hydro")
            if name in self.names
        )

        base_write_names = [*self.names, *always_dims]
        write_names = [*base_write_names, *(("worldclim",) if has_worldclim else ())]
        self.write_names = tuple(write_names)
        self.write_type = nn.ParameterDict({
            n: nn.Parameter(torch.randn(d_model) * 0.02)
            for n in base_write_names
        })
        if has_worldclim:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260825)
                self.write_type["worldclim"] = nn.Parameter(
                    torch.randn(d_model) * 0.02
                )
        residual_rng = torch.random.get_rng_state()
        self.fiber_residual = nn.ModuleDict({
            name: nn.Linear(d_model, d_model, bias=False)
            for name in write_names
        })
        for residual in self.fiber_residual.values():
            nn.init.zeros_(residual.weight)
        torch.random.set_rng_state(residual_rng)
        self.write_gate = nn.ParameterDict({n: nn.Parameter(torch.zeros(levels)) for n in write_names})
        self.write_norm = nn.LayerNorm(d_model)
        self.neighbor_norm = nn.LayerNorm(d_model)

        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.read_norm = nn.LayerNorm(d_model)
        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.decode_query = nn.Parameter(torch.randn(len(variables), d_model) * 0.02)
        self.decoders = nn.ModuleDict()
        for v in variables:
            if v.name == self.species_variable:
                continue
            width = v.dim if v.kind == "continuous" else v.num_classes
            self.decoders[v.name] = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, width)
            )

        self.community_head = nn.Linear(d_model, source.n_classes)
        self.poll_head = nn.Linear(d_model, source.n_pollinators) if hasattr(source, "n_pollinators") else None
        self.pollinator_reader = None
        self.poll_transfer_head = None
        if self.poll_head is not None:
            transfer_rng = torch.random.get_rng_state()
            self.poll_transfer_head = nn.Linear(d_model, source.n_pollinators)
            self.poll_transfer_head.load_state_dict(self.poll_head.state_dict())
            self.pollinator_transfer_router = nn.Linear(1, 1)
            nn.init.zeros_(self.pollinator_transfer_router.weight)
            nn.init.constant_(self.pollinator_transfer_router.bias, -2.0)
            torch.random.set_rng_state(transfer_rng)
            self.pollinator_log_temperature = nn.Parameter(
                torch.zeros(()), requires_grad=False
            )
            self.register_buffer("poll_species_idx", source.poll_idx.long(), persistent=False)
            self.register_buffer("poll_species_frq", source.poll_frq.float(), persistent=False)
            interaction_rng = torch.random.get_rng_state()
            self.pollinator_reader_query = nn.Parameter(torch.randn(2, d_model) * 0.02)
            self.pollinator_reader_norm = nn.LayerNorm(d_model)
            self.pollinator_reader = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )
            self.pollinator_reader_output_norm = nn.LayerNorm(d_model)
            self.pollinator_reader_gate = nn.Parameter(torch.tensor(0.05))
            self.pollinator_reader_cell_key = nn.Parameter(torch.zeros(2, d_model))
            self.pollinator_reader_level_key = nn.Parameter(torch.zeros(levels, d_model))
            self.pollinator_reader_lens_key = nn.Parameter(
                torch.zeros(len(LENSES), d_model)
            )
            torch.random.set_rng_state(interaction_rng)
        identity_reader_rng = torch.random.get_rng_state()
        self.identity_detail_query = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.identity_detail_norm = nn.LayerNorm(d_model)
        self.identity_detail_reader = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.identity_detail_output_norm = nn.LayerNorm(d_model)
        self.identity_detail_gate = nn.Parameter(torch.tensor(0.05))
        self.identity_detail_cell_key = nn.Parameter(torch.zeros(2, d_model))
        self.identity_detail_level_key = nn.Parameter(torch.zeros(levels, d_model))
        self.identity_detail_lens_key = nn.Parameter(
            torch.zeros(len(LENSES), d_model)
        )
        torch.random.set_rng_state(identity_reader_rng)
        self.lfmc_head = nn.Linear(d_model, 1) if hasattr(source, "lfmc") else None
        if self.lfmc_head is not None:
            lfmc_reader_rng = torch.random.get_rng_state()
            self.lfmc_lens_reader_norm = nn.LayerNorm(d_model)
            self.lfmc_lens_reader = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )
            self.lfmc_lens_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.lfmc_lens_head.weight)
            nn.init.zeros_(self.lfmc_lens_head.bias)
            torch.random.set_rng_state(lfmc_reader_rng)
        self.myco_head = nn.Linear(d_model, 5) if hasattr(source, "myco") else None
        self.flower_head = nn.Linear(d_model, 1) if hasattr(source, "flower") else None
        self.species_myco_head = None
        if self.myco_head is not None:
            myco_rng = torch.random.get_rng_state()
            self.species_myco_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 5)
            )
            self.myco_relation_gate = nn.Parameter(torch.tensor(math.atanh(0.75)))
            torch.random.set_rng_state(myco_rng)
            train_species = torch.zeros(
                source.n_classes, dtype=torch.bool, device=source.cls.device
            )
            train_species[source.cls[source.train]] = True
            valid = source.myco_valid.bool() & train_species
            counts = torch.bincount(
                source.myco[valid].long(), minlength=5
            ).to(torch.float32)
            self.register_buffer("species_myco", source.myco.long())
            self.register_buffer("species_myco_valid", valid)
            self.register_buffer(
                "species_myco_prior", counts / counts.sum().clamp_min(1.0)
            )
        self.community_metric = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        variable_kind = {v.name: v.kind for v in variables}
        self.write_lens = {
            name: LENS_INDEX[signal_lens(name, variable_kind.get(name))]
            for name in write_names
        }
        sidecar_rng = torch.random.get_rng_state()
        self.fiber_level_gate = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(levels)) for name in write_names
        })
        self.fiber_reliability = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(())) for name in write_names
        })
        self.fiber_type = nn.Parameter(torch.randn(len(LENSES), d_model) * 0.02)
        self.fiber_prior = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
            for _ in LENSES
        ])
        self.fiber_information_gate = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.fiber_norm = nn.LayerNorm(d_model)
        self.fiber_latents = 4
        self.fiber_query = nn.Parameter(
            torch.randn(len(LENSES), self.fiber_latents, d_model) * 0.02
        )
        self.fiber_decode_query = nn.Parameter(torch.randn(len(variables), d_model) * 0.02)
        scientific_reads = ["community"]
        if self.poll_head is not None:
            scientific_reads.append("pollinator")
        if self.lfmc_head is not None:
            scientific_reads.append("lfmc")
        if self.myco_head is not None:
            scientific_reads.append("myco")
        if self.flower_head is not None:
            scientific_reads.append("flower")
        self.mesh_read_names = (*self.names, *scientific_reads)
        self.mesh_read_query = nn.ParameterDict({
            name: nn.Parameter(torch.randn(d_model) * 0.02)
            for name in self.mesh_read_names
        })
        self.mesh_read_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.05))
            for name in self.mesh_read_names
        })
        self.mesh_scale_read_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.05))
            for name in self.mesh_read_names
        })
        self.mesh_scale_attention_gate = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in self.mesh_read_names
        })
        task_reader_rng = torch.random.get_rng_state()
        self.task_mesh_reader = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        torch.random.set_rng_state(task_reader_rng)
        scale_reader_rng = torch.random.get_rng_state()
        self.scale_mesh_reader = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        torch.random.set_rng_state(scale_reader_rng)
        deep_reader_rng = torch.random.get_rng_state()
        self.deep_mesh_reader = nn.ModuleList([
            CrossFiberReaderBlock(d_model, n_heads) for _ in range(4)
        ])
        self.deep_mesh_reader_gate = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in self.mesh_read_names
        })
        self.deep_mesh_reader_output_norm = nn.LayerNorm(d_model)
        torch.random.set_rng_state(deep_reader_rng)
        self.scale_mesh_reader_mix = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(
                -2.0 if name == self.species_variable else 0.0
            ))
            for name in self.mesh_read_names
        })
        router_rng = torch.random.get_rng_state()
        self.scale_mesh_reader_router = nn.Sequential(
            nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, 1)
        )
        nn.init.zeros_(self.scale_mesh_reader_router[-1].weight)
        nn.init.zeros_(self.scale_mesh_reader_router[-1].bias)
        torch.random.set_rng_state(router_rng)
        self.task_mesh_reader_gate = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in self.mesh_read_names
        })
        self.task_mesh_reader_norm = nn.LayerNorm(d_model)
        self.task_mesh_reader_output_norm = nn.LayerNorm(d_model)
        self.mesh_prior_read_gate = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in self.mesh_read_names
        })
        information_rng = torch.random.get_rng_state()
        self.mesh_prior_information_gate = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        torch.random.set_rng_state(information_rng)
        conditioned_reads = [name for name in ("pollinator",) if name in self.mesh_read_names]
        self.mesh_condition_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.05)) for name in conditioned_reads
        })
        self.mesh_task_norm = nn.LayerNorm(d_model)
        self.mesh_scale_task_norm = nn.LayerNorm(d_model)
        self.mesh_prior_task_norm = nn.LayerNorm(d_model)
        self.mesh_condition_norm = nn.LayerNorm(d_model)
        self.mesh_cell_key = nn.Parameter(torch.zeros(2, d_model))
        self.mesh_level_key = nn.Parameter(torch.zeros(levels, d_model))
        self.mesh_lens_key = nn.Parameter(torch.zeros(len(LENSES), d_model))
        self.fiber_read_norm = nn.LayerNorm(d_model)
        self.fiber_read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fiber_fuse_norm = nn.LayerNorm(d_model)
        self.fiber_fuse = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fiber_reconstruct = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
            for name in base_write_names
        })
        if has_worldclim:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260826)
                self.fiber_reconstruct["worldclim"] = nn.Sequential(
                    nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
                )
        self.fiber_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.sparse_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.coarse_scale_exchange = nn.Linear(d_model, d_model, bias=False)
        self.fine_scale_exchange = nn.Linear(d_model, d_model, bias=False)
        self.scale_exchange_gate = nn.Parameter(torch.full((len(LENSES),), 0.05))
        self.scale_message_norm = nn.LayerNorm(d_model)
        self.mesh_linear_reconstruct = nn.ModuleDict({
            name: nn.Linear(d_model, d_model, bias=False)
            for name in base_write_names
        })
        if has_worldclim:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260827)
                self.mesh_linear_reconstruct["worldclim"] = nn.Linear(
                    d_model, d_model, bias=False
                )
        self.lens_exchange_norm = nn.LayerNorm(d_model)
        self.lens_exchange = nn.Parameter(
            torch.zeros(levels, len(LENSES), len(LENSES))
        )
        torch.random.set_rng_state(sidecar_rng)
        self._fiber_summary = None
        self._fiber_mesh = None
        self._fiber_prior_mesh = None
        self._latest_fiber_prior = None
        self._pool_cache = {}
        self._mesh_reader_cache = None
        specialist_rng = torch.random.get_rng_state()
        self.specialist_meshes = nn.ModuleList([
            nn.ModuleList([
                SpecialistMesh(d_model, n_heads, n_latents=2)
                for _ in range(2)
            ])
            for _ in LENSES
        ])
        self.specialist_pair_mix = nn.Parameter(
            torch.zeros(len(LENSES), 2)
        )
        self.specialist_type = nn.Parameter(
            torch.randn(len(LENSES), d_model) * 0.02
        )
        self.specialist_aggregate_norm = nn.LayerNorm(d_model)
        self.specialist_fusion = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.specialist_output_norm = nn.LayerNorm(d_model)
        self.specialist_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.raw_residual_norm = nn.LayerNorm(d_model)
        self.raw_residual_read = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.raw_residual_output_norm = nn.LayerNorm(d_model)
        self.raw_residual_gate = nn.Parameter(torch.tensor(0.05))
        self.specialist_decode_query = nn.Parameter(
            torch.randn(len(variables), d_model) * 0.02
        )
        self.specialist_reconstruct = nn.ModuleDict({
            name: nn.Linear(d_model, d_model) for name in self.names
        })
        relation_names = ["identity"]
        if self.poll_head is not None:
            relation_names.extend(("pollinator", "pollinator_transfer"))
        if self.myco_head is not None:
            relation_names.append("myco")
        self.relation_names = tuple(relation_names)
        self.relation_meshes = nn.ModuleDict({
            name: nn.ModuleList([
                SpecialistMesh(d_model, n_heads, n_latents=2)
                for _ in range(2)
            ])
            for name in self.relation_names
        })
        self.relation_pair_mix = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(2))
            for name in self.relation_names
        })
        self.relation_readers = nn.ModuleDict({
            name: nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for name in self.relation_names
        })
        self.relation_reader_norms = nn.ModuleDict({
            name: nn.LayerNorm(d_model) for name in self.relation_names
        })
        self.relation_output_norms = nn.ModuleDict({
            name: nn.LayerNorm(d_model) for name in self.relation_names
        })
        self.relation_query = nn.ParameterDict({
            name: nn.Parameter(torch.randn(d_model) * 0.02)
            for name in self.relation_names
        })
        self.relation_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.05))
            for name in self.relation_names
        })
        corruption = {
            "abiotic": (0.05, 0.05, 0.15, 0.010),
            "visual": (0.15, 0.10, 0.10, 0.020),
            "biological": (0.08, 0.08, 0.10, 0.010),
            "ecological": (0.15, 0.12, 0.15, 0.020),
            "identity": (0.08, 0.08, 0.10, 0.010),
            "pollinator": (0.15, 0.12, 0.15, 0.020),
            "pollinator_transfer": (0.10, 0.10, 0.12, 0.015),
            "myco": (0.10, 0.10, 0.12, 0.015),
        }
        segment_names = (*LENSES, *self.relation_names)
        self.segment_denoisers = nn.ModuleDict({
            name: SegmentDenoiser(
                d_model, n_heads, levels,
                token_drop=corruption[name][0],
                cell_drop=corruption[name][1],
                level_drop=corruption[name][2],
                jitter=corruption[name][3],
            )
            for name in segment_names
        })
        self.segment_type = nn.ParameterDict({
            name: nn.Parameter(torch.randn(d_model) * 0.02)
            for name in segment_names
        })
        self.segment_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(math.atanh(0.5)))
            for name in segment_names
        })
        self.segment_fusion_norm = nn.LayerNorm(d_model)
        self.segment_fusion = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.segment_output_norm = nn.LayerNorm(d_model)
        trait_reads = {
            "seasonality", "water", "soil_drainage", "form",
            "plant_type", "growth_rate", "sun", "ease_of_care",
        }
        denoised_reads = {
            self.species_variable, "community", "pollinator",
            "lfmc", "myco", "flower", *trait_reads,
        }
        self.segment_task_gate = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.1))
            for name in self.mesh_read_names
            if name in denoised_reads
        })
        torch.random.set_rng_state(specialist_rng)
        self._specialist_mesh = None
        self._specialist_latents = None
        self._relation_mesh = {}
        self._relation_latents = {}
        self._denoised_pool_cache = {}
        self._identity_graph_uncertainty = None
        self._pollinator_route = None
        self._raw_state_tokens = None
        self._raw_state_mask = None
        self._position_species_state = None
        self._position_species_patch_state = None
        self._position_species_patch_mask = None
        self._position_species_patch_valid = None

    @staticmethod
    def _mesh_pair(
        meshes: nn.ModuleList,
        state: torch.Tensor,
        mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branches = [mesh(state) for mesh in meshes]
        weight = mix.softmax(0)
        combined = state + sum(
            weight[index] * (branch_state - state)
            for index, (branch_state, _) in enumerate(branches)
        )
        latent = torch.cat([branch_latent for _, branch_latent in branches], 1)
        return combined, latent

    def _species(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        refined = self.species_graph._seed() if self._ablate_species else self.species_graph(mask)
        self._refined_species = refined
        return refined

    def _adapt(self, name: str, value: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        if name == self.species_variable:
            return species[value.long().clamp(0, species.shape[0] - 1)]
        if name in self.adapters:
            return self.adapters[name](value)
        return self.category_inputs[name](value.long().clamp_min(0))

    def _raw_residuals(
        self,
        state: torch.Tensor,
        values: Dict[str, torch.Tensor],
        present: Dict[str, torch.Tensor],
        species: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep one situated residual per modality after specialist aggregation."""
        address = state.mean(-2)
        tokens, masks = [], []
        for name in self.write_names:
            if name == "worldclim":
                continue
            if name not in values or name not in present:
                continue
            valid = present[name].bool()
            token = (
                self._adapt(name, values[name], species) + self.write_type[name] + address
            ).detach()
            tokens.append(token * valid.unsqueeze(-1).to(token.dtype))
            masks.append(valid)
        return torch.stack(tokens, 1), torch.stack(masks, 1)

    def _write(
        self,
        state: torch.Tensor,
        values: Dict[str, torch.Tensor],
        present: Dict[str, torch.Tensor],
        species: torch.Tensor,
    ) -> torch.Tensor:
        updates = torch.zeros_like(state)
        count = state.new_zeros((*state.shape[:-2], 1, 1))
        for name, mask in present.items():
            if name == "worldclim":
                continue
            if name not in values or name not in self.write_gate:
                continue
            edit = self._adapt(name, values[name], species) + self.write_type[name]
            gate = torch.sigmoid(self.write_gate[name]).view(*([1] * (state.dim() - 2)), self.levels, 1)
            valid = mask.to(state.dtype).view(*mask.shape, 1, 1)
            updates = updates + valid * gate * edit.unsqueeze(-2)
            count = count + valid
        return self.write_norm(state + updates / count.clamp_min(1.0).sqrt())

    def _fiber_write(
        self,
        state: torch.Tensor,
        values: Dict[str, torch.Tensor],
        present: Dict[str, torch.Tensor],
        species: torch.Tensor,
    ) -> torch.Tensor:
        priors = torch.stack([
            prior(state) for prior in self.fiber_prior
        ], -2)
        fiber_type = self.fiber_type.view(
            *([1] * (priors.dim() - 2)), len(LENSES), self.d_model
        )
        priors = priors + fiber_type
        self._latest_fiber_prior = self.fiber_norm(priors)
        updates = [torch.zeros_like(state) for _ in LENSES]
        precision = [state.new_zeros((*state.shape[:-1], 1)) for _ in LENSES]
        for name, mask in present.items():
            if name not in values or name not in self.fiber_level_gate:
                continue
            lens = self.write_lens[name]
            prior = priors[..., lens, :]
            adapted = self._adapt(name, values[name], species).detach()
            evidence = (
                adapted + self.fiber_residual[name](adapted) + self.write_type[name]
            ).unsqueeze(-2)
            evidence = evidence.expand_as(prior)
            innovation = evidence - prior
            features = torch.cat((prior, evidence, prior * evidence, innovation.abs()), -1)
            gate = self.fiber_information_gate(features)
            level_gate = self.fiber_level_gate[name].view(
                *([1] * (prior.dim() - 2)), self.levels, 1
            )
            gate = torch.sigmoid(gate + level_gate + self.fiber_reliability[name])
            valid = mask.to(state.dtype).view(*mask.shape, 1, 1)
            weight = valid * gate
            updates[lens] = updates[lens] + weight * innovation
            precision[lens] = precision[lens] + weight
        fibers = self.fiber_norm(torch.stack([
            updates[index] / (1.0 + precision[index])
            for index in range(len(LENSES))
        ], -2))
        coarse = torch.cat((
            torch.zeros_like(fibers[..., :1, :, :]),
            fibers[..., :-1, :, :],
        ), -3)
        fine = torch.cat((
            fibers[..., 1:, :, :],
            torch.zeros_like(fibers[..., :1, :, :]),
        ), -3)
        exchange = self.coarse_scale_exchange(coarse) + self.fine_scale_exchange(fine)
        gate = torch.tanh(self.scale_exchange_gate).view(
            *([1] * (fibers.dim() - 2)), len(LENSES), 1
        )
        fibers = fibers + gate * self.scale_message_norm(exchange)
        off_diagonal = 1.0 - torch.eye(
            len(LENSES), device=fibers.device, dtype=fibers.dtype
        )
        normalized_lenses = self.lens_exchange_norm(fibers)
        availability = normalized_lenses.square().mean(-1).clamp(0.0, 1.0)
        agreement = torch.einsum(
            "...lid,...ljd->...lij", normalized_lenses, normalized_lenses
        ) / self.d_model
        evidence_gate = availability.unsqueeze(-1) * availability.unsqueeze(-2) \
                        * torch.sigmoid(4.0 * agreement)
        lens_exchange = torch.tanh(self.lens_exchange) * off_diagonal * evidence_gate
        lens_message = torch.einsum(
            "...lid,...lij->...ljd", normalized_lenses, lens_exchange
        )
        return fibers + lens_message

    def context(self, query_coords, neighbor_coords, manifold_positions=None, neighbor_values=None):
        spatial, temporal = self.mesh.raw(query_coords)
        query = self.absolute_proj_s(spatial) + self.absolute_proj_t(temporal)

        n_spatial, n_temporal = self.mesh.raw(neighbor_coords)
        neighbor = self.absolute_proj_s(n_spatial) + self.absolute_proj_t(n_temporal)
        relative = self.neighbors.space_time(query_coords, neighbor_coords)
        # The fixed ablation returns one zero vector per neighbor; the live field
        # retains a separate vector per resolution level.
        if relative.dim() == neighbor.dim() - 1:
            relative = relative.unsqueeze(-2)
        neighbor = neighbor + relative
        return {
            "coordinates": query_coords,
            "position_s": query.mean(-2),
            "position_t": self.absolute_proj_t(temporal).mean(-2),
            "position": query.mean(-2),
            "query_state": query,
            "neighbor_state": neighbor,
            "neighbor_values": neighbor_values or {},
        }

    def encode(
        self,
        values: Dict[str, torch.Tensor],
        present: Dict[str, torch.Tensor],
        context: dict,
        detach_species: bool = False,
        species_mask: torch.Tensor | None = None,
    ):
        self._position_species_state = context["query_state"].detach()
        self._position_species_patch_state = self._positioned_patch_state(
            context, values, present
        )
        species = self._species(species_mask)
        identity = values.get(self.species_variable)
        identity_present = present.get(self.species_variable)
        if identity is not None and identity_present is not None:
            index = identity.long().clamp(0, species.shape[0] - 1)
            seed = self.species_graph._seed().detach()[index]
            uncertainty = 1.0 - F.cosine_similarity(
                species.detach()[index], seed, dim=-1
            )
            self._identity_graph_uncertainty = uncertainty.mul(
                identity_present.to(uncertainty.dtype)
            ).unsqueeze(-1)
        else:
            self._identity_graph_uncertainty = None
        if detach_species:
            species = species.detach()
        write_mask = dict(present)
        for name in self.always_names:
            if name in values:
                valid = values[name].isfinite().all(-1) & (values[name].norm(dim=-1) > 1e-6)
                if name == "worldclim":
                    valid &= present.get(name, torch.zeros_like(valid))
                    valid &= present.get("climate", torch.zeros_like(valid))
                write_mask[name] = valid
        self._raw_state_tokens, self._raw_state_mask = self._raw_residuals(
            context["query_state"], values, write_mask, species
        )
        query_fibers = self._fiber_write(context["query_state"], values, write_mask, species)
        query_priors = self._latest_fiber_prior
        query = self._write(context["query_state"], values, write_mask, species)

        neighbor = context["neighbor_state"]
        neighbor_values = context["neighbor_values"]
        masks = {}
        if neighbor_values:
            masks = {name: torch.ones(value.shape[:-1] if value.dim() > 2 else value.shape,
                                      dtype=torch.bool, device=value.device)
                     for name, value in neighbor_values.items()}
        neighbor_fibers = self._fiber_write(neighbor, neighbor_values, masks, species)
        neighbor_priors = self._latest_fiber_prior
        if neighbor_values:
            neighbor = self._write(neighbor, neighbor_values, masks, species)
        neighbor = self.neighbor_norm(neighbor).flatten(1, 2)
        tokens = torch.cat((query, neighbor), 1)

        latent = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        latent = latent + self.read(latent, self.read_norm(tokens), self.read_norm(tokens), need_weights=False)[0]
        for block in self.blocks:
            latent = block(latent)
        fiber_mesh = torch.cat((query_fibers.unsqueeze(1), neighbor_fibers), 1)
        self._fiber_mesh = fiber_mesh
        self._fiber_prior_mesh = torch.cat((query_priors.unsqueeze(1), neighbor_priors), 1)
        fiber_tokens = fiber_mesh.permute(0, 3, 1, 2, 4).flatten(2, 3).flatten(0, 1)
        fiber_query = self.fiber_query.unsqueeze(0).expand(latent.shape[0], -1, -1, -1).flatten(0, 1)
        normalized = self.fiber_read_norm(fiber_tokens)
        fiber_summary = fiber_query + self.fiber_read(
            fiber_query, normalized, normalized, need_weights=False
        )[0]
        fiber_summary = fiber_summary.reshape(
            latent.shape[0], len(LENSES), self.fiber_latents, self.d_model
        )
        self._fiber_summary = fiber_summary
        self._pool_cache = {}
        self._denoised_pool_cache = {}
        self._mesh_reader_cache = None
        normalized = self.fiber_fuse_norm(fiber_summary.flatten(1, 2))
        latent = latent + torch.tanh(self.fiber_fusion_gate) * self.fiber_fuse(
            latent.detach(), normalized, normalized, need_weights=False
        )[0]
        mesh_tokens = fiber_mesh.flatten(1, 3)
        normalized = self.fiber_fuse_norm(mesh_tokens)
        score = torch.einsum(
            "bld,bkd->blk", latent.detach(), normalized
        ) / math.sqrt(self.d_model)
        dense_weight = score.softmax(-1)
        selected_score, selected = score.topk(min(16, score.shape[-1]), dim=-1)
        sparse_weight = torch.zeros_like(score).scatter(
            -1, selected, selected_score.softmax(-1).to(score.dtype)
        )
        route = sparse_weight.detach() + dense_weight - dense_weight.detach()
        mesh_read = torch.einsum("blk,bkd->bld", route, mesh_tokens)
        latent = latent + torch.tanh(self.sparse_fusion_gate) * mesh_read

        specialist_states, specialist_latents = [], []
        for lens, specialists in enumerate(self.specialist_meshes):
            state, expert = self._mesh_pair(
                specialists,
                fiber_mesh[..., lens, :],
                self.specialist_pair_mix[lens],
            )
            specialist_states.append(state)
            specialist_latents.append(expert)
        self._specialist_mesh = torch.stack(specialist_states, 3)
        self._specialist_latents = torch.stack(specialist_latents, 1)
        biological = self._specialist_mesh[..., LENS_INDEX["biological"], :]
        relation_sources = {"identity": biological}
        if "pollinator" in self.relation_meshes:
            ecological = self._specialist_mesh[..., LENS_INDEX["ecological"], :]
            relation_sources["pollinator"] = 0.5 * (biological + ecological)
            relation_sources["pollinator_transfer"] = biological
        if "myco" in self.relation_meshes:
            abiotic = self._specialist_mesh[..., LENS_INDEX["abiotic"], :]
            relation_sources["myco"] = 0.5 * (biological + abiotic)
        self._relation_mesh = {}
        self._relation_latents = {}
        for name, state in relation_sources.items():
            relation_state, relation_latent = self._mesh_pair(
                self.relation_meshes[name],
                state,
                self.relation_pair_mix[name],
            )
            self._relation_mesh[name] = relation_state
            self._relation_latents[name] = relation_latent
        expert_tokens = self._specialist_latents \
                        + self.specialist_type.view(1, len(LENSES), 1, self.d_model)
        expert_tokens = self.specialist_aggregate_norm(expert_tokens.flatten(1, 2))
        expert_read = self.specialist_fusion(
            latent.detach(), expert_tokens, expert_tokens, need_weights=False
        )[0]
        latent = latent + torch.tanh(self.specialist_fusion_gate) \
                 * self.specialist_output_norm(expert_read)

        raw_tokens = self.raw_residual_norm(self._raw_state_tokens)
        raw_mask = self._raw_state_mask.clone()
        raw_mask[:, 0] |= ~raw_mask.any(-1)
        raw_read = self.raw_residual_read(
            latent.detach(), raw_tokens, raw_tokens,
            key_padding_mask=~raw_mask, need_weights=False,
        )[0]
        latent = latent + torch.tanh(self.raw_residual_gate) \
                 * self.raw_residual_output_norm(raw_read)
        return latent

    def _task_segments(self, name: str) -> tuple[str, ...]:
        traits = {
            "seasonality", "water", "soil_drainage", "form",
            "plant_type", "growth_rate", "sun", "ease_of_care",
        }
        if name == self.species_variable:
            return "abiotic", "visual", "biological", "identity"
        if name == "pollinator":
            return "biological", "ecological", "pollinator", "pollinator_transfer"
        if name == "community":
            return "biological", "ecological"
        if name == "myco":
            return "abiotic", "biological", "myco"
        if name == "lfmc":
            return "abiotic", "biological"
        if name == "flower":
            return "abiotic", "ecological"
        if name in traits:
            return "visual", "biological"
        lens = LENSES[self.write_lens.get(name, LENS_INDEX["ecological"])]
        return (lens,)

    def _segment_state(self, name: str) -> torch.Tensor | None:
        if name in LENS_INDEX and self._specialist_mesh is not None:
            return self._specialist_mesh[..., LENS_INDEX[name], :]
        return self._relation_mesh.get(name)

    def _query_denoised_pool(
        self, pooled: torch.Tensor, name: str
    ) -> torch.Tensor:
        if name not in self.segment_task_gate:
            return pooled
        if name in self._denoised_pool_cache:
            return self._denoised_pool_cache[name]
        query = pooled + self.mesh_read_query[name]
        reads = []
        for segment in self._task_segments(name):
            state = self._segment_state(segment)
            if state is None or segment not in self.segment_denoisers:
                continue
            denoiser = self.segment_denoisers[segment]
            read = checkpoint(
                denoiser, state, query, use_reentrant=False
            ) if self.training and ACTIVATION_CHECKPOINTING \
              else denoiser(state, query)
            read = torch.tanh(self.segment_gate[segment]) * read
            if self.training:
                keep = 0.9
                path = (
                    torch.rand(read.shape[0], 1, device=read.device) < keep
                ).to(read.dtype) / keep
                read = read * path
            reads.append(read + self.segment_type[segment])
        if not reads:
            self._denoised_pool_cache[name] = pooled
            return pooled
        tokens = self.segment_fusion_norm(torch.stack(reads, 1))
        update = self.segment_fusion(
            self.segment_fusion_norm(query).unsqueeze(1),
            tokens,
            tokens,
            need_weights=False,
        )[0].squeeze(1)
        pooled = pooled + torch.tanh(self.segment_task_gate[name]) \
                          * self.segment_output_norm(update)
        self._denoised_pool_cache[name] = pooled
        return pooled

    def _pool(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        if name in self._denoised_pool_cache:
            return self._denoised_pool_cache[name]
        if name in self._pool_cache:
            return self._query_denoised_pool(self._pool_cache[name], name)
        base_name = name if name in self.names else self.species_variable
        query = self.decode_query[self.names.index(base_name)]
        weight = torch.softmax((latent @ query) / math.sqrt(self.d_model), -1)
        pooled = torch.einsum("bl,bld->bd", weight, latent)
        if self._fiber_summary is None or name not in self.mesh_read_query:
            self._pool_cache[name] = pooled
            return self._query_denoised_pool(pooled, name)
        if self._mesh_reader_cache is None:
            fibers = self._fiber_summary.flatten(1, 2)
            cells = self._fiber_mesh.shape[1]
            cell_key = torch.cat((
                self.mesh_cell_key[:1],
                self.mesh_cell_key[1:].expand(cells - 1, -1),
            ))
            scale_fibers = self._fiber_mesh.flatten(1, 3)
            scale_keys = (
                self._fiber_mesh
                + cell_key.view(1, cells, 1, 1, self.d_model)
                + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
                + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
            ).flatten(1, 3)
            prior_mesh = self._fiber_prior_mesh.detach()
            prior_fibers = prior_mesh.flatten(1, 3)
            prior_keys = (
                prior_mesh
                + cell_key.view(1, cells, 1, 1, self.d_model)
                + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
                + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
            ).flatten(1, 3)
            self._mesh_reader_cache = {
                "fibers": fibers,
                "task_tokens": self.task_mesh_reader_norm(fibers),
                "scale_fibers": scale_fibers,
                "scale_keys": scale_keys,
                "prior_fibers": prior_fibers,
                "prior_keys": prior_keys,
            }
        fibers = self._mesh_reader_cache["fibers"]
        mesh_query = self.mesh_read_query[name]
        if self._fiber_mesh is not None and name in self.mesh_condition_gate:
            mesh_query = mesh_query.unsqueeze(0).expand(fibers.shape[0], -1)
            query_lenses = self._fiber_mesh[:, 0].mean(1)
            lens_score = torch.einsum(
                "bld,bd->bl", query_lenses, mesh_query
            ) / math.sqrt(self.d_model)
            condition = torch.einsum(
                "bl,bld->bd", lens_score.softmax(-1), query_lenses
            )
            mesh_query = mesh_query + torch.tanh(
                self.mesh_condition_gate[name]
            ) * self.mesh_condition_norm(condition)
            score = torch.einsum(
                "bkd,bd->bk", fibers, mesh_query
            ) / math.sqrt(self.d_model)
        else:
            score = (fibers @ mesh_query) / math.sqrt(self.d_model)
        task_query = mesh_query if mesh_query.dim() == 2 else mesh_query.unsqueeze(0).expand(
            fibers.shape[0], -1
        )
        task_tokens = self._mesh_reader_cache["task_tokens"]
        task_read = self.task_mesh_reader(
            task_query.unsqueeze(1), task_tokens, task_tokens, need_weights=False
        )[0].squeeze(1)
        pooled = pooled + torch.tanh(self.task_mesh_reader_gate[name]) \
                 * self.task_mesh_reader_output_norm(task_read)
        selected_score, selected = score.topk(min(4, score.shape[-1]), dim=-1)
        selected_fibers = fibers.gather(
            1, selected[..., None].expand(-1, -1, self.d_model)
        )
        mesh_read = torch.einsum(
            "bk,bkd->bd", selected_score.softmax(-1), selected_fibers
        )
        pooled = pooled + torch.tanh(self.mesh_read_gate[name]) * self.mesh_task_norm(mesh_read)
        if self._fiber_mesh is None:
            self._pool_cache[name] = pooled
            return self._query_denoised_pool(pooled, name)
        cells = self._fiber_mesh.shape[1]
        scale_fibers = self._mesh_reader_cache["scale_fibers"]
        scale_keys = self._mesh_reader_cache["scale_keys"]
        if mesh_query.dim() == 2:
            scale_score = torch.einsum(
                "bkd,bd->bk", scale_keys, mesh_query
            ) / math.sqrt(self.d_model)
        else:
            scale_score = (scale_keys @ mesh_query) / math.sqrt(self.d_model)
        dense_weight = scale_score.softmax(-1)
        selected_score, scale_index = scale_score.topk(
            min(8, scale_score.shape[-1]), dim=-1
        )
        sparse_weight = torch.zeros_like(scale_score).scatter(
            -1, scale_index, selected_score.softmax(-1)
        )
        route = sparse_weight.detach() + dense_weight - dense_weight.detach()
        scale_read = torch.einsum(
            "bk,bkd->bd", route, scale_fibers
        )
        pooled = pooled + torch.tanh(self.mesh_scale_read_gate[name]) * self.mesh_scale_task_norm(
            scale_read
        )
        score_grid = scale_score.reshape(
            -1, cells, self.levels, len(LENSES)
        )
        lens = torch.arange(len(LENSES), device=scale_score.device)
        query_level = score_grid[:, 0].argmax(1)
        query_index = query_level * len(LENSES) + lens
        neighbor_position = score_grid[:, 1:].reshape(
            scale_score.shape[0], -1, len(LENSES)
        ).argmax(1)
        neighbor_cell = neighbor_position.div(self.levels, rounding_mode="floor") + 1
        neighbor_level = neighbor_position.remainder(self.levels)
        neighbor_index = (
            neighbor_cell * self.levels + neighbor_level
        ) * len(LENSES) + lens
        attention_index = torch.cat((query_index, neighbor_index), -1)
        selected_keys = scale_keys.gather(
            1, attention_index[..., None].expand(-1, -1, self.d_model)
        )
        selected_fibers = scale_fibers.gather(
            1, attention_index[..., None].expand(-1, -1, self.d_model)
        )
        scale_query = task_query + self.task_mesh_reader_output_norm(task_read)
        shared_scale_attention = self.task_mesh_reader(
            scale_query.unsqueeze(1),
            self.task_mesh_reader_norm(selected_keys),
            self.task_mesh_reader_norm(selected_fibers),
            need_weights=False,
        )[0].squeeze(1)
        dedicated_scale_attention = self.scale_mesh_reader(
            scale_query.unsqueeze(1),
            self.task_mesh_reader_norm(selected_keys),
            self.task_mesh_reader_norm(selected_fibers),
            need_weights=False,
        )[0].squeeze(1)
        reader_features = torch.cat((
            task_query,
            shared_scale_attention,
            dedicated_scale_attention,
            (shared_scale_attention - dedicated_scale_attention).abs(),
        ), -1)
        reader_mix = torch.sigmoid(
            self.scale_mesh_reader_mix[name]
            + self.scale_mesh_reader_router(reader_features).squeeze(-1)
        ).unsqueeze(-1)
        scale_attention = torch.lerp(
            shared_scale_attention, dedicated_scale_attention, reader_mix
        )
        pooled = pooled + torch.tanh(self.mesh_scale_attention_gate[name]) \
                 * self.mesh_scale_task_norm(scale_attention)
        if name != "community" or self.training:
            deep_read = scale_query
            for block in self.deep_mesh_reader:
                deep_read = block(deep_read, selected_keys, selected_fibers)
            pooled = pooled + torch.tanh(self.deep_mesh_reader_gate[name]) \
                     * self.deep_mesh_reader_output_norm(deep_read - scale_query)
        prior_fibers = self._mesh_reader_cache["prior_fibers"]
        prior_keys = self._mesh_reader_cache["prior_keys"]
        if mesh_query.dim() == 2:
            prior_score = torch.einsum("bkd,bd->bk", prior_keys, mesh_query)
        else:
            prior_score = prior_keys @ mesh_query
        prior_score = prior_score / math.sqrt(self.d_model)
        selected_score, selected = prior_score.topk(min(16, prior_score.shape[-1]), dim=-1)
        prior_read = torch.einsum(
            "bk,bkd->bd",
            selected_score.softmax(-1),
            prior_fibers.gather(1, selected[..., None].expand(-1, -1, self.d_model)),
        )
        confidence = torch.sigmoid(self.mesh_prior_information_gate(torch.cat((
            pooled, prior_read, pooled * prior_read, (pooled - prior_read).abs(),
        ), -1)))
        pooled = pooled + torch.tanh(self.mesh_prior_read_gate[name]) * confidence \
                          * self.mesh_prior_task_norm(prior_read)
        self._pool_cache[name] = pooled
        return self._query_denoised_pool(pooled, name)

    def _prime_pool_cache(self, latent: torch.Tensor) -> None:
        names = tuple(
            name for name in self.mesh_read_names
            if name not in self.mesh_condition_gate and name != "community"
        )
        if not names or self._fiber_summary is None or self._fiber_mesh is None:
            return
        batch = latent.shape[0]
        tasks = len(names)
        queries = torch.stack([
            self.decode_query[self.names.index(
                name if name in self.names else self.species_variable
            )]
            for name in names
        ])
        weight = torch.einsum("bld,td->btl", latent, queries) \
                 .div(math.sqrt(self.d_model)).softmax(-1)
        pooled = torch.einsum("btl,bld->btd", weight, latent)

        if self._mesh_reader_cache is None:
            fibers = self._fiber_summary.flatten(1, 2)
            cells = self._fiber_mesh.shape[1]
            cell_key = torch.cat((
                self.mesh_cell_key[:1],
                self.mesh_cell_key[1:].expand(cells - 1, -1),
            ))
            scale_fibers = self._fiber_mesh.flatten(1, 3)
            scale_keys = (
                self._fiber_mesh
                + cell_key.view(1, cells, 1, 1, self.d_model)
                + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
                + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
            ).flatten(1, 3)
            prior_mesh = self._fiber_prior_mesh.detach()
            prior_fibers = prior_mesh.flatten(1, 3)
            prior_keys = (
                prior_mesh
                + cell_key.view(1, cells, 1, 1, self.d_model)
                + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
                + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
            ).flatten(1, 3)
            self._mesh_reader_cache = {
                "fibers": fibers,
                "task_tokens": self.task_mesh_reader_norm(fibers),
                "scale_fibers": scale_fibers,
                "scale_keys": scale_keys,
                "prior_fibers": prior_fibers,
                "prior_keys": prior_keys,
            }
        fibers = self._mesh_reader_cache["fibers"]
        mesh_queries = torch.stack([self.mesh_read_query[name] for name in names])
        task_query = mesh_queries.unsqueeze(0).expand(batch, -1, -1)
        task_tokens = self._mesh_reader_cache["task_tokens"]
        task_tokens = task_tokens.unsqueeze(1).expand(-1, tasks, -1, -1) \
            .reshape(batch * tasks, task_tokens.shape[1], self.d_model)
        task_read = self.task_mesh_reader(
            task_query.reshape(batch * tasks, 1, self.d_model),
            task_tokens,
            task_tokens,
            need_weights=False,
        )[0].reshape(batch, tasks, self.d_model)
        task_gates = torch.stack([
            self.task_mesh_reader_gate[name] for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(task_gates) \
                 * self.task_mesh_reader_output_norm(task_read)

        fiber_score = torch.einsum(
            "bfd,td->btf", fibers, mesh_queries
        ) / math.sqrt(self.d_model)
        selected_score, selected = fiber_score.topk(
            min(4, fiber_score.shape[-1]), dim=-1
        )
        selected_fibers = fibers.unsqueeze(1).expand(-1, tasks, -1, -1).gather(
            2, selected[..., None].expand(-1, -1, -1, self.d_model)
        )
        mesh_read = torch.einsum(
            "btk,btkd->btd", selected_score.softmax(-1), selected_fibers
        )
        read_gates = torch.stack([
            self.mesh_read_gate[name] for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(read_gates) * self.mesh_task_norm(mesh_read)

        scale_fibers = self._mesh_reader_cache["scale_fibers"]
        scale_keys = self._mesh_reader_cache["scale_keys"]
        scale_score = torch.einsum(
            "bkd,td->btk", scale_keys, mesh_queries
        ) / math.sqrt(self.d_model)
        dense_weight = scale_score.softmax(-1)
        selected_score, scale_index = scale_score.topk(
            min(8, scale_score.shape[-1]), dim=-1
        )
        sparse_weight = torch.zeros_like(scale_score).scatter(
            -1, scale_index, selected_score.softmax(-1)
        )
        route = sparse_weight.detach() + dense_weight - dense_weight.detach()
        scale_read = torch.einsum(
            "btk,bkd->btd", route, scale_fibers
        )
        scale_gates = torch.stack([
            self.mesh_scale_read_gate[name] for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(scale_gates) \
                 * self.mesh_scale_task_norm(scale_read)

        cells = self._fiber_mesh.shape[1]
        score_grid = scale_score.reshape(
            batch, tasks, cells, self.levels, len(LENSES)
        )
        lens = torch.arange(len(LENSES), device=scale_score.device)
        query_level = score_grid[:, :, 0].argmax(2)
        query_index = query_level * len(LENSES) + lens
        neighbor_position = score_grid[:, :, 1:].reshape(
            batch, tasks, -1, len(LENSES)
        ).argmax(2)
        neighbor_cell = neighbor_position.div(
            self.levels, rounding_mode="floor"
        ) + 1
        neighbor_level = neighbor_position.remainder(self.levels)
        neighbor_index = (
            neighbor_cell * self.levels + neighbor_level
        ) * len(LENSES) + lens
        attention_index = torch.cat((query_index, neighbor_index), -1)
        expanded_scale_keys = scale_keys.unsqueeze(1).expand(-1, tasks, -1, -1)
        expanded_scale_fibers = scale_fibers.unsqueeze(1).expand(-1, tasks, -1, -1)
        selected_keys = expanded_scale_keys.gather(
            2, attention_index[..., None].expand(-1, -1, -1, self.d_model)
        )
        selected_fibers = expanded_scale_fibers.gather(
            2, attention_index[..., None].expand(-1, -1, -1, self.d_model)
        )
        scale_query = task_query + self.task_mesh_reader_output_norm(task_read)
        selected_keys = self.task_mesh_reader_norm(selected_keys).reshape(
            batch * tasks, -1, self.d_model
        )
        selected_fibers = self.task_mesh_reader_norm(selected_fibers).reshape(
            batch * tasks, -1, self.d_model
        )
        flat_query = scale_query.reshape(batch * tasks, 1, self.d_model)
        shared_attention = self.task_mesh_reader(
            flat_query, selected_keys, selected_fibers, need_weights=False
        )[0].reshape(batch, tasks, self.d_model)
        dedicated_attention = self.scale_mesh_reader(
            flat_query, selected_keys, selected_fibers, need_weights=False
        )[0].reshape(batch, tasks, self.d_model)
        reader_features = torch.cat((
            task_query,
            shared_attention,
            dedicated_attention,
            (shared_attention - dedicated_attention).abs(),
        ), -1)
        reader_bias = torch.stack([
            self.scale_mesh_reader_mix[name] for name in names
        ]).view(1, tasks)
        reader_mix = torch.sigmoid(
            reader_bias + self.scale_mesh_reader_router(reader_features).squeeze(-1)
        ).unsqueeze(-1)
        scale_attention = torch.lerp(
            shared_attention, dedicated_attention, reader_mix
        )
        attention_gates = torch.stack([
            self.mesh_scale_attention_gate[name] for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(attention_gates) \
                 * self.mesh_scale_task_norm(scale_attention)

        deep_read = flat_query.squeeze(1)
        for block in self.deep_mesh_reader:
            deep_read = block(deep_read, selected_keys, selected_fibers)
        deep_read = self.deep_mesh_reader_output_norm(
            deep_read - flat_query.squeeze(1)
        ).reshape(batch, tasks, self.d_model)
        deep_gates = torch.stack([
            self.deep_mesh_reader_gate[name]
            if name not in {"community", "identity"} or self.training
            else self.deep_mesh_reader_gate[name] * 0.0
            for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(deep_gates) * deep_read

        prior_fibers = self._mesh_reader_cache["prior_fibers"]
        prior_keys = self._mesh_reader_cache["prior_keys"]
        prior_score = torch.einsum(
            "bkd,td->btk", prior_keys, mesh_queries
        ) / math.sqrt(self.d_model)
        selected_score, selected = prior_score.topk(
            min(16, prior_score.shape[-1]), dim=-1
        )
        selected_prior = prior_fibers.unsqueeze(1).expand(
            -1, tasks, -1, -1
        ).gather(2, selected[..., None].expand(-1, -1, -1, self.d_model))
        prior_read = torch.einsum(
            "btk,btkd->btd", selected_score.softmax(-1), selected_prior
        )
        confidence = torch.sigmoid(self.mesh_prior_information_gate(torch.cat((
            pooled,
            prior_read,
            pooled * prior_read,
            (pooled - prior_read).abs(),
        ), -1)))
        prior_gates = torch.stack([
            self.mesh_prior_read_gate[name] for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(prior_gates) * confidence \
                 * self.mesh_prior_task_norm(prior_read)
        self._pool_cache.update({
            name: pooled[:, index] for index, name in enumerate(names)
        })

    def _pollinator_pool(self, latent: torch.Tensor, *, isolated: bool = False) -> torch.Tensor:
        pooled = self._pool(latent, "pollinator")
        if self.pollinator_reader is None or self._fiber_mesh is None:
            return self._relation_pool(pooled, "pollinator", isolated=isolated)
        cells = self._fiber_mesh.shape[1]
        fibers = self._fiber_mesh.flatten(1, 3)
        if isolated:
            pooled = pooled.detach()
            fibers = fibers.detach()
        keys = self.pollinator_reader_norm(fibers)
        cell_key = torch.cat((
            self.pollinator_reader_cell_key[:1],
            self.pollinator_reader_cell_key[1:].expand(cells - 1, -1),
        ))
        route_keys = (
            keys.reshape(-1, cells, self.levels, len(LENSES), self.d_model)
            + cell_key.view(1, cells, 1, 1, self.d_model)
            + self.pollinator_reader_level_key.view(
                1, 1, self.levels, 1, self.d_model
            )
            + self.pollinator_reader_lens_key.view(
                1, 1, 1, len(LENSES), self.d_model
            )
        ).flatten(1, 3)
        score = torch.einsum(
            "bkd,bd->bk", route_keys, pooled
        ) / math.sqrt(self.d_model)
        selected_score, selected_index = score.topk(
            min(16, score.shape[-1]), dim=-1
        )
        selected = keys.gather(
            1, selected_index[..., None].expand(-1, -1, self.d_model)
        )
        routed = torch.einsum(
            "bk,bkd->bd", selected_score.softmax(-1), selected
        )
        query = self.pollinator_reader_query.unsqueeze(0) \
                + pooled.unsqueeze(1) + routed.unsqueeze(1)
        read = self.pollinator_reader(query, selected, selected, need_weights=False)[0].mean(1)
        pooled = pooled + torch.tanh(self.pollinator_reader_gate) \
                          * self.pollinator_reader_output_norm(read)
        return self._relation_pool(pooled, "pollinator", isolated=isolated)

    def _pollinator_logits(
        self, latent: torch.Tensor, *, isolated: bool = False
    ) -> torch.Tensor:
        ordinary_pool = self._pollinator_pool(latent, isolated=isolated)
        transfer_pool = self._relation_pool(
            self._pool(latent, "pollinator"),
            "pollinator_transfer",
            isolated=isolated,
        )
        ordinary = F.log_softmax(self.poll_head(ordinary_pool), -1)
        transfer = F.log_softmax(self.poll_transfer_head(transfer_pool), -1)
        route = self._pollinator_transfer_probability(
            ordinary.shape[0], ordinary.device
        )
        self._pollinator_route = route
        return torch.logaddexp(
            ordinary + torch.log1p(-route.clamp(max=1.0 - 1e-6)),
            transfer + route.clamp_min(1e-6).log(),
        )

    def _pollinator_transfer_probability(
        self, batch: int, device: torch.device
    ) -> torch.Tensor:
        uncertainty = self._identity_graph_uncertainty
        if uncertainty is None:
            uncertainty = torch.zeros((batch, 1), device=device)
        return torch.sigmoid(
            self.pollinator_transfer_router(10.0 * uncertainty)
        )

    def _calibrate_pollinator_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.pollinator_log_temperature.clamp(-2.0, 2.0).exp()
        return logits / temperature

    def _lfmc_lens_residual(self, pooled: torch.Tensor) -> torch.Tensor:
        query = self.lfmc_lens_reader_norm(
            pooled.detach().float()
        ).unsqueeze(1)
        lenses = self.lfmc_lens_reader_norm(
            self._fiber_mesh[:, 0].mean(1).detach().float()
        )
        read = self.lfmc_lens_reader(
            query, lenses, lenses, need_weights=False
        )[0].squeeze(1)
        return self.lfmc_lens_head(read).squeeze(-1)

    def _lfmc_log_prediction(self, pooled: torch.Tensor) -> torch.Tensor:
        prediction = self.lfmc_head(pooled).squeeze(-1)
        if self._fiber_mesh is not None:
            prediction = prediction + self._lfmc_lens_residual(pooled)
        return prediction

    def _decode_pooled(self, pooled: torch.Tensor, name: str) -> torch.Tensor:
        if name == self.species_variable:
            return pooled @ self._refined_species.t()
        return self.decoders[name](pooled)

    def _species_lens_residual(
        self, pooled: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        query = self.species_lens_reader_norm(
            pooled.detach().float()
        ).unsqueeze(1)
        lenses = self.species_lens_reader_norm(
            self._fiber_mesh[:, 0].mean(1).detach().float()
        )
        read = self.species_lens_reader(
            query, lenses, lenses, need_weights=False
        )[0].squeeze(1)
        residual = read @ key.detach().t()
        family_sum = residual.new_zeros(residual.shape[0], self.family_count)
        family_sum.scatter_add_(
            1, self.species_family.expand(residual.shape[0], -1), residual
        )
        family_size = torch.bincount(
            self.species_family, minlength=self.family_count
        ).clamp_min(1).to(residual.dtype)
        family_mean = family_sum / family_size
        return residual - family_mean.gather(
            1, self.species_family.expand(residual.shape[0], -1)
        )

    def _niche_species_logits(
        self, pooled: torch.Tensor, include_lens: bool = True
    ) -> torch.Tensor:
        key = self._refined_species.detach().float() \
              + self.species_niche_key.float()
        pooled = pooled.float()
        base = pooled @ key.t()
        residual = self.species_niche_adapter(pooled) @ key.t()
        family_sum = residual.new_zeros(residual.shape[0], self.family_count)
        family_sum.scatter_add_(
            1, self.species_family.expand(residual.shape[0], -1), residual
        )
        family_size = torch.bincount(
            self.species_family, minlength=self.family_count
        ).clamp_min(1).to(residual.dtype)
        family_mean = family_sum / family_size
        residual = residual - family_mean.gather(
            1, self.species_family.expand(residual.shape[0], -1)
        )
        logits = base + residual + self._position_species_logits(pooled, key)
        if include_lens:
            logits = logits + self._species_lens_residual(pooled, key)
        return logits

    def _position_species_logits(
        self, pooled: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        state = self._position_species_state
        patches = self._position_species_patch_state
        if state is None or patches is None:
            return key.new_zeros(pooled.shape[0], key.shape[0])
        position = torch.einsum(
            "l,bld->bd", self.position_species_level.softmax(0), state.float()
        )
        query = pooled + position
        read = self.position_species_patch_attention(
            query.unsqueeze(1), patches, patches,
            key_padding_mask=self._position_species_patch_mask,
            need_weights=False,
        )[0].squeeze(1)
        query = query + self.position_species_patch_output(read)
        residual = self.position_species_adapter(query) @ key.detach().t()
        family = self.species_family.expand(residual.shape[0], -1)
        family_sum = residual.new_zeros(residual.shape[0], self.family_count)
        family_sum.scatter_add_(1, family, residual)
        family_size = torch.bincount(
            self.species_family, minlength=self.family_count
        ).clamp_min(1).to(residual.dtype)
        residual = residual - (family_sum / family_size).gather(1, family)
        valid = self._position_species_patch_valid.to(residual.dtype).unsqueeze(1)
        return torch.tanh(self.position_species_gate) * residual * valid

    def _positioned_patch_state(
        self,
        context: dict,
        values: Dict[str, torch.Tensor],
        present: Dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        index = values.get("naip_patch_index")
        if index is None:
            self._position_species_patch_mask = None
            self._position_species_patch_valid = None
            return None
        index = index.long()
        valid = (index >= 0) & present.get(
            "naip_rgb", torch.zeros_like(index, dtype=torch.bool)
        )
        coords = context["coordinates"]
        coarse_stop = self.levels // 3
        mid_stop = 2 * self.levels // 3
        fine_levels = self.levels - mid_stop
        mid_levels = mid_stop - coarse_stop
        token_count = 64 * fine_levels + 16 * mid_levels + coarse_stop
        state = coords.new_zeros(len(coords), token_count, self.d_model)
        if valid.any():
            active_coords = coords[valid]

            def shifted(offsets: torch.Tensor) -> torch.Tensor:
                patch_coords = active_coords[:, None].expand(
                    -1, len(offsets), -1
                ).clone()
                patch_coords[..., 0] += offsets[None, :, 0] / 111_320.0
                longitude_scale = 111_320.0 * torch.cos(
                    torch.deg2rad(active_coords[:, 0])
                ).clamp_min(0.2)
                patch_coords[..., 1] += (
                    offsets[None, :, 1] / longitude_scale[:, None]
                )
                return patch_coords

            def addressed(patch_coords: torch.Tensor) -> torch.Tensor:
                spatial, temporal = self.mesh.raw(
                    patch_coords.float().flatten(0, 1)
                )
                position = self.absolute_proj_s(spatial) \
                           + self.absolute_proj_t(temporal)
                position = position.reshape(
                    len(active_coords), patch_coords.shape[1],
                    self.levels, self.d_model,
                )
                relative = self.neighbors.space_time(
                    active_coords.float(), patch_coords.float()
                )
                if relative.dim() == position.dim() - 1:
                    relative = relative.unsqueeze(-2)
                return (position + relative).detach()

            patches = self.position_species_patch_tokens[index[valid]]
            patch_coords = None
            if self.position_species_patch_coords is not None:
                patch_coords = self.position_species_patch_coords[index[valid]]
                patch_coords = patch_coords.float()
                patch_coords[..., 2:] = torch.nan_to_num(
                    patch_coords[..., 2:], nan=0.0
                )
            if patches.dim() == 4:
                patch_grid = patches.float().permute(0, 3, 1, 2)
                fine_raw = F.adaptive_avg_pool2d(patch_grid, (8, 8))
                mid_raw = F.adaptive_avg_pool2d(patch_grid, (4, 4))
                coarse_raw = F.adaptive_avg_pool2d(patch_grid, (1, 1))
                fine_evidence = self.position_species_rgb_adapter(
                    fine_raw.permute(0, 2, 3, 1).reshape(-1, 64, patches.shape[-1])
                )
                fine_grid = fine_evidence.reshape(
                    -1, 8, 8, self.d_model
                ).permute(0, 3, 1, 2)
                fine_grid = fine_grid + self.position_species_local(fine_grid)
                fine_evidence = fine_grid.permute(0, 2, 3, 1).reshape(
                    -1, 64, self.d_model
                )
                mid_evidence = self.position_species_rgb_adapter(
                    mid_raw.permute(0, 2, 3, 1).reshape(-1, 16, patches.shape[-1])
                )
                coarse_evidence = self.position_species_rgb_adapter(
                    coarse_raw.permute(0, 2, 3, 1).reshape(-1, 1, patches.shape[-1])
                )
                if patch_coords is not None:
                    coord_grid = patch_coords.permute(0, 3, 1, 2)
                    fine_coords = F.adaptive_avg_pool2d(
                        coord_grid, (8, 8)
                    ).permute(0, 2, 3, 1).reshape(-1, 64, 4)
                    mid_coords = F.adaptive_avg_pool2d(
                        coord_grid, (4, 4)
                    ).permute(0, 2, 3, 1).reshape(-1, 16, 4)
                    global_coords = F.adaptive_avg_pool2d(
                        coord_grid, (1, 1)
                    ).permute(0, 2, 3, 1).reshape(-1, 1, 4)
                else:
                    fine_coords = shifted(self.position_species_patch_offsets)
                    mid_coords = shifted(self.position_species_mid_offsets)
                    global_coords = active_coords[:, None]
            else:
                evidence = self.position_species_rgb_adapter(patches[:, 0].float())
                if self.position_species_ir_adapter is not None:
                    evidence = (
                        evidence
                        + self.position_species_ir_adapter(patches[:, 1].float())
                    ).mul_(1.0 / math.sqrt(2.0))
                grid = evidence.reshape(-1, 8, 8, self.d_model).permute(0, 3, 1, 2)
                grid = grid + self.position_species_local(grid)
                fine_evidence = grid.permute(0, 2, 3, 1).reshape(
                    -1, 64, self.d_model
                )
                mid_grid = F.avg_pool2d(grid, 2, 2)
                mid_evidence = mid_grid.permute(0, 2, 3, 1).reshape(
                    -1, 16, self.d_model
                )
                coarse_evidence = mid_evidence.mean(1, keepdim=True)
                fine_coords = shifted(self.position_species_patch_offsets)
                mid_coords = shifted(self.position_species_mid_offsets)
                global_coords = active_coords[:, None]
            with torch.autocast(device_type=coords.device.type, enabled=False):
                fine_position = addressed(fine_coords)[:, :, mid_stop:]
                mid_position = addressed(mid_coords)[:, :, coarse_stop:mid_stop]
                coarse_position = addressed(global_coords)[:, :, :coarse_stop]
            scale = self.position_species_scale_type
            fine_state = self.position_species_patch_norm(
                fine_evidence.unsqueeze(2) + fine_position + scale[0]
            ).flatten(1, 2)
            mid_state = self.position_species_patch_norm(
                mid_evidence.unsqueeze(2) + mid_position + scale[1]
            ).flatten(1, 2)
            coarse_state = self.position_species_patch_norm(
                coarse_evidence.unsqueeze(2) + coarse_position + scale[2]
            ).flatten(1, 2)
            state[valid] = torch.cat(
                (fine_state, mid_state, coarse_state), dim=1
            )
        token_valid = valid[:, None].expand(-1, token_count).clone()
        token_valid[~valid, 0] = True
        self._position_species_patch_mask = ~token_valid
        self._position_species_patch_valid = valid
        return state

    def _hierarchical_family_read(self, species_logits: torch.Tensor) -> torch.Tensor:
        """Read the strongest species inside the mesh posterior's strongest family."""
        logits = species_logits.float()
        family = self.species_family.expand(logits.shape[0], -1)
        family_mass = logits.new_zeros(logits.shape[0], self.family_count)
        family_mass.scatter_add_(1, family, logits.softmax(-1))
        winning_family = family_mass.argmax(-1)
        eligible = self.species_family.unsqueeze(0) == winning_family.unsqueeze(1)
        selected = logits.masked_fill(~eligible, -torch.inf).argmax(-1)
        top = species_logits.amax(-1)
        return species_logits.scatter(1, selected.unsqueeze(1), top.unsqueeze(1) + 1e-4)

    def _pollinator_species_posterior(self, species_logits: torch.Tensor) -> torch.Tensor:
        """Marginalize uncertain plant identity through known pollinator relations."""
        k = min(64, species_logits.shape[-1])
        weight, species = species_logits.float().softmax(-1).topk(k, -1)
        index = self.poll_species_idx[species].clamp(0, self.poll_head.out_features - 1)
        mass = weight.unsqueeze(-1) * self.poll_species_frq[species]
        mixture = species_logits.new_zeros(
            species_logits.shape[0], self.poll_head.out_features, dtype=torch.float32
        )
        mixture.scatter_add_(1, index.flatten(1), mass.flatten(1))
        mixture = mixture / mixture.sum(-1, keepdim=True).clamp_min(1e-8)
        return mixture.clamp_min(1e-8).log()

    def _identity_detail_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        cells = self._fiber_mesh.shape[1]
        fibers = self._fiber_mesh.flatten(1, 3).detach()
        keys = self.identity_detail_norm(fibers)
        cell_key = torch.cat((
            self.identity_detail_cell_key[:1],
            self.identity_detail_cell_key[1:].expand(cells - 1, -1),
        ))
        route_keys = (
            keys.reshape(-1, cells, self.levels, len(LENSES), self.d_model)
            + cell_key.view(1, cells, 1, 1, self.d_model)
            + self.identity_detail_level_key.view(
                1, 1, self.levels, 1, self.d_model
            )
            + self.identity_detail_lens_key.view(
                1, 1, 1, len(LENSES), self.d_model
            )
        ).flatten(1, 3)
        score = torch.einsum(
            "bkd,bd->bk", route_keys, pooled
        ) / math.sqrt(self.d_model)
        selected_score, selected_index = score.topk(
            min(16, score.shape[-1]), dim=-1
        )
        selected = keys.gather(
            1, selected_index[..., None].expand(-1, -1, self.d_model)
        )
        routed = torch.einsum(
            "bk,bkd->bd", selected_score.softmax(-1), selected
        )
        query = self.identity_detail_query.unsqueeze(0) \
                + pooled.unsqueeze(1) + routed.unsqueeze(1)
        read = self.identity_detail_reader(
            query, selected, selected, need_weights=False
        )[0].mean(1)
        read = torch.tanh(self.identity_detail_gate) \
               * self.identity_detail_output_norm(read)
        logits = read @ self._refined_species.detach().t()
        family = self.species_family.expand(logits.shape[0], -1)
        family_sum = logits.new_zeros(logits.shape[0], self.family_count)
        family_sum.scatter_add_(1, family, logits)
        family_size = torch.bincount(
            self.species_family, minlength=self.family_count
        ).clamp_min(1).to(logits.dtype)
        family_mean = family_sum / family_size
        return logits - family_mean.gather(1, family)

    def _myco_logits(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = self._relation_pool(self._pool(latent, "myco"), "myco")
        base = self.myco_head(pooled)
        if self.species_myco_head is None or self.training:
            return base
        species_logits = self.decode(latent, self.species_variable)
        species_to_myco = self.species_myco_head(
            self._refined_species.detach()
        ).softmax(-1)
        myco = species_logits.softmax(-1) @ species_to_myco
        evidence = myco.clamp_min(1e-8).log() \
                   - self.species_myco_prior.clamp_min(1e-8).log()
        return base + torch.tanh(self.myco_relation_gate) * evidence

    def _pool_fiber(self, fiber: torch.Tensor, name: str) -> torch.Tensor:
        query = self.fiber_decode_query[self.names.index(name)]
        weight = torch.softmax((fiber @ query) / math.sqrt(self.d_model), -1)
        return torch.einsum("bl,bld->bd", weight, fiber)

    def _pool_specialist(self, name: str) -> torch.Tensor:
        lens = self.write_lens[name]
        expert = self._specialist_latents[:, lens]
        query = self.specialist_decode_query[self.names.index(name)]
        weight = torch.softmax((expert @ query) / math.sqrt(self.d_model), -1)
        return torch.einsum("bl,bld->bd", weight, expert)

    def _relation_pool(
        self, pooled: torch.Tensor, name: str, *, isolated: bool = False
    ) -> torch.Tensor:
        expert = self._relation_latents.get(name)
        if expert is None:
            return pooled
        if isolated:
            pooled = pooled.detach()
            expert = expert.detach()
        tokens = self.relation_reader_norms[name](expert)
        query = pooled.unsqueeze(1) + self.relation_query[name].view(1, 1, -1)
        read = self.relation_readers[name](
            query, tokens, tokens, need_weights=False
        )[0].squeeze(1)
        return pooled + torch.tanh(self.relation_gate[name]) \
                        * self.relation_output_norms[name](read)

    def decode(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        pooled = self._pool(latent, name)
        if name == self.species_variable:
            pooled = self._relation_pool(pooled, "identity")
        logits = self._decode_pooled(pooled, name)
        if name == self.species_variable and not self.training \
                and self._fiber_mesh is not None:
            logits = logits + self._identity_detail_logits(pooled)
        return logits

    @torch.no_grad()
    def infer(self, values, given, targets, context, observed=None):
        batch = context["position"].shape[0]
        observed = observed or {name: torch.ones(batch, dtype=torch.bool, device=context["position"].device)
                                for name in self.names}
        present = {name: torch.zeros(batch, dtype=torch.bool, device=context["position"].device)
                   for name in self.names}
        for name in given:
            if name in present:
                present[name] = observed.get(name, torch.ones_like(present[name]))
        present = self._with_worldclim_observed(present, observed)
        latent = self.encode(values, present, context)
        environment_species = None
        if tuple(given) == self.environment_names \
                and self.species_variable in targets:
            pooled = self._pool(latent, self.species_variable)
            environment_species = self._niche_species_logits(pooled) \
                                  + self._identity_detail_logits(pooled)
            environment_species = self._hierarchical_family_read(
                environment_species
            )
        pollinator_species = None
        if self.poll_head is not None and "pollinator" in targets \
                and self.species_variable not in given:
            species_logits = self.decode(latent, self.species_variable)
            if not given or tuple(given) == self.environment_names:
                species_logits = self._hierarchical_family_read(species_logits)
            pollinator_species = self._pollinator_species_posterior(species_logits)
        out = {}
        for name in targets:
            if name == "community":
                pooled = self._pool(latent, "community")
                out[name] = self.community_metric(pooled) @ self._refined_species.t()
            elif name == "pollinator":
                prediction = self._pollinator_logits(latent)
                if pollinator_species is None:
                    prediction = self._calibrate_pollinator_logits(prediction)
                else:
                    prediction = torch.logaddexp(
                        F.log_softmax(prediction.float(), -1) + math.log(0.5),
                        pollinator_species + math.log(0.5),
                    )
                out[name] = prediction
            elif name == "lfmc":
                pooled = self._pool(latent, "lfmc")
                out[name] = self._lfmc_log_prediction(pooled).exp()
            elif name == "myco":
                out[name] = self._myco_logits(latent)
            elif name == "flower":
                pooled = self._pool(latent, "flower")
                out[name] = torch.sigmoid(self.flower_head(pooled).squeeze(-1))
            else:
                prediction = environment_species \
                    if name == self.species_variable \
                    and environment_species is not None \
                    else self.decode(latent, name)
                if name == self.species_variable and not given:
                    prediction = self._hierarchical_family_read(prediction)
                out[name] = prediction
        return out

    def _with_worldclim_observed(self, present, observed):
        if "worldclim" not in self.always_names or "worldclim" not in observed:
            return present
        present = dict(present)
        present["worldclim"] = observed["worldclim"]
        return present

    def reconstruction_loss(self, values, observed, context, hide_probability: float = 0.5):
        batch = context["position"].shape[0]
        present = {name: (torch.rand(batch, device=context["position"].device) > hide_probability) & observed[name]
                   for name in self.names}
        blank = torch.rand(batch, device=context["position"].device) < 0.15
        for name in present:
            present[name] &= ~blank
        present = self._with_worldclim_observed(present, observed)
        latent = self.encode(values, present, context)
        self._prime_pool_cache(latent)
        terms = []
        fiber_terms = []
        mesh_terms = []
        specialist_terms = []
        pollinator_target = None
        pollinator_valid = None
        pollinator_structured_term = None
        pollinator_calibration_term = None
        for variable in self.variables:
            hidden = (~present[variable.name]) & observed[variable.name]
            if not hidden.any():
                continue
            prediction = self.decode(latent, variable.name)
            if variable.kind == "categorical":
                error = F.cross_entropy(prediction, values[variable.name].long(), reduction="none") \
                        / math.log(max(variable.num_classes, 2))
            else:
                target = values[variable.name].float()
                valid = target.norm(dim=-1) > 1e-6
                mean = target[valid].mean(0, keepdim=True).detach() if valid.any() else target.mean(0, keepdim=True)
                error = 1.0 - F.cosine_similarity(prediction - mean, target - mean, dim=-1)
            terms.append((error * hidden).sum() / hidden.sum().clamp_min(1))
            lens = self.write_lens[variable.name]
            fiber_prediction = self.fiber_reconstruct[variable.name](
                self._pool_fiber(self._fiber_summary[:, lens], variable.name)
            )
            fiber_target = self._adapt(
                variable.name, values[variable.name], self._refined_species
            ).detach()
            fiber_error = 1.0 - F.cosine_similarity(fiber_prediction, fiber_target, dim=-1)
            fiber_terms.append((fiber_error * hidden).sum() / hidden.sum().clamp_min(1))
            mesh_state = self._fiber_mesh[:, 0, :, lens, :].mean(1)
            mesh_prediction = self.mesh_linear_reconstruct[variable.name](mesh_state)
            mesh_error = 1.0 - F.cosine_similarity(mesh_prediction, fiber_target, dim=-1)
            mesh_terms.append((mesh_error * hidden).sum() / hidden.sum().clamp_min(1))
            specialist_prediction = self.specialist_reconstruct[variable.name](
                self._pool_specialist(variable.name)
            )
            specialist_error = 1.0 - F.cosine_similarity(
                specialist_prediction, fiber_target, dim=-1
            )
            specialist_terms.append(
                (specialist_error * hidden).sum() / hidden.sum().clamp_min(1)
            )

        if self.poll_head is not None and "_poll_idx" in values:
            pooled = self._pool(latent, "pollinator")
            logits = self.poll_head(pooled)
            pollinator_target = torch.zeros_like(logits).scatter_add_(
                1, values["_poll_idx"].clamp_min(0), values["_poll_frq"].float()
            )
            pollinator_valid = values["_poll_valid"].float()
            error = -(pollinator_target * F.log_softmax(logits, -1)).sum(-1)
            terms.append(
                0.1 * (error * pollinator_valid).sum()
                / pollinator_valid.sum().clamp_min(1)
            )
            if getattr(self, "reader_phase", False):
                calibrated = self._calibrate_pollinator_logits(logits.detach())
                calibration_error = -(
                    pollinator_target * F.log_softmax(calibrated, -1)
                ).sum(-1)
                pollinator_calibration_term = (
                    calibration_error * pollinator_valid
                ).sum() / pollinator_valid.sum().clamp_min(1)
            structured = self._pollinator_logits(latent, isolated=True)
            structured_error = -(
                pollinator_target * F.log_softmax(structured, -1)
            ).sum(-1)
            pollinator_structured_term = (
                (structured_error * pollinator_valid).sum()
                / pollinator_valid.sum().clamp_min(1)
                / math.log(self.poll_head.out_features)
            )
        if self.lfmc_head is not None and "_lfmc" in values:
            valid = values["_lfmc_valid"].float()
            lfmc_pool = self._pool(latent, "lfmc")
            target_lfmc = torch.log(values["_lfmc"].clamp_min(1.0))
            error = (self.lfmc_head(lfmc_pool).squeeze(-1)
                     - target_lfmc).square()
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        if self.myco_head is not None and "_myco" in values:
            valid = values["_myco_valid"].float()
            error = F.cross_entropy(self._myco_logits(latent),
                                    values["_myco"].long().clamp_min(0), reduction="none")
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        if self.flower_head is not None and "_flower" in values:
            valid = values["_flower_valid"].float()
            error = F.binary_cross_entropy_with_logits(
                                                        self.flower_head(self._pool(latent, "flower")).squeeze(-1),
                                                        values["_flower"].float(), reduction="none")
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        loss = torch.stack(terms).mean() \
               + 0.05 * torch.stack(fiber_terms).mean() \
               + 0.05 * torch.stack(mesh_terms).mean() \
               + 0.05 * torch.stack(specialist_terms).mean()
        if pollinator_calibration_term is not None:
            loss = loss + pollinator_calibration_term
        if self.species_myco_head is not None and self.species_myco_valid.any():
            species_myco = self.species_myco_head(
                self._refined_species.detach()[self.species_myco_valid]
            )
            loss = loss + 0.1 * F.cross_entropy(
                species_myco, self.species_myco[self.species_myco_valid]
            )
        if pollinator_structured_term is not None:
            loss = loss + 0.1 * pollinator_structured_term
        environment_present = {
            name: observed[name] if name in self.environment_names else torch.zeros_like(observed[name])
            for name in self.names
        }
        environment_present = self._with_worldclim_observed(
            environment_present, observed
        )
        environment_latent = self.encode(values, environment_present, context, detach_species=True)
        environment_pool = self._pool(environment_latent, self.species_variable)
        family_logits = environment_pool.float() \
                        @ self._refined_species.detach().float().t()
        target_species = values[self.species_variable].long()
        family_valid = observed[self.species_variable]
        niche_input = environment_pool if getattr(
            self, "rank_aligned_expansion", False
        ) else environment_pool.detach()
        niche_logits = self._niche_species_logits(
            niche_input, include_lens=False
        )
        species_error = F.cross_entropy(
            niche_logits, target_species, reduction="none"
        ) / math.log(max(self._refined_species.shape[0], 2))
        loss = loss + 0.1 * (species_error * family_valid).sum() \
                      / family_valid.sum().clamp_min(1)
        if getattr(self, "reader_phase", False):
            target_score = niche_logits.gather(1, target_species[:, None])
            soft_rank = 0.5 + torch.sigmoid(
                (niche_logits - target_score) / 0.25
            ).sum(-1)
            rank_error = soft_rank.clamp_min(1.0).log() \
                         / math.log(max(niche_logits.shape[-1], 2))
            loss = loss + 0.25 * (rank_error * family_valid).sum() \
                          / family_valid.sum().clamp_min(1)
            key = self._refined_species.detach().float() \
                  + self.species_niche_key.detach().float()
            lens_logits = niche_logits.detach() \
                          + self._species_lens_residual(
                              environment_pool.detach(), key
                          )
            lens_error = F.cross_entropy(
                lens_logits, target_species, reduction="none"
            ) / math.log(max(lens_logits.shape[-1], 2))
            loss = loss + 0.1 * (lens_error * family_valid).sum() \
                          / family_valid.sum().clamp_min(1)
            lens_target = lens_logits.gather(1, target_species[:, None])
            lens_rank = 0.5 + torch.sigmoid(
                (lens_logits - lens_target) / 0.25
            ).sum(-1)
            lens_rank = lens_rank.clamp_min(1.0).log() \
                        / math.log(max(lens_logits.shape[-1], 2))
            loss = loss + 0.25 * (lens_rank * family_valid).sum() \
                          / family_valid.sum().clamp_min(1)
            lens_probability = lens_logits.softmax(-1)
            lens_family_probability = lens_probability.new_zeros(
                batch, self.family_count
            )
            lens_family_probability.scatter_add_(
                1,
                self.species_family.expand(batch, -1),
                lens_probability,
            )
            lens_target_family = self.species_family[target_species]
            lens_family_error = -lens_family_probability.gather(
                1, lens_target_family[:, None]
            ).squeeze(1).clamp_min(1e-8).log() \
             / math.log(max(self.family_count, 2))
            loss = loss + 0.25 * (lens_family_error * family_valid).sum() \
                          / family_valid.sum().clamp_min(1)
        probability = family_logits.softmax(-1)
        family_probability = probability.new_zeros(batch, self.family_count)
        family_probability.scatter_add_(1, self.species_family.expand(batch, -1), probability)
        target_family = self.species_family[target_species]
        family_error = -family_probability.gather(
            1, target_family[:, None]
        ).squeeze(1).clamp_min(1e-8).log()
        family_term = (family_error * family_valid).sum() / family_valid.sum().clamp_min(1) \
                      / math.log(max(self.family_count, 2))
        loss = loss + 0.1 * family_term
        if getattr(self, "reader_phase", False):
            relation_logits = self._identity_detail_logits(environment_pool)
            species_logits = niche_logits if getattr(
                self, "rank_aligned_expansion", False
            ) else family_logits.detach()
            calibrated_logits = species_logits + relation_logits
            target_family = self.species_family[target_species]
            same_family = self.species_family.unsqueeze(0) == target_family.unsqueeze(1)
            within_family = calibrated_logits.masked_fill(
                ~same_family, -1e4
            )
            relation_error = F.cross_entropy(
                within_family, target_species, reduction="none"
            ) / math.log(max(self._refined_species.shape[0], 2))
            loss = loss + 0.25 * (relation_error * family_valid).sum() \
                          / family_valid.sum().clamp_min(1)
            target_score = calibrated_logits.gather(1, target_species[:, None])
            soft_rank = 0.5 + torch.sigmoid(
                (calibrated_logits - target_score) / 0.25
            ).sum(-1)
            rank_error = soft_rank.clamp_min(1.0).log() \
                         / math.log(max(self._refined_species.shape[0], 2))
            loss = loss + 0.25 * (rank_error * family_valid).sum() \
                          / family_valid.sum().clamp_min(1)
        devices = [torch.cuda.current_device()] if loss.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            mask = torch.rand(self._refined_species.shape[0], device=loss.device) < 0.15
            reconstructed = self.species_graph(mask)
            loss = loss + 0.1 * self.species_graph.masked_reconstruction_loss(
                mask, self._refined_species.detach(), metric="mse",
                reconstructed=reconstructed,
            )
            reference = self._refined_species.detach()[~mask]
            reference_family = self.species_family[~mask]
            prototypes = reference.new_zeros(self.family_count, self.d_model)
            prototypes.index_add_(0, reference_family, reference)
            counts = torch.bincount(
                reference_family, minlength=self.family_count
            ).to(reference.dtype)
            prototypes = prototypes / counts[:, None].clamp_min(1.0)
            logits = F.normalize(reconstructed[mask], dim=-1) \
                     @ F.normalize(prototypes, dim=-1).t() / 0.1
            logits = logits.masked_fill((counts == 0).unsqueeze(0), -1e4)
            target_family = self.species_family[mask]
            valid = counts[target_family] > 0
            if valid.any():
                family_loss = F.cross_entropy(logits[valid], target_family[valid]) \
                              / math.log(max(self.family_count, 2))
                loss = loss + 0.03 * family_loss
        neighbor_identity = context["neighbor_values"].get("identity")
        if neighbor_identity is not None:
            query = self.community_metric(self._pool(latent, "community").detach())
            logits = query @ self._refined_species.detach().t()
            target = torch.zeros_like(logits)
            target.scatter_(1, neighbor_identity.long(), 1.0)
            target.scatter_(1, values[self.species_variable].long().unsqueeze(1), 1.0)
            target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
            community = -(target * F.log_softmax(logits, -1)).sum(-1) / math.log(logits.shape[-1])
            loss = loss + 0.5 * community.mean()

            if getattr(self, "reader_phase", False):
                masked_valid = family_valid * mask[values[self.species_variable].long()]
                empty_present = {
                    name: torch.zeros_like(observed[name]) for name in self.names
                }
                empty_present = self._with_worldclim_observed(
                    empty_present, observed
                )
                with torch.no_grad():
                    empty_latent = self.encode(
                        values, empty_present, context, detach_species=True, species_mask=mask
                    )
                    baseline = self.community_metric(self._pool(empty_latent, "community"))
                identity_present = dict(empty_present)
                identity_present[self.species_variable] = observed[self.species_variable]
                identity_latent = self.encode(
                    values, identity_present, context, detach_species=True, species_mask=mask
                )
                identity_query = self.community_metric(self._pool(identity_latent, "community"))
                conditional_logits = (identity_query - baseline) @ self._refined_species.detach().t()
                conditional_target = torch.zeros_like(conditional_logits)
                conditional_target.scatter_(1, neighbor_identity.long(), 1.0)
                conditional_target = conditional_target / conditional_target.sum(
                    -1, keepdim=True
                ).clamp_min(1.0)
                conditional = -(conditional_target * F.log_softmax(
                    conditional_logits, -1
                )).sum(-1) / math.log(conditional_logits.shape[-1])
                loss = loss + 0.1 * (conditional * masked_valid).sum() \
                       / family_valid.sum().clamp_min(1)
        if getattr(self, "reader_phase", False) and pollinator_target is not None:
            ordinary_present = {
                name: observed[name]
                if name == self.species_variable or name in self.environment_names
                else torch.zeros_like(observed[name])
                for name in self.names
            }
            ordinary_present = self._with_worldclim_observed(
                ordinary_present, observed
            )
            devices = [torch.cuda.current_device()] if loss.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                ordinary_latent = self.encode(
                    values, ordinary_present, context
                )
            ordinary_logits = self.poll_head(
                self._pollinator_pool(ordinary_latent, isolated=True)
            )
            ordinary = -(
                pollinator_target * F.log_softmax(ordinary_logits, -1)
            ).sum(-1) / math.log(self.poll_head.out_features)
            loss = loss + 0.25 * (ordinary * pollinator_valid).sum() \
                   / pollinator_valid.sum().clamp_min(1)
            ordinary_route = self._pollinator_transfer_probability(
                batch, loss.device
            ).squeeze(-1)
            loss = loss - 0.25 * (
                torch.log1p(-ordinary_route.clamp(max=1.0 - 1e-6))
                * pollinator_valid
            ).sum() / pollinator_valid.sum().clamp_min(1)

            masked_present = {
                name: observed[name] if name == self.species_variable
                else torch.zeros_like(observed[name])
                for name in self.names
            }
            masked_present = self._with_worldclim_observed(
                masked_present, observed
            )
            with torch.random.fork_rng(devices=devices):
                pollinator_latent = self.encode(
                    values, masked_present, context, species_mask=mask
                )
            transfer_pool = self._relation_pool(
                self._pool(pollinator_latent, "pollinator"),
                "pollinator_transfer",
            )
            pollinator_logits = self.poll_transfer_head(transfer_pool)
            interaction = -(
                pollinator_target * F.log_softmax(pollinator_logits, -1)
            ).sum(-1) / math.log(self.poll_head.out_features)
            masked_valid = pollinator_valid * mask[target_species]
            loss = loss + 0.25 * (interaction * masked_valid).sum() \
                   / masked_valid.sum().clamp_min(1)
            masked_route = self._pollinator_transfer_probability(
                batch, loss.device
            ).squeeze(-1)
            loss = loss - 0.25 * (
                masked_route.clamp_min(1e-6).log() * masked_valid
            ).sum() / masked_valid.sum().clamp_min(1)
        if getattr(self, "reader_phase", False):
            photo_row = torch.arange(batch, device=loss.device).remainder(2).bool()
            structured_present = {
                name: observed[name] & photo_row
                if name in {"vision_dino", "vision_bio"}
                else torch.zeros_like(observed[name])
                for name in self.names
            }
            structured_present = self._with_worldclim_observed(
                structured_present, observed
            )
            structured_latent = self.encode(
                values, structured_present, context, detach_species=True
            )
            identity_pool = self._pool(structured_latent, self.species_variable)
            identity_logits = self._decode_pooled(
                identity_pool, self.species_variable
            ) + self._identity_detail_logits(identity_pool)
            identity_error = F.cross_entropy(
                identity_logits, target_species, reduction="none"
            ) / math.log(max(identity_logits.shape[-1], 2))
            structured_terms = [
                (identity_error * family_valid).sum()
                / family_valid.sum().clamp_min(1)
            ]
            probability = identity_logits.softmax(-1)
            family_probability = probability.new_zeros(batch, self.family_count)
            family_probability.scatter_add_(
                1, self.species_family.expand(batch, -1), probability
            )
            family_error = -family_probability.gather(
                1, self.species_family[target_species, None]
            ).squeeze(1).clamp_min(1e-8).log() / math.log(max(self.family_count, 2))
            structured_terms.append(
                (family_error * family_valid).sum()
                / family_valid.sum().clamp_min(1)
            )
            trait_names = {
                "seasonality", "water", "soil_drainage", "form",
                "plant_type", "growth_rate", "sun", "ease_of_care",
            }
            for variable in self.variables:
                if variable.name not in trait_names:
                    continue
                valid = observed[variable.name] & photo_row
                if valid.any():
                    error = F.cross_entropy(
                        self.decode(structured_latent, variable.name),
                        values[variable.name].long(), reduction="none"
                    ) / math.log(max(variable.num_classes, 2))
                    structured_terms.append(
                        (error * valid).sum() / valid.sum().clamp_min(1)
                    )
            if pollinator_target is not None:
                pollinator_logits = self._pollinator_logits(structured_latent)
                error = -(pollinator_target * F.log_softmax(
                    pollinator_logits, -1
                )).sum(-1) / math.log(self.poll_head.out_features)
                structured_terms.append(
                    (error * pollinator_valid).sum()
                    / pollinator_valid.sum().clamp_min(1)
                )
            structured_loss = 0.25 * torch.stack(structured_terms).mean()
            return loss, structured_loss
        return loss

def build_model(source, variable_specs, always_dims, device: str, design: Experiment = EXPERIMENT) -> MeshModel:
    variables = [Variable(**spec) for spec in variable_specs]
    return MeshModel(
        variables,
        always_dims,
        source,
        d_model=design.width,
        levels=design.levels,
        log2_size=design.hash_log2,
        n_latents=design.latents,
        n_layers=design.layers,
    ).to(device)


def attach_naip_patches(source, cache: str, device: str) -> None:
    root = Path(cache).expanduser()
    patch32 = root / "gbif_naip_dinov3_patch32_v1"
    if (patch32 / "manifest.npz").exists():
        manifest = np.load(patch32 / "manifest.npz")
        bytes_per_row = 32 * 32 * 1024 * np.dtype(np.float16).itemsize
        eager_gb = source.n * bytes_per_row / 1e9
        max_eager_gb = float(os.environ.get("MESH_NAIP_PATCH_MAX_EAGER_GB", "24"))
        if eager_gb > max_eager_gb and os.environ.get("MESH_NAIP_PATCH_ALLOW_EAGER") != "1":
            raise RuntimeError(
                "DINOv3 patch32 cache requires streaming for this source size: "
                f"{source.n:,} rows would eagerly stage about {eager_gb:.1f}GB. "
                "Run a smaller proxy slice, raise MESH_NAIP_PATCH_MAX_EAGER_GB, "
                "or set MESH_NAIP_PATCH_ALLOW_EAGER=1 if this is intentional."
            )
        manifest_row = {
            int(g): i for i, g in enumerate(manifest["gbifID"].astype(np.int64))
        }
        source_ids = np.asarray(source.gbifID).astype(np.int64)
        source_row = {int(g): i for i, g in enumerate(source_ids)}
        lookup = np.full(source.n, -1, np.int64)
        tokens, coords = [], []
        for file in sorted(patch32.glob("chunk[0-9]*.npz")):
            z = np.load(file, allow_pickle=True)
            patch = z["patch"]
            patch_lat = z["patch_lat"].astype(np.float32)
            patch_lon = z["patch_lon"].astype(np.float32)
            for j, gid in enumerate(z["gbifID"].astype(np.int64)):
                row = source_row.get(int(gid))
                if row is None or lookup[row] >= 0:
                    continue
                lookup[row] = len(tokens)
                tokens.append(patch[j])
                k = manifest_row[int(gid)]
                elev = np.full(patch_lat[j].shape, manifest["elev_m"][k], np.float32)
                day = np.full(patch_lat[j].shape, manifest["event_day"][k], np.float32)
                coords.append(np.stack((patch_lat[j], patch_lon[j], elev, day), -1))
        if not tokens:
            raise FileNotFoundError(f"empty NAIP DINOv3 patch32 cache: {patch32}")
        source.naip_patch_tokens = torch.from_numpy(np.stack(tokens)).to(device)
        source.naip_patch_coords = torch.from_numpy(np.stack(coords)).to(device)
        source.naip_patch_dim = int(source.naip_patch_tokens.shape[-1])
        source.naip_patch_grid = int(source.naip_patch_tokens.shape[1])
        source.naip_patch_views = 1
        index = torch.from_numpy(lookup).to(device)
        have = index >= 0
        source.extra["naip_patch_index"] = (index, have, 1)
        print(
            f"NAIP DINOv3 patch32 state {int(have.sum()):,}/{source.n:,} observations  "
            f"tokens {tuple(source.naip_patch_tokens.shape)}",
            flush=True,
        )
        return

    directory = root / "naip_dinov2_patch8_v1"
    metadata = directory / "metadata.json"
    if not metadata.exists():
        raise FileNotFoundError(f"incomplete NAIP patch cache: {directory}")
    rows = np.load(directory / "rows.npy")
    gbif_id = np.load(directory / "gbifID.npy")
    valid = np.load(directory / "valid.npy")
    if not np.array_equal(np.asarray(source.gbifID)[rows], gbif_id):
        raise ValueError("NAIP patch cache does not match the assembled dataset")
    lookup = np.full(source.n, -1, np.int64)
    lookup[rows[valid]] = np.arange(int(valid.sum()))
    tokens = np.load(directory / "tokens.npy", mmap_mode="r")[valid]
    source.naip_patch_tokens = torch.from_numpy(
        np.asarray(tokens).copy()
    ).to(device)
    source.naip_patch_coords = None
    source.naip_patch_dim = int(source.naip_patch_tokens.shape[-1])
    source.naip_patch_grid = 8
    source.naip_patch_views = int(source.naip_patch_tokens.shape[1])
    index = torch.from_numpy(lookup).to(device)
    have = index >= 0
    source.extra["naip_patch_index"] = (index, have, 1)
    print(
        f"NAIP DINOv2 patch state {int(have.sum()):,}/{source.n:,} observations  "
        f"tokens {tuple(source.naip_patch_tokens.shape)}",
        flush=True,
    )


def train(
    cache: str,
    device: str,
    design: Experiment = EXPERIMENT,
    *,
    checkpoint_steps: frozenset[int] = frozenset(),
    checkpoint_dir: Path | None = None,
):
    if not design.reader_only and not 0 <= design.reader_steps < design.steps:
        raise ValueError("reader_steps must fall between 0 and total steps")
    if design.reader_only and not design.init_checkpoint:
        raise ValueError("MESH_READER_ONLY requires MESH_INIT_CHECKPOINT")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(design.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(design.seed)
    source, variable_specs, always_dims = load_data(cache, device)
    attach_naip_patches(source, cache, device)
    if design.width != 128:
        candidate_rng = torch.random.get_rng_state()
        candidate_cuda_rng = torch.cuda.get_rng_state_all() \
            if device.startswith("cuda") else None
        control = build_model(
            source, variable_specs, always_dims, device,
            replace(design, width=128),
        )
        control_rng = torch.random.get_rng_state()
        control_cuda_rng = torch.cuda.get_rng_state_all() \
            if device.startswith("cuda") else None
        del control
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.set_rng_state_all(candidate_cuda_rng)
        torch.random.set_rng_state(candidate_rng)
        model = build_model(source, variable_specs, always_dims, device, design)
        torch.random.set_rng_state(control_rng)
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(control_cuda_rng)
    else:
        model = build_model(source, variable_specs, always_dims, device, design)
    if design.init_checkpoint:
        checkpoint = Path(design.init_checkpoint).expanduser()
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        print(
            f"initialized from {checkpoint}  "
            f"missing={len(incompatible.missing_keys)}  "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    if checkpoint_steps:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_steps are requested")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if 0 in checkpoint_steps:
            torch.save(model.state_dict(), checkpoint_dir / "step_000000.pt")
    relation_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("species_myco_head.")
    ]
    relation_ids = {
        id(parameter) for name, parameter in model.named_parameters()
        if name.startswith(RELATION_PARAMETERS)
    }
    lens_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(SPECIES_LENS_PARAMETERS + LFMC_LENS_PARAMETERS)
    ]
    lens_ids = {id(parameter) for parameter in lens_parameters}
    calibration_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(CALIBRATION_PARAMETERS)
    ]
    calibration_ids = {id(parameter) for parameter in calibration_parameters}
    position_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(POSITION_PARAMETERS)
    ]
    position_ids = {id(parameter) for parameter in position_parameters}
    base_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in relation_ids
        and id(parameter) not in lens_ids
        and id(parameter) not in calibration_ids
        and id(parameter) not in position_ids
    ]
    optimizer = torch.optim.AdamW(
        base_parameters,
        lr=design.learning_rate,
        weight_decay=design.weight_decay,
        fused=device.startswith("cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, design.steps)
    relation_optimizer = None
    relation_scheduler = None
    if relation_parameters:
        relation_optimizer = torch.optim.AdamW(
            relation_parameters,
            lr=design.learning_rate,
            weight_decay=design.weight_decay,
            fused=device.startswith("cuda"),
        )
        relation_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            relation_optimizer, design.steps
        )
    detail_optimizer = None
    detail_scheduler = None
    lens_optimizer = None
    lens_scheduler = None
    calibration_optimizer = None
    calibration_scheduler = None
    position_optimizer = torch.optim.AdamW(
        position_parameters,
        lr=design.learning_rate * 2.0,
        weight_decay=design.weight_decay,
        fused=device.startswith("cuda"),
    )
    position_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        position_optimizer, design.steps
    )
    reader_budget = design.steps if design.reader_only else design.reader_steps
    reader_start = 0 if design.reader_only else design.steps - design.reader_steps
    lfmc_train_index = None
    if model.lfmc_head is not None and hasattr(source, "lfmc_valid"):
        lfmc_mask = source.lfmc_valid[source.cls[source.train_index]]
        lfmc_train_index = source.train_index[lfmc_mask]
        print(f"LFMC reader examples {len(lfmc_train_index):,}", flush=True)
    model.train()
    started = time.time()
    for step in range(design.steps):
        if design.reader_steps and step == reader_start:
            model.reader_phase = True
            model.rank_aligned_expansion = design.reader_only
            for name, parameter in model.named_parameters():
                is_reader = name.startswith(READER_PARAMETERS) \
                            or name.startswith(SPECIES_LENS_PARAMETERS) \
                            or name.startswith(LFMC_LENS_PARAMETERS) \
                            or name.startswith(CALIBRATION_PARAMETERS)
                if design.reader_only:
                    is_reader = name.startswith(EXPANSION_PARAMETERS)
                if design.reader_only and name.startswith("species_graph."):
                    is_reader = False
                parameter.requires_grad_(is_reader)
            graph_parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith("species_graph.") and parameter.requires_grad
            ]
            graph_ids = {id(parameter) for parameter in graph_parameters}
            detail_parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith(IDENTITY_DETAIL_PARAMETERS)
            ]
            detail_ids = {id(parameter) for parameter in detail_parameters}
            reader_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) not in graph_ids
                and id(parameter) not in detail_ids
                and id(parameter) not in relation_ids
                and id(parameter) not in lens_ids
                and id(parameter) not in calibration_ids
                and id(parameter) not in position_ids
            ]
            base_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) not in detail_ids
                and id(parameter) not in relation_ids
                and id(parameter) not in lens_ids
                and id(parameter) not in calibration_ids
                and id(parameter) not in position_ids
            ]
            del optimizer, scheduler, position_optimizer, position_scheduler
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            optimizer = torch.optim.AdamW(
                (
                    {"params": reader_parameters, "lr": design.learning_rate * 0.2},
                    {"params": graph_parameters,
                     "lr": design.learning_rate * design.graph_learning_rate_scale},
                ),
                lr=design.learning_rate * 0.2,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, reader_budget
            )
            detail_optimizer = torch.optim.AdamW(
                detail_parameters,
                lr=design.learning_rate * 0.2,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            detail_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                detail_optimizer, reader_budget
            )
            lens_optimizer = torch.optim.AdamW(
                lens_parameters,
                lr=design.learning_rate * 0.4,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            lens_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                lens_optimizer, reader_budget
            )
            calibration_optimizer = torch.optim.AdamW(
                calibration_parameters,
                lr=design.learning_rate * 10.0,
                weight_decay=0.0,
                fused=device.startswith("cuda"),
            )
            calibration_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                calibration_optimizer, reader_budget
            )
            position_optimizer = torch.optim.AdamW(
                position_parameters,
                lr=design.learning_rate * 2.0,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            position_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                position_optimizer, reader_budget
            )
            print(
                f"reader phase {reader_budget} steps  "
                f"parameters {sum(parameter.numel() for parameter in reader_parameters):,}  "
                f"graph parameters {sum(parameter.numel() for parameter in graph_parameters):,}  "
                f"detail parameters {sum(parameter.numel() for parameter in detail_parameters):,}  "
                f"lens parameters {sum(parameter.numel() for parameter in lens_parameters):,}  "
                f"position parameters {sum(parameter.numel() for parameter in position_parameters):,}  "
                f"graph lr scale {design.graph_learning_rate_scale:g}",
                flush=True,
            )
        index = source.train_index[torch.randint(len(source.train_index), (design.batch,), device=device)]
        values, observed, coords, neighbors, manifolds, neighbor_values = source.batch(index)
        context = model.context(
            coords, neighbors, manifolds, neighbor_values
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=TRAIN_BFLOAT16 and device.startswith("cuda"),
        ):
            objective = model.reconstruction_loss(
                values, observed, context, design.hide_probability
            )
        if isinstance(objective, tuple):
            loss, structured_loss = objective
        else:
            loss, structured_loss = objective, None
        if getattr(model, "reader_phase", False) \
                and lfmc_train_index is not None \
                and len(lfmc_train_index) > 2:
            devices = [torch.cuda.current_device()] \
                      if device.startswith("cuda") else []
            with torch.random.fork_rng(devices=devices):
                auxiliary_seed = design.seed + 100_000 + step
                torch.manual_seed(auxiliary_seed)
                if devices:
                    torch.cuda.manual_seed_all(auxiliary_seed)
                lfmc_index = lfmc_train_index[torch.randint(
                    len(lfmc_train_index), (design.batch,), device=device
                )]
                lfmc_values, lfmc_observed, lfmc_coords, lfmc_neighbors, \
                    lfmc_manifolds, lfmc_neighbor_values = source.batch(
                        lfmc_index
                    )
                lfmc_context = model.context(
                    lfmc_coords, lfmc_neighbors, lfmc_manifolds,
                    lfmc_neighbor_values
                )
                lfmc_present = {
                    name: lfmc_observed[name]
                    if name in model.environment_names
                    else torch.zeros_like(lfmc_observed[name])
                    for name in model.names
                }
                lfmc_present = model._with_worldclim_observed(
                    lfmc_present, lfmc_observed
                )
                lfmc_latent = model.encode(
                    lfmc_values, lfmc_present, lfmc_context,
                    detach_species=True
                )
                lfmc_pool = model._pool(lfmc_latent, "lfmc")
                prediction = model.lfmc_head(
                    lfmc_pool.detach()
                ).squeeze(-1).detach() + model._lfmc_lens_residual(
                    lfmc_pool.detach()
                )
                target = torch.log(lfmc_values["_lfmc"].clamp_min(1.0))
                valid = lfmc_values["_lfmc_valid"].bool()
                prediction = prediction[valid]
                target = target[valid]
                prediction = prediction - prediction.mean()
                target = target - target.mean()
                correlation = (prediction * target).sum() / (
                    prediction.square().sum().sqrt()
                    * target.square().sum().sqrt()
                ).clamp_min(1e-8)
            loss = loss + 1.0 - correlation
        total_loss = loss if structured_loss is None else loss + structured_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        if relation_optimizer is not None:
            relation_optimizer.zero_grad(set_to_none=True)
        if detail_optimizer is not None:
            detail_optimizer.zero_grad(set_to_none=True)
        if lens_optimizer is not None:
            lens_optimizer.zero_grad(set_to_none=True)
        if calibration_optimizer is not None:
            calibration_optimizer.zero_grad(set_to_none=True)
        position_optimizer.zero_grad(set_to_none=True)
        gradient_cosine = None
        if structured_loss is None:
            loss.backward()
        else:
            trainable = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ]
            loss.backward(retain_graph=True)
            base_grad = {
                id(parameter): parameter.grad.detach().clone()
                for parameter in trainable if parameter.grad is not None
            }
            optimizer.zero_grad(set_to_none=True)
            if relation_optimizer is not None:
                relation_optimizer.zero_grad(set_to_none=True)
            if detail_optimizer is not None:
                detail_optimizer.zero_grad(set_to_none=True)
            if lens_optimizer is not None:
                lens_optimizer.zero_grad(set_to_none=True)
            if calibration_optimizer is not None:
                calibration_optimizer.zero_grad(set_to_none=True)
            position_optimizer.zero_grad(set_to_none=True)
            structured_loss.backward()
            shared = [
                parameter for parameter in trainable
                if parameter.grad is not None and id(parameter) in base_grad
            ]
            dot = sum(
                (parameter.grad * base_grad[id(parameter)]).sum()
                for parameter in shared
            )
            base_norm = sum(
                base_grad[id(parameter)].square().sum()
                for parameter in shared
            ).clamp_min(1e-12)
            structured_norm = sum(
                parameter.grad.square().sum() for parameter in shared
            ).clamp_min(1e-12)
            gradient_cosine = float(
                dot / (base_norm.sqrt() * structured_norm.sqrt())
            )
            projection = dot / base_norm if dot < 0 else dot.new_zeros(())
            for parameter in trainable:
                base = base_grad.get(id(parameter))
                auxiliary = parameter.grad
                if base is None:
                    continue
                if auxiliary is None:
                    parameter.grad = base
                else:
                    parameter.grad = base + auxiliary - projection * base
        torch.nn.utils.clip_grad_norm_(base_parameters, 5.0)
        if relation_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(relation_parameters, 5.0)
        if detail_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(detail_parameters, 5.0)
        if lens_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(lens_parameters, 5.0)
        if calibration_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(calibration_parameters, 5.0)
        torch.nn.utils.clip_grad_norm_(position_parameters, 5.0)
        optimizer.step()
        scheduler.step()
        if relation_optimizer is not None:
            relation_optimizer.step()
            relation_scheduler.step()
        if detail_optimizer is not None:
            detail_optimizer.step()
            detail_scheduler.step()
        if lens_optimizer is not None:
            lens_optimizer.step()
            lens_scheduler.step()
        if calibration_optimizer is not None:
            calibration_optimizer.step()
            calibration_scheduler.step()
        position_optimizer.step()
        position_scheduler.step()
        for module in model.modules():
            if hasattr(module, "clamp_per_level_scale"):
                module.clamp_per_level_scale()
        completed = step + 1
        if completed in checkpoint_steps:
            torch.save(model.state_dict(), checkpoint_dir / f"step_{completed:06d}.pt")
        if step % 100 == 0 or step + 1 == design.steps:
            conflict = "" if gradient_cosine is None \
                       else f"  gradient_cosine {gradient_cosine:+.3f}"
            print(
                f"step {step:>5}  loss {float(total_loss):.4f}{conflict}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )

    checkpoint = Path(__file__).with_name("checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"checkpoint: {checkpoint}", flush=True)
    return model, source


def reader_screen(cache: str, device: str = "cuda", batch: int = 512):
    """Cheap, non-promotional capability screen for a frozen-mesh reader."""
    from deepearth.autoresearch.main.harness import evaluate as canonical

    model, source = train(cache, device)
    model.eval()
    raw = canonical._evaluate_benchmarks_once(model, source, device, batch=batch)
    harmonic = canonical.net_score(raw)
    arithmetic = canonical.arithmetic_net(raw)
    print(canonical.format_benchmarks(raw), flush=True)
    print("READER SCREEN RECEIPT: " + json.dumps({
        "protocol": f"{canonical.BENCHMARK_PROTOCOL}+reader-screen-v1",
        "batch": batch,
        "scores": raw,
        "harmonic": harmonic,
        "arithmetic": arithmetic,
    }, sort_keys=True), flush=True)
    return model, source, raw
