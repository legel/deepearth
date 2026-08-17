"""Editable mesh thesis: situated signals write state; fusion reads only state.

Run through the fixed evaluator:
    python mesh_research/evaluate.py --cache /path/to/deepcal --device cuda

`data.py` and the canonical evaluator are fixed. This is the only research-editable
file: architecture, writes, fusion, objectives, optimization, and training live here.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    width: int = 128
    levels: int = 12
    hash_log2: int = 14
    latents: int = 16
    layers: int = 2
    hide_probability: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    reader_steps: int = int(os.environ.get("MESH_READER_STEPS", "100"))
    graph_learning_rate_scale: float = float(os.environ.get("MESH_GRAPH_LR_SCALE", "0.02"))


EXPERIMENT = Experiment()

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
    "identity_detail_query", "identity_detail_reader.",
    "identity_detail_norm.", "identity_detail_output_norm.",
    "identity_detail_gate", "identity_detail_cell_key",
    "identity_detail_level_key", "identity_detail_lens_key",
    "lfmc_head.", "myco_head.", "species_myco_head.", "myco_relation_gate",
    "flower_head.",
    "mesh_read_query.", "mesh_read_gate.", "mesh_scale_read_gate.",
    "mesh_scale_attention_gate.",
    "task_mesh_reader.", "task_mesh_reader_gate.", "task_mesh_reader_norm.",
    "task_mesh_reader_output_norm.",
    "mesh_prior_read_gate.", "mesh_prior_information_gate.",
    "mesh_task_norm.", "mesh_scale_task_norm.", "mesh_prior_task_norm.",
    "mesh_condition_gate.", "mesh_condition_norm.",
    "mesh_cell_key", "mesh_level_key", "mesh_lens_key",
)
IDENTITY_DETAIL_PARAMETERS = ("identity_detail_",)
RELATION_PARAMETERS = ("species_myco_head.", "myco_relation_gate")


def signal_lens(name: str, kind: str | None = None) -> str:
    if name in {"climate", "soil", "clay", "topo", "hydro", "water", "soil_drainage"}:
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
        projections = ((0, 1, 3), (1, 2, 3), (0, 2, 3))
        temporal = torch.cat([
            encoder(xyzt[..., axes].contiguous(), size=1.0).reshape(
                *lead, self.levels, self.features)
            for encoder, axes in zip(self.temporal, projections)
        ], -1)
        return spatial, temporal


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
        self.always_names = tuple(always_dims)
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
        self.register_buffer("species_family", source.class_group)
        self.family_count = len(source.group_names)
        self.environment_names = tuple(
            name for name in ("climate", "soil", "naip_rgb", "naip_ir", "clay", "topo", "chm", "hydro")
            if name in self.names
        )

        write_names = [*self.names, *self.always_names]
        self.write_type = nn.ParameterDict({n: nn.Parameter(torch.randn(d_model) * 0.02) for n in write_names})
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
        if self.poll_head is not None:
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
            for name in write_names
        })
        self.fiber_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.sparse_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.coarse_scale_exchange = nn.Linear(d_model, d_model, bias=False)
        self.fine_scale_exchange = nn.Linear(d_model, d_model, bias=False)
        self.scale_exchange_gate = nn.Parameter(torch.full((len(LENSES),), 0.05))
        self.scale_message_norm = nn.LayerNorm(d_model)
        self.mesh_linear_reconstruct = nn.ModuleDict({
            name: nn.Linear(d_model, d_model, bias=False) for name in write_names
        })
        self.lens_exchange_norm = nn.LayerNorm(d_model)
        self.lens_exchange = nn.Parameter(
            torch.zeros(levels, len(LENSES), len(LENSES))
        )
        torch.random.set_rng_state(sidecar_rng)
        self._fiber_summary = None
        self._fiber_mesh = None
        self._fiber_prior_mesh = None
        self._latest_fiber_prior = None

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
        species = self._species(species_mask)
        if detach_species:
            species = species.detach()
        write_mask = dict(present)
        for name in self.always_names:
            if name in values:
                write_mask[name] = values[name].isfinite().all(-1) & (values[name].norm(dim=-1) > 1e-6)
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
            -1, selected, selected_score.softmax(-1)
        )
        route = sparse_weight.detach() + dense_weight - dense_weight.detach()
        mesh_read = torch.einsum("blk,bkd->bld", route, mesh_tokens)
        latent = latent + torch.tanh(self.sparse_fusion_gate) * mesh_read
        return latent

    def _pool(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        base_name = name if name in self.names else self.species_variable
        query = self.decode_query[self.names.index(base_name)]
        weight = torch.softmax((latent @ query) / math.sqrt(self.d_model), -1)
        pooled = torch.einsum("bl,bld->bd", weight, latent)
        if self._fiber_summary is None or name not in self.mesh_read_query:
            return pooled
        fibers = self._fiber_summary.flatten(1, 2)
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
        task_tokens = self.task_mesh_reader_norm(fibers)
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
            return pooled
        cells = self._fiber_mesh.shape[1]
        cell_key = torch.cat((
            self.mesh_cell_key[:1],
            self.mesh_cell_key[1:].expand(cells - 1, -1),
        ))
        scale_keys = self._fiber_mesh \
            + cell_key.view(1, cells, 1, 1, self.d_model) \
            + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model) \
            + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
        scale_fibers = self._fiber_mesh.flatten(1, 3)
        scale_keys = scale_keys.flatten(1, 3)
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
        selected_keys = scale_keys.gather(
            1, scale_index[..., None].expand(-1, -1, self.d_model)
        )
        selected_fibers = scale_fibers.gather(
            1, scale_index[..., None].expand(-1, -1, self.d_model)
        )
        scale_query = task_query + self.task_mesh_reader_output_norm(task_read)
        scale_attention = self.task_mesh_reader(
            scale_query.unsqueeze(1),
            self.task_mesh_reader_norm(selected_keys),
            self.task_mesh_reader_norm(selected_fibers),
            need_weights=False,
        )[0].squeeze(1)
        pooled = pooled + torch.tanh(self.mesh_scale_attention_gate[name]) \
                 * self.mesh_scale_task_norm(scale_attention)
        prior_mesh = self._fiber_prior_mesh.detach()
        prior_fibers = prior_mesh.flatten(1, 3)
        prior_keys = (
            prior_mesh
            + cell_key.view(1, cells, 1, 1, self.d_model)
            + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
            + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
        ).flatten(1, 3)
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
        return pooled + torch.tanh(self.mesh_prior_read_gate[name]) * confidence \
                        * self.mesh_prior_task_norm(prior_read)

    def _pollinator_pool(self, latent: torch.Tensor, *, isolated: bool = False) -> torch.Tensor:
        pooled = self._pool(latent, "pollinator")
        if self.pollinator_reader is None or self._fiber_mesh is None:
            return pooled
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
        return pooled + torch.tanh(self.pollinator_reader_gate) \
                        * self.pollinator_reader_output_norm(read)

    def _decode_pooled(self, pooled: torch.Tensor, name: str) -> torch.Tensor:
        if name == self.species_variable:
            return pooled @ self._refined_species.t()
        return self.decoders[name](pooled)

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
        pooled = pooled.detach()
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
        base = self.myco_head(self._pool(latent, "myco"))
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

    def decode(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        pooled = self._pool(latent, name)
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
        latent = self.encode(values, present, context)
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
                pooled = self._pollinator_pool(latent)
                prediction = self.poll_head(pooled)
                if pollinator_species is not None:
                    prediction = torch.logaddexp(
                        F.log_softmax(prediction.float(), -1) + math.log(0.5),
                        pollinator_species + math.log(0.5),
                    )
                out[name] = prediction
            elif name == "lfmc":
                pooled = self._pool(latent, "lfmc")
                out[name] = self.lfmc_head(pooled).squeeze(-1).exp()
            elif name == "myco":
                out[name] = self._myco_logits(latent)
            elif name == "flower":
                pooled = self._pool(latent, "flower")
                out[name] = torch.sigmoid(self.flower_head(pooled).squeeze(-1))
            else:
                prediction = self.decode(latent, name)
                if name == self.species_variable and (
                    not given or tuple(given) == self.environment_names
                ):
                    prediction = self._hierarchical_family_read(prediction)
                out[name] = prediction
        return out

    def reconstruction_loss(self, values, observed, context, hide_probability: float = 0.5):
        batch = context["position"].shape[0]
        present = {name: (torch.rand(batch, device=context["position"].device) > hide_probability) & observed[name]
                   for name in self.names}
        blank = torch.rand(batch, device=context["position"].device) < 0.15
        for name in present:
            present[name] &= ~blank
        latent = self.encode(values, present, context)
        terms = []
        fiber_terms = []
        mesh_terms = []
        pollinator_target = None
        pollinator_valid = None
        pollinator_structured_term = None
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
            structured = self.poll_head(
                self._pollinator_pool(latent, isolated=True)
            )
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
            error = (self.lfmc_head(self._pool(latent, "lfmc")).squeeze(-1)
                     - torch.log(values["_lfmc"].clamp_min(1.0))).square()
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
               + 0.05 * torch.stack(mesh_terms).mean()
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
        environment_latent = self.encode(values, environment_present, context, detach_species=True)
        environment_pool = self._pool(environment_latent, self.species_variable)
        family_logits = environment_pool.float() \
                        @ self._refined_species.detach().float().t()
        probability = family_logits.softmax(-1)
        family_probability = probability.new_zeros(batch, self.family_count)
        family_probability.scatter_add_(1, self.species_family.expand(batch, -1), probability)
        target_family = self.species_family[values[self.species_variable].long()]
        family_error = -family_probability.gather(
            1, target_family[:, None]
        ).squeeze(1).clamp_min(1e-8).log()
        family_valid = observed[self.species_variable]
        family_term = (family_error * family_valid).sum() / family_valid.sum().clamp_min(1) \
                      / math.log(max(self.family_count, 2))
        loss = loss + 0.1 * family_term
        if getattr(self, "reader_phase", False):
            relation_logits = self._identity_detail_logits(environment_pool)
            target_species = values[self.species_variable].long()
            calibrated_logits = family_logits.detach() + relation_logits
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
            pollinator_present = {
                name: observed[name]
                if name == self.species_variable or name in self.environment_names
                else torch.zeros_like(observed[name])
                for name in self.names
            }
            devices = [torch.cuda.current_device()] if loss.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                pollinator_latent = self.encode(values, pollinator_present, context)
            pollinator_logits = self.poll_head(
                self._pollinator_pool(pollinator_latent, isolated=True)
            )
            interaction = -(
                pollinator_target * F.log_softmax(pollinator_logits, -1)
            ).sum(-1) / math.log(self.poll_head.out_features)
            loss = loss + 0.25 * (interaction * pollinator_valid).sum() \
                   / pollinator_valid.sum().clamp_min(1)
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


def train(
    cache: str,
    device: str,
    design: Experiment = EXPERIMENT,
    *,
    checkpoint_steps: frozenset[int] = frozenset(),
    checkpoint_dir: Path | None = None,
):
    if not 0 <= design.reader_steps < design.steps:
        raise ValueError("reader_steps must fall between 0 and total steps")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(design.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(design.seed)
    source, variable_specs, always_dims = load_data(cache, device)
    model = build_model(source, variable_specs, always_dims, device, design)
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
    base_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in relation_ids
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
    reader_start = design.steps - design.reader_steps
    model.train()
    started = time.time()
    for step in range(design.steps):
        if design.reader_steps and step == reader_start:
            model.reader_phase = True
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(READER_PARAMETERS))
            graph_parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith("species_graph.")
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
            ]
            base_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) not in detail_ids
                and id(parameter) not in relation_ids
            ]
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
                optimizer, design.reader_steps
            )
            detail_optimizer = torch.optim.AdamW(
                detail_parameters,
                lr=design.learning_rate * 0.2,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            detail_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                detail_optimizer, design.reader_steps
            )
            print(
                f"reader phase {design.reader_steps} steps  "
                f"parameters {sum(parameter.numel() for parameter in reader_parameters):,}  "
                f"graph parameters {sum(parameter.numel() for parameter in graph_parameters):,}  "
                f"detail parameters {sum(parameter.numel() for parameter in detail_parameters):,}  "
                f"graph lr scale {design.graph_learning_rate_scale:g}",
                flush=True,
            )
        index = source.train_index[torch.randint(len(source.train_index), (design.batch,), device=device)]
        values, observed, coords, neighbors, manifolds, neighbor_values = source.batch(index)
        context = model.context(coords, neighbors, manifolds, neighbor_values)
        loss = model.reconstruction_loss(values, observed, context, design.hide_probability)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        if relation_optimizer is not None:
            relation_optimizer.zero_grad(set_to_none=True)
        if detail_optimizer is not None:
            detail_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_parameters, 5.0)
        if relation_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(relation_parameters, 5.0)
        if detail_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(detail_parameters, 5.0)
        optimizer.step()
        scheduler.step()
        if relation_optimizer is not None:
            relation_optimizer.step()
            relation_scheduler.step()
        if detail_optimizer is not None:
            detail_optimizer.step()
            detail_scheduler.step()
        for module in model.modules():
            if hasattr(module, "clamp_per_level_scale"):
                module.clamp_per_level_scale()
        completed = step + 1
        if completed in checkpoint_steps:
            torch.save(model.state_dict(), checkpoint_dir / f"step_{completed:06d}.pt")
        if step % 100 == 0 or step + 1 == design.steps:
            print(f"step {step:>5}  loss {float(loss):.4f}  elapsed {time.time() - started:.1f}s", flush=True)

    checkpoint = Path(__file__).with_name("checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"checkpoint: {checkpoint}", flush=True)
    return model, source
