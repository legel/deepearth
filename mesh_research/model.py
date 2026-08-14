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


EXPERIMENT = Experiment()


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

        write_names = [*self.names, *self.always_names]
        self.write_type = nn.ParameterDict({n: nn.Parameter(torch.randn(d_model) * 0.02) for n in write_names})
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
        self.lfmc_head = nn.Linear(d_model, 1) if hasattr(source, "lfmc") else None
        self.myco_head = nn.Linear(d_model, 5) if hasattr(source, "myco") else None
        self.flower_head = nn.Linear(d_model, 1) if hasattr(source, "flower") else None
        self.species_myco_head = None

    def _species(self) -> torch.Tensor:
        refined = self.species_graph._seed() if self._ablate_species else self.species_graph()
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

    def encode(self, values: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor], context: dict):
        species = self._species()
        write_mask = dict(present)
        for name in self.always_names:
            if name in values:
                write_mask[name] = values[name].isfinite().all(-1) & (values[name].norm(dim=-1) > 1e-6)
        query = self._write(context["query_state"], values, write_mask, species)

        neighbor = context["neighbor_state"]
        neighbor_values = context["neighbor_values"]
        if neighbor_values:
            masks = {name: torch.ones(value.shape[:-1] if value.dim() > 2 else value.shape,
                                      dtype=torch.bool, device=value.device)
                     for name, value in neighbor_values.items()}
            neighbor = self._write(neighbor, neighbor_values, masks, species)
        neighbor = self.neighbor_norm(neighbor).flatten(1, 2)
        tokens = torch.cat((query, neighbor), 1)

        latent = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        latent = latent + self.read(latent, self.read_norm(tokens), self.read_norm(tokens), need_weights=False)[0]
        for block in self.blocks:
            latent = block(latent)
        return latent

    def _pool(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        query = self.decode_query[self.names.index(name)]
        weight = torch.softmax((latent @ query) / math.sqrt(self.d_model), -1)
        return torch.einsum("bl,bld->bd", weight, latent)

    def decode(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        pooled = self._pool(latent, name)
        if name == self.species_variable:
            return pooled @ self._refined_species.t()
        return self.decoders[name](pooled)

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
        pooled = self._pool(latent, self.species_variable)
        out = {}
        for name in targets:
            if name == "community":
                out[name] = self.community_head(pooled)
            elif name == "pollinator":
                out[name] = self.poll_head(pooled)
            elif name == "lfmc":
                out[name] = self.lfmc_head(pooled).squeeze(-1).exp()
            elif name == "myco":
                out[name] = self.myco_head(pooled)
            elif name == "flower":
                out[name] = torch.sigmoid(self.flower_head(pooled).squeeze(-1))
            else:
                out[name] = self.decode(latent, name)
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

        pooled = self._pool(latent, self.species_variable)
        if self.poll_head is not None and "_poll_idx" in values:
            logits = self.poll_head(pooled)
            target = torch.zeros_like(logits).scatter_add_(1, values["_poll_idx"].clamp_min(0),
                                                           values["_poll_frq"].float())
            valid = values["_poll_valid"].float()
            error = -(target * F.log_softmax(logits, -1)).sum(-1)
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        if self.lfmc_head is not None and "_lfmc" in values:
            valid = values["_lfmc_valid"].float()
            error = (self.lfmc_head(pooled).squeeze(-1) - torch.log(values["_lfmc"].clamp_min(1.0))).square()
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        if self.myco_head is not None and "_myco" in values:
            valid = values["_myco_valid"].float()
            error = F.cross_entropy(self.myco_head(pooled), values["_myco"].long().clamp_min(0), reduction="none")
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        if self.flower_head is not None and "_flower" in values:
            valid = values["_flower_valid"].float()
            error = F.binary_cross_entropy_with_logits(self.flower_head(pooled).squeeze(-1),
                                                        values["_flower"].float(), reduction="none")
            terms.append(0.1 * (error * valid).sum() / valid.sum().clamp_min(1))
        return torch.stack(terms).mean()

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


def train(cache: str, device: str, design: Experiment = EXPERIMENT):
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(design.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(design.seed)
    source, variable_specs, always_dims = load_data(cache, device)
    model = build_model(source, variable_specs, always_dims, device, design)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=design.learning_rate,
        weight_decay=design.weight_decay,
        fused=device.startswith("cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, design.steps)
    model.train()
    started = time.time()
    for step in range(design.steps):
        index = source.train_index[torch.randint(len(source.train_index), (design.batch,), device=device)]
        values, observed, coords, neighbors, manifolds, neighbor_values = source.batch(index)
        context = model.context(coords, neighbors, manifolds, neighbor_values)
        loss = model.reconstruction_loss(values, observed, context, design.hide_probability)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        for module in model.modules():
            if hasattr(module, "clamp_per_level_scale"):
                module.clamp_per_level_scale()
        if step % 100 == 0 or step + 1 == design.steps:
            print(f"step {step:>5}  loss {float(loss):.4f}  elapsed {time.time() - started:.1f}s", flush=True)

    checkpoint = Path(__file__).with_name("checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"checkpoint: {checkpoint}", flush=True)
    return model, source
