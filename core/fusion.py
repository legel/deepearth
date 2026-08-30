"""DeepEarth fusion over a fibered, Earth4D-addressed world state."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.core.ecology import EcologicalReadoutMixin
from deepearth.core.objective import TrainingObjectiveMixin
from deepearth.core.layers import consume_rng, mlp, per_name, preserve_rng
from deepearth.core.reader import (
    MeshQueryReader,
    RoutedMeshReader,
    SegmentDenoiser,
    ScientificReadoutMixin,
    SpecialistMesh,
)
from deepearth.core.world_mesh import (
    FiberAdapter,
    LENSES,
    LENS_INDEX,
    MeshNeighborhood,
    WorldMesh,
    signal_lens,
)
from deepearth.encoders.biological.phylogenomic import SpeciesGraph

@dataclass(frozen=True)
class Variable:
    name: str
    kind: str
    dim: int = 0
    num_classes: int = 0
    reconstruct: bool = True


class DeepEarth(
    EcologicalReadoutMixin, ScientificReadoutMixin, TrainingObjectiveMixin, nn.Module
):
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
        self.always_names = (
            *always_dims, *(("worldclim",) if has_worldclim else ())
        )
        self._ablate_species = False

        self.reader_phase = False
        self.rank_aligned_expansion = False
        self._init_mesh(d_model, levels, log2_size)
        self._init_inputs(variables, always_dims, source, d_model)
        self._init_species(source, d_model)
        self._init_ecological_readers(source, d_model)
        write_names = [*self.names, *self.always_names]
        self._init_backbone(
            variables, write_names, d_model, levels, n_latents, n_layers, n_heads
        )
        consume_rng(nn.Linear(d_model, source.n_classes))
        self._init_scientific_heads(source, d_model, levels, n_heads)
        with preserve_rng():
            self._init_fiber_mesh(
                variables, write_names, d_model, levels, n_heads
            )
        with preserve_rng():
            self._init_specialists(variables, d_model, levels, n_heads)
        self._reset_runtime_state()

    def _init_mesh(self, d_model: int, levels: int, log2_size: int) -> None:
        self.mesh = WorldMesh(d_model, levels, log2_size)
        self.absolute_proj_s = self.mesh.spatial_projection
        self.absolute_proj_t = self.mesh.temporal_projection
        self.neighbors = MeshNeighborhood(d_model, levels, max(10, log2_size - 2))

    def _init_inputs(
        self, variables: Sequence[Variable], always_dims: Dict[str, int],
        source, d_model: int,
    ) -> None:
        self.adapters = nn.ModuleDict()
        self.category_inputs = nn.ModuleDict()
        for v in variables:
            if v.kind == "continuous":
                self.adapters[v.name] = FiberAdapter(v.dim, d_model)
            elif v.name != self.species_variable:
                self.category_inputs[v.name] = nn.Embedding(v.num_classes, d_model)
        for name, dim in always_dims.items():
            self.adapters[name] = FiberAdapter(dim, d_model)
        if "worldclim" in source.extra:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260824)
                self.adapters["worldclim"] = FiberAdapter(
                    int(source.extra["worldclim"][2]), d_model
                )

    def _init_species(self, source, d_model: int) -> None:
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
        with preserve_rng():
            self.species_niche_adapter = mlp(d_model, d_model, d_model)
            nn.init.zeros_(self.species_niche_adapter[-1].weight)
            nn.init.zeros_(self.species_niche_adapter[-1].bias)
        with preserve_rng():
            self.species_lens_reader_norm = nn.LayerNorm(d_model)
            self.species_lens_reader = nn.MultiheadAttention(
                d_model, 4, batch_first=True
            )
            nn.init.zeros_(self.species_lens_reader.out_proj.weight)
            nn.init.zeros_(self.species_lens_reader.out_proj.bias)
        self.register_buffer("species_family", source.class_group)
        self.family_count = len(source.group_names)
        self.environment_names = tuple(
            name for name in ("climate", "soil", "naip_rgb", "naip_ir", "clay", "topo", "chm", "hydro")
            if name in self.names
        )

    def _init_backbone(
        self,
        variables: Sequence[Variable],
        write_names: Sequence[str],
        d_model: int,
        levels: int,
        n_latents: int,
        n_layers: int,
        n_heads: int,
    ) -> None:
        self.write_names = tuple(write_names)
        base_write_names = [name for name in write_names if name != "worldclim"]
        self.write_type = per_name(
            base_write_names, lambda _: torch.randn(d_model) * 0.02
        )
        if "worldclim" in write_names:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260825)
                self.write_type["worldclim"] = nn.Parameter(
                    torch.randn(d_model) * 0.02
                )
        with preserve_rng():
            self.fiber_residual = nn.ModuleDict({
                name: nn.Linear(d_model, d_model, bias=False)
                for name in write_names
            })
            for residual in self.fiber_residual.values():
                nn.init.zeros_(residual.weight)
        self.write_gate = per_name(write_names, lambda _: torch.zeros(levels))
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
            self.decoders[v.name] = mlp(d_model, width, 2 * d_model)

    def _init_scientific_heads(
        self, source, d_model: int, levels: int, n_heads: int
    ) -> None:
        self.poll_head = nn.Linear(d_model, source.n_pollinators) if hasattr(source, "n_pollinators") else None
        self.pollinator_reader = None
        self.poll_transfer_head = None
        if self.poll_head is not None:
            with preserve_rng():
                self.poll_transfer_head = nn.Linear(
                    d_model, source.n_pollinators
                )
                self.poll_transfer_head.load_state_dict(
                    self.poll_head.state_dict()
                )
                self.pollinator_transfer_router = nn.Linear(1, 1)
                nn.init.zeros_(self.pollinator_transfer_router.weight)
                nn.init.constant_(self.pollinator_transfer_router.bias, -2.0)
            self.pollinator_log_temperature = nn.Parameter(
                torch.zeros(()), requires_grad=False
            )
            self.register_buffer("poll_species_idx", source.poll_idx.long(), persistent=False)
            self.register_buffer("poll_species_frq", source.poll_frq.float(), persistent=False)
            with preserve_rng():
                self.pollinator_reader = RoutedMeshReader(
                    d_model, n_heads, levels
                )
        with preserve_rng():
            self.identity_detail_reader = RoutedMeshReader(
                d_model, n_heads, levels
            )
        self.lfmc_head = nn.Linear(d_model, 1) if hasattr(source, "lfmc") else None
        if self.lfmc_head is not None:
            with preserve_rng():
                self.lfmc_lens_reader_norm = nn.LayerNorm(d_model)
                self.lfmc_lens_reader = nn.MultiheadAttention(
                    d_model, n_heads, batch_first=True
                )
                self.lfmc_lens_head = nn.Linear(d_model, 1)
                nn.init.zeros_(self.lfmc_lens_head.weight)
                nn.init.zeros_(self.lfmc_lens_head.bias)
        self.myco_head = nn.Linear(d_model, 5) if hasattr(source, "myco") else None
        self.flower_head = nn.Linear(d_model, 1) if hasattr(source, "flower") else None
        self.species_myco_head = None
        if self.myco_head is not None:
            with preserve_rng():
                self.species_myco_head = mlp(
                    d_model, 5, d_model, normalize=False
                )
                self.myco_relation_gate = nn.Parameter(
                    torch.tensor(math.atanh(0.75))
                )
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
        self.community_metric = mlp(d_model, d_model, d_model)

    def _init_fiber_mesh(
        self,
        variables: Sequence[Variable],
        write_names: Sequence[str],
        d_model: int,
        levels: int,
        n_heads: int,
    ) -> None:
        variable_kind = {v.name: v.kind for v in variables}
        self.write_lens = {
            name: LENS_INDEX[signal_lens(name, variable_kind.get(name))]
            for name in write_names
        }
        self.fiber_level_gate = per_name(
            write_names, lambda _: torch.zeros(levels)
        )
        self.fiber_reliability = per_name(
            write_names, lambda _: torch.zeros(())
        )
        self.fiber_type = nn.Parameter(torch.randn(len(LENSES), d_model) * 0.02)
        self.fiber_prior = nn.ModuleList([
            mlp(d_model, d_model) for _ in LENSES
        ])
        self.fiber_information_gate = mlp(4 * d_model, 1, d_model)
        self.fiber_norm = nn.LayerNorm(d_model)
        self.fiber_latents = 4
        self.fiber_query = nn.Parameter(
            torch.randn(len(LENSES), self.fiber_latents, d_model) * 0.02
        )
        self.fiber_decode_query = nn.Parameter(
            torch.randn(len(variables), d_model) * 0.02
        )
        scientific_reads = ["community"] + [
            name for name, head in (
                ("pollinator", self.poll_head), ("lfmc", self.lfmc_head),
                ("myco", self.myco_head), ("flower", self.flower_head),
            ) if head is not None
        ]
        self.mesh_read_names = (*self.names, *scientific_reads)
        self.mesh_reader = MeshQueryReader(
            self.mesh_read_names, d_model, levels, n_heads,
            self.species_variable,
        )
        self.fiber_read_norm = nn.LayerNorm(d_model)
        self.fiber_read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fiber_fuse_norm = nn.LayerNorm(d_model)
        self.fiber_fuse = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        reconstruct_names = [variable.name for variable in variables]
        self.fiber_reconstruct = nn.ModuleDict({
            name: mlp(d_model, d_model) for name in reconstruct_names
        })
        consume_rng(*(
            mlp(d_model, d_model)
            for name in self.always_names if name != "worldclim"
        ))
        self.fiber_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.sparse_fusion_gate = nn.Parameter(torch.tensor(0.05))
        self.coarse_scale_exchange = nn.Linear(d_model, d_model, bias=False)
        self.fine_scale_exchange = nn.Linear(d_model, d_model, bias=False)
        self.scale_exchange_gate = nn.Parameter(torch.full((len(LENSES),), 0.05))
        self.scale_message_norm = nn.LayerNorm(d_model)
        self.mesh_linear_reconstruct = nn.ModuleDict({
            name: nn.Linear(d_model, d_model, bias=False)
            for name in reconstruct_names
        })
        consume_rng(*(
            nn.Linear(d_model, d_model, bias=False)
            for name in self.always_names if name != "worldclim"
        ))
        self.lens_exchange_norm = nn.LayerNorm(d_model)
        self.lens_exchange = nn.Parameter(
            torch.zeros(levels, len(LENSES), len(LENSES))
        )

    def _init_specialists(
        self, variables, d_model: int, levels: int, n_heads: int
    ) -> None:
        self.specialist_meshes = nn.ModuleList([
            nn.ModuleList([
                SpecialistMesh(d_model, n_heads) for _ in range(2)
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
            variable.name: nn.Linear(d_model, d_model)
            for variable in variables
        })

        relations = ["identity"]
        if self.poll_head is not None:
            relations.extend(("pollinator", "pollinator_transfer"))
        if self.myco_head is not None:
            relations.append("myco")
        self.relation_names = tuple(relations)
        self.relation_meshes = nn.ModuleDict({
            name: nn.ModuleList([
                SpecialistMesh(d_model, n_heads) for _ in range(2)
            ])
            for name in relations
        })
        self.relation_pair_mix = per_name(relations, lambda _: torch.zeros(2))
        self.relation_readers = nn.ModuleDict({
            name: nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for name in relations
        })
        self.relation_reader_norms = nn.ModuleDict({
            name: nn.LayerNorm(d_model) for name in relations
        })
        self.relation_output_norms = nn.ModuleDict({
            name: nn.LayerNorm(d_model) for name in relations
        })
        self.relation_query = per_name(
            relations, lambda _: torch.randn(d_model) * 0.02
        )
        self.relation_gate = per_name(relations, lambda _: torch.tensor(0.05))

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
        segments = (*LENSES, *relations)
        self.segment_denoisers = nn.ModuleDict({
            name: SegmentDenoiser(
                d_model, n_heads, levels,
                token_drop=corruption[name][0],
                cell_drop=corruption[name][1],
                level_drop=corruption[name][2],
                jitter=corruption[name][3],
            )
            for name in segments
        })
        self.segment_type = per_name(
            segments, lambda _: torch.randn(d_model) * 0.02
        )
        self.segment_gate = per_name(
            segments, lambda _: torch.tensor(math.atanh(0.5))
        )
        self.segment_fusion_norm = nn.LayerNorm(d_model)
        self.segment_fusion = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.segment_output_norm = nn.LayerNorm(d_model)
        denoised = {
            self.species_variable, "community", "pollinator", "lfmc",
            "myco", "flower", "seasonality", "water", "soil_drainage",
            "form", "plant_type", "growth_rate", "sun", "ease_of_care",
        }
        self.segment_task_gate = per_name(
            [name for name in self.mesh_read_names if name in denoised],
            lambda _: torch.tensor(0.1),
        )

    def _reset_runtime_state(self) -> None:
        self._fiber_summary = None
        self._fiber_mesh = None
        self._fiber_prior_mesh = None
        self._latest_fiber_prior = None
        self._specialist_mesh = None
        self._specialist_latents = None
        self._relation_mesh = {}
        self._relation_latents = {}
        self._denoised_pool_cache = {}
        self._identity_graph_uncertainty = None
        self._pollinator_route = None
        self._raw_state_tokens = None
        self._raw_state_mask = None
        self.mesh_reader.bind(None, None, None)

    @staticmethod
    def _mesh_pair(meshes, state, mix):
        branches = [mesh(state) for mesh in meshes]
        weight = mix.softmax(0)
        combined = state + sum(
            weight[index] * (branch_state - state)
            for index, (branch_state, _) in enumerate(branches)
        )
        latent = torch.cat([
            branch_latent for _, branch_latent in branches
        ], 1)
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

    def _raw_residuals(self, state, values, present, species):
        address = state.mean(-2)
        tokens, masks = [], []
        for name in self.write_names:
            if name == "worldclim":
                continue
            if name not in values or name not in present:
                continue
            valid = present[name].bool()
            token = (
                self._adapt(name, values[name], species)
                + self.write_type[name] + address
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
        if relative.dim() == neighbor.dim() - 1:
            relative = relative.unsqueeze(-2)
        neighbor = neighbor + relative
        return {
            "coords": query_coords,
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
        identity = values.get(self.species_variable)
        identity_present = present.get(self.species_variable)
        if identity is not None and identity_present is not None:
            index = identity.long().clamp(0, species.shape[0] - 1)
            seed = self.species_graph._seed().detach()[index]
            uncertainty = 1 - F.cosine_similarity(
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
                valid = values[name].isfinite().all(-1) \
                        & (values[name].norm(dim=-1) > 1e-6)
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
        self.mesh_reader.bind(
            fiber_summary, self._fiber_mesh, self._fiber_prior_mesh
        )
        self._denoised_pool_cache = {}
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

        specialist_states, specialist_latents = [], []
        for lens, specialists in enumerate(self.specialist_meshes):
            state, expert = self._mesh_pair(
                specialists, fiber_mesh[..., lens, :],
                self.specialist_pair_mix[lens],
            )
            specialist_states.append(state)
            specialist_latents.append(expert)
        self._specialist_mesh = torch.stack(specialist_states, 3)
        self._specialist_latents = torch.stack(specialist_latents, 1)

        biological = self._specialist_mesh[
            ..., LENS_INDEX["biological"], :
        ]
        relation_sources = {"identity": biological}
        if "pollinator" in self.relation_meshes:
            ecological = self._specialist_mesh[
                ..., LENS_INDEX["ecological"], :
            ]
            relation_sources["pollinator"] = 0.5 * (
                biological + ecological
            )
            relation_sources["pollinator_transfer"] = biological
        if "myco" in self.relation_meshes:
            abiotic = self._specialist_mesh[..., LENS_INDEX["abiotic"], :]
            relation_sources["myco"] = 0.5 * (biological + abiotic)
        self._relation_mesh = {}
        self._relation_latents = {}
        for name, state in relation_sources.items():
            relation_state, relation_latent = self._mesh_pair(
                self.relation_meshes[name], state,
                self.relation_pair_mix[name],
            )
            self._relation_mesh[name] = relation_state
            self._relation_latents[name] = relation_latent

        expert_tokens = self._specialist_latents + self.specialist_type.view(
            1, len(LENSES), 1, self.d_model
        )
        expert_tokens = self.specialist_aggregate_norm(
            expert_tokens.flatten(1, 2)
        )
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
        return latent + torch.tanh(self.raw_residual_gate) \
               * self.raw_residual_output_norm(raw_read)

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
            if self.ecological_readers:
                environment_species = self._ecological_species_read(
                    latent, values, observed, context["coords"]
                )
            else:
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
