"""Query-conditioned reads from the shared Earth4D mesh."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from deepearth.core.layers import mlp, per_name, preserve_rng
from deepearth.core.world_mesh import LENSES


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


class RoutedMeshReader(nn.Module):
    def __init__(self, d_model: int, n_heads: int, levels: int):
        super().__init__()
        self.d_model = d_model
        self.levels = levels
        self.query = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(0.05))
        self.cell_key = nn.Parameter(torch.zeros(2, d_model))
        self.level_key = nn.Parameter(torch.zeros(levels, d_model))
        self.lens_key = nn.Parameter(torch.zeros(len(LENSES), d_model))

    def residual(self, pooled, mesh, *, detach_mesh=False):
        cells = mesh.shape[1]
        fibers = mesh.flatten(1, 3)
        if detach_mesh:
            fibers = fibers.detach()
        keys = self.norm(fibers)
        cell_key = torch.cat((
            self.cell_key[:1], self.cell_key[1:].expand(cells - 1, -1)
        ))
        route_keys = (
            keys.reshape(-1, cells, self.levels, len(LENSES), self.d_model)
            + cell_key.view(1, cells, 1, 1, self.d_model)
            + self.level_key.view(1, 1, self.levels, 1, self.d_model)
            + self.lens_key.view(1, 1, 1, len(LENSES), self.d_model)
        ).flatten(1, 3)
        score = torch.einsum("bkd,bd->bk", route_keys, pooled)
        score = score / math.sqrt(self.d_model)
        weight, index = score.topk(min(16, score.shape[-1]), dim=-1)
        selected = keys.gather(
            1, index[..., None].expand(-1, -1, self.d_model)
        )
        routed = torch.einsum(
            "bk,bkd->bd", weight.softmax(-1), selected
        )
        query = self.query[None] + pooled[:, None] + routed[:, None]
        read = self.attention(
            query, selected, selected, need_weights=False
        )[0].mean(1)
        return torch.tanh(self.gate) * self.output_norm(read)

    def forward(self, pooled, mesh, *, isolated=False):
        if isolated:
            pooled = pooled.detach()
        return pooled + self.residual(pooled, mesh, detach_mesh=isolated)


class MeshQueryReader(nn.Module):
    """Route task queries through the fibered mesh."""

    def __init__(
        self, names, d_model: int, levels: int, n_heads: int,
        species_variable: str,
    ):
        super().__init__()
        self.names = tuple(names)
        self.d_model = d_model
        self.levels = levels
        self.species_variable = species_variable
        self.mesh_read_query = per_name(names, lambda _: torch.randn(d_model) * 0.02)
        self.mesh_read_gate = per_name(names, lambda _: torch.tensor(0.05))
        self.mesh_scale_read_gate = per_name(names, lambda _: torch.tensor(0.05))
        self.mesh_scale_attention_gate = per_name(names, lambda _: torch.zeros(()))
        with preserve_rng():
            self.task_mesh_reader = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        with preserve_rng():
            self.scale_mesh_reader = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        with preserve_rng():
            self.deep_mesh_reader = nn.ModuleList([
                CrossFiberReaderBlock(d_model, n_heads) for _ in range(4)
            ])
            self.deep_mesh_reader_gate = per_name(names, lambda _: torch.zeros(()))
            self.deep_mesh_reader_output_norm = nn.LayerNorm(d_model)
        self.scale_mesh_reader_mix = per_name(
            names,
            lambda name: torch.tensor(
                -2.0 if name == species_variable else 0.0
            ),
        )
        with preserve_rng():
            self.scale_mesh_reader_router = mlp(4 * d_model, 1)
            nn.init.zeros_(self.scale_mesh_reader_router[-1].weight)
            nn.init.zeros_(self.scale_mesh_reader_router[-1].bias)
        self.task_mesh_reader_gate = per_name(names, lambda _: torch.zeros(()))
        self.task_mesh_reader_norm = nn.LayerNorm(d_model)
        self.task_mesh_reader_output_norm = nn.LayerNorm(d_model)
        self.mesh_prior_read_gate = per_name(names, lambda _: torch.zeros(()))
        with preserve_rng():
            self.mesh_prior_information_gate = mlp(4 * d_model, 1, d_model)
        conditioned = [name for name in ("pollinator",) if name in names]
        self.mesh_condition_gate = per_name(conditioned, lambda _: torch.tensor(0.05))
        self.mesh_task_norm = nn.LayerNorm(d_model)
        self.mesh_scale_task_norm = nn.LayerNorm(d_model)
        self.mesh_prior_task_norm = nn.LayerNorm(d_model)
        self.mesh_condition_norm = nn.LayerNorm(d_model)
        self.mesh_cell_key = nn.Parameter(torch.zeros(2, d_model))
        self.mesh_level_key = nn.Parameter(torch.zeros(levels, d_model))
        self.mesh_lens_key = nn.Parameter(torch.zeros(len(LENSES), d_model))
        self.bind(None, None, None)

    def bind(self, summary, mesh, priors) -> None:
        self.summary = summary
        self.mesh = mesh
        self.priors = priors
        self.cache = {}
        self.tokens = None

    def _reader_tokens(self) -> dict[str, torch.Tensor]:
        if self.tokens is not None:
            return self.tokens
        fibers = self.summary.flatten(1, 2)
        cells = self.mesh.shape[1]
        cell_key = torch.cat((
            self.mesh_cell_key[:1],
            self.mesh_cell_key[1:].expand(cells - 1, -1),
        ))
        scale_fibers = self.mesh.flatten(1, 3)
        scale_keys = (
            self.mesh
            + cell_key.view(1, cells, 1, 1, self.d_model)
            + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
            + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
        ).flatten(1, 3)
        prior_mesh = self.priors.detach()
        prior_fibers = prior_mesh.flatten(1, 3)
        prior_keys = (
            prior_mesh
            + cell_key.view(1, cells, 1, 1, self.d_model)
            + self.mesh_level_key.view(1, 1, self.levels, 1, self.d_model)
            + self.mesh_lens_key.view(1, 1, 1, len(LENSES), self.d_model)
        ).flatten(1, 3)
        self.tokens = {
            "fibers": fibers,
            "task_tokens": self.task_mesh_reader_norm(fibers),
            "scale_fibers": scale_fibers,
            "scale_keys": scale_keys,
            "prior_fibers": prior_fibers,
            "prior_keys": prior_keys,
        }
        return self.tokens

    def _topk_read(
        self, scores: torch.Tensor, values: torch.Tensor, count: int
    ) -> torch.Tensor:
        weight, index = scores.topk(min(count, scores.shape[-1]), dim=-1)
        batch, tasks = scores.shape[:2]
        selected = values.unsqueeze(1).expand(-1, tasks, -1, -1).gather(
            2, index[..., None].expand(-1, -1, -1, self.d_model)
        )
        return torch.einsum("btk,btkd->btd", weight.softmax(-1), selected)

    @staticmethod
    def _task_parameters(parameters, names) -> torch.Tensor:
        return torch.stack([parameters[name] for name in names]).view(
            1, len(names), -1
        )

    def _gated_read(self, pooled, gates, names, read):
        return pooled + torch.tanh(self._task_parameters(gates, names)) * read

    def _conditioned_queries(self, query, names):
        gates = torch.stack([
            self.mesh_condition_gate[name]
            if name in self.mesh_condition_gate else query.new_zeros(())
            for name in names
        ]).view(1, len(names), 1)
        if not torch.count_nonzero(gates):
            return query
        lenses = self.mesh[:, 0].mean(1)
        score = torch.einsum("bld,btd->btl", lenses, query)
        condition = torch.einsum(
            "btl,bld->btd",
            (score / math.sqrt(self.d_model)).softmax(-1),
            lenses,
        )
        return query + torch.tanh(gates) * self.mesh_condition_norm(condition)

    def _task_attention(self, reader, query):
        batch, tasks = query.shape[:2]
        tokens = reader["task_tokens"][:, None].expand(-1, tasks, -1, -1)
        tokens = tokens.reshape(batch * tasks, -1, self.d_model)
        read = self.task_mesh_reader(
            query.reshape(batch * tasks, 1, self.d_model),
            tokens, tokens, need_weights=False,
        )[0]
        return read.reshape(batch, tasks, self.d_model)

    def _scale_route(self, reader, query):
        score = torch.einsum("bkd,btd->btk", reader["scale_keys"], query)
        score = score / math.sqrt(self.d_model)
        dense = score.softmax(-1)
        selected_score, selected = score.topk(min(8, score.shape[-1]), -1)
        sparse = torch.zeros_like(score).scatter(
            -1, selected, selected_score.softmax(-1)
        )
        route = sparse.detach() + dense - dense.detach()
        read = torch.einsum("btk,bkd->btd", route, reader["scale_fibers"])
        return score, read

    def _scale_attention(self, reader, query, task_read, scale_score, names):
        batch, tasks = query.shape[:2]
        cells = self.mesh.shape[1]
        grid = scale_score.reshape(
            batch, tasks, cells, self.levels, len(LENSES)
        )
        lens = torch.arange(len(LENSES), device=scale_score.device)
        query_index = grid[:, :, 0].argmax(2) * len(LENSES) + lens
        neighbor = grid[:, :, 1:].reshape(
            batch, tasks, -1, len(LENSES)
        ).argmax(2)
        neighbor_index = (
            (neighbor.div(self.levels, rounding_mode="floor") + 1) * self.levels
            + neighbor.remainder(self.levels)
        ) * len(LENSES) + lens
        index = torch.cat((query_index, neighbor_index), -1)
        expanded = (-1, tasks, -1, -1)
        gather = index[..., None].expand(-1, -1, -1, self.d_model)
        keys = reader["scale_keys"][:, None].expand(*expanded).gather(2, gather)
        values = reader["scale_fibers"][:, None].expand(*expanded).gather(2, gather)
        keys = self.task_mesh_reader_norm(keys).flatten(0, 1)
        values = self.task_mesh_reader_norm(values).flatten(0, 1)
        scale_query = query + self.task_mesh_reader_output_norm(task_read)
        flat_query = scale_query.flatten(0, 1)
        attention_query = flat_query.unsqueeze(1)
        shared = self.task_mesh_reader(
            attention_query, keys, values, need_weights=False
        )[0].reshape(batch, tasks, self.d_model)
        dedicated = self.scale_mesh_reader(
            attention_query, keys, values, need_weights=False
        )[0].reshape(batch, tasks, self.d_model)
        features = torch.cat((
            query, shared, dedicated, (shared - dedicated).abs()
        ), -1)
        bias = self._task_parameters(
            self.scale_mesh_reader_mix, names
        ).squeeze(-1)
        mix = torch.sigmoid(
            bias + self.scale_mesh_reader_router(features).squeeze(-1)
        ).unsqueeze(-1)
        return torch.lerp(shared, dedicated, mix), attention_query, keys, values

    def read(self, pooled: torch.Tensor, names) -> None:
        names = tuple(names)
        batch, tasks = pooled.shape[:2]
        if self.summary is None or self.mesh is None:
            self.cache.update({
                name: pooled[:, index] for index, name in enumerate(names)
            })
            return

        reader = self._reader_tokens()
        fibers = reader["fibers"]
        task_query = torch.stack([
            self.mesh_read_query[name] for name in names
        ]).unsqueeze(0).expand(batch, -1, -1)
        task_query = self._conditioned_queries(task_query, names)
        task_read = self._task_attention(reader, task_query)
        pooled = self._gated_read(
            pooled, self.task_mesh_reader_gate, names,
            self.task_mesh_reader_output_norm(task_read),
        )

        fiber_score = torch.einsum("bfd,btd->btf", fibers, task_query)
        fiber_score = fiber_score / math.sqrt(self.d_model)
        mesh_read = self._topk_read(fiber_score, fibers, 4)
        pooled = self._gated_read(
            pooled, self.mesh_read_gate, names, self.mesh_task_norm(mesh_read)
        )

        scale_score, scale_read = self._scale_route(reader, task_query)
        pooled = self._gated_read(
            pooled, self.mesh_scale_read_gate, names,
            self.mesh_scale_task_norm(scale_read),
        )

        scale_attention, flat_query, keys, values = self._scale_attention(
            reader, task_query, task_read, scale_score, names
        )
        pooled = self._gated_read(
            pooled, self.mesh_scale_attention_gate, names,
            self.mesh_scale_task_norm(scale_attention),
        )

        deep = flat_query.squeeze(1)
        for block in self.deep_mesh_reader:
            deep = block(deep, keys, values)
        deep = self.deep_mesh_reader_output_norm(
            deep - flat_query.squeeze(1)
        ).reshape(batch, tasks, self.d_model)
        deep_gates = torch.stack([
            self.deep_mesh_reader_gate[name]
            if self.training or name not in {"community", "identity"}
            else self.deep_mesh_reader_gate[name] * 0.0
            for name in names
        ]).view(1, tasks, 1)
        pooled = pooled + torch.tanh(deep_gates) * deep

        prior_keys = reader["prior_keys"]
        prior_score = torch.einsum("bkd,btd->btk", prior_keys, task_query)
        prior_score = prior_score / math.sqrt(self.d_model)
        prior_read = self._topk_read(
            prior_score, reader["prior_fibers"], 16
        )
        confidence = torch.sigmoid(self.mesh_prior_information_gate(torch.cat((
            pooled, prior_read, pooled * prior_read, (pooled - prior_read).abs()
        ), -1)))
        pooled = self._gated_read(
            pooled, self.mesh_prior_read_gate, names,
            confidence * self.mesh_prior_task_norm(prior_read),
        )
        self.cache.update({
            name: pooled[:, index] for index, name in enumerate(names)
        })

    def missing(self, names):
        return tuple(name for name in names if name not in self.cache)


class ScientificReadoutMixin:
    """Scientific predictions built on task-conditioned mesh reads."""

    def _latent_pool(self, latent, names):
        queries = torch.stack([
            self.decode_query[self.names.index(
                name if name in self.names else self.species_variable
            )]
            for name in names
        ])
        weights = torch.einsum("bld,td->btl", latent, queries)
        weights = (weights / math.sqrt(self.d_model)).softmax(-1)
        return torch.einsum("btl,bld->btd", weights, latent)

    def _read_tasks(self, latent, names) -> None:
        names = self.mesh_reader.missing(names)
        if names:
            self.mesh_reader.read(self._latent_pool(latent, names), names)

    def _pool(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        self._read_tasks(latent, (name,))
        return self.mesh_reader.cache[name]

    def _prime_pool_cache(self, latent: torch.Tensor) -> None:
        self._read_tasks(latent, self.mesh_read_names)

    def _pollinator_pool(self, latent: torch.Tensor, *, isolated: bool = False) -> torch.Tensor:
        pooled = self._pool(latent, "pollinator")
        if self.pollinator_reader is None or self._fiber_mesh is None:
            return pooled
        return self.pollinator_reader(
            pooled, self._fiber_mesh, isolated=isolated
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

    def _center_within_family(self, values: torch.Tensor) -> torch.Tensor:
        family = self.species_family.expand(values.shape[0], -1)
        total = values.new_zeros(values.shape[0], self.family_count)
        total.scatter_add_(1, family, values)
        size = torch.bincount(
            self.species_family, minlength=self.family_count
        ).clamp_min(1).to(values.dtype)
        return values - (total / size).gather(1, family)

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
        return self._center_within_family(read @ key.detach().t())

    def _niche_species_logits(
        self, pooled: torch.Tensor, include_lens: bool = True
    ) -> torch.Tensor:
        key = self._refined_species.detach().float() \
              + self.species_niche_key.float()
        pooled = pooled.float()
        base = pooled @ key.t()
        residual = self._center_within_family(
            self.species_niche_adapter(pooled) @ key.t()
        )
        logits = base + residual
        if include_lens:
            logits = logits + self._species_lens_residual(pooled, key)
        return logits

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
        read = self.identity_detail_reader.residual(
            pooled, self._fiber_mesh, detach_mesh=True
        )
        return self._center_within_family(
            read @ self._refined_species.detach().t()
        )

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
        return base + 0.75 * evidence

    def _pool_fiber(self, fiber: torch.Tensor, name: str) -> torch.Tensor:
        query = self.fiber_decode_query[self.names.index(name)]
        weight = torch.softmax((fiber @ query) / math.sqrt(self.d_model), -1)
        return torch.einsum("bl,bld->bd", weight, fiber)
