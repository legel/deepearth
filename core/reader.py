"""Query-conditioned reads from the shared Earth4D mesh."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

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


class MeshReaderMixin:
    """Read scientific targets from the mesh without owning model state."""

    def _reader_tokens(self) -> dict[str, torch.Tensor]:
        if self._mesh_reader_cache is not None:
            return self._mesh_reader_cache
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
        return self._mesh_reader_cache

    def _pool(self, latent: torch.Tensor, name: str) -> torch.Tensor:
        if name in self._pool_cache:
            return self._pool_cache[name]
        base_name = name if name in self.names else self.species_variable
        query = self.decode_query[self.names.index(base_name)]
        weight = torch.softmax((latent @ query) / math.sqrt(self.d_model), -1)
        pooled = torch.einsum("bl,bld->bd", weight, latent)
        if self._fiber_summary is None or name not in self.mesh_read_query:
            self._pool_cache[name] = pooled
            return pooled
        reader = self._reader_tokens()
        fibers = reader["fibers"]
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
        task_tokens = reader["task_tokens"]
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
            return pooled
        cells = self._fiber_mesh.shape[1]
        scale_fibers = reader["scale_fibers"]
        scale_keys = reader["scale_keys"]
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
        prior_fibers = reader["prior_fibers"]
        prior_keys = reader["prior_keys"]
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
        return pooled

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

        reader = self._reader_tokens()
        fibers = reader["fibers"]
        mesh_queries = torch.stack([self.mesh_read_query[name] for name in names])
        task_query = mesh_queries.unsqueeze(0).expand(batch, -1, -1)
        task_tokens = reader["task_tokens"]
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

        scale_fibers = reader["scale_fibers"]
        scale_keys = reader["scale_keys"]
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

        prior_fibers = reader["prior_fibers"]
        prior_keys = reader["prior_keys"]
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
