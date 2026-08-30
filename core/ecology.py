"""Ecological readers over the Earth4D world state."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunityScaleMesh(nn.Module):
    def __init__(self, input_dim: int, species: int, heads: int = 4, width: int = 64):
        super().__init__()
        self.heads = heads
        self.width = width
        self.query = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, heads * width),
        )
        self.state = nn.Parameter(torch.randn(species, heads, width) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = F.normalize(
            self.query(features).reshape(-1, self.heads, self.width), dim=-1
        )
        state = F.normalize(self.state, dim=-1)
        temperature = self.log_temperature.clamp(
            math.log(1.0), math.log(100.0)
        ).exp()
        score = temperature * torch.einsum("bhd,shd->bsh", query, state)
        return torch.logsumexp(score, -1) - math.log(self.heads)


class EcologicalReadoutMixin:
    def _init_ecological_readers(self, source, d_model: int) -> None:
        self.ecological_readers = all(
            name in source.extra for name in ("alphaearth", "worldclim")
        )
        if not self.ecological_readers:
            return

        with torch.random.fork_rng(devices=[]):
            confidence_width = max(32, d_model // 2)
            nn.Sequential(
                nn.LayerNorm(d_model + 4),
                nn.Linear(d_model + 4, confidence_width),
                nn.GELU(),
                nn.Linear(confidence_width, 1),
            )
            self.environment_family_reader = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.family_count),
            )
            self.environment_species_reader = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            nn.init.zeros_(self.environment_family_reader[-1].weight)
            nn.init.zeros_(self.environment_family_reader[-1].bias)
            nn.init.zeros_(self.environment_species_reader[-1].weight)
            nn.init.zeros_(self.environment_species_reader[-1].bias)

        habitat = torch.cat((
            source.extra["alphaearth"][0].float(),
            source.extra["worldclim"][0].float(),
        ), -1)
        self.register_buffer("habitat_mean", habitat[source.train_index].mean(0))
        self.register_buffer(
            "habitat_scale", habitat[source.train_index].std(0).clamp_min(1e-4)
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1337)
            self.habitat_family_expert = nn.Sequential(
                nn.LayerNorm(habitat.shape[-1] + 29),
                nn.Linear(habitat.shape[-1] + 29, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, self.family_count),
            )
        self.register_buffer("habitat_family_mix", torch.tensor(0.35))
        self.register_buffer("habitat_family_margin", torch.tensor(0.01))

        self.habitat_species_heads = 4
        self.habitat_species_dim = 64
        self.habitat_species_candidates = 256
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260830)
            self.habitat_species_query = nn.Sequential(
                nn.LayerNorm(habitat.shape[-1] + 29),
                nn.Linear(habitat.shape[-1] + 29, 256),
                nn.GELU(),
                nn.Linear(256, self.habitat_species_heads * self.habitat_species_dim),
            )
            self.habitat_species_mesh = nn.Parameter(torch.randn(
                source.n_classes,
                self.habitat_species_heads,
                self.habitat_species_dim,
            ) * 0.02)
            self.habitat_species_log_temperature = nn.Parameter(
                torch.tensor(math.log(10.0))
            )
        self.register_buffer("habitat_species_mix", torch.zeros(()))

        self.habitat_family_multimodal_names = (
            "alphaearth", "worldclim", *self.environment_names
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260901)
            self.habitat_family_multimodal_query = nn.Parameter(
                torch.randn(4, d_model) * 0.02
            )
            self.habitat_family_multimodal_type = nn.Parameter(torch.randn(
                len(self.habitat_family_multimodal_names), d_model
            ) * 0.02)
            self.habitat_family_multimodal_norm = nn.LayerNorm(d_model)
            self.habitat_family_multimodal_reader = nn.MultiheadAttention(
                d_model, 4, batch_first=True
            )
            self.habitat_family_multimodal_head = nn.Sequential(
                nn.LayerNorm(4 * d_model),
                nn.Linear(4 * d_model, 512),
                nn.GELU(),
                nn.Linear(512, self.family_count),
            )
        self.register_buffer("habitat_family_multimodal_mix", torch.zeros(()))

        self.distribution_heads = 4
        self.distribution_dim = 64
        self.distribution_candidates = 256
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260907)
            width = habitat.shape[-1] + 29
            self.distribution_query = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 512),
                nn.GELU(),
                nn.Linear(512, self.distribution_heads * self.distribution_dim),
            )
            self.distribution_mesh = nn.Parameter(torch.randn(
                source.n_classes,
                self.distribution_heads,
                self.distribution_dim,
            ) * 0.02)
            self.distribution_log_temperature = nn.Parameter(
                torch.tensor(math.log(10.0))
            )
        self.register_buffer("distribution_family_mix", torch.zeros(()))
        self.register_buffer("distribution_species_mix", torch.zeros(()))
        self.register_buffer("distribution_tail_mix", torch.tensor(0.07))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260913)
            self.community_scale_meshes = nn.ModuleDict({
                scale: CommunityScaleMesh(habitat.shape[-1] + 29, source.n_classes)
                for scale in ("30m", "3km")
            })
        self.register_buffer("community_scale_family_mix", torch.zeros(()))
        self.register_buffer("community_scale_species_mix", torch.zeros(()))
        self._init_seasonality(source)

    def _init_seasonality(self, source) -> None:
        seasonal = torch.ones(
            12, self.family_count, device=source.class_group.device
        )
        phase_offset = torch.zeros((), device=source.class_group.device)
        annual_cycles = float(getattr(source, "time_span_days", 365.2425)) / 365.2425
        if hasattr(source, "obs_month"):
            training = source.train_index
            family = self.species_family[source.cls[training]]
            month = torch.as_tensor(
                source.obs_month[training.cpu().numpy()], device=training.device
            ).long()
            count = torch.zeros_like(seasonal)
            count.index_put_(
                (month, family), torch.ones_like(family, dtype=count.dtype),
                accumulate=True,
            )
            prior = count.sum(0)
            prior = (prior + 1.0) / (prior.sum() + self.family_count)
            monthly = count + 2.0 * prior.unsqueeze(0)
            seasonal = monthly / monthly.sum(-1, keepdim=True) \
                       / prior.clamp_min(1e-8)

            sample = training[torch.linspace(
                0, len(training) - 1, min(len(training), 8192),
                device=training.device,
            ).long()]
            phase = source.coords[sample, 3].float() * annual_cycles
            known = torch.as_tensor(
                source.obs_month[sample.cpu().numpy()], device=training.device
            ).long()
            offsets = torch.linspace(0.0, 1.0, 1461, device=training.device)[:-1]
            predicted = torch.floor(
                torch.remainder(phase.unsqueeze(0) + offsets[:, None], 1.0) * 12
            ).long()
            phase_offset = offsets[
                predicted.eq(known.unsqueeze(0)).sum(-1).argmax()
            ]
        self.register_buffer("seasonal_family_likelihood", seasonal)
        self.register_buffer("seasonal_phase_offset", phase_offset)
        self.register_buffer(
            "seasonal_annual_cycles",
            torch.tensor(annual_cycles, device=source.class_group.device),
        )
        self.register_buffer(
            "seasonal_family_strength",
            torch.tensor(0.35, device=source.class_group.device),
        )

    def _family_posterior(self, species_logits: torch.Tensor):
        logits = species_logits.float()
        family = self.species_family.expand(logits.shape[0], -1)
        mass = logits.new_zeros(logits.shape[0], self.family_count)
        mass.scatter_add_(1, family, logits.softmax(-1))
        return family, mass

    def _factorized_environment_logits(
        self, pooled: torch.Tensor, species_logits: torch.Tensor
    ) -> torch.Tensor:
        logits = species_logits.detach().float()
        family, family_mass = self._family_posterior(logits)
        family_log = family_mass.clamp_min(1e-8).log()
        within_family = F.log_softmax(logits, -1) - family_log.gather(1, family)
        family_logits = family_log + self.environment_family_reader(
            pooled.detach().float()
        )
        key = self._refined_species.detach().float() \
              + self.species_niche_key.detach().float()
        residual = self.environment_species_reader(pooled.detach().float()) @ key.t()
        family_sum = residual.new_zeros(residual.shape[0], self.family_count)
        family_sum.scatter_add_(1, family, residual)
        family_size = torch.bincount(
            self.species_family, minlength=self.family_count
        ).clamp_min(1).to(residual.dtype)
        residual = residual - (family_sum / family_size).gather(1, family)
        return within_family \
               + F.log_softmax(family_logits, -1).gather(1, family) \
               + residual

    def _habitat_features(self, values: Dict[str, torch.Tensor], coords: torch.Tensor):
        continuous = torch.cat((
            values["alphaearth"].float(), values["worldclim"].float()
        ), -1)
        continuous = (continuous - self.habitat_mean) / self.habitat_scale
        latitude = torch.deg2rad(coords[:, 0].float())
        longitude = torch.deg2rad(coords[:, 1].float())
        geo = [
            latitude.sin(), latitude.cos(), longitude.sin(), longitude.cos(),
            (coords[:, 2].float() / 1000.0).clamp(-1, 5),
        ]
        for frequency in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            geo.extend((
                (frequency * latitude).sin(), (frequency * latitude).cos(),
                (frequency * longitude).sin(), (frequency * longitude).cos(),
            ))
        return torch.cat((continuous, torch.stack(geo, -1)), -1)

    def _habitat_family_multimodal_logits(self, values, observed=None):
        tokens = torch.stack([
            self.adapters[name](values[name]).float()
            for name in self.habitat_family_multimodal_names
        ], 1)
        if observed is None:
            available = torch.ones(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
        else:
            available = torch.stack([
                observed.get(name, torch.ones(
                    len(tokens), dtype=torch.bool, device=tokens.device
                )).bool()
                for name in self.habitat_family_multimodal_names
            ], 1)
        tokens = self.habitat_family_multimodal_norm(
            tokens + self.habitat_family_multimodal_type.unsqueeze(0)
        )
        query = self.habitat_family_multimodal_query.unsqueeze(0).expand(
            len(tokens), -1, -1
        )
        read = self.habitat_family_multimodal_reader(
            query, tokens, tokens, key_padding_mask=~available,
            need_weights=False,
        )[0]
        return self.habitat_family_multimodal_head(read.flatten(1))

    def _habitat_family_read(
        self, species_logits, values, coords, observed=None,
        multimodal_mix=None, multimodal_logits=None,
    ):
        family, family_mass = self._family_posterior(species_logits)
        base_family = family_mass.clamp_min(1e-8).log()
        conditional = F.log_softmax(species_logits.float(), -1) \
                      - base_family.gather(1, family)
        habitat_family = F.log_softmax(
            self.habitat_family_expert(self._habitat_features(values, coords)), -1
        )
        if multimodal_logits is None:
            multimodal_logits = self._habitat_family_multimodal_logits(
                values, observed
            )
        multimodal_family = F.log_softmax(multimodal_logits, -1)
        expert_mix = self.habitat_family_multimodal_mix \
            if multimodal_mix is None else multimodal_mix
        if not torch.is_tensor(expert_mix):
            expert_mix = habitat_family.new_tensor(expert_mix)
        expert_family = torch.logaddexp(
            habitat_family + torch.log1p(-expert_mix),
            multimodal_family + expert_mix.log(),
        )
        mixture = self.habitat_family_mix
        mixed_family = torch.logaddexp(
            base_family + torch.log1p(-mixture),
            expert_family + mixture.log(),
        )
        mixed = conditional + mixed_family.gather(1, family)
        expert_top = expert_family.exp().topk(2, -1)
        route = (
            expert_top.values[:, 0] - expert_top.values[:, 1]
            >= self.habitat_family_margin
        ) & (expert_top.indices[:, 0] == base_family.argmax(-1))
        promoted = self._hierarchical_family_read(mixed)
        return torch.where(route[:, None], promoted, mixed)

    def _distribution_scores(self, values, coords):
        query = self.distribution_query(
            self._habitat_features(values, coords)
        ).reshape(-1, self.distribution_heads, self.distribution_dim)
        query = F.normalize(query, dim=-1)
        prototype = F.normalize(self.distribution_mesh, dim=-1)
        temperature = self.distribution_log_temperature.clamp(
            math.log(1.0), math.log(100.0)
        ).exp()
        score = temperature * torch.einsum("bhd,shd->bsh", query, prototype)
        return torch.logsumexp(score, -1) - math.log(self.distribution_heads)

    def _distribution_family_read(
        self, species_logits, values, coords, mix=None, expert_scores=None
    ):
        family, family_mass = self._family_posterior(species_logits)
        base_family = family_mass.clamp_min(1e-8).log()
        conditional = F.log_softmax(species_logits.float(), -1) \
                      - base_family.gather(1, family)
        if expert_scores is None:
            expert_scores = self._distribution_scores(values, coords)
        expert_probability = F.softmax(expert_scores, -1)
        expert_family = expert_probability.new_zeros(
            len(species_logits), self.family_count
        )
        expert_family.scatter_add_(1, family, expert_probability)
        mixture = self.distribution_family_mix if mix is None else mix
        if not torch.is_tensor(mixture):
            mixture = base_family.new_tensor(mixture)
        family_log = torch.logaddexp(
            base_family + torch.log1p(-mixture),
            expert_family.clamp_min(1e-8).log() + mixture.log(),
        )
        return conditional + family_log.gather(1, family)

    def _rank_within_family(self, species_logits, expert_scores, mix):
        count = min(self.distribution_candidates, species_logits.shape[-1])
        candidate_values, candidates = species_logits.detach().topk(count, -1)
        expert = expert_scores.gather(1, candidates)
        family = self.species_family[candidates]
        same_family = family[:, None, :] == family[:, :, None]
        family_count = same_family.sum(-1).clamp_min(1)
        mean = (same_family.to(expert.dtype) * expert[:, None, :]).sum(-1) \
               / family_count
        variance = (
            same_family.to(expert.dtype)
            * (expert[:, None, :] - mean[:, :, None]).square()
        ).sum(-1) / family_count
        adjusted = candidate_values + mix * (
            expert - mean
        ) / variance.clamp_min(1e-4).sqrt()
        position = torch.arange(count, device=candidates.device)
        precedes = position.view(1, 1, -1) < position.view(1, -1, 1)
        adjusted_rank = (same_family & (
            (adjusted[:, None, :] > adjusted[:, :, None])
            | ((adjusted[:, None, :] == adjusted[:, :, None]) & precedes)
        )).sum(-1)
        original_rank = (same_family & (
            (candidate_values[:, None, :] > candidate_values[:, :, None])
            | ((candidate_values[:, None, :] == candidate_values[:, :, None])
               & precedes)
        )).sum(-1)
        assignment = same_family & (
            original_rank[:, None, :] == adjusted_rank[:, :, None]
        )
        replacement = (
            assignment.to(candidate_values.dtype) * candidate_values[:, None, :]
        ).sum(-1)
        return species_logits.scatter(1, candidates, replacement)

    def _distribution_species_read(
        self, species_logits, values, coords, mix=None, expert_scores=None
    ):
        if expert_scores is None:
            expert_scores = self._distribution_scores(values, coords)
        mixture = self.distribution_species_mix if mix is None else mix
        return self._rank_within_family(species_logits, expert_scores, mixture)

    def _habitat_species_scores(self, values, coords):
        query = self.habitat_species_query(
            self._habitat_features(values, coords)
        ).reshape(-1, self.habitat_species_heads, self.habitat_species_dim)
        query = F.normalize(query, dim=-1)
        prototype = F.normalize(self.habitat_species_mesh, dim=-1)
        temperature = self.habitat_species_log_temperature.clamp(
            math.log(1.0), math.log(100.0)
        ).exp()
        score = temperature * torch.einsum("bhd,shd->bsh", query, prototype)
        return torch.logsumexp(score, -1) - math.log(self.habitat_species_heads)

    def _habitat_species_read(
        self, species_logits, values, coords, mix=None, expert_scores=None
    ):
        if expert_scores is None:
            expert_scores = self._habitat_species_scores(values, coords)
        mixture = self.habitat_species_mix if mix is None else mix
        return self._rank_within_family(species_logits, expert_scores, mixture)

    def _scores_to_family(self, species_scores):
        family = self.species_family.expand(len(species_scores), -1)
        probability = F.softmax(species_scores, -1)
        result = probability.new_zeros(len(species_scores), self.family_count)
        result.scatter_add_(1, family, probability)
        return result

    def _protected_family_tail(self, species_logits, expert_family, mixture):
        if (not torch.is_tensor(mixture) and mixture == 0.0) or (
            torch.is_tensor(mixture) and mixture.numel() == 1
            and float(mixture) == 0.0
        ):
            return species_logits
        family, base_probability = self._family_posterior(species_logits)
        base_family = base_probability.clamp_min(1e-8).log()
        if not torch.is_tensor(mixture):
            mixture = base_family.new_tensor(mixture)
        candidate = torch.logaddexp(
            base_family + torch.log1p(-mixture),
            expert_family.clamp_min(1e-8).log() + mixture.log(),
        )
        offset = candidate - base_family
        winner = species_logits.argmax(-1)
        protected = self.species_family[winner]
        offset = offset.scatter(
            1, protected[:, None], offset.new_zeros(len(offset), 1)
        )
        adjusted = species_logits + offset.gather(1, family)
        ceiling = species_logits.gather(1, winner[:, None]) - 1e-6
        return torch.where(
            family != protected[:, None], torch.minimum(adjusted, ceiling), adjusted
        )

    def _distribution_tail_read(self, species_logits, expert_scores, mix=None):
        mixture = self.distribution_tail_mix if mix is None else mix
        return self._protected_family_tail(
            species_logits, self._scores_to_family(expert_scores), mixture
        )

    def _community_scale_scores(self, values, coords):
        features = self._habitat_features(values, coords)
        return tuple(
            self.community_scale_meshes[scale](features)
            for scale in ("30m", "3km")
        )

    def _community_scale_read(self, species_logits, scale_scores):
        species_expert = torch.logsumexp(torch.stack(scale_scores), 0) \
                         - math.log(len(scale_scores))
        output = self._distribution_species_read(
            species_logits, {}, species_logits.new_empty((len(species_logits), 4)),
            mix=self.community_scale_species_mix, expert_scores=species_expert,
        )
        family = torch.stack([
            self._scores_to_family(scores).clamp_min(1e-8).log()
            for scores in scale_scores
        ]).mean(0).exp()
        family = family / family.sum(-1, keepdim=True)
        return self._protected_family_tail(
            output, family, self.community_scale_family_mix
        )

    def _seasonal_family_read(self, species_logits, coords):
        phase = coords[:, 3].float() * self.seasonal_annual_cycles
        month = torch.floor(torch.remainder(
            phase + self.seasonal_phase_offset, 1.0
        ) * 12).long()
        _, family = self._family_posterior(species_logits)
        family = family * self.seasonal_family_likelihood[month].pow(
            self.seasonal_family_strength
        )
        family = family / family.sum(-1, keepdim=True)
        return self._protected_family_tail(species_logits, family, 1.0)

    def _with_worldclim_observed(self, present, observed):
        if "worldclim" not in self.always_names or "worldclim" not in observed:
            return present
        present = dict(present)
        present["worldclim"] = observed["worldclim"]
        return present

    def _ecological_species_read(self, latent, values, observed, coords):
        pooled = self._pool(latent, self.species_variable)
        species = self._niche_species_logits(pooled, include_lens=False)
        species = self._factorized_environment_logits(pooled, species)
        species = self._habitat_family_read(species, values, coords, observed)
        distribution = self._distribution_scores(values, coords)
        species = self._distribution_family_read(
            species, values, coords, expert_scores=distribution
        )
        species = self._habitat_species_read(species, values, coords)
        species = self._distribution_species_read(
            species, values, coords, expert_scores=distribution
        )
        species = self._distribution_tail_read(species, distribution)
        species = self._community_scale_read(
            species, self._community_scale_scores(values, coords)
        )
        return self._seasonal_family_read(species, coords)
