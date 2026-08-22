"""Training objectives for the DeepEarth world model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


TRAITS = {
    "seasonality", "water", "soil_drainage", "form",
    "plant_type", "growth_rate", "sun", "ease_of_care",
}


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def class_error(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, reduction="none") \
           / math.log(max(logits.shape[-1], 2))


def rank_error(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_score = logits.gather(1, target[:, None])
    rank = 0.5 + torch.sigmoid((logits - target_score) / 0.25).sum(-1)
    return rank.clamp_min(1).log() / math.log(max(logits.shape[-1], 2))


def distribution_error(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, -1)).sum(-1)


@dataclass
class SpeciesState:
    pool: torch.Tensor
    base: torch.Tensor
    niche: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor


@dataclass
class PollinatorState:
    target: torch.Tensor
    valid: torch.Tensor


class TrainingObjectiveMixin:
    def _masked_observations(self, observed, batch, device, probability):
        present = {
            name: (torch.rand(batch, device=device) > probability) & observed[name]
            for name in self.names
        }
        blank = torch.rand(batch, device=device) < 0.15
        return {name: mask & ~blank for name, mask in present.items()}

    def _family_error(self, logits, target):
        probability = logits.softmax(-1)
        family = probability.new_zeros(logits.shape[0], self.family_count)
        family.scatter_add_(
            1, self.species_family.expand(logits.shape[0], -1), probability
        )
        target_family = self.species_family[target]
        return -family.gather(1, target_family[:, None]).squeeze(1) \
               .clamp_min(1e-8).log() / math.log(max(self.family_count, 2))

    def _reconstruction_terms(self, values, observed, present, latent):
        terms, fiber_terms, mesh_terms = [], [], []
        for variable in self.variables:
            name = variable.name
            hidden = ~present[name] & observed[name]
            if not hidden.any():
                continue
            prediction = self.decode(latent, name)
            if variable.kind == "categorical":
                error = class_error(prediction, values[name].long())
            else:
                target = values[name].float()
                valid = target.norm(dim=-1) > 1e-6
                mean = target[valid].mean(0, keepdim=True).detach() \
                       if valid.any() else target.mean(0, keepdim=True)
                error = 1 - F.cosine_similarity(
                    prediction - mean, target - mean, dim=-1
                )
            terms.append(masked_mean(error, hidden))

            lens = self.write_lens[name]
            target = self._adapt(name, values[name], self._refined_species).detach()
            prediction = self.fiber_reconstruct[name](
                self._pool_fiber(self._fiber_summary[:, lens], name)
            )
            fiber_terms.append(masked_mean(
                1 - F.cosine_similarity(prediction, target, dim=-1), hidden
            ))
            state = self._fiber_mesh[:, 0, :, lens].mean(1)
            prediction = self.mesh_linear_reconstruct[name](state)
            mesh_terms.append(masked_mean(
                1 - F.cosine_similarity(prediction, target, dim=-1), hidden
            ))
        return torch.stack(terms).mean() \
               + 0.05 * torch.stack(fiber_terms).mean() \
               + 0.05 * torch.stack(mesh_terms).mean()

    def _scientific_terms(self, values, latent):
        loss = latent.new_zeros(())
        pollinator = None
        if self.poll_head is not None and "_poll_idx" in values:
            logits = self.poll_head(self._pool(latent, "pollinator"))
            target = torch.zeros_like(logits).scatter_add_(
                1, values["_poll_idx"].clamp_min(0), values["_poll_frq"].float()
            )
            valid = values["_poll_valid"].float()
            loss = loss + 0.1 * masked_mean(
                distribution_error(logits, target), valid
            )
            if self.reader_phase:
                calibrated = self._calibrate_pollinator_logits(logits.detach())
                loss = loss + masked_mean(
                    distribution_error(calibrated, target), valid
                )
            structured = self.poll_head(
                self._pollinator_pool(latent, isolated=True)
            )
            loss = loss + 0.1 * masked_mean(
                distribution_error(structured, target), valid
            ) / math.log(self.poll_head.out_features)
            pollinator = PollinatorState(target, valid)

        if self.lfmc_head is not None and "_lfmc" in values:
            valid = values["_lfmc_valid"].float()
            target = torch.log(values["_lfmc"].clamp_min(1))
            prediction = self.lfmc_head(
                self._pool(latent, "lfmc")
            ).squeeze(-1)
            loss = loss + 0.1 * masked_mean((prediction - target).square(), valid)
        if self.myco_head is not None and "_myco" in values:
            error = F.cross_entropy(
                self._myco_logits(latent),
                values["_myco"].long().clamp_min(0),
                reduction="none",
            )
            loss = loss + 0.1 * masked_mean(error, values["_myco_valid"])
        if self.flower_head is not None and "_flower" in values:
            prediction = self.flower_head(
                self._pool(latent, "flower")
            ).squeeze(-1)
            error = F.binary_cross_entropy_with_logits(
                prediction, values["_flower"].float(), reduction="none"
            )
            loss = loss + 0.1 * masked_mean(error, values["_flower_valid"])
        if self.species_myco_head is not None and self.species_myco_valid.any():
            prediction = self.species_myco_head(
                self._refined_species.detach()[self.species_myco_valid]
            )
            target = self.species_myco[self.species_myco_valid]
            loss = loss + 0.1 * F.cross_entropy(prediction, target)
        return loss, pollinator

    def _species_state(self, values, observed, context):
        present = {
            name: observed[name]
            if name in self.environment_names else torch.zeros_like(observed[name])
            for name in self.names
        }
        latent = self.encode(
            values, present, context, detach_species=True
        )
        pool = self._pool(latent, self.species_variable)
        base = pool.float() @ self._refined_species.detach().float().t()
        niche_input = pool if self.rank_aligned_expansion else pool.detach()
        niche = self._niche_species_logits(niche_input, include_lens=False)
        return SpeciesState(
            pool, base, niche, values[self.species_variable].long(),
            observed[self.species_variable],
        )

    def _species_loss(self, state: SpeciesState) -> torch.Tensor:
        loss = 0.1 * masked_mean(
            class_error(state.niche, state.target), state.valid
        )
        loss = loss + 0.1 * masked_mean(
            self._family_error(state.base, state.target), state.valid
        )
        if not self.reader_phase:
            return loss

        loss = loss + 0.25 * masked_mean(
            rank_error(state.niche, state.target), state.valid
        )
        key = self._refined_species.detach().float() \
              + self.species_niche_key.detach().float()
        lens = state.niche.detach() + self._species_lens_residual(
            state.pool.detach(), key
        )
        loss = loss + 0.1 * masked_mean(
            class_error(lens, state.target), state.valid
        )
        loss = loss + 0.25 * masked_mean(
            rank_error(lens, state.target), state.valid
        )
        loss = loss + 0.25 * masked_mean(
            self._family_error(lens, state.target), state.valid
        )

        logits = state.niche if self.rank_aligned_expansion else state.base.detach()
        logits = logits + self._identity_detail_logits(state.pool)
        target_family = self.species_family[state.target]
        same_family = self.species_family[None] == target_family[:, None]
        within_family = logits.masked_fill(~same_family, -1e4)
        loss = loss + 0.25 * masked_mean(
            class_error(within_family, state.target), state.valid
        )
        loss = loss + 0.25 * masked_mean(
            rank_error(logits, state.target), state.valid
        )
        return loss
    def _graph_loss(self, reference):
        devices = [torch.cuda.current_device()] if reference.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            mask = torch.rand(
                reference.shape[0], device=reference.device
            ) < 0.15
            reconstructed = self.species_graph(mask)
            loss = 0.1 * self.species_graph.masked_reconstruction_loss(
                mask, reference, metric="mse", reconstructed=reconstructed
            )
            family = self.species_family[~mask]
            prototypes = reference.new_zeros(self.family_count, self.d_model)
            prototypes.index_add_(0, family, reference[~mask])
            counts = torch.bincount(
                family, minlength=self.family_count
            ).to(reference.dtype)
            prototypes = prototypes / counts[:, None].clamp_min(1)
            logits = F.normalize(reconstructed[mask], dim=-1) \
                     @ F.normalize(prototypes, dim=-1).t() / 0.1
            logits = logits.masked_fill((counts == 0)[None], -1e4)
            target = self.species_family[mask]
            valid = counts[target] > 0
            if valid.any():
                loss = loss + 0.03 * F.cross_entropy(
                    logits[valid], target[valid]
                ) / math.log(max(self.family_count, 2))
        return loss, mask

    def _community_loss(
        self, values, observed, context, latent, species: SpeciesState, mask
    ):
        neighbors = context["neighbor_values"].get("identity")
        if neighbors is None:
            return latent.new_zeros(())
        query = self.community_metric(self._pool(latent, "community").detach())
        logits = query @ self._refined_species.detach().t()
        target = torch.zeros_like(logits)
        target.scatter_(1, neighbors.long(), 1)
        target.scatter_(1, species.target[:, None], 1)
        target = target / target.sum(-1, keepdim=True).clamp_min(1)
        loss = 0.5 * distribution_error(logits, target).mean() \
               / math.log(logits.shape[-1])
        if not self.reader_phase:
            return loss

        empty = {name: torch.zeros_like(observed[name]) for name in self.names}
        with torch.no_grad():
            empty_latent = self.encode(
                values, empty, context, detach_species=True, species_mask=mask
            )
            baseline = self.community_metric(
                self._pool(empty_latent, "community")
            )
        present = dict(empty)
        present[self.species_variable] = observed[self.species_variable]
        identity_latent = self.encode(
            values, present, context, detach_species=True, species_mask=mask
        )
        identity = self.community_metric(
            self._pool(identity_latent, "community")
        )
        logits = (identity - baseline) @ self._refined_species.detach().t()
        target = torch.zeros_like(logits)
        target.scatter_(1, neighbors.long(), 1)
        target = target / target.sum(-1, keepdim=True).clamp_min(1)
        valid = species.valid * mask[species.target]
        loss = loss + 0.1 * masked_mean(
            distribution_error(logits, target), valid
        ) / math.log(logits.shape[-1])
        return loss

    def _pollinator_interaction(
        self, values, observed, context, pollinator: PollinatorState | None
    ):
        if not self.reader_phase or pollinator is None:
            return context["position"].new_zeros(())
        present = {
            name: observed[name]
            if name == self.species_variable or name in self.environment_names
            else torch.zeros_like(observed[name])
            for name in self.names
        }
        devices = [torch.cuda.current_device()] \
                  if context["position"].is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            latent = self.encode(values, present, context)
        logits = self.poll_head(self._pollinator_pool(latent, isolated=True))
        return 0.25 * masked_mean(
            distribution_error(logits, pollinator.target), pollinator.valid
        ) / math.log(self.poll_head.out_features)

    def _structured_loss(
        self, values, observed, context, species, pollinator
    ):
        batch = species.target.shape[0]
        photo = torch.arange(
            batch, device=species.target.device
        ).remainder(2).bool()
        present = {
            name: observed[name] & photo
            if name in {"vision_dino", "vision_bio"}
            else torch.zeros_like(observed[name])
            for name in self.names
        }
        latent = self.encode(
            values, present, context, detach_species=True
        )
        pool = self._pool(latent, self.species_variable)
        logits = self._decode_pooled(
            pool, self.species_variable
        ) + self._identity_detail_logits(pool)
        terms = [
            masked_mean(class_error(logits, species.target), species.valid),
            masked_mean(self._family_error(logits, species.target), species.valid),
        ]
        for variable in self.variables:
            if variable.name not in TRAITS:
                continue
            valid = observed[variable.name] & photo
            if valid.any():
                terms.append(masked_mean(
                    class_error(
                        self.decode(latent, variable.name),
                        values[variable.name].long(),
                    ),
                    valid,
                ))
        if pollinator is not None:
            logits = self.poll_head(self._pollinator_pool(latent))
            terms.append(masked_mean(
                distribution_error(logits, pollinator.target), pollinator.valid
            ) / math.log(self.poll_head.out_features))
        return 0.25 * torch.stack(terms).mean()

    def reconstruction_loss(
        self, values, observed, context, hide_probability: float = 0.5
    ):
        batch = context["position"].shape[0]
        present = self._masked_observations(
            observed, batch, context["position"].device, hide_probability
        )
        latent = self.encode(values, present, context)
        self._prime_pool_cache(latent)
        loss = self._reconstruction_terms(values, observed, present, latent)
        scientific, pollinator = self._scientific_terms(values, latent)
        loss = loss + scientific

        species = self._species_state(values, observed, context)
        loss = loss + self._species_loss(species)
        graph, mask = self._graph_loss(self._refined_species.detach())
        loss = loss + graph
        loss = loss + self._community_loss(
            values, observed, context, latent, species, mask
        )
        loss = loss + self._pollinator_interaction(
            values, observed, context, pollinator
        )
        if self.reader_phase:
            return loss, self._structured_loss(
                values, observed, context, species, pollinator
            )
        return loss
