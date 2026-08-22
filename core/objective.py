"""Self-supervised objectives for the DeepEarth world model."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class TrainingObjectiveMixin:
    """Train reconstruction, graph, and scientific read paths together."""

    def reconstruction_loss(self, values, observed, context, hide_probability: float = 0.5):
        batch = context["position"].shape[0]
        present = {name: (torch.rand(batch, device=context["position"].device) > hide_probability) & observed[name]
                   for name in self.names}
        blank = torch.rand(batch, device=context["position"].device) < 0.15
        for name in present:
            present[name] &= ~blank
        latent = self.encode(values, present, context)
        self._prime_pool_cache(latent)
        terms = []
        fiber_terms = []
        mesh_terms = []
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
            if self.reader_phase:
                calibrated = self._calibrate_pollinator_logits(logits.detach())
                calibration_error = -(
                    pollinator_target * F.log_softmax(calibrated, -1)
                ).sum(-1)
                pollinator_calibration_term = (
                    calibration_error * pollinator_valid
                ).sum() / pollinator_valid.sum().clamp_min(1)
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
               + 0.05 * torch.stack(mesh_terms).mean()
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
        environment_latent = self.encode(values, environment_present, context, detach_species=True)
        environment_pool = self._pool(environment_latent, self.species_variable)
        family_logits = environment_pool.float() \
                        @ self._refined_species.detach().float().t()
        target_species = values[self.species_variable].long()
        family_valid = observed[self.species_variable]
        niche_input = environment_pool if self.rank_aligned_expansion \
                      else environment_pool.detach()
        niche_logits = self._niche_species_logits(
            niche_input, include_lens=False
        )
        species_error = F.cross_entropy(
            niche_logits, target_species, reduction="none"
        ) / math.log(max(self._refined_species.shape[0], 2))
        loss = loss + 0.1 * (species_error * family_valid).sum() \
                      / family_valid.sum().clamp_min(1)
        if self.reader_phase:
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
        if self.reader_phase:
            relation_logits = self._identity_detail_logits(environment_pool)
            species_logits = niche_logits if self.rank_aligned_expansion \
                             else family_logits.detach()
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

            if self.reader_phase:
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
        if self.reader_phase and pollinator_target is not None:
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
        if self.reader_phase:
            photo_row = torch.arange(batch, device=loss.device).remainder(2).bool()
            structured_present = {
                name: observed[name] & photo_row
                if name in {"vision_dino", "vision_bio"}
                else torch.zeros_like(observed[name])
                for name in self.names
            }
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
                pollinator_logits = self.poll_head(
                    self._pollinator_pool(structured_latent)
                )
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
