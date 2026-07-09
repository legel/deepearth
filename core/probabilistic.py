"""Probabilistic scores for the diffusion decoder: does its ensemble of samples match the truth and know its own
uncertainty?

A deterministic decoder gives one number; a diffusion decoder gives a distribution, drawn as an ensemble of
samples. We score that ensemble the way GenCast scores a weather ensemble:

  - CRPS (continuous ranked probability score): how well the sampled distribution matches the observed value,
    minimized by a well-calibrated forecast. Computed from samples as ``mean|x - y| - 0.5 mean|x - x'|``.
  - spread/skill: the ratio of ensemble spread to ensemble-mean error; near 1 means the model's stated
    uncertainty matches its actual error (calibrated), below 1 means overconfident.
"""
from __future__ import annotations
import torch


def crps_ensemble(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CRPS from an ensemble. ``samples`` ``[B, N, D]``, ``target`` ``[B, D]`` -> per-row CRPS ``[B]`` (mean over D).

    The pairwise spread term is accumulated over ensemble members (loop over N) rather than materializing the
    ``[B, N, N, D]`` tensor, so it stays memory-light for wide variables."""
    accuracy = (samples - target.unsqueeze(1)).abs().mean(dim=1)          # [B, D]  mean_i |x_i - y|
    n = samples.shape[1]
    spread = torch.zeros_like(accuracy)
    for i in range(n):                                                    # mean_ij |x_i - x_j|, streamed over i
        spread = spread + (samples[:, i:i + 1] - samples).abs().mean(dim=1)
    spread = spread / n
    return (accuracy - 0.5 * spread).mean(dim=-1)                         # [B]


def spread_skill(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Ratio of ensemble spread (std) to ensemble-mean RMSE, averaged; ~1 is calibrated, <1 overconfident."""
    mean = samples.mean(dim=1)
    skill = (mean - target).pow(2).mean().sqrt()
    spread = samples.std(dim=1).pow(2).mean().sqrt()
    return (spread / skill.clamp_min(1e-9)).item()


def rank_histogram_flatness(samples: torch.Tensor, target: torch.Tensor) -> float:
    """How flat is the rank histogram (Talagrand diagram)? For each scalar the truth falls at some rank among the
    ``N`` sorted samples; a calibrated ensemble spreads those ranks uniformly. Returns the total-variation distance
    from uniform (0 = perfectly flat/calibrated, higher = miscalibrated: U-shaped under-, n-shaped over-dispersed)."""
    n = samples.shape[1]
    rank = (samples < target.unsqueeze(1)).sum(dim=1).flatten()          # 0..N, per scalar
    hist = torch.bincount(rank, minlength=n + 1).float()
    hist = hist / hist.sum().clamp_min(1.0)
    return 0.5 * (hist - 1.0 / (n + 1)).abs().sum().item()


@torch.no_grad()
def probabilistic_scores(model, source, indices, device, n_samples: int = 16, batch: int = 2048) -> dict:
    """For each diffusion (continuous, reconstructable) variable, condition on the config's given inputs and score
    the sampled ensemble on held-out ``indices`` by CRPS and spread/skill."""
    model.eval()
    targets = [n for n in model.diffusion_heads]
    if not targets:
        return {}
    crps = {t: [0.0, 0.0] for t in targets}; ss = {t: [] for t in targets}; rh = {t: [] for t in targets}
    import numpy as np
    given = [v.name for v in model.variables if not v.reconstruct] or ["vision_dino", "vision_bio"]
    given = [g for g in given if g in [v.name for v in model.variables]]
    for c0 in range(0, len(indices), batch):
        idx = torch.tensor(indices[c0:c0 + batch], device=device)
        values, observed, coords, nbr_coords, manifold_coords, nbr_values = source.batch(idx)
        ctx = model.context(coords, nbr_coords, manifold_coords, nbr_values)
        present = {n: torch.zeros(len(idx), dtype=torch.bool, device=device) for n in [v.name for v in model.variables]}
        for g in given:
            present[g] = torch.ones(len(idx), dtype=torch.bool, device=device)
        z = model.encode(values, present, ctx)
        for t in targets:
            samples = torch.stack([model.diffusion_heads[t].sample(model._pooled(z, t)) for _ in range(n_samples)], 1)
            m = observed[t]
            if m.any():
                crps[t][0] += crps_ensemble(samples[m], values[t][m]).sum().item(); crps[t][1] += m.sum().item()
                ss[t].append(spread_skill(samples[m], values[t][m]))
                rh[t].append(rank_histogram_flatness(samples[m], values[t][m]))
    return {t: {"crps": crps[t][0] / max(crps[t][1], 1), "spread_skill": float(np.mean(ss[t]) if ss[t] else 0.0),
                "rank_flatness": float(np.mean(rh[t]) if rh[t] else 0.0)} for t in targets}


# --------------------------------------------------------------------------------------- standalone unit test
def _test():
    torch.manual_seed(0)
    B, N, D = 64, 32, 4
    truth = torch.randn(B, D)
    # a perfectly-centered, appropriately-spread ensemble scores far better than a biased or collapsed one
    good = truth.unsqueeze(1) + 0.3 * torch.randn(B, N, D)
    biased = truth.unsqueeze(1) + 3.0 + 0.3 * torch.randn(B, N, D)
    collapsed = truth.unsqueeze(1).expand(B, N, D) + 5.0     # confident and wrong
    assert crps_ensemble(good, truth).mean() < crps_ensemble(biased, truth).mean(), "CRPS must penalize bias"
    # a matched ensemble is better calibrated (spread/skill closer to 1) than a collapsed, overconfident one
    assert spread_skill(good, truth) > spread_skill(collapsed, truth)
    # a well-spread ensemble has a flatter rank histogram than a biased one whose truth always ranks at an extreme
    assert rank_histogram_flatness(good, truth) < rank_histogram_flatness(biased, truth)
    # exact samples give ~0 CRPS
    exact = truth.unsqueeze(1).expand(B, N, D)
    assert crps_ensemble(exact, truth).abs().mean() < 1e-5
    print(f"probabilistic.py: all unit tests passed (good CRPS {crps_ensemble(good, truth).mean():.3f} "
          f"< biased {crps_ensemble(biased, truth).mean():.3f}; rank-flatness good {rank_histogram_flatness(good, truth):.2f} "
          f"< biased {rank_histogram_flatness(biased, truth):.2f})")


if __name__ == "__main__":
    _test()
