"""Fixed census of the learned world mesh.

This is a diagnostic instrument, not a promotion evaluator. It measures the
persistent address field separately from transient modality writes and can
compare checkpoints captured along one uninterrupted training trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(HERE))

PROTOCOL = "mesh-census-v2"
INFORMATION_PROTOCOL = "mesh-information-v3-task-readout"
DEFAULT_STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
ORDERING_EPSILON = 1e-5


def _effective_rank(x: torch.Tensor) -> float:
    """Entropy effective rank, normalized to [0, 1]."""
    x = x.float()
    if x.shape[0] < 2:
        return 0.0
    x = x - x.mean(0, keepdim=True)
    covariance = x.t() @ x / (x.shape[0] - 1)
    spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = spectrum.sum()
    if float(total) <= 1e-12:
        return 0.0
    probability = spectrum / total
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp() / min(x.shape[0] - 1, x.shape[1]))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def _spatial_shift(coords: torch.Tensor, kilometers: float) -> torch.Tensor:
    shifted = coords.clone()
    denominator = 111.320 * torch.cos(torch.deg2rad(coords[:, 0])).clamp_min(0.1)
    shifted[:, 1] = shifted[:, 1] + kilometers / denominator
    return shifted


def _time_shift(coords: torch.Tensor, fraction: float) -> torch.Tensor:
    shifted = coords.clone()
    forward = coords[:, 3] <= 1.0 - fraction
    shifted[:, 3] = torch.where(forward, coords[:, 3] + fraction, coords[:, 3] - fraction)
    return shifted


def _field(model, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial, temporal = model.mesh.raw(coords)
    spatial_state = model.absolute_proj_s(spatial)
    temporal_state = model.absolute_proj_t(temporal)
    return spatial_state, temporal_state, spatial_state + temporal_state


def _address_stats(encoder, inputs: torch.Tensor) -> dict:
    """Measure sampled center-cell occupancy and true hash aliasing by level."""
    count = inputs.reshape(-1, inputs.shape[-1]).shape[0]
    addresses = torch.full(
        (count, encoder.num_levels), -1, dtype=torch.int32, device=inputs.device
    )
    encoder(inputs.contiguous(), size=1.0, collision_tracking={
        "collision_indices": addresses,
        "example_offset": 0,
        "max_tracked_examples": count,
    })

    unit = ((inputs.reshape(count, -1).float() + 1.0) / 2.0).clamp(0, 1)
    scale = torch.exp2(encoder.per_level_scale.float()) * encoder.base_resolution - 1.0
    cells = torch.floor(unit[:, None, :] * scale[None, :, :]).to(torch.int64).cpu()
    addresses = addresses.cpu()
    offsets = encoder.offsets.cpu()
    levels = []
    for level in range(encoder.num_levels):
        pairs = {
            tuple(int(v) for v in cells[row, level].tolist()): int(addresses[row, level])
            for row in range(count)
        }
        unique_cells = len(pairs)
        unique_addresses = len(set(pairs.values()))
        table_size = int(offsets[level + 1] - offsets[level])
        levels.append({
            "level": level,
            "sampled_unique_cells": unique_cells,
            "sampled_unique_addresses": unique_addresses,
            "center_collision_rate": 1.0 - unique_addresses / max(unique_cells, 1),
            "sampled_table_load": unique_addresses / max(table_size, 1),
            "table_size": table_size,
        })
    return {
        "levels": levels,
        "collision_free_mean": 1.0 - _mean([v["center_collision_rate"] for v in levels]),
    }


@torch.no_grad()
def _writes(model, values: dict, observed: dict, base: torch.Tensor) -> tuple[dict, torch.Tensor, list[str]]:
    species = model._species()
    names = []
    deltas = []
    masks = []
    summaries = {}
    for name in model.write_gate:
        if name not in values:
            continue
        if name in observed:
            valid = observed[name]
        else:
            value = values[name]
            valid = value.isfinite().all(-1) & (value.norm(dim=-1) > 1e-6)
        if not valid.any():
            continue
        edit = model._adapt(name, values[name], species) + model.write_type[name]
        gate = torch.sigmoid(model.write_gate[name])
        delta = edit.unsqueeze(1) * gate.view(1, -1, 1)
        norm = delta[valid].float().norm(dim=-1).mean(0)
        allocation = norm / norm.sum().clamp_min(1e-12)
        entropy = -(allocation * allocation.clamp_min(1e-12).log()).sum() / math.log(model.levels)
        summaries[name] = {
            "observations": int(valid.sum()),
            "gate_mean": float(gate.mean()),
            "write_norm_mean": float(norm.mean()),
            "level_allocation_entropy": float(entropy),
            "level_write_norm": norm.cpu().tolist(),
        }
        names.append(name)
        deltas.append(delta)
        masks.append(valid)

    level_ranks = []
    mean_directions = []
    for level in range(model.levels):
        population = torch.cat([delta[valid, level] for delta, valid in zip(deltas, masks)], 0)
        level_ranks.append(_effective_rank(population))
        mean_directions.append(torch.stack([
            delta[valid, level].float().mean(0) for delta, valid in zip(deltas, masks)
        ]))

    interference = []
    for directions in mean_directions:
        directions = F.normalize(directions, dim=-1)
        similarity = (directions @ directions.t()).abs()
        off_diagonal = ~torch.eye(len(names), dtype=torch.bool, device=similarity.device)
        interference.append(float(similarity[off_diagonal].mean()))

    update = torch.zeros_like(base)
    count = base.new_zeros((base.shape[0], 1, 1))
    atlas_norms = []
    for delta, valid in zip(deltas, masks):
        mask = valid.to(base.dtype).view(-1, 1, 1)
        update += mask * delta
        count += mask
        atlas_norms.append((mask * delta).float().norm(dim=-1))
    update = update / count.clamp_min(1.0).sqrt()
    ratio = update.float().norm(dim=-1) / base.float().norm(dim=-1).clamp_min(1e-8)

    return ({
        "persistent_or_transient": "transient",
        "modalities": summaries,
        "write_effective_rank_by_level": level_ranks,
        "write_effective_rank_mean": _mean(level_ranks),
        "cross_modal_abs_cosine_by_level": interference,
        "cross_modal_abs_cosine_mean": _mean(interference),
        "write_to_base_norm_ratio": float(ratio.mean()),
    }, torch.stack(atlas_norms, 1), names)


def _harmonic(values: list[float]) -> float:
    return len(values) / sum(1.0 / max(value, 1e-8) for value in values)


def _sample_indices(source, split: str, count: int, seed: int) -> np.ndarray:
    population = np.asarray(getattr(source, split))
    generator = np.random.default_rng(seed)
    return population[generator.permutation(len(population))[:min(count, len(population))]]


@torch.no_grad()
def _stage_batch(model, source, selected: np.ndarray, mask_seed: int) -> dict:
    """Expose coordinate, written, fused, and task-read states."""
    index = torch.as_tensor(selected, device=source.device)
    values, observed, coords, neighbors, manifolds, neighbor_values = source.batch(index)
    generator = torch.Generator(device=coords.device).manual_seed(mask_seed)
    present = {
        name: (torch.rand(observed[name].shape, device=coords.device, generator=generator) > 0.5)
              & observed[name]
        for name in model.names
    }
    blank = torch.rand(len(selected), device=coords.device, generator=generator) < 0.15
    for name in present:
        present[name] &= ~blank

    context = model.context(coords, neighbors, manifolds, neighbor_values)
    species = model._species()
    write_mask = dict(present)
    for name in model.always_names:
        if name in values:
            write_mask[name] = values[name].isfinite().all(-1) \
                               & (values[name].norm(dim=-1) > 1e-6)

    query_fibers = model._fiber_write(context["query_state"], values, write_mask, species)
    query_written = model._write(context["query_state"], values, write_mask, species)
    neighbor_state = context["neighbor_state"]
    masks = {
        name: torch.ones(
            value.shape[:-1] if value.dim() > 2 else value.shape,
            dtype=torch.bool,
            device=value.device,
        )
        for name, value in context["neighbor_values"].items()
    }
    neighbor_fibers = model._fiber_write(
        neighbor_state, context["neighbor_values"], masks, species
    )
    neighbor_written = model._write(
        neighbor_state, context["neighbor_values"], masks, species
    ) if masks else neighbor_state

    latent = model.encode(values, present, context)
    position = torch.cat((
        context["query_state"].flatten(1),
        context["neighbor_state"].mean(1).flatten(1),
    ), -1)
    written = torch.cat((
        query_written.flatten(1),
        neighbor_written.mean(1).flatten(1),
        query_fibers.flatten(1),
        neighbor_fibers.mean(1).flatten(1),
    ), -1)
    lens_features = {}
    for lens_index, lens_name in enumerate(getattr(model, "write_lens", {}) and (
            "abiotic", "visual", "biological", "ecological")):
        lens_features[f"lens.{lens_name}"] = torch.cat((
            query_fibers[..., lens_index, :].flatten(1),
            neighbor_fibers[..., lens_index, :].mean(1).flatten(1),
        ), -1)

    read_names = list(model.names)
    special_targets = {}
    special_hidden = {}
    for name, target_key, valid_key, kind in (
        ("lfmc", "_lfmc", "_lfmc_valid", "continuous"),
        ("myco", "_myco", "_myco_valid", "categorical"),
        ("flower", "_flower", "_flower_valid", "categorical"),
    ):
        if target_key not in values:
            continue
        target = values[target_key]
        if kind == "continuous":
            target = target.float().clamp_min(1.0).log().unsqueeze(-1)
        special_targets[name] = target.detach().cpu()
        special_hidden[name] = values[valid_key].bool().detach().cpu()
        read_names.append(name)
    if "_poll_idx" in values and getattr(model, "poll_head", None) is not None:
        pollinator = values["_poll_frq"].new_zeros((len(selected), model.poll_head.out_features))
        pollinator.scatter_add_(
            1,
            values["_poll_idx"].clamp(0, model.poll_head.out_features - 1),
            values["_poll_frq"].float(),
        )
        special_targets["pollinator"] = pollinator.detach().cpu()
        special_hidden["pollinator"] = values["_poll_valid"].bool().detach().cpu()
        read_names.append("pollinator")
    readouts = {}
    task_queries = getattr(model, "mesh_read_query", {})
    for name in read_names:
        pool_name = name if name in model.names or name in task_queries else model.species_variable
        readouts[name] = model._pool(latent, pool_name).float().cpu()

    return {
        "features": {
            "position": position.float().cpu(),
            "written": written.float().cpu(),
            "latent": latent.flatten(1).float().cpu(),
            **{name: value.float().cpu() for name, value in lens_features.items()},
        },
        "readouts": readouts,
        "targets": {
            **{name: values[name].detach().cpu() for name in model.names},
            **special_targets,
        },
        "hidden": {
            **{
                name: ((~present[name]) & observed[name]).detach().cpu()
                for name in model.names
            },
            **special_hidden,
        },
    }


@torch.no_grad()
def _collect_probe_data(model, source, indices: np.ndarray, seed: int, batch: int) -> dict:
    chunks = [
        _stage_batch(model, source, indices[start:start + batch], seed + start)
        for start in range(0, len(indices), batch)
    ]
    return {
        group: {
            name: torch.cat([chunk[group][name] for chunk in chunks])
            for name in chunks[0][group]
        }
        for group in ("features", "readouts", "targets", "hidden")
    }


def _probe_features(features: torch.Tensor, device: str) -> torch.Tensor:
    features = features.to(device=device, dtype=torch.float32)
    features = F.layer_norm(features, (features.shape[-1],)) / math.sqrt(features.shape[-1])
    return torch.cat((features, torch.ones_like(features[:, :1])), -1)


def _ridge_predict(
    train: torch.Tensor,
    target: torch.Tensor,
    test: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    gram = train @ train.t()
    gram.diagonal().add_(ridge)
    weights = torch.linalg.solve(gram, target)
    return (test @ train.t()) @ weights


def _probe_target(
    variable,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_target: torch.Tensor,
    test_target: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    ridge: float,
) -> dict | None:
    train_rows = train_mask.nonzero(as_tuple=False).flatten()
    test_rows = test_mask.nonzero(as_tuple=False).flatten()
    if len(train_rows) < 16 or len(test_rows) < 16:
        return None
    x_train = train_features[train_rows]
    x_test = test_features[test_rows]
    if variable.kind == "categorical":
        y_train = train_target[train_rows].long()
        y_test = test_target[test_rows].long()
        classes = variable.num_classes
        prediction = _ridge_predict(
            x_train, F.one_hot(y_train, classes).float(), x_test, ridge
        ).argmax(-1)
        raw = float((prediction == y_test).float().mean())
        majority = torch.bincount(y_train, minlength=classes).argmax()
        null = float((y_test == majority).float().mean())
        skill = (raw - null) / max(1.0 - null, 1e-8)
        metric = "top1_skill_above_train_majority"
    else:
        y_train = train_target[train_rows].float().flatten(1)
        y_test = test_target[test_rows].float().flatten(1)
        prediction = _ridge_predict(x_train, y_train, x_test, ridge)
        center = y_train.mean(0, keepdim=True)
        error = float((prediction - y_test).square().mean())
        null_error = float((center - y_test).square().mean())
        raw = 1.0 - error / max(null_error, 1e-12)
        null = 0.0
        skill = raw
        metric = "variance_explained_over_train_mean"
    return {
        "metric": metric,
        "raw": raw,
        "null": null,
        "normalized_skill": skill,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
    }


@torch.no_grad()
def measure_information(
    model,
    source,
    *,
    train_samples: int,
    test_samples: int,
    sample_seed: int,
    batch: int,
    ridge: float,
    device: str,
) -> dict:
    """Measure held-out linear accessibility through the task reader."""
    model.eval()
    train_indices = _sample_indices(source, "train", train_samples, sample_seed)
    test_indices = _sample_indices(source, "test", test_samples, sample_seed + 1)
    train = _collect_probe_data(model, source, train_indices, sample_seed + 10, batch)
    test = _collect_probe_data(model, source, test_indices, sample_seed + 20, batch)
    train_features = {
        name: _probe_features(value, device) for name, value in train["features"].items()
    }
    test_features = {
        name: _probe_features(value, device) for name, value in test["features"].items()
    }
    variables = {variable.name: variable for variable in model.variables}
    if "pollinator" in train["targets"]:
        variables["pollinator"] = SimpleNamespace(kind="continuous", num_classes=0)
    if "lfmc" in train["targets"]:
        variables["lfmc"] = SimpleNamespace(kind="continuous", num_classes=0)
    if "myco" in train["targets"]:
        variables["myco"] = SimpleNamespace(kind="categorical", num_classes=5)
    if "flower" in train["targets"]:
        variables["flower"] = SimpleNamespace(kind="categorical", num_classes=2)
    train_readouts = {
        name: _probe_features(value, device) for name, value in train["readouts"].items()
    }
    test_readouts = {
        name: _probe_features(value, device) for name, value in test["readouts"].items()
    }
    targets = {}
    for name, variable in variables.items():
        stages = {}
        for stage in train_features:
            result = _probe_target(
                variable,
                train_features[stage],
                test_features[stage],
                train["targets"][name].to(device),
                test["targets"][name].to(device),
                train["hidden"][name].to(device),
                test["hidden"][name].to(device),
                ridge,
            )
            if result is not None:
                stages[stage] = result
        if name in train_readouts:
            result = _probe_target(
                variable,
                train_readouts[name],
                test_readouts[name],
                train["targets"][name].to(device),
                test["targets"][name].to(device),
                train["hidden"][name].to(device),
                test["hidden"][name].to(device),
                ridge,
            )
            if result is not None:
                stages["readout"] = result
        if all(stage in stages for stage in ("position", "written", "latent", "readout")):
            position = stages["position"]["normalized_skill"]
            written = stages["written"]["normalized_skill"]
            latent = stages["latent"]["normalized_skill"]
            readout = stages["readout"]["normalized_skill"]
            written_gain = written - position
            reader_gap = written - latent
            targets[name] = {
                "kind": variable.kind,
                "stages": stages,
                "written_gain_over_position": written_gain,
                "reader_gap_written_minus_latent": reader_gap,
                "task_read_gain_over_latent": readout - latent,
                "task_reader_gap_written_minus_readout": written - readout,
                "linear_retention": (latent - position) / written_gain
                                    if written_gain >= 0.01 else None,
            }

    stage_names = ("position", "written", "latent", "readout")
    stage_summary = {}
    for stage in stage_names:
        values = [target["stages"][stage]["normalized_skill"] for target in targets.values()]
        ordered = sorted(values)
        quartile = max(1, math.ceil(len(ordered) / 4))
        stage_summary[stage] = {
            "mean_normalized_skill": _mean(values),
            "worst_quartile_mean": _mean(ordered[:quartile]),
            "at_or_below_null": sum(value <= 0.0 for value in values),
            "targets": len(values),
        }
    lens_summary = {}
    lens_names = sorted(
        stage for stage in train_features if stage.startswith("lens.")
    )
    for lens in lens_names:
        values = [
            target["stages"][lens]["normalized_skill"]
            for target in targets.values()
            if lens in target["stages"]
        ]
        lens_summary[lens.removeprefix("lens.")] = {
            "mean_normalized_skill": _mean(values),
            "at_or_below_null": sum(value <= 0.0 for value in values),
            "targets": len(values),
        }
    written_gains = [target["written_gain_over_position"] for target in targets.values()]
    reader_gaps = [target["reader_gap_written_minus_latent"] for target in targets.values()]
    task_read_gains = [target["task_read_gain_over_latent"] for target in targets.values()]
    task_reader_gaps = [target["task_reader_gap_written_minus_readout"] for target in targets.values()]
    retention = [
        target["linear_retention"] for target in targets.values()
        if target["linear_retention"] is not None
    ]
    return {
        "protocol": INFORMATION_PROTOCOL,
        "interpretation": "Fixed-protocol linear accessibility diagnostic; not a promotion score.",
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "sample_seed": sample_seed,
        "mask_probability": 0.5,
        "blank_probability": 0.15,
        "ridge": ridge,
        "stages": stage_summary,
        "mean_written_gain_over_position": _mean(written_gains),
        "mean_reader_gap_written_minus_latent": _mean(reader_gaps),
        "mean_task_read_gain_over_latent": _mean(task_read_gains),
        "mean_task_reader_gap_written_minus_readout": _mean(task_reader_gaps),
        "mean_linear_retention": _mean(retention),
        "retention_targets": len(retention),
        "lenses": lens_summary,
        "targets": targets,
    }


@torch.no_grad()
def measure(model, source, sample_count: int, seed: int) -> tuple[dict, dict]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(source.test), generator=generator)[:sample_count].numpy()
    selected = np.asarray(source.test)[order]
    index = torch.as_tensor(selected, device=source.device)
    values, observed, coords, _, _, _ = source.batch(index)

    spatial, temporal, base = _field(model, coords)
    near_spatial, _, _ = _field(model, _spatial_shift(coords, 1.0))
    far_spatial, _, _ = _field(model, coords.roll(coords.shape[0] // 2, 0))
    near_geo = _cosine(spatial, near_spatial)
    far_geo = _cosine(spatial, far_spatial)

    _, near_temporal, _ = _field(model, _time_shift(coords, 1.0 / 12.0))
    shuffled_time = coords.clone()
    shuffled_time[:, 3] = coords[:, 3].roll(coords.shape[0] // 2)
    _, far_temporal, _ = _field(model, shuffled_time)
    near_time = _cosine(temporal, near_temporal)
    far_time = _cosine(temporal, far_temporal)

    state_ranks = [_effective_rank(base[:, level]) for level in range(model.levels)]
    state_variance = base.float().var(0).mean(-1)
    xyz, xyzt = model.mesh.coordinates(coords)
    spatial_addresses = _address_stats(model.mesh.spatial, xyz)
    projections = ((0, 1, 3), (1, 2, 3), (0, 2, 3))
    temporal_addresses = [
        _address_stats(encoder, xyzt[..., axes])
        for encoder, axes in zip(model.mesh.temporal, projections)
    ]
    collision_free = _mean([
        spatial_addresses["collision_free_mean"],
        *[item["collision_free_mean"] for item in temporal_addresses],
    ])
    writes, write_norms, write_names = _writes(model, values, observed, base)

    geo_order = float((near_geo - far_geo > ORDERING_EPSILON).float().mean())
    time_order = float((near_time - far_time > ORDERING_EPSILON).float().mean())
    geo_margin = float((near_geo - far_geo).mean())
    time_margin = float((near_time - far_time).mean())
    rank_mean = _mean(state_ranks)
    axes = {
        "spatial_ordering_accuracy": geo_order,
        "temporal_ordering_accuracy": time_order,
        "state_effective_rank": rank_mean,
        "sampled_collision_free": collision_free,
    }
    report = {
        "protocol": PROTOCOL,
        "samples": len(selected),
        "sample_seed": seed,
        "sample_index_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "census_score": _harmonic(list(axes.values())),
        "census_score_note": "Structural breadth diagnostic, not a model-quality or promotion score.",
        "census_axes": axes,
        "persistent_field": {
            "state_effective_rank_by_level": state_ranks,
            "state_effective_rank_mean": rank_mean,
            "state_variance_by_level": state_variance.cpu().tolist(),
            "spatial_near_vs_shuffled_margin_by_level": (near_geo - far_geo).mean(0).cpu().tolist(),
            "spatial_near_vs_shuffled_margin": geo_margin,
            "spatial_ordering_accuracy": geo_order,
            "temporal_near_vs_shuffled_margin_by_level": (near_time - far_time).mean(0).cpu().tolist(),
            "temporal_near_vs_shuffled_margin": time_margin,
            "temporal_ordering_accuracy": time_order,
        },
        "addresses": {
            "spatial": spatial_addresses,
            "temporal_projections": temporal_addresses,
            "sampled_collision_free_mean": collision_free,
            "note": "Collision rates use distinct sampled center cells, not repeated observations or interpolation corners.",
        },
        "writes": writes,
    }
    atlas = {
        "protocol": PROTOCOL,
        "indices": torch.as_tensor(selected),
        "coordinates": coords.cpu(),
        "persistent_state": base.float().cpu(),
        "persistent_state_norm": base.float().norm(dim=-1).cpu(),
        "write_norm": write_norms.cpu(),
        "write_names": write_names,
    }
    return report, atlas


def _print_scorecard(report: dict, step: int | None = None) -> None:
    prefix = f"step {step:>6}" if step is not None else "checkpoint"
    axes = report["census_axes"]
    field = report["persistent_field"]
    print(
        f"{prefix}  census={report['census_score']:.6f}  "
        f"space={axes['spatial_ordering_accuracy']:.4f}  "
        f"space_margin={field['spatial_near_vs_shuffled_margin']:.4f}  "
        f"time={axes['temporal_ordering_accuracy']:.4f}  "
        f"time_margin={field['temporal_near_vs_shuffled_margin']:.4f}  "
        f"rank={axes['state_effective_rank']:.4f}  "
        f"collision_free={axes['sampled_collision_free']:.4f}  "
        f"write_rank={report['writes']['write_effective_rank_mean']:.4f}  "
        f"write/base={report['writes']['write_to_base_norm_ratio']:.4f}",
        flush=True,
    )
    if "information" in report:
        information = report["information"]
        stages = information["stages"]
        print(
            "  information  "
            f"position={stages['position']['mean_normalized_skill']:+.4f}  "
            f"written={stages['written']['mean_normalized_skill']:+.4f}  "
            f"latent={stages['latent']['mean_normalized_skill']:+.4f}  "
            f"readout={stages['readout']['mean_normalized_skill']:+.4f}  "
            f"write_gain={information['mean_written_gain_over_position']:+.4f}  "
            f"reader_gap={information['mean_reader_gap_written_minus_latent']:+.4f}  "
            f"task_gain={information['mean_task_read_gain_over_latent']:+.4f}  "
            f"retention={information['mean_linear_retention']:+.4f}",
            flush=True,
        )


def _save_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def _fusion_scorecard(model, source, device: str, batch: int) -> tuple[dict, str]:
    from deepearth.autoresearch.main.harness import evaluate as canonical

    raw = canonical.evaluate_benchmarks(model, source, device, batch=batch)
    scorecard = {
        "protocol": canonical.BENCHMARK_PROTOCOL,
        "harmonic": canonical.net_score(raw),
        "arithmetic": canonical.arithmetic_net(raw),
        "scores": raw,
    }
    return scorecard, canonical.format_benchmarks(raw)


def _load_model(cache: str, device: str, checkpoint: Path):
    import model as experiment
    from data import load

    source, variables, always = load(cache, device)
    model = experiment.build_model(source, variables, always, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, source


def _rank_correlation(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    value = torch.tensor(values)
    ranks = torch.argsort(torch.argsort(value)).float()
    steps = torch.arange(len(values), dtype=torch.float32)
    return float(torch.corrcoef(torch.stack((steps, ranks)))[0, 1])


def _correlation(left: list[float], right: list[float], *, ranked: bool = False) -> float:
    a, b = torch.tensor(left), torch.tensor(right)
    if ranked:
        a = torch.argsort(torch.argsort(a)).float()
        b = torch.argsort(torch.argsort(b)).float()
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(torch.corrcoef(torch.stack((a.float(), b.float())))[0, 1])


def _monotonic_summary(rows: list[tuple[int, dict]]) -> dict:
    paths = {
        "census_score": lambda report: report["census_score"],
        "spatial_ordering_accuracy": lambda report: report["census_axes"]["spatial_ordering_accuracy"],
        "spatial_locality_margin": lambda report: report["persistent_field"]["spatial_near_vs_shuffled_margin"],
        "temporal_ordering_accuracy": lambda report: report["census_axes"]["temporal_ordering_accuracy"],
        "temporal_locality_margin": lambda report: report["persistent_field"]["temporal_near_vs_shuffled_margin"],
        "state_effective_rank": lambda report: report["census_axes"]["state_effective_rank"],
        "sampled_collision_free": lambda report: report["census_axes"]["sampled_collision_free"],
        "write_effective_rank": lambda report: report["writes"]["write_effective_rank_mean"],
    }
    if rows and all("fusion" in report for _, report in rows):
        paths.update({
            "fusion_harmonic": lambda report: report["fusion"]["harmonic"],
            "fusion_arithmetic": lambda report: report["fusion"]["arithmetic"],
        })
    if rows and all("information" in report for _, report in rows):
        paths.update({
            "position_mean_skill": lambda report: report["information"]["stages"]["position"]["mean_normalized_skill"],
            "written_mean_skill": lambda report: report["information"]["stages"]["written"]["mean_normalized_skill"],
            "latent_mean_skill": lambda report: report["information"]["stages"]["latent"]["mean_normalized_skill"],
            "readout_mean_skill": lambda report: report["information"]["stages"]["readout"]["mean_normalized_skill"],
            "written_gain": lambda report: report["information"]["mean_written_gain_over_position"],
            "reader_gap": lambda report: report["information"]["mean_reader_gap_written_minus_latent"],
            "task_read_gain": lambda report: report["information"]["mean_task_read_gain_over_latent"],
            "linear_retention": lambda report: report["information"]["mean_linear_retention"],
        })
    result = {"protocol": PROTOCOL, "steps": [step for step, _ in rows], "metrics": {}}
    for name, get in paths.items():
        values = [float(get(report)) for _, report in rows]
        result["metrics"][name] = {
            "values": values,
            "nondecreasing": all(right >= left - 1e-6 for left, right in zip(values, values[1:])),
            "increasing_transitions": sum(right > left + 1e-6 for left, right in zip(values, values[1:])),
            "decreasing_transitions": sum(right < left - 1e-6 for left, right in zip(values, values[1:])),
            "step_rank_correlation": _rank_correlation(values),
        }
    if "fusion_harmonic" in result["metrics"]:
        harmonic = result["metrics"]["fusion_harmonic"]["values"]
        arithmetic = result["metrics"]["fusion_arithmetic"]["values"]
        result["fusion_correlations"] = {}
        for name, metric in result["metrics"].items():
            if name.startswith("fusion_"):
                continue
            values = metric["values"]
            result["fusion_correlations"][name] = {
                "harmonic_pearson": _correlation(values, harmonic),
                "harmonic_spearman": _correlation(values, harmonic, ranked=True),
                "arithmetic_pearson": _correlation(values, arithmetic),
                "arithmetic_spearman": _correlation(values, arithmetic, ranked=True),
            }
    return result


def _measure_command(args) -> None:
    model, source = _load_model(args.cache, args.device, args.checkpoint)
    report, atlas = measure(model, source, args.samples, args.sample_seed)
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.fusion:
        report["fusion"], formatted = _fusion_scorecard(model, source, args.device, args.fusion_batch)
        print(formatted, flush=True)
    if args.information:
        report["information"] = measure_information(
            model,
            source,
            train_samples=args.probe_train_samples,
            test_samples=args.probe_test_samples,
            sample_seed=args.probe_seed,
            batch=args.probe_batch,
            ridge=args.probe_ridge,
            device=args.device,
        )
    _save_report(report, args.output)
    if args.atlas:
        args.atlas.parent.mkdir(parents=True, exist_ok=True)
        torch.save(atlas, args.atlas)
    _print_scorecard(report)
    print(f"report: {args.output}", flush=True)


def _trajectory_command(args) -> None:
    import model as experiment

    steps = tuple(sorted(set(args.checkpoints)))
    if not steps or steps[0] < 0 or steps[-1] > args.steps:
        raise ValueError("checkpoint steps must fall between 0 and --steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupied = list(args.output_dir.glob("step_*.pt")) + list(args.output_dir.glob("census_*.json"))
    if occupied:
        raise FileExistsError(f"trajectory output is not empty: {args.output_dir}")

    design = replace(experiment.EXPERIMENT, steps=args.steps, seed=args.seed)
    model, source = experiment.train(
        args.cache,
        args.device,
        design,
        checkpoint_steps=frozenset(steps),
        checkpoint_dir=args.output_dir,
    )
    rows = []
    for step in steps:
        checkpoint = args.output_dir / f"step_{step:06d}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
        report, atlas = measure(model, source, args.samples, args.sample_seed)
        report.update({
            "training_seed": args.seed,
            "training_step": step,
            "training_total_steps": args.steps,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
        if args.information:
            report["information"] = measure_information(
                model,
                source,
                train_samples=args.probe_train_samples,
                test_samples=args.probe_test_samples,
                sample_seed=args.probe_seed,
                batch=args.probe_batch,
                ridge=args.probe_ridge,
                device=args.device,
            )
        if args.fusion:
            report["fusion"], formatted = _fusion_scorecard(
                model, source, args.device, args.fusion_batch
            )
            print(formatted, flush=True)
        _save_report(report, args.output_dir / f"census_{step:06d}.json")
        torch.save(atlas, args.output_dir / f"atlas_{step:06d}.pt")
        rows.append((step, report))
        _print_scorecard(report, step)

    summary = _monotonic_summary(rows)
    _save_report(summary, args.output_dir / "monotonicity.json")
    print("\nMONOTONICITY", flush=True)
    for name, result in summary["metrics"].items():
        print(
            f"  {name:<29} nondecreasing={str(result['nondecreasing']):<5} "
            f"up={result['increasing_transitions']} down={result['decreasing_transitions']} "
            f"rho={result['step_rank_correlation']:+.3f}",
            flush=True,
        )
    print(f"trajectory: {args.output_dir}", flush=True)


def _steps(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("measure", help="measure one existing checkpoint")
    single.add_argument("--cache", required=True)
    single.add_argument("--checkpoint", type=Path, required=True)
    single.add_argument("--device", default="cuda")
    single.add_argument("--samples", type=int, default=512)
    single.add_argument("--sample-seed", type=int, default=20260814)
    single.add_argument("--output", type=Path, required=True)
    single.add_argument("--atlas", type=Path)
    single.add_argument("--fusion", action="store_true", help="also run the immutable human-capability suite")
    single.add_argument("--fusion-batch", type=int, default=1280)
    single.add_argument("--information", action="store_true", help="measure S0 -> S1 -> Z accessibility")
    single.add_argument("--probe-train-samples", type=int, default=1024)
    single.add_argument("--probe-test-samples", type=int, default=512)
    single.add_argument("--probe-seed", type=int, default=20260815)
    single.add_argument("--probe-batch", type=int, default=128)
    single.add_argument("--probe-ridge", type=float, default=1.0)
    single.set_defaults(run=_measure_command)

    trajectory = commands.add_parser("trajectory", help="train once and census intermediate checkpoints")
    trajectory.add_argument("--cache", required=True)
    trajectory.add_argument("--device", default="cuda")
    trajectory.add_argument("--steps", type=int, default=8000)
    trajectory.add_argument("--checkpoints", type=_steps, default=DEFAULT_STEPS)
    trajectory.add_argument("--seed", type=int, default=1337)
    trajectory.add_argument("--samples", type=int, default=512)
    trajectory.add_argument("--sample-seed", type=int, default=20260814)
    trajectory.add_argument("--output-dir", type=Path, required=True)
    trajectory.add_argument("--fusion", action="store_true", help="run canonical capabilities at each checkpoint")
    trajectory.add_argument("--fusion-batch", type=int, default=1280)
    trajectory.add_argument("--information", action="store_true", help="measure S0 -> S1 -> Z accessibility")
    trajectory.add_argument("--probe-train-samples", type=int, default=1024)
    trajectory.add_argument("--probe-test-samples", type=int, default=512)
    trajectory.add_argument("--probe-seed", type=int, default=20260815)
    trajectory.add_argument("--probe-batch", type=int, default=128)
    trajectory.add_argument("--probe-ridge", type=float, default=1.0)
    trajectory.set_defaults(run=_trajectory_command)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
