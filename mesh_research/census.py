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

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(HERE))

PROTOCOL = "mesh-census-v1"
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
    trajectory.set_defaults(run=_trajectory_command)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
