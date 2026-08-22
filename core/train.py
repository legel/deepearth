"""Train the production fibered world model and score it with the public evaluator."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from deepearth.core import data as base_data
from deepearth.core.fusion import DeepEarth, Variable


@dataclass(frozen=True)
class Experiment:
    seed: int = 1337
    steps: int = 2291
    batch: int = 256
    width: int = 192
    levels: int = 12
    hash_log2: int = 14
    latents: int = 16
    layers: int = 2
    hide_probability: float = 0.5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    reader_steps: int = 100
    graph_learning_rate_scale: float = 0.02
    init_checkpoint: str = ""
    reader_only: bool = False


EXPERIMENT = Experiment()


CONTINUOUS_SIGNALS = (
    "vision_dino", "vision_bio", "phylo", "climate", "soil", "naip_rgb",
    "naip_ir", "clay", "topo", "chm", "hydro", "phenology",
)


def prepared_cache(cache: Path) -> Path:
    preferred = cache / "prepared_mesh.pt"
    if preferred.exists():
        return preferred
    legacy = sorted(cache.glob("prepared_*.pt"))
    if len(legacy) <= 1:
        return legacy[0] if legacy else preferred
    names = ", ".join(path.name for path in legacy)
    raise RuntimeError(f"multiple prepared caches found: {names}")


def load_data(cache_dir: str, device: str, *, subset: dict | None = None):
    settings = {
        "adapter": "california",
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "n_neighbors": 16,
        "holdout": "spatial",
        "subset": subset,
        "time_axis": True,
        "time_km": 50.0,
        "clay_v2": False,
    }
    cache = Path(settings["cache_dir"])
    prepared = prepared_cache(cache)
    source = base_data.build(
        settings["adapter"], cache_dir=settings["cache_dir"],
        n_neighbors=settings["n_neighbors"], device=device,
        holdout=settings["holdout"], subset=settings["subset"],
        time_axis=settings["time_axis"], time_km=settings["time_km"],
        clay_v2=settings["clay_v2"], prepared=str(prepared),
    )
    expected = {"n_neighbors": 16, "holdout": "spatial", "time_axis": True}
    mismatched = {key: (getattr(source, key), value) for key, value in expected.items()
                  if getattr(source, key) != value}
    if mismatched:
        raise ValueError(f"prepared cache violates the production data contract: {mismatched}")
    dims = source.variable_dims()
    variables = [
        {"name": name, "kind": "continuous", "dim": int(dims[name])}
        for name in CONTINUOUS_SIGNALS if int(dims.get(name, 0)) > 0
    ]
    variables.insert(2, {
        "name": "identity", "kind": "categorical",
        "num_classes": int(dims["identity_classes"]),
    })
    variables.extend(
        {"name": name, "kind": "categorical", "num_classes": int(classes)}
        for name, classes in dims["trait_classes"].items()
    )
    always = {}
    if "alphaearth" in source.extra:
        always["alphaearth"] = int(source.extra["alphaearth"][2])
    return source, variables, always


READER_PARAMETERS = (
    "latents", "read.", "read_norm.", "blocks.",
    "fiber_query", "fiber_read", "fiber_fuse", "fiber_fusion_gate",
    "sparse_fusion_gate", "decode_query", "decoders.", "community_metric.",
    "species_graph.",
    "poll_head.", "pollinator_reader_query", "pollinator_reader.",
    "pollinator_reader_norm.", "pollinator_reader_output_norm.",
    "pollinator_reader_gate", "pollinator_reader_cell_key",
    "pollinator_reader_level_key", "pollinator_reader_lens_key",
    "identity_detail_query", "identity_detail_reader.",
    "identity_detail_norm.", "identity_detail_output_norm.",
    "identity_detail_gate", "identity_detail_cell_key",
    "identity_detail_level_key", "identity_detail_lens_key",
    "lfmc_head.", "myco_head.", "species_myco_head.", "myco_relation_gate",
    "flower_head.",
    "mesh_read_query.", "mesh_read_gate.", "mesh_scale_read_gate.",
    "mesh_scale_attention_gate.",
    "task_mesh_reader.", "task_mesh_reader_gate.", "task_mesh_reader_norm.",
    "task_mesh_reader_output_norm.", "scale_mesh_reader.",
    "scale_mesh_reader_mix.", "scale_mesh_reader_router.",
    "deep_mesh_reader.", "deep_mesh_reader_gate.",
    "deep_mesh_reader_output_norm.",
    "mesh_prior_read_gate.", "mesh_prior_information_gate.",
    "mesh_task_norm.", "mesh_scale_task_norm.", "mesh_prior_task_norm.",
    "mesh_condition_gate.", "mesh_condition_norm.",
    "mesh_cell_key", "mesh_level_key", "mesh_lens_key",
    "species_niche_key", "species_niche_adapter.",
)
EXPANSION_PARAMETERS = (
    "deep_mesh_reader.", "deep_mesh_reader_gate.",
    "deep_mesh_reader_output_norm.",
)
SPECIES_LENS_PARAMETERS = (
    "species_lens_reader.", "species_lens_reader_norm."
)
LFMC_LENS_PARAMETERS = (
    "lfmc_lens_reader.", "lfmc_lens_reader_norm.", "lfmc_lens_head."
)
IDENTITY_DETAIL_PARAMETERS = ("identity_detail_",)
RELATION_PARAMETERS = ("species_myco_head.", "myco_relation_gate")
CALIBRATION_PARAMETERS = ("pollinator_log_temperature",)



def build_model(source, variable_specs, always_dims, device: str, design: Experiment = EXPERIMENT) -> DeepEarth:
    variables = [Variable(**spec) for spec in variable_specs]
    return DeepEarth(
        variables,
        always_dims,
        source,
        d_model=design.width,
        levels=design.levels,
        log2_size=design.hash_log2,
        n_latents=design.latents,
        n_layers=design.layers,
    ).to(device)


def train(
    cache: str,
    device: str,
    design: Experiment = EXPERIMENT,
    *,
    checkpoint_steps: frozenset[int] = frozenset(),
    checkpoint_dir: Path | None = None,
):
    if not design.reader_only and not 0 <= design.reader_steps < design.steps:
        raise ValueError("reader_steps must fall between 0 and total steps")
    if design.reader_only and not design.init_checkpoint:
        raise ValueError("MESH_READER_ONLY requires MESH_INIT_CHECKPOINT")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(design.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(design.seed)
    source, variable_specs, always_dims = load_data(cache, device)
    if design.width != 128:
        candidate_rng = torch.random.get_rng_state()
        candidate_cuda_rng = torch.cuda.get_rng_state_all() \
            if device.startswith("cuda") else None
        control = build_model(
            source, variable_specs, always_dims, device,
            replace(design, width=128),
        )
        control_rng = torch.random.get_rng_state()
        control_cuda_rng = torch.cuda.get_rng_state_all() \
            if device.startswith("cuda") else None
        del control
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.set_rng_state_all(candidate_cuda_rng)
        torch.random.set_rng_state(candidate_rng)
        model = build_model(source, variable_specs, always_dims, device, design)
        torch.random.set_rng_state(control_rng)
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(control_cuda_rng)
    else:
        model = build_model(source, variable_specs, always_dims, device, design)
    if design.init_checkpoint:
        checkpoint = Path(design.init_checkpoint).expanduser()
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        print(
            f"initialized from {checkpoint}  "
            f"missing={len(incompatible.missing_keys)}  "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    if checkpoint_steps:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_steps are requested")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if 0 in checkpoint_steps:
            torch.save(model.state_dict(), checkpoint_dir / "step_000000.pt")
    relation_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("species_myco_head.")
    ]
    relation_ids = {
        id(parameter) for name, parameter in model.named_parameters()
        if name.startswith(RELATION_PARAMETERS)
    }
    lens_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(SPECIES_LENS_PARAMETERS + LFMC_LENS_PARAMETERS)
    ]
    lens_ids = {id(parameter) for parameter in lens_parameters}
    calibration_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(CALIBRATION_PARAMETERS)
    ]
    calibration_ids = {id(parameter) for parameter in calibration_parameters}
    base_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in relation_ids
        and id(parameter) not in lens_ids
        and id(parameter) not in calibration_ids
    ]
    optimizer = torch.optim.AdamW(
        base_parameters,
        lr=design.learning_rate,
        weight_decay=design.weight_decay,
        fused=device.startswith("cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, design.steps)
    relation_optimizer = None
    relation_scheduler = None
    if relation_parameters:
        relation_optimizer = torch.optim.AdamW(
            relation_parameters,
            lr=design.learning_rate,
            weight_decay=design.weight_decay,
            fused=device.startswith("cuda"),
        )
        relation_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            relation_optimizer, design.steps
        )
    detail_optimizer = None
    detail_scheduler = None
    lens_optimizer = None
    lens_scheduler = None
    calibration_optimizer = None
    calibration_scheduler = None
    reader_budget = design.steps if design.reader_only else design.reader_steps
    reader_start = 0 if design.reader_only else design.steps - design.reader_steps
    lfmc_train_index = None
    if model.lfmc_head is not None and hasattr(source, "lfmc_valid"):
        lfmc_mask = source.lfmc_valid[source.cls[source.train_index]]
        lfmc_train_index = source.train_index[lfmc_mask]
        print(f"LFMC reader examples {len(lfmc_train_index):,}", flush=True)
    model.train()
    started = time.time()
    for step in range(design.steps):
        if design.reader_steps and step == reader_start:
            model.reader_phase = True
            model.rank_aligned_expansion = design.reader_only
            for name, parameter in model.named_parameters():
                is_reader = name.startswith(READER_PARAMETERS) \
                            or name.startswith(SPECIES_LENS_PARAMETERS) \
                            or name.startswith(LFMC_LENS_PARAMETERS) \
                            or name.startswith(CALIBRATION_PARAMETERS)
                if design.reader_only:
                    is_reader = name.startswith(EXPANSION_PARAMETERS)
                if design.reader_only and name.startswith("species_graph."):
                    is_reader = False
                parameter.requires_grad_(is_reader)
            graph_parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith("species_graph.") and parameter.requires_grad
            ]
            graph_ids = {id(parameter) for parameter in graph_parameters}
            detail_parameters = [
                parameter for name, parameter in model.named_parameters()
                if name.startswith(IDENTITY_DETAIL_PARAMETERS)
            ]
            detail_ids = {id(parameter) for parameter in detail_parameters}
            reader_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) not in graph_ids
                and id(parameter) not in detail_ids
                and id(parameter) not in relation_ids
                and id(parameter) not in lens_ids
                and id(parameter) not in calibration_ids
            ]
            base_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) not in detail_ids
                and id(parameter) not in relation_ids
                and id(parameter) not in lens_ids
                and id(parameter) not in calibration_ids
            ]
            optimizer = torch.optim.AdamW(
                (
                    {"params": reader_parameters, "lr": design.learning_rate * 0.2},
                    {"params": graph_parameters,
                     "lr": design.learning_rate * design.graph_learning_rate_scale},
                ),
                lr=design.learning_rate * 0.2,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, reader_budget
            )
            detail_optimizer = torch.optim.AdamW(
                detail_parameters,
                lr=design.learning_rate * 0.2,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            detail_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                detail_optimizer, reader_budget
            )
            lens_optimizer = torch.optim.AdamW(
                lens_parameters,
                lr=design.learning_rate * 0.4,
                weight_decay=design.weight_decay,
                fused=device.startswith("cuda"),
            )
            lens_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                lens_optimizer, reader_budget
            )
            calibration_optimizer = torch.optim.AdamW(
                calibration_parameters,
                lr=design.learning_rate * 10.0,
                weight_decay=0.0,
                fused=device.startswith("cuda"),
            )
            calibration_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                calibration_optimizer, reader_budget
            )
            print(
                f"reader phase {reader_budget} steps  "
                f"parameters {sum(parameter.numel() for parameter in reader_parameters):,}  "
                f"graph parameters {sum(parameter.numel() for parameter in graph_parameters):,}  "
                f"detail parameters {sum(parameter.numel() for parameter in detail_parameters):,}  "
                f"lens parameters {sum(parameter.numel() for parameter in lens_parameters):,}  "
                f"graph lr scale {design.graph_learning_rate_scale:g}",
                flush=True,
            )
        index = source.train_index[torch.randint(len(source.train_index), (design.batch,), device=device)]
        values, observed, coords, neighbors, manifolds, neighbor_values = source.batch(index)
        context = model.context(coords, neighbors, manifolds, neighbor_values)
        objective = model.reconstruction_loss(
            values, observed, context, design.hide_probability
        )
        if isinstance(objective, tuple):
            loss, structured_loss = objective
        else:
            loss, structured_loss = objective, None
        if getattr(model, "reader_phase", False) \
                and lfmc_train_index is not None \
                and len(lfmc_train_index) > 2:
            devices = [torch.cuda.current_device()] \
                      if device.startswith("cuda") else []
            with torch.random.fork_rng(devices=devices):
                auxiliary_seed = design.seed + 100_000 + step
                torch.manual_seed(auxiliary_seed)
                if devices:
                    torch.cuda.manual_seed_all(auxiliary_seed)
                lfmc_index = lfmc_train_index[torch.randint(
                    len(lfmc_train_index), (design.batch,), device=device
                )]
                lfmc_values, lfmc_observed, lfmc_coords, lfmc_neighbors, \
                    lfmc_manifolds, lfmc_neighbor_values = source.batch(
                        lfmc_index
                    )
                lfmc_context = model.context(
                    lfmc_coords, lfmc_neighbors, lfmc_manifolds,
                    lfmc_neighbor_values
                )
                lfmc_present = {
                    name: lfmc_observed[name]
                    if name in model.environment_names
                    else torch.zeros_like(lfmc_observed[name])
                    for name in model.names
                }
                lfmc_latent = model.encode(
                    lfmc_values, lfmc_present, lfmc_context,
                    detach_species=True
                )
                lfmc_pool = model._pool(lfmc_latent, "lfmc")
                prediction = model.lfmc_head(
                    lfmc_pool.detach()
                ).squeeze(-1).detach() + model._lfmc_lens_residual(
                    lfmc_pool.detach()
                )
                target = torch.log(lfmc_values["_lfmc"].clamp_min(1.0))
                valid = lfmc_values["_lfmc_valid"].bool()
                prediction = prediction[valid]
                target = target[valid]
                prediction = prediction - prediction.mean()
                target = target - target.mean()
                correlation = (prediction * target).sum() / (
                    prediction.square().sum().sqrt()
                    * target.square().sum().sqrt()
                ).clamp_min(1e-8)
            loss = loss + 1.0 - correlation
        total_loss = loss if structured_loss is None else loss + structured_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        if relation_optimizer is not None:
            relation_optimizer.zero_grad(set_to_none=True)
        if detail_optimizer is not None:
            detail_optimizer.zero_grad(set_to_none=True)
        if lens_optimizer is not None:
            lens_optimizer.zero_grad(set_to_none=True)
        if calibration_optimizer is not None:
            calibration_optimizer.zero_grad(set_to_none=True)
        gradient_cosine = None
        if structured_loss is None:
            loss.backward()
        else:
            trainable = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ]
            loss.backward(retain_graph=True)
            base_grad = {
                id(parameter): parameter.grad.detach().clone()
                for parameter in trainable if parameter.grad is not None
            }
            optimizer.zero_grad(set_to_none=True)
            if relation_optimizer is not None:
                relation_optimizer.zero_grad(set_to_none=True)
            if detail_optimizer is not None:
                detail_optimizer.zero_grad(set_to_none=True)
            if lens_optimizer is not None:
                lens_optimizer.zero_grad(set_to_none=True)
            if calibration_optimizer is not None:
                calibration_optimizer.zero_grad(set_to_none=True)
            structured_loss.backward()
            shared = [
                parameter for parameter in trainable
                if parameter.grad is not None and id(parameter) in base_grad
            ]
            dot = sum(
                (parameter.grad * base_grad[id(parameter)]).sum()
                for parameter in shared
            )
            base_norm = sum(
                base_grad[id(parameter)].square().sum()
                for parameter in shared
            ).clamp_min(1e-12)
            structured_norm = sum(
                parameter.grad.square().sum() for parameter in shared
            ).clamp_min(1e-12)
            gradient_cosine = float(
                dot / (base_norm.sqrt() * structured_norm.sqrt())
            )
            projection = dot / base_norm if dot < 0 else dot.new_zeros(())
            for parameter in trainable:
                base = base_grad.get(id(parameter))
                auxiliary = parameter.grad
                if base is None:
                    continue
                if auxiliary is None:
                    parameter.grad = base
                else:
                    parameter.grad = base + auxiliary - projection * base
        torch.nn.utils.clip_grad_norm_(base_parameters, 5.0)
        if relation_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(relation_parameters, 5.0)
        if detail_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(detail_parameters, 5.0)
        if lens_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(lens_parameters, 5.0)
        if calibration_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(calibration_parameters, 5.0)
        optimizer.step()
        scheduler.step()
        if relation_optimizer is not None:
            relation_optimizer.step()
            relation_scheduler.step()
        if detail_optimizer is not None:
            detail_optimizer.step()
            detail_scheduler.step()
        if lens_optimizer is not None:
            lens_optimizer.step()
            lens_scheduler.step()
        if calibration_optimizer is not None:
            calibration_optimizer.step()
            calibration_scheduler.step()
        for module in model.modules():
            if hasattr(module, "clamp_per_level_scale"):
                module.clamp_per_level_scale()
        completed = step + 1
        if completed in checkpoint_steps:
            torch.save(model.state_dict(), checkpoint_dir / f"step_{completed:06d}.pt")
        if step % 100 == 0 or step + 1 == design.steps:
            conflict = "" if gradient_cosine is None \
                       else f"  gradient_cosine {gradient_cosine:+.3f}"
            print(
                f"step {step:>5}  loss {float(total_loss):.4f}{conflict}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )

    checkpoint = Path(__file__).with_name("checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print(f"checkpoint: {checkpoint}", flush=True)
    return model, source



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=2291)
    parser.add_argument("--checkpoint")
    parser.add_argument("--reader-only", action="store_true")
    args = parser.parse_args()

    base = Experiment()
    design = replace(
        base, seed=args.seed, steps=args.steps,
        init_checkpoint=args.checkpoint or "",
        reader_only=args.reader_only,
        reader_steps=args.steps if args.reader_only else base.reader_steps,
    )
    model, source = train(args.cache, args.device, design)
    from deepearth.autoresearch import evaluate

    scores = evaluate.evaluate_benchmarks(model, source, args.device, batch=1280)
    print(evaluate.format_benchmarks(scores), flush=True)
    print("BENCHMARK RECEIPT: " + json.dumps({
        "scores": scores,
        "harmonic": evaluate.net_score(scores),
        "arithmetic": evaluate.arithmetic_net(scores),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
