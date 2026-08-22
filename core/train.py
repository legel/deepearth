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
from deepearth.core.optimization import Optimizers, adamw_group


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
    cache = Path(cache_dir).expanduser().resolve()
    source = base_data.build(
        "california",
        cache_dir=str(cache),
        n_neighbors=16,
        device=device,
        holdout="spatial",
        subset=subset,
        time_axis=True,
        time_km=50,
        clay_v2=False,
        prepared=str(prepared_cache(cache)),
    )
    expected = {"n_neighbors": 16, "holdout": "spatial", "time_axis": True}
    mismatched = {
        key: (getattr(source, key), value)
        for key, value in expected.items() if getattr(source, key) != value
    }
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
    always = {
        "alphaearth": int(source.extra["alphaearth"][2])
    } if "alphaearth" in source.extra else {}
    return source, variables, always


READER_PARAMETERS = (
    "latents", "read.", "read_norm.", "blocks.",
    "fiber_query", "fiber_read", "fiber_fuse", "fiber_fusion_gate",
    "sparse_fusion_gate", "decode_query", "decoders.", "community_metric.",
    "species_graph.",
    "poll_head.", "pollinator_reader.", "identity_detail_reader.",
    "lfmc_head.", "myco_head.", "species_myco_head.",
    "flower_head.", "mesh_reader.",
    "species_niche_key", "species_niche_adapter.",
)
EXPANSION_PARAMETERS = (
    "mesh_reader.deep_mesh_reader.",
    "mesh_reader.deep_mesh_reader_gate.",
    "mesh_reader.deep_mesh_reader_output_norm.",
)
SPECIES_LENS_PARAMETERS = (
    "species_lens_reader.", "species_lens_reader_norm."
)
LFMC_LENS_PARAMETERS = (
    "lfmc_lens_reader.", "lfmc_lens_reader_norm.", "lfmc_lens_head."
)
IDENTITY_DETAIL_PARAMETERS = ("identity_detail_",)
RELATION_PARAMETERS = ("species_myco_head.",)
CALIBRATION_PARAMETERS = ("pollinator_log_temperature",)


@dataclass(frozen=True)
class ParameterSets:
    relation_ids: set[int]
    lens: list
    lens_ids: set[int]
    calibration: list
    calibration_ids: set[int]


def matching_parameters(model: DeepEarth, prefixes) -> list:
    return [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ]


def build_model(
    source, variable_specs, always_dims, device: str,
    design: Experiment = EXPERIMENT,
) -> DeepEarth:
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


def initialize_model(source, variable_specs, always_dims, device, design):
    model = build_model(source, variable_specs, always_dims, device, design)
    if design.init_checkpoint:
        checkpoint = Path(design.init_checkpoint).expanduser()
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"initialized from {checkpoint}", flush=True)
    for name, parameter in model.named_parameters():
        expansion = name.startswith(EXPANSION_PARAMETERS)
        parameter.requires_grad_(
            expansion if design.reader_only else not expansion
        )
    return model


def initialize_optimizers(model, design, device):
    relation = [
        parameter for parameter in matching_parameters(
            model, "species_myco_head."
        ) if parameter.requires_grad
    ]
    relation_ids = {
        id(parameter) for parameter in matching_parameters(model, RELATION_PARAMETERS)
    }
    lens = matching_parameters(
        model, SPECIES_LENS_PARAMETERS + LFMC_LENS_PARAMETERS
    )
    lens_ids = {id(parameter) for parameter in lens}
    calibration = matching_parameters(model, CALIBRATION_PARAMETERS)
    calibration_ids = {id(parameter) for parameter in calibration}
    excluded = relation_ids | lens_ids | calibration_ids
    base = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    optimizers = Optimizers(
        base=adamw_group(
            base, lr=design.learning_rate,
            weight_decay=design.weight_decay, steps=design.steps, device=device,
        ),
        relation=adamw_group(
            relation, lr=design.learning_rate,
            weight_decay=design.weight_decay, steps=design.steps, device=device,
        ),
    )
    parameters = ParameterSets(
        relation_ids, lens, lens_ids, calibration, calibration_ids
    )
    return optimizers, parameters


def enter_reader_phase(model, optimizers, parameters, design, device, budget):
    model.reader_phase = True
    model.rank_aligned_expansion = design.reader_only
    for name, parameter in model.named_parameters():
        if design.reader_only:
            trainable = name.startswith(EXPANSION_PARAMETERS)
        else:
            trainable = (
                name.startswith(READER_PARAMETERS)
                or name.startswith(SPECIES_LENS_PARAMETERS)
                or name.startswith(LFMC_LENS_PARAMETERS)
                or name.startswith(CALIBRATION_PARAMETERS)
            ) and not name.startswith(EXPANSION_PARAMETERS)
        parameter.requires_grad_(trainable)

    graph = matching_parameters(model, "species_graph.")
    graph = [parameter for parameter in graph if parameter.requires_grad]
    graph_ids = {id(parameter) for parameter in graph}
    detail = matching_parameters(model, IDENTITY_DETAIL_PARAMETERS)
    detail_ids = {id(parameter) for parameter in detail}
    excluded = (
        detail_ids | parameters.relation_ids | parameters.lens_ids
        | parameters.calibration_ids
    )
    readers = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in graph_ids
        and id(parameter) not in excluded
    ]
    base = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    optimizers.add("base", adamw_group(
        base,
        lr=design.learning_rate * 0.2,
        weight_decay=design.weight_decay,
        steps=budget,
        device=device,
        parameter_groups=(
            {"params": readers, "lr": design.learning_rate * 0.2},
            {"params": graph,
             "lr": design.learning_rate * design.graph_learning_rate_scale},
        ),
    ))
    optimizers.add("detail", adamw_group(
        detail, lr=design.learning_rate * 0.2,
        weight_decay=design.weight_decay, steps=budget, device=device,
    ))
    optimizers.add("lens", adamw_group(
        parameters.lens, lr=design.learning_rate * 0.4,
        weight_decay=design.weight_decay, steps=budget, device=device,
    ))
    optimizers.add("calibration", adamw_group(
        parameters.calibration, lr=design.learning_rate * 10.0,
        weight_decay=0.0, steps=budget, device=device,
    ))
    print(
        f"reader phase {budget} steps  "
        f"parameters {sum(parameter.numel() for parameter in readers):,}  "
        f"graph parameters {sum(parameter.numel() for parameter in graph):,}  "
        f"detail parameters {sum(parameter.numel() for parameter in detail):,}  "
        f"lens parameters {sum(parameter.numel() for parameter in parameters.lens):,}  "
        f"graph lr scale {design.graph_learning_rate_scale:g}",
        flush=True,
    )


def lfmc_correlation_loss(
    model, source, train_index, design, device, step
) -> torch.Tensor:
    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        auxiliary_seed = design.seed + 100_000 + step
        torch.manual_seed(auxiliary_seed)
        if devices:
            torch.cuda.manual_seed_all(auxiliary_seed)
        index = train_index[torch.randint(
            len(train_index), (design.batch,), device=device
        )]
        values, observed, coords, neighbors, manifolds, neighbor_values = \
            source.batch(index)
        context = model.context(
            coords, neighbors, manifolds, neighbor_values
        )
        present = {
            name: observed[name]
            if name in model.environment_names
            else torch.zeros_like(observed[name])
            for name in model.names
        }
        latent = model.encode(
            values, present, context, detach_species=True
        )
        pooled = model._pool(latent, "lfmc")
        prediction = model.lfmc_head(
            pooled.detach()
        ).squeeze(-1).detach() + model._lfmc_lens_residual(
            pooled.detach()
        )
        target = torch.log(values["_lfmc"].clamp_min(1.0))
        valid = values["_lfmc_valid"].bool()
        prediction = prediction[valid]
        target = target[valid]
        prediction = prediction - prediction.mean()
        target = target - target.mean()
        correlation = (prediction * target).sum() / (
            prediction.square().sum().sqrt()
            * target.square().sum().sqrt()
        ).clamp_min(1e-8)
    return 1.0 - correlation


def backward(model, optimizers, loss, structured_loss):
    optimizers.zero_grad()
    if structured_loss is None:
        loss.backward()
        return None
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    loss.backward(retain_graph=True)
    base_grad = {
        id(parameter): parameter.grad.detach().clone()
        for parameter in trainable if parameter.grad is not None
    }
    optimizers.zero_grad()
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
        base_grad[id(parameter)].square().sum() for parameter in shared
    ).clamp_min(1e-12)
    structured_norm = sum(
        parameter.grad.square().sum() for parameter in shared
    ).clamp_min(1e-12)
    cosine = float(dot / (base_norm.sqrt() * structured_norm.sqrt()))
    projection = dot / base_norm if dot < 0 else dot.new_zeros(())
    for parameter in trainable:
        base = base_grad.get(id(parameter))
        auxiliary = parameter.grad
        if base is None:
            continue
        parameter.grad = base if auxiliary is None \
            else base + auxiliary - projection * base
    return cosine


def train(
    cache: str,
    device: str,
    design: Experiment = EXPERIMENT,
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
    model = initialize_model(
        source, variable_specs, always_dims, device, design
    )
    optimizers, parameters = initialize_optimizers(model, design, device)
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
            enter_reader_phase(
                model, optimizers, parameters, design, device, reader_budget
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
        if model.reader_phase and lfmc_train_index is not None \
                and len(lfmc_train_index) > 2:
            loss = loss + lfmc_correlation_loss(
                model, source, lfmc_train_index, design, device, step
            )
        total_loss = loss if structured_loss is None else loss + structured_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        gradient_cosine = backward(
            model, optimizers, loss, structured_loss
        )
        optimizers.clip_grad_norm(5.0)
        optimizers.step()
        for module in model.modules():
            if hasattr(module, "clamp_per_level_scale"):
                module.clamp_per_level_scale()
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
    parser.add_argument("--reader-steps", type=int, default=100)
    parser.add_argument("--checkpoint")
    parser.add_argument("--reader-only", action="store_true")
    args = parser.parse_args()

    base = Experiment()
    design = replace(
        base, seed=args.seed, steps=args.steps,
        init_checkpoint=args.checkpoint or "",
        reader_only=args.reader_only,
        reader_steps=args.steps if args.reader_only else args.reader_steps,
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
