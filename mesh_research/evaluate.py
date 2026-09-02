"""Fixed measurement runner. The research loop may edit only model.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import MethodType

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


EVAL_BATCH = int(os.environ.get("MESH_EVAL_BATCH", "1280"))
VAL_BATCHES = int(os.environ.get("MESH_VAL_BATCHES", "48"))
BENCHMARK_ROWS = int(os.environ.get("MESH_BENCHMARK_ROWS", "0"))
BENCHMARK_TRAIN_ROWS = int(os.environ.get("MESH_BENCHMARK_TRAIN_ROWS", "0"))
SKIP_LIKELIHOOD = os.environ.get("MESH_SKIP_LIKELIHOOD", "0") == "1"
SKIP_CENSUS = os.environ.get("MESH_SKIP_CENSUS", "0") == "1"
EVAL_SEED = 20260806
HIDE_PROBABILITY = 0.5
CENSUS_SAMPLES = 512
CENSUS_SEED = 20260814


def load_model(experiment, cache: str, device: str):
    """Build the experiment model and load a checkpoint without taking a train step."""
    source, variable_specs, always_dims = experiment.load_data(cache, device)
    experiment.attach_naip_patches(source, cache, device)
    design = experiment.EXPERIMENT
    if design.width != 128:
        candidate_rng = torch.random.get_rng_state()
        candidate_cuda_rng = torch.cuda.get_rng_state_all() \
            if str(device).startswith("cuda") else None
        control = experiment.build_model(
            source, variable_specs, always_dims, device,
            experiment.replace(design, width=128),
        )
        control_rng = torch.random.get_rng_state()
        control_cuda_rng = torch.cuda.get_rng_state_all() \
            if str(device).startswith("cuda") else None
        del control
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.set_rng_state_all(candidate_cuda_rng)
        torch.random.set_rng_state(candidate_rng)
        model = experiment.build_model(
            source, variable_specs, always_dims, device, design
        )
        torch.random.set_rng_state(control_rng)
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(control_cuda_rng)
    else:
        model = experiment.build_model(
            source, variable_specs, always_dims, device, design
        )
    checkpoint = Path(
        design.init_checkpoint
        or os.environ.get("MESH_INIT_CHECKPOINT", "")
        or Path(experiment.__file__).with_name("checkpoint.pt")
    ).expanduser()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    print(
        f"loaded checkpoint {checkpoint}  "
        f"missing={len(incompatible.missing_keys)}  "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    return model, source


@torch.no_grad()
def likelihood(model, source, args):
    """Run the production autoresearch likelihood methods on this model's decoder."""
    from deepearth.autoresearch.main.editable_files.fusion.fusion import DeepEarth
    from deepearth.autoresearch.scoring import objective

    # Measurement belongs to the fixed runner. Bind the production implementation
    # without adding scoring behavior to editable model.py.
    model._pooled = model._pool
    model.diffusion_heads = {}
    for name in ("calibrate_nats", "_retrieval_floors", "retrieval_floors",
                 "_reconstruction_nats", "variable_losses"):
        setattr(model, name, MethodType(getattr(DeepEarth, name), model))

    model.eval()
    test = torch.as_tensor(source.test, device=args.device)
    generator = torch.Generator(device=args.device).manual_seed(EVAL_SEED)
    reference_index = test[torch.randint(
        len(test), (min(4096, len(test)),), device=args.device, generator=generator
    )]
    model.calibrate_nats(source.batch(reference_index)[0])

    totals = {}
    for step in range(VAL_BATCHES):
        generator.manual_seed(EVAL_SEED + step)
        index = test[torch.randint(len(test), (EVAL_BATCH,), device=args.device, generator=generator)]
        values, observed, coords, neighbors, manifolds, neighbor_values = source.batch(index)
        present = {
            name: (torch.rand(EVAL_BATCH, device=args.device, generator=generator) > HIDE_PROBABILITY)
                  & observed[name]
            for name in model.names
        }
        blank = torch.rand(EVAL_BATCH, device=args.device, generator=generator) < 0.15
        for name in present:
            present[name] &= ~blank
        latent = model.encode(values, present, model.context(coords, neighbors, manifolds, neighbor_values))

        for name, (nats, dimensions) in model.variable_losses(
                latent, values, observed, present).items():
            old_nats, old_dimensions = totals.get(name, (0.0, 0))
            totals[name] = old_nats + nats, old_dimensions + dimensions
    return objective.aggregate(totals), objective.decompose(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import model as experiment
    from deepearth.autoresearch.main.harness import evaluate as canonical
    os.environ.setdefault("MESH_SAVE_CHECKPOINT", "0")
    model, source = load_model(experiment, args.cache, args.device)
    if BENCHMARK_ROWS:
        source.test = source.test[:BENCHMARK_ROWS]
    if BENCHMARK_TRAIN_ROWS:
        source.train = source.train[:BENCHMARK_TRAIN_ROWS]

    raw = canonical.evaluate_benchmarks(model, source, args.device, batch=EVAL_BATCH)
    print(canonical.format_benchmarks(raw), flush=True)
    harmonic = canonical.net_score(raw)
    arithmetic = canonical.arithmetic_net(raw)
    print("BENCHMARK RECEIPT: " + json.dumps({
        "protocol": canonical.BENCHMARK_PROTOCOL,
        "scores": raw,
        "harmonic": harmonic,
        "arithmetic": arithmetic,
    }, sort_keys=True), flush=True)

    if not SKIP_LIKELIHOOD:
        torch.cuda.empty_cache()
        val_bpb, decomposition = likelihood(model, source, args)
        print(f"val_bpb: {val_bpb:.6f}", flush=True)
        for name in sorted(decomposition, key=decomposition.get, reverse=True):
            print(f"  val_bpb.{name:<22} {decomposition[name]:.6f}", flush=True)

    if not SKIP_CENSUS:
        torch.cuda.empty_cache()
        from census import measure as measure_mesh
        census, _ = measure_mesh(model, source, CENSUS_SAMPLES, CENSUS_SEED)
        field = census["persistent_field"]
        writes = census["writes"]
        diagnostics = {
            "protocol": census["protocol"],
            "samples": census["samples"],
            "sample_seed": census["sample_seed"],
            "spatial_locality_margin": field["spatial_near_vs_shuffled_margin"],
            "temporal_locality_margin": field["temporal_near_vs_shuffled_margin"],
            "write_effective_rank": writes["write_effective_rank_mean"],
            "state_effective_rank": field["state_effective_rank_mean"],
            "sampled_collision_free": census["addresses"]["sampled_collision_free_mean"],
        }
        print(
            "MESH DIAGNOSTICS: "
            f"space_margin={diagnostics['spatial_locality_margin']:.6f}  "
            f"time_margin={diagnostics['temporal_locality_margin']:.6f}  "
            f"write_rank={diagnostics['write_effective_rank']:.6f}  "
            f"state_rank={diagnostics['state_effective_rank']:.6f}  "
            f"collision_free={diagnostics['sampled_collision_free']:.6f}",
            flush=True,
        )
        print("MESH DIAGNOSTIC RECEIPT: " + json.dumps(diagnostics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
