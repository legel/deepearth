"""Fixed measurement runner. The research loop may edit only model.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


EVAL_BATCH = 1280
VAL_BATCHES = 48
EVAL_SEED = 20260806
HIDE_PROBABILITY = 0.5


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
    model, source = experiment.train(args.cache, args.device)

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

    val_bpb, decomposition = likelihood(model, source, args)
    print(f"val_bpb: {val_bpb:.6f}", flush=True)
    for name in sorted(decomposition, key=decomposition.get, reverse=True):
        print(f"  val_bpb.{name:<22} {decomposition[name]:.6f}", flush=True)


if __name__ == "__main__":
    main()
