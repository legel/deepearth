"""Install feedback hooks, run training, and emit a reusable run receipt.

This is how the two encoder loops launch an experiment -- through the harness, so the fast-feedback
signal (biological ``[profile] refined_seed_norm``; spacetime ``*_spacetime_gain`` deltas) lands in the
run log with NO edit to train.py / evaluate.py / fusion.py.

Usage (identical to train.py; the canonical evaluator always measures Earth4D gain):
  python -m deepearth.autoresearch.main.harness.run_experiment autoresearch/main/editable_files/champion.yaml --tag bio_maskw --cache_dir ...
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from deepearth.autoresearch.main.harness import hooks


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _arg(argv: list[str], name: str, default=None):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _receipt(argv: list[str]) -> dict:
    from deepearth.autoresearch.main.harness import evaluate
    from deepearth.autoresearch.scoring import definitions

    here = Path(__file__).resolve()
    root = next(parent for parent in here.parents if parent.name == "deepearth")
    train_path = root / "autoresearch/main/editable_files/train.py"
    config_arg = argv[0] if argv and not argv[0].startswith("-") else None
    config_path = Path(config_arg).resolve() if config_arg else train_path.with_name("deepcal.yaml")
    config = yaml.safe_load(config_path.read_text())
    if (value := _arg(argv, "--steps")) is not None:
        config["training"]["steps"] = int(value)
    if (value := _arg(argv, "--time_budget")) is not None:
        config["training"]["time_budget_s"] = float(value)
    if (value := _arg(argv, "--seed")) is not None:
        config["training"]["seed"] = int(value)
    if (value := _arg(argv, "--cache_dir")) is not None:
        config["data"]["cache_dir"] = value

    data_keys = ("adapter", "cache_dir", "n_neighbors", "holdout", "subset", "time_axis", "time_km")
    data_identity = {key: config["data"].get(key) for key in data_keys}
    tag = hashlib.md5(json.dumps(data_identity, sort_keys=True, default=str).encode()).hexdigest()[:10]
    prepared = root / "autoresearch/main/data/deepcal" / f"prepared_{tag}.pt"
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    parent_tree = _git(root, "rev-parse", "HEAD^{tree}")
    training = config.get("training", {})

    runtime = {}
    try:
        import torch
        runtime = {"torch": torch.__version__, "cuda": torch.version.cuda}
        device = _arg(argv, "--device", "cuda")
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index or 0
            runtime["gpu"] = torch.cuda.get_device_name(index)
    except Exception:
        pass

    return {
        "schema": "fusion-run-v1",
        "source": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "parent_tree": parent_tree,
            "dirty": bool(status),
        },
        "config": {
            "sha256": _sha(config_path),
            "effective_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode()
            ).hexdigest(),
        },
        "judge": {
            "protocol": evaluate.BENCHMARK_PROTOCOL,
            "evaluate_sha256": _sha(Path(evaluate.__file__).resolve()),
            "definitions_sha256": _sha(Path(definitions.__file__).resolve()),
        },
        "data": {
            "identity": hashlib.sha256(
                json.dumps(data_identity, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "prepared_sha256": _sha(prepared),
        },
        "training": {
            "seed": int(training.get("seed", 0)),
            "steps": training.get("steps"),
            "time_budget_s": training.get("time_budget_s"),
            "batch": training.get("batch", 512),
            "precision": training.get("precision", "fp32"),
        },
        "runner": {"train_sha256": _sha(train_path), "hooks_sha256": _sha(Path(hooks.__file__).resolve())},
        "runtime": runtime,
    }


def main():
    # Consume the old flag for command compatibility, but it no longer changes the benchmark suite.
    argv = [x for x in sys.argv[1:] if x != "--st-gain"]
    hooks.instrument()
    sys.argv = [sys.argv[0]] + argv          # hand the remaining args to train's argparse unchanged
    from deepearth.autoresearch.main.editable_files import train
    train.main()
    print("RUN RECEIPT: " + json.dumps(_receipt(argv), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
