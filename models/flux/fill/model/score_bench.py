"""Score a FluxFill checkpoint with the benchmark runner (bench/run.py `run`) on a fixed window list, so that its rows
are directly comparable with the baselines scored on the same list (same masks, same scoring code).

Usage: python score_bench.py --ckpt CK --windows windows_val_list_list.json --mode marginal --name fluxfill-v0 \
           --out parts/fluxfill-v0.parquet [--families M2] [--samples 64] [--steps 16] [--device cuda]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))

from data import load_split  # noqa: E402
from method import FluxFill  # noqa: E402
import run as bench  # noqa: E402


def main():
    import torch
    torch.set_num_threads(2)            # machine rule: at most 2 CPU threads per job
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", required=True, help="JSON list of {site, start, table}")
    ap.add_argument("--mode", default="marginal")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="+", default=["M2"])
    ap.add_argument("--regime", default="R1")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagery-sites", default="", help="comma list: Clay imagery only at these sites")
    a = ap.parse_args()
    specs = json.loads(Path(a.windows).read_text())
    split, name = load_split(sorted({s["site"] for s in specs}))
    bad = [s["site"] for s in specs if split.get(s["site"]) != "val"]
    if bad:
        raise SystemExit(f"refusing to score non-validation sites: {sorted(set(bad))}")
    m = FluxFill(a.ckpt, mode=a.mode, n_samples=a.samples, steps=a.steps, chunk=a.chunk, device=a.device, name=a.name,
                 imagery_sites=a.imagery_sites.split(",") if a.imagery_sites else None)
    t0, parts = time.time(), []
    from gpu_pause import wait_if_paused
    for i, spec in enumerate(specs):
        wait_if_paused(f"{a.name} window {i + 1}")
        tables = {spec["site"]: pd.read_parquet(spec["table"])}
        parts.append(bench.run([m], [spec], tables, a.regime, a.families))
        df = pd.concat(parts, ignore_index=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.out, index=False)
        print(f"{i + 1}/{len(specs)} {spec['site']} {spec['start']} {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {a.out}: {len(df)} rows, checkpoint {a.ckpt}, {name} split")


if __name__ == "__main__":
    main()
