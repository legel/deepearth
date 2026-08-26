#!/usr/bin/env python3
"""Inference-cost benchmark for GridTransformerSurrogate at site3's TRUE full resolution
(1363x1372 cells, 5m, the real production grid `run_site3_ian.py`/`flood_probability.py` use) —
the direct analog of `benchmark_gnn_forward.py --inference-only`, which found the mesh-GNN
surrogate averaged 304.8s/forward-pass on CPU (OOM on MPS) at site3's full MESH resolution
(8.67M edges), ~56x SLOWER than the real 72hr Ian event's own solver wall time (397s total).

Uses the SAME trained weights from training_grid_transformer_surrogate.py's small-crop
(272x274-cell, 25m) run — the architecture is fully convolutional + attention-over-actual-token-
count, so it runs unmodified at any input resolution (see grid_transformer_surrogate.py's
docstring). Real input data: 4 bands of `analysis/data/flood_depth_by_return_period_site3.tif`
(true full-resolution site3 peak-depth fields from the existing 9-return-period ensemble) stand
in as "context frames" — legitimate real full-scale depth fields for a pure compute-cost
measurement (this benchmark is not evaluating temporal accuracy at full scale, matching exactly
what the GNN benchmark itself did: apply trained weights to the full graph/field and TIME it).

Run: python3 simulation/benchmark_grid_transformer_forward.py --tag baseline
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import rasterio
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
from grid_transformer_surrogate import GridTransformerSurrogate, pad_to_multiple, crop_to, DOWNSAMPLE  # noqa: E402
from train_grid_transformer_surrogate import N_CONTEXT, get_device  # noqa: E402

DEPTH_TIF = os.path.join(PROJ_DIR, "analysis", "data", "flood_depth_by_return_period_site3.tif")

# Real full-solver wall-time figures already established in this project, reused
# here rather than re-derived, so the comparison table below is self-contained.
GRID_SOLVER_FULL_IAN_WALL_S = 397.0     # run_site3_ian.py, 72hr real Ian event, 1.87M cells
GRID_SOLVER_FULL_IAN_STEPS = 12960      # same run: 72hr @ dt_s=20s fixed step spacing
GRID_SOLVER_SIMULATED_S = 72 * 3600.0   # 72hr, what those 12960 steps actually cover
FRAME_INTERVAL_MIN = 20.0               # one surrogate forward pass predicts this much sim-time
                                        # ahead — must match train_grid_transformer_surrogate's
                                        # own build_grid_surrogate_dataset_site3.py setting
MESH_GNN_CPU_S_PER_PASS = 304.8         # benchmark_gnn_forward.py --inference-only, 8.67M edges
MESH_GNN_MPS_OOM = True                 # RuntimeError: MPS backend out of memory


def load_full_res_context():
    with rasterio.open(DEPTH_TIF) as src:
        bands = src.read()   # [9, 1363, 1372] — one band per return period, real peak depths
    # 4 bands as context frames — any 4 distinct real full-resolution depth fields; adequate
    # for a pure forward-pass timing measurement (see module docstring).
    ctx = bands[[0, 2, 4, 6]].astype(np.float32)   # [4, 1363, 1372]
    return ctx, bands.shape[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n-timed-passes", type=int, default=5)
    args = ap.parse_args()

    ckpt_path = os.path.join(CKPT_DIR, f"grid_transformer_site3_{args.tag}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ctx_np, native_shape = load_full_res_context()
    print(f"  site3 true full resolution: {native_shape}  ({native_shape[0]*native_shape[1]:,} cells)")

    results = {}
    for device_name in ("cpu", "mps"):
        if device_name == "mps" and not torch.backends.mps.is_available():
            continue
        device = torch.device(device_name)
        model = GridTransformerSurrogate(n_context=N_CONTEXT).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ctx = torch.from_numpy(ctx_np).unsqueeze(0).unsqueeze(2).float()   # [1,K,1,H,W]
        ctx_pad, hw = pad_to_multiple(ctx.reshape(-1, 1, *ctx.shape[-2:]), DOWNSAMPLE)
        ctx_pad = ctx_pad.reshape(1, N_CONTEXT, 1, *ctx_pad.shape[-2:]).to(device)
        print(f"\n  [{device_name}] padded to {tuple(ctx_pad.shape[-2:])}  "
              f"(latent grid {ctx_pad.shape[-2]//DOWNSAMPLE}x{ctx_pad.shape[-1]//DOWNSAMPLE} "
              f"= {(ctx_pad.shape[-2]//DOWNSAMPLE)*(ctx_pad.shape[-1]//DOWNSAMPLE):,} tokens/frame)")
        forcing = torch.tensor([50.0], dtype=torch.float32, device=device)   # a real mid-range rate

        try:
            # Warm-up pass (excluded from timing — matches benchmark_gnn_forward.py convention)
            with torch.no_grad():
                _ = model(ctx_pad, forcing)
            if device_name == "mps":
                torch.mps.synchronize()

            times = []
            with torch.no_grad():
                for _ in range(args.n_timed_passes):
                    t0 = time.time()
                    pred, _ = model(ctx_pad, forcing)
                    if device_name == "mps":
                        torch.mps.synchronize()
                    times.append(time.time() - t0)
            mean_s = float(np.mean(times))
            print(f"  [{device_name}] {args.n_timed_passes} timed passes: "
                  f"{[round(t,3) for t in times]}  mean={mean_s:.3f}s")
            results[device_name] = dict(status="ok", passes_s=[round(t, 4) for t in times],
                                        mean_s_per_pass=round(mean_s, 4))
        except RuntimeError as e:
            print(f"  [{device_name}] FAILED: {e}")
            results[device_name] = dict(status="failed", error=str(e)[:300])

    # Wall time the PHYSICS SOLVER needs to advance the same amount of simulated time one
    # surrogate forward pass predicts (one FRAME_INTERVAL_MIN chunk) — NOT a raw per-step
    # comparison, since one solver step only covers dt_s=20s of simulated time while one
    # surrogate pass covers FRAME_INTERVAL_MIN=20min (60x more simulated time per call).
    solver_wall_s_per_sim_s = GRID_SOLVER_FULL_IAN_WALL_S / GRID_SOLVER_SIMULATED_S
    solver_wall_s_per_frame_interval = solver_wall_s_per_sim_s * FRAME_INTERVAL_MIN * 60.0

    n_params = sum(p.numel() for p in model.parameters())
    summary = dict(
        native_grid_shape=list(native_shape),
        native_cells=int(native_shape[0] * native_shape[1]),
        n_params=n_params,
        results=results,
        grid_solver_full_ian_wall_s=GRID_SOLVER_FULL_IAN_WALL_S,
        grid_solver_full_ian_steps=GRID_SOLVER_FULL_IAN_STEPS,
        frame_interval_min=FRAME_INTERVAL_MIN,
        grid_solver_wall_s_per_frame_interval=round(solver_wall_s_per_frame_interval, 4),
        mesh_gnn_cpu_s_per_pass=MESH_GNN_CPU_S_PER_PASS,
        mesh_gnn_mps_oom=MESH_GNN_MPS_OOM,
    )
    for dname in ("cpu", "mps"):
        if dname in results and results[dname]["status"] == "ok":
            s_per_pass = results[dname]["mean_s_per_pass"]
            summary[f"speedup_vs_mesh_gnn_cpu_{dname}"] = round(MESH_GNN_CPU_S_PER_PASS / s_per_pass, 1)
            summary[f"speedup_vs_grid_solver_equivalent_{dname}"] = round(
                solver_wall_s_per_frame_interval / s_per_pass, 2)

    out_path = os.path.join(BASE_DIR, "outputs", f"grid_transformer_site3_fullres_benchmark_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {os.path.relpath(out_path)}")
    print(f"\n  Physics solver: {solver_wall_s_per_frame_interval:.3f}s to advance "
          f"{FRAME_INTERVAL_MIN:.0f}min of simulated time at full site3 resolution.")
    for dname in ("cpu", "mps"):
        k1, k2 = f"speedup_vs_mesh_gnn_cpu_{dname}", f"speedup_vs_grid_solver_equivalent_{dname}"
        if k1 in summary:
            print(f"  [{dname}] {summary[k1]:.1f}x faster than the mesh-GNN's 304.8s/pass CPU "
                  f"figure; {summary[k2]:.1f}x faster than the physics solver's own equivalent "
                  f"wall time for the same simulated-time span.")


if __name__ == "__main__":
    main()
