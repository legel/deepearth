#!/usr/bin/env python3
"""The real-event test every rollout result so far has been missing: run the trained grid-
transformer checkpoints against the REAL Hurricane Ian reconstruction
(`build_ian_rollout_dataset_site3.py`'s output), not another synthetic Atlas-14 design storm.
Reuses evaluate_grid_transformer_checkpoints.py's own rollout/one-step/naive functions directly
(same metrics: volume drift, RMSE, wet-cell IoU/F1) against a synthetic `storms` dict with one
real entry so nothing has to be duplicated.

Run: python3 simulation/evaluate_grid_transformer_on_ian.py --tags naive baseline_s0 volloss_e20
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "grid_surrogate_site3")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
from grid_transformer_surrogate import GridTransformerSurrogate, pad_to_multiple, crop_to, DOWNSAMPLE  # noqa: E402
from train_grid_transformer_surrogate import N_CONTEXT, get_device  # noqa: E402
from evaluate_grid_transformer_checkpoints import (   # noqa: E402
    one_step_eval, rollout_eval, naive_persistence_eval)


def load_ian_storm():
    d = np.load(os.path.join(DATA_DIR, "storm_ian.npz"))
    return {"ian": dict(frames=d["frames"], rain_mm_hr=d["rain_mm_hr"], split="held_out",
                        dx=float(d["dx"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["naive", "baseline_s0", "volloss_e20"])
    args = ap.parse_args()

    device = get_device()
    storms = load_ian_storm()
    print(f"  Real Ian event: {len(storms['ian']['frames'])} frames, "
          f"peak depth {storms['ian']['frames'].max():.3f}m")

    results = {}
    if "naive" in args.tags:
        print("\n=== naive (persistence baseline, no model) — REAL Ian ===")
        rollout = naive_persistence_eval(storms)
        r = rollout["ian"]
        print(f"  final drift={r['final_volume_drift_pct']:+.1f}%  "
              f"(10%={r['volume_drift_pct_at_10pct_steps']:+.1f}%  "
              f"50%={r['volume_drift_pct_at_50pct_steps']:+.1f}%)  "
              f"RMSE={r['final_frame_rmse_m']:.4f}m  wet IoU/F1={r['final_wet_iou']}/{r['final_wet_f1']}")
        results["naive"] = dict(rollout=rollout)

    for tag in [t for t in args.tags if t != "naive"]:
        ckpt_path = os.path.join(CKPT_DIR, f"grid_transformer_site3_{tag}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  (skip) {ckpt_path} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = GridTransformerSurrogate(n_context=ckpt["n_context"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        hw = tuple(ckpt["hw"])
        print(f"\n=== {tag} — REAL Ian ===")
        one_step = one_step_eval(model, storms, device, hw)
        rollout = rollout_eval(model, storms, device, hw)
        r, o = rollout["ian"], one_step["ian"]
        print(f"  one-step RMSE(mean/max)={o['one_step_rmse_mean_m']:.4f}/{o['one_step_rmse_max_m']:.4f}m  "
              f"final drift={r['final_volume_drift_pct']:+.1f}%  "
              f"(10%={r['volume_drift_pct_at_10pct_steps']:+.1f}%  "
              f"50%={r['volume_drift_pct_at_50pct_steps']:+.1f}%)  "
              f"RMSE={r['final_frame_rmse_m']:.4f}m  wet IoU/F1={r['final_wet_iou']}/{r['final_wet_f1']}")
        results[tag] = dict(one_step=one_step, rollout=rollout)

    out_path = os.path.join(BASE_DIR, "outputs", "grid_transformer_site3_ian_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
