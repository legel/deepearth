#!/usr/bin/env python3
"""Evaluate one or more trained GridTransformerSurrogate checkpoints on the site3 held-out
storms, reporting BOTH diagnostics the mesh-GNN study found it necessary to separate (CLAUDE.md:
"a model can have low single-step loss and still fail badly once its own predictions are fed
back in"):

  1. ONE-STEP error: predict frame[i+K] from the REAL frames[i:i+K] every time (never fed its
     own output back in) — this is what the training loss already optimizes for.
  2. ROLLOUT error: autoregressive — seed with real frames[0:K], then feed predictions back in
     as context for every subsequent step, the way a surrogate actually has to run.

If (1) is good and (2) is bad, that's the SAME exposure-bias-shaped failure the GNN study found.
If (1) is ALSO bad, the problem is more basic (the architecture/data isn't learning the dynamics
at all) — a materially different, and less interesting, finding. Distinguishing these is the
point of this script.

Run: python3 simulation/evaluate_grid_transformer_checkpoints.py \
        --tags baseline reweighted
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
from grid_transformer_surrogate import GridTransformerSurrogate, pad_to_multiple, crop_to, DOWNSAMPLE  # noqa: E402
from train_grid_transformer_surrogate import load_storms, N_CONTEXT, get_device  # noqa: E402


def one_step_eval(model, storms, device, hw):
    out = {}
    for T, s in storms.items():
        if s["split"] != "held_out":
            continue
        frames, rain = s["frames"], s["rain_mm_hr"]
        n = len(frames)
        errs = []
        with torch.no_grad():
            for i in range(n - N_CONTEXT):
                ctx = torch.from_numpy(frames[i:i + N_CONTEXT]).unsqueeze(0).unsqueeze(2).float()
                ctx, _ = pad_to_multiple(ctx.reshape(-1, 1, *ctx.shape[-2:]), DOWNSAMPLE)
                ctx = ctx.reshape(1, N_CONTEXT, 1, *ctx.shape[-2:]).to(device)
                fb = torch.tensor([rain[i + N_CONTEXT]], dtype=torch.float32, device=device)
                pred, _ = model(ctx, fb)
                pred_crop = crop_to(pred, hw)[0, 0].cpu().numpy()
                errs.append(float(np.sqrt(np.mean((pred_crop - frames[i + N_CONTEXT]) ** 2))))
        out[T] = dict(one_step_rmse_mean_m=round(float(np.mean(errs)), 5),
                     one_step_rmse_max_m=round(float(np.max(errs)), 5))
    return out


def rollout_eval(model, storms, device, hw):
    out = {}
    for T, s in storms.items():
        if s["split"] != "held_out":
            continue
        frames, rain = s["frames"], s["rain_mm_hr"]
        n = len(frames)
        cell_ha = (s["dx"] ** 2) / 1e4
        ctx = torch.from_numpy(frames[:N_CONTEXT]).unsqueeze(0).unsqueeze(2).float()
        ctx, _ = pad_to_multiple(ctx.reshape(-1, 1, *ctx.shape[-2:]), DOWNSAMPLE)
        ctx = ctx.reshape(1, N_CONTEXT, 1, *ctx.shape[-2:]).to(device)
        pred_vol, real_vol, rmse_ts = [], [], []
        pred_final = None
        with torch.no_grad():
            for i in range(N_CONTEXT, n):
                fb = torch.tensor([rain[i]], dtype=torch.float32, device=device)
                pred, _ = model(ctx, fb)
                pred_crop = crop_to(pred, hw)[0, 0].cpu().numpy()
                pred_vol.append(float(pred_crop.sum()) * cell_ha * 100)
                real_vol.append(float(frames[i].sum()) * cell_ha * 100)
                rmse_ts.append(float(np.sqrt(np.mean((pred_crop - frames[i]) ** 2))))
                pred_final = pred_crop
                ctx = torch.cat([ctx[:, 1:], pred.unsqueeze(1)], dim=1)
        out[T] = _rollout_summary(pred_vol, real_vol, rmse_ts, pred_final, frames[-1])
    return out


def naive_persistence_eval(storms):
    """Zero-effort baseline: 'predict no change from the last real/predicted frame' forever
    (rain rate is ignored entirely). Calibrates how bad the trained models' own rollout
    collapse really is — a model with no learned dynamics at all still gets SOME volume-drift
    number on a growing storm, and until this was computed, -88.9% had no reference point."""
    out = {}
    for T, s in storms.items():
        if s["split"] != "held_out":
            continue
        frames = s["frames"]
        n = len(frames)
        cell_ha = (s["dx"] ** 2) / 1e4
        pred_vol, real_vol, rmse_ts = [], [], []
        last = frames[N_CONTEXT - 1]
        for i in range(N_CONTEXT, n):
            pred_vol.append(float(last.sum()) * cell_ha * 100)
            real_vol.append(float(frames[i].sum()) * cell_ha * 100)
            rmse_ts.append(float(np.sqrt(np.mean((last - frames[i]) ** 2))))
        out[T] = _rollout_summary(pred_vol, real_vol, rmse_ts, last, frames[-1])
    return out


WET_THR_M = 0.05   # same "flooded" threshold flood_sim_ian.py's own DEPTH_THR uses — reused for
                    # consistency rather than picking a new one for this script specifically.


def _wet_iou_f1(pred, real, thr=WET_THR_M):
    """Wet-cell IoU/precision/recall/F1 — the SPATIAL PATTERN question volume drift and RMSE
    both fail to answer: does the model know WHERE the water is, not just how much of it there
    is in total? Same metric family FloodTransformer (Gu, Kang et al. 2026) reports (IoU/F1 for
    inundation classification) — directly comparable framing, not just an internal convention.
    A model could match real total volume closely while still getting the spatial pattern wrong
    (e.g. flooding the wrong region by the same total amount) — this is the check for exactly
    that failure mode, not yet run before this pass."""
    p_wet = pred > thr
    r_wet = real > thr
    inter = float((p_wet & r_wet).sum())
    union = float((p_wet | r_wet).sum())
    iou = inter / union if union > 0 else float("nan")
    precision = inter / p_wet.sum() if p_wet.sum() > 0 else float("nan")
    recall = inter / r_wet.sum() if r_wet.sum() > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and precision + recall > 0) else float("nan"))
    return iou, precision, recall, f1, int(r_wet.sum())


def _rollout_summary(pred_vol, real_vol, rmse_ts, pred_final, real_final):
    pred_vol, real_vol = np.array(pred_vol), np.array(real_vol)
    iou, prec, rec, f1, n_wet = _wet_iou_f1(pred_final, real_final)
    return dict(
        final_volume_drift_pct=round(float(100 * (pred_vol[-1] - real_vol[-1]) /
                                          max(real_vol[-1], 1e-9)), 2),
        volume_drift_pct_at_10pct_steps=round(float(100 * (pred_vol[int(0.1*len(pred_vol))] -
                                          real_vol[int(0.1*len(real_vol))]) /
                                          max(real_vol[int(0.1*len(real_vol))], 1e-9)), 2),
        volume_drift_pct_at_50pct_steps=round(float(100 * (pred_vol[int(0.5*len(pred_vol))] -
                                          real_vol[int(0.5*len(real_vol))]) /
                                          max(real_vol[int(0.5*len(real_vol))], 1e-9)), 2),
        final_frame_rmse_m=round(float(np.sqrt(np.mean((pred_final - real_final) ** 2))), 5),
        rollout_rmse_at_10pct_steps_m=round(float(rmse_ts[int(0.1 * len(rmse_ts))]), 5),
        rollout_rmse_at_50pct_steps_m=round(float(rmse_ts[int(0.5 * len(rmse_ts))]), 5),
        rollout_rmse_at_100pct_steps_m=round(float(rmse_ts[-1]), 5),
        final_wet_iou=round(iou, 4) if iou == iou else None,
        final_wet_precision=round(prec, 4) if prec == prec else None,
        final_wet_recall=round(rec, 4) if rec == rec else None,
        final_wet_f1=round(f1, 4) if f1 == f1 else None,
        real_wet_cells_final=n_wet,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["baseline", "reweighted"])
    ap.add_argument("--dataset", default="grid_surrogate_site3")
    args = ap.parse_args()

    device = get_device()
    manifest, storms = load_storms(args.dataset)

    results = {}

    if "naive" in args.tags:
        print("\n=== naive (persistence baseline, no model) ===")
        rollout = naive_persistence_eval(storms)
        for T in rollout:
            print(f"  T={T:>4}yr  rollout final drift={rollout[T]['final_volume_drift_pct']:+.1f}%  "
                  f"(10%={rollout[T]['volume_drift_pct_at_10pct_steps']:+.1f}%  "
                  f"50%={rollout[T]['volume_drift_pct_at_50pct_steps']:+.1f}%)  "
                  f"final RMSE={rollout[T]['final_frame_rmse_m']:.4f}m  "
                  f"wet IoU/F1={rollout[T]['final_wet_iou']}/{rollout[T]['final_wet_f1']}")
        results["naive"] = dict(one_step={}, rollout=rollout)

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
        print(f"\n=== {tag} ===")
        one_step = one_step_eval(model, storms, device, hw)
        rollout = rollout_eval(model, storms, device, hw)
        for T in one_step:
            print(f"  T={T:>4}yr  one-step RMSE(mean/max)={one_step[T]['one_step_rmse_mean_m']:.4f}/"
                  f"{one_step[T]['one_step_rmse_max_m']:.4f}m   "
                  f"rollout final drift={rollout[T]['final_volume_drift_pct']:+.1f}%  "
                  f"(10%={rollout[T]['volume_drift_pct_at_10pct_steps']:+.1f}%  "
                  f"50%={rollout[T]['volume_drift_pct_at_50pct_steps']:+.1f}%)  "
                  f"rollout final RMSE={rollout[T]['final_frame_rmse_m']:.4f}m  "
                  f"wet IoU/F1={rollout[T]['final_wet_iou']}/{rollout[T]['final_wet_f1']}")
        results[tag] = dict(one_step=one_step, rollout=rollout)

    out_path = os.path.join(BASE_DIR, "outputs", "grid_transformer_site3_eval_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
