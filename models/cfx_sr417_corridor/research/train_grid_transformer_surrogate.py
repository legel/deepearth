#!/usr/bin/env python3
"""Train GridTransformerSurrogate on the site3 multi-storm grid dataset, then validate via a
real autoregressive rollout on the 2 held-out storms — same "don't trust single-step loss alone"
discipline `validate_gnn_rollout.py` already established for the mesh-GNN experiment: a
model can have low single-step loss and still fail badly once its own predictions are fed back
in as the next input. Reports the same class of diagnostic here (predicted vs. real total
water volume trajectory, plus final-frame RMSE — the latter chosen specifically to be
comparable to FloodSformer's own reported ~10cm RMSE, Pianforini et al. 2025).

Run: python3 simulation/train_grid_transformer_surrogate.py
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
from grid_transformer_surrogate import GridTransformerSurrogate, pad_to_multiple, crop_to, DOWNSAMPLE  # noqa: E402

N_CONTEXT = 4


def load_storms(dataset="grid_surrogate_site3"):
    data_dir = os.path.join(BASE_DIR, "data", dataset)
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    storms = {}
    for s in manifest["storms"]:
        d = np.load(os.path.join(data_dir, s["file"]))
        storms[s["return_period_yr"]] = dict(
            frames=d["frames"], times_min=d["times_min"], rain_mm_hr=d["rain_mm_hr"],
            split=s["split"], dx=float(d["dx"]), grid_shape=s["grid_shape"])
    return manifest, storms


def build_windows(frames, rain_mm_hr, n_context):
    """Sliding windows: input = frames[i:i+K], target = frames[i+K], forcing = rain at i+K."""
    n = len(frames)
    X, y, f = [], [], []
    for i in range(n - n_context):
        X.append(frames[i:i + n_context])
        y.append(frames[i + n_context])
        f.append(rain_mm_hr[i + n_context])
    return np.stack(X), np.stack(y), np.array(f, dtype=np.float32)


def build_rollout_windows(frames, rain_mm_hr, n_context, r_steps):
    """Sliding windows for MULTI-STEP rollout training: input = frames[i:i+K] (real context),
    targets = frames[i+K : i+K+R] (R real future frames, in order), forcing = rain at each of
    those R future times. Used to train THROUGH a short autoregressive rollout (see train()'s
    r_steps handling) rather than single-step-only — the standard, most direct fix for
    exposure bias in autoregressive prediction (the model only ever sees real context during
    single-step training, but has to run on its OWN accumulating output at inference)."""
    n = len(frames)
    X, Y, F = [], [], []
    for i in range(n - n_context - r_steps + 1):
        X.append(frames[i:i + n_context])
        Y.append(frames[i + n_context: i + n_context + r_steps])
        F.append(rain_mm_hr[i + n_context: i + n_context + r_steps])
    return np.stack(X), np.stack(Y), np.stack(F).astype(np.float32)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train(model, storms, device, epochs, batch_size, lr, loss_weight_alpha=0.0, rollout_steps=1,
          vol_loss_weight=0.0):
    """rollout_steps=1 is plain single-step training (the original baseline/reweighted runs).
    rollout_steps=R>1 trains THROUGH an R-step autoregressive rollout: the model's own (still
    differentiable — gradients flow through the whole chain, standard truncated-BPTT-through-
    rollout) prediction at step t becomes part of the context for step t+1, and the loss is the
    mean single-step MSE across all R steps against REAL targets at each step. This is the
    direct, standard fix for exposure bias — the model is trained under the same distribution
    (its own imperfect output feeding forward) it has to run under at inference — as opposed to
    loss reweighting, which changes WHAT is penalized but not WHAT DISTRIBUTION of inputs the
    model ever sees during training."""
    train_T = [T for T, s in storms.items() if s["split"] == "train"]
    print(f"  training storms: {sorted(train_T)}  rollout_steps={rollout_steps}")

    Xs, Ys, Fs = [], [], []
    for T in train_T:
        s = storms[T]
        X, Y, F = build_rollout_windows(s["frames"], s["rain_mm_hr"], N_CONTEXT, rollout_steps)
        Xs.append(X); Ys.append(Y); Fs.append(F)
    X = np.concatenate(Xs); Y = np.concatenate(Ys); F = np.concatenate(Fs)
    print(f"  {len(X)} windowed training samples "
          f"(K={N_CONTEXT} context frames, R={rollout_steps} rollout targets each)")

    X_t = torch.from_numpy(X).unsqueeze(2).float()    # [N, K, 1, H, W]
    Y_t = torch.from_numpy(Y).unsqueeze(2).float()    # [N, R, 1, H, W]
    F_t = torch.from_numpy(F).float()                 # [N, R]

    # Pad once, up front (uniform grid shape across all storms — same site3 domain/cell size).
    X_pad, hw = pad_to_multiple(X_t.reshape(-1, 1, X_t.shape[-2], X_t.shape[-1]), DOWNSAMPLE)
    X_pad = X_pad.reshape(X_t.shape[0], X_t.shape[1], 1, X_pad.shape[-2], X_pad.shape[-1])
    Y_pad, _ = pad_to_multiple(Y_t.reshape(-1, 1, Y_t.shape[-2], Y_t.shape[-1]), DOWNSAMPLE)
    Y_pad = Y_pad.reshape(Y_t.shape[0], Y_t.shape[1], 1, Y_pad.shape[-2], Y_pad.shape[-1])
    print(f"  frame grid {hw} padded to {tuple(X_pad.shape[-2:])} (mult. of {DOWNSAMPLE})")

    # Per-pixel target-diff magnitude (step-1 only), for optional loss reweighting — the same
    # fix that closed the mesh-GNN's own volume-collapse failure (alpha=8.0, "per-
    # node loss reweighting ... worked, adopted"). Kept available here (rollout_steps=1 +
    # loss_weight_alpha>0 reproduces that exact experiment on this architecture); combining it
    # with rollout_steps>1 is not implemented (the two fixes target the same failure via
    # different mechanisms — see module docstring — comparing them separately is the point).
    diff_std = float((Y_t[:, 0, 0] - X_t[:, -1, 0]).std()) + 1e-9
    diff_pad0 = None
    if loss_weight_alpha > 0 and rollout_steps == 1:
        diff_map = (Y_t[:, 0, 0] - X_t[:, -1, 0])
        diff_pad0, _ = pad_to_multiple(diff_map.unsqueeze(1), DOWNSAMPLE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_pad.shape[0]
    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ctx = X_pad[idx].to(device)
            yb = Y_pad[idx].to(device)
            fb = F_t[idx].to(device)

            step_loss = 0.0
            for r in range(rollout_steps):
                pred, _ = model(ctx, fb[:, r])
                sq_err = (pred - yb[:, r]) ** 2
                if diff_pad0 is not None:
                    w = 1.0 + loss_weight_alpha * diff_pad0[idx].abs().to(device) / diff_std
                    step_loss = step_loss + (sq_err * w).mean()
                else:
                    step_loss = step_loss + sq_err.mean()
                if vol_loss_weight > 0:
                    # Mass-conservation-informed term: match PREDICTED mean depth (proportional
                    # to total water volume, since domain area is fixed) to the REAL target's
                    # mean depth. Pointwise MSE alone has no incentive to get the domain-wide
                    # total right as long as it gets most pixels' small errors right — this term
                    # penalizes exactly the aggregate quantity that collapsed in the unweighted
                    # baseline (see research/README.md: rollout volume drift up to
                    # -99.9% despite near-identical RMSE). Same spirit as the domain-wise volume-
                    # conservation loss FloodTransformer's own authors name as their unbuilt
                    # future work (Gu, Kang et al. 2026) — built here as a real, tested attempt,
                    # not just cited as an idea.
                    vol_err = (pred.mean(dim=(-2, -1)) - yb[:, r].mean(dim=(-2, -1))) ** 2
                    step_loss = step_loss + vol_loss_weight * vol_err.mean()
                ctx = torch.cat([ctx[:, 1:], pred.unsqueeze(1)], dim=1)   # own prediction, gradients flow
            loss = step_loss / rollout_steps

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        losses.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:>3}/{epochs}  loss={epoch_loss:.6e}")
    wall_s = time.time() - t0
    return wall_s, losses, hw


def rollout_validate(model, storms, device, hw):
    """Autoregressive rollout on each held-out storm: seed with the first K REAL frames, then
    feed the model's own predictions back in as context for every subsequent step — the actual
    way a surrogate would run if it replaced the solver (same principle validate_gnn_rollout.py
    already established for the mesh-GNN experiment)."""
    results = {}
    for T, s in storms.items():
        if s["split"] != "held_out":
            continue
        frames = s["frames"]
        rain = s["rain_mm_hr"]
        n = len(frames)
        cell_ha = (s["dx"] ** 2) / 1e4

        ctx = torch.from_numpy(frames[:N_CONTEXT]).unsqueeze(0).unsqueeze(2).float()  # [1,K,1,H,W]
        ctx, _ = pad_to_multiple(ctx.reshape(-1, 1, *ctx.shape[-2:]), DOWNSAMPLE)
        ctx = ctx.reshape(1, N_CONTEXT, 1, *ctx.shape[-2:]).to(device)

        pred_vol, real_vol = [], []
        pred_final = None
        with torch.no_grad():
            for i in range(N_CONTEXT, n):
                fb = torch.tensor([rain[i]], dtype=torch.float32, device=device)
                pred, _ = model(ctx, fb)
                pred_crop = crop_to(pred, hw)[0, 0].cpu().numpy()
                pred_vol.append(float(pred_crop.sum()) * cell_ha * 100)  # m3 (ha->m2 *100, *depth)
                real_vol.append(float(frames[i].sum()) * cell_ha * 100)
                pred_final = pred_crop
                ctx = torch.cat([ctx[:, 1:], pred.unsqueeze(1)], dim=1)   # slide window with OWN prediction

        pred_vol = np.array(pred_vol); real_vol = np.array(real_vol)
        final_rmse_m = float(np.sqrt(np.mean((pred_final - frames[-1]) ** 2)))
        final_vol_drift_pct = float(100.0 * (pred_vol[-1] - real_vol[-1]) / max(real_vol[-1], 1e-9))
        results[T] = dict(
            n_rollout_steps=int(n - N_CONTEXT),
            real_final_volume_m3=float(real_vol[-1]),
            pred_final_volume_m3=float(pred_vol[-1]),
            final_volume_drift_pct=round(final_vol_drift_pct, 2),
            final_frame_rmse_m=round(final_rmse_m, 4),
            real_peak_depth_m=float(frames.max()),
            pred_peak_depth_m=float(pred_final.max()),
        )
        print(f"  held-out T={T:>4}yr  rollout {n - N_CONTEXT} steps  "
              f"final volume drift={final_vol_drift_pct:+.1f}%  "
              f"final-frame RMSE={final_rmse_m:.4f}m")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--loss-weight-alpha", type=float, default=0.0,
                    help="Per-pixel loss reweighting by target-diff magnitude (0 = plain MSE, "
                         "the mesh-GNN study's own baseline). alpha=8.0 is what fixed that "
                         "study's volume-collapse failure —.")
    ap.add_argument("--rollout-steps", type=int, default=1,
                    help="Train through an R-step autoregressive rollout (R=1 = plain "
                         "single-step training). The direct fix for exposure bias — see train().")
    ap.add_argument("--vol-loss-weight", type=float, default=0.0,
                    help="Mass-conservation-informed loss term weight (0 = off). Penalizes "
                         "predicted-vs-real MEAN depth mismatch per step — see train().")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (torch/numpy/python).")
    ap.add_argument("--dataset", default="grid_surrogate_site3",
                    help="Subdirectory under simulation/data/ to load storms from — e.g. "
                         "grid_surrogate_site3_crop for the apples-to-apples GNN comparison.")
    ap.add_argument("--tag", default="", help="Suffix for checkpoint/summary filenames")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"  device: {device}")

    manifest, storms = load_storms(args.dataset)
    model = GridTransformerSurrogate(n_context=N_CONTEXT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    print("\n[1/2] Training …")
    wall_s, losses, hw = train(model, storms, device, args.epochs, args.batch_size, args.lr,
                               loss_weight_alpha=args.loss_weight_alpha,
                               rollout_steps=args.rollout_steps,
                               vol_loss_weight=args.vol_loss_weight)
    print(f"  training wall time: {wall_s:.1f}s ({args.epochs} epochs, seed={args.seed}, "
          f"loss_weight_alpha={args.loss_weight_alpha}, rollout_steps={args.rollout_steps}, "
          f"vol_loss_weight={args.vol_loss_weight})")

    tag = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(CKPT_DIR, f"grid_transformer_site3{tag}.pt")
    torch.save(dict(state_dict=model.state_dict(), n_context=N_CONTEXT, hw=hw), ckpt_path)
    print(f"  saved checkpoint: {os.path.relpath(ckpt_path)}")

    print("\n[2/2] Autoregressive rollout validation on held-out storms …")
    rollout = rollout_validate(model, storms, device, hw)

    summary = dict(
        device=str(device), n_params=n_params, epochs=args.epochs, seed=args.seed,
        batch_size=args.batch_size, lr=args.lr, loss_weight_alpha=args.loss_weight_alpha,
        rollout_steps=args.rollout_steps, vol_loss_weight=args.vol_loss_weight,
        train_wall_s=round(wall_s, 1), final_train_loss=losses[-1],
        train_storms=sorted([T for T, s in storms.items() if s["split"] == "train"]),
        held_out_storms=sorted([T for T, s in storms.items() if s["split"] == "held_out"]),
        rollout=rollout,
    )
    out_path = os.path.join(BASE_DIR, "outputs", f"grid_transformer_site3_summary{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
