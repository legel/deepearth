"""
MeshGraphKAN surrogate — trained on site3's OWN real per-point 3D mesh physics
==================================================================================
The real successor to train_mesh_gnn.py's proof-of-concept (which trained on a synthetic
COARSE GRID built from the original AOI's 2.5D flood_sim_ian.py output, one single Ian event).
This trains on the REAL unstructured mesh graph from site3_crop_coarse (real edges, real
per-triangle area/elevation/soil/impervious data) and REAL multi-scenario shallow-water output
from run_gnn_training_sweep.py (8 distinct rain scenarios at site3's own pour point, GPU-solved,
mass-conservative to ~0.000% residual on every run).

Held-out validation:
2 of the 8 scenarios are held out ENTIRELY from training, not just individual timesteps within
seen scenarios -- this tests real generalization to unseen rain intensity/duration, not just
interpolation within a seen event. "mediumhigh_short" (130mm/hr, interpolated intensity between
trained levels) and "medium_long" (100mm/hr but 8min/15min instead of 4min/10min -- a different
DURATION shape) are held out, testing both intensity- and duration-generalization separately.

Usage:
    .venv/bin/python3 simulation/train_mesh_gnn_site3.py --epochs 30
"""
import os, sys, argparse, pickle, glob
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)

from mesh_shallow_water import synthetic_rain_rate_ms  # noqa: E402
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN  # noqa: E402

N_TIME_STEPS = 2   # sliding-window length, matches HydroGraphNet's own default + the earlier
                     # proof-of-concept's convention
HELD_OUT_SCENARIOS = {"mediumhigh_short", "medium_long"}


def standardize(a, mean=None, std=None):
    a = a.astype(np.float32).ravel()
    if mean is None:
        mean, std = a.mean(), a.std() + 1e-8
    return (a - mean) / std, mean, std


def build_static_features(mesh, gim, lm, gh, gng):
    T = mesh["T"]
    area = mesh["area"]; z = mesh["z"]; is_roof = mesh["is_roof"].astype(np.float32)
    xy = mesh["xy"]

    # Ground-only arrays (gim/gh/gng are (Tg,)) padded to full T -- roofs get 0/no-infiltration
    # sentinel values, consistent with how the solver itself treats roofs (always impervious).
    Tg = mesh["Tg"]
    impervious_full = np.zeros(T, dtype=np.float32)
    impervious_full[:Tg] = gim.astype(np.float32)
    impervious_full[Tg:] = 1.0   # roofs are always impervious

    fc_full = np.zeros(T, dtype=np.float32)
    if gh is not None:
        fc_full[:Tg] = gh["fc"]
    nlcd_full = np.ones(T, dtype=np.float32)
    if gng is not None:
        nlcd_full[:Tg] = gng

    x_s, _, _ = standardize(xy[:, 0])
    y_s, _, _ = standardize(xy[:, 1])
    area_s, _, _ = standardize(area)
    z_s, _, _ = standardize(z)
    fc_s, _, _ = standardize(fc_full)
    nlcd_s, _, _ = standardize(nlcd_full)

    static = np.stack([x_s, y_s, area_s, z_s, is_roof, impervious_full, fc_s, nlcd_s], axis=1)
    return static.astype(np.float32)   # (T, 8)


def build_edge_features(mesh):
    L = mesh["edges"]["L"].astype(np.float32)
    W = mesh["edges"]["W"].astype(np.float32)
    is_roof = mesh["is_roof"]
    i_all, j_all = mesh["edges"]["i"], mesh["edges"]["j"]
    roof_edge = (is_roof[i_all] | is_roof[j_all]).astype(np.float32)
    return np.stack([L, W, roof_edge], axis=1).astype(np.float32)   # (n_edges, 3)


def load_scenarios(scenario_dir):
    files = sorted(glob.glob(os.path.join(scenario_dir, "*.pkl")))
    scenarios = {}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        scenarios[d["scenario"]["name"]] = d
    return scenarios


def build_samples(scenario, depth_mean, depth_std, noise_std=0.0, rng=None):
    """noise_std>0 implements the "training noise" trick from Sanchez-Gonzalez et al. 2020
    (Learning to Simulate Complex Physics with Graph Networks): perturb the most-recent frame
    of the input depth window with Gaussian noise, and compute the target diff FROM that noised
    state rather than the clean one. This teaches the model to correct back toward the true
    trajectory when its own input is slightly off -- exactly the situation it's in during
    autoregressive rollout (each step's input is the model's OWN, imperfect prior prediction),
    which a pure teacher-forced/clean-input training signal never exposes it to."""
    frames_h = scenario["frames_h"]
    frame_times = scenario["frame_times"]
    sc = scenario["scenario"]
    n_frames = len(frames_h)

    depths = [(h.astype(np.float32) - depth_mean) / depth_std for h in frames_h]
    rain_at_frame = [synthetic_rain_rate_ms(t, sc["rain_duration_min"] * 60, sc["peak_mm_hr"])
                      for t in frame_times]

    samples = []
    for t in range(n_frames - N_TIME_STEPS - 1):
        depth_window = np.stack(depths[t:t + N_TIME_STEPS], axis=1).copy()
        last_clean = depth_window[:, -1].copy()
        if noise_std > 0:
            noise = rng.normal(0.0, noise_std, size=depth_window.shape).astype(np.float32)
            depth_window += noise
            last_noised = depth_window[:, -1]
        else:
            last_noised = last_clean
        current_rain = np.full((depth_window.shape[0], 1), rain_at_frame[t + N_TIME_STEPS - 1],
                                dtype=np.float32)
        x_dynamic = np.hstack([current_rain, depth_window])
        target_depth = depths[t + N_TIME_STEPS] - last_noised
        samples.append((x_dynamic.astype(np.float32), target_depth.astype(np.float32)))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3_crop_coarse")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--noise-std-frac", type=float, default=0.5,
                     help="training-noise std as a fraction of the real single-step diff std "
                          "(Sanchez-Gonzalez et al. 2020); 0 reproduces the original "
                          "teacher-forced-only training (real, measured 87-91%% volume collapse "
                          "under autoregressive rollout — see validate_gnn_rollout.py)")
    ap.add_argument("--loss-weight-alpha", type=float, default=8.0,
                     help="training-loss reweighting: per-node weight = 1 + alpha*|target|/diff_std. "
                          "Real root cause found via rollout validation: with plain per-node MSE, "
                          "the vast dry majority of cells (near-zero target diff every step) lets "
                          "the model minimize loss by predicting ~no change everywhere, which is a "
                          "self-reinforcing fixed point once its own near-flat output is fed back in "
                          "during rollout (measured: -88 to -97% total-volume collapse). Weighting "
                          "actively-changing cells more heavily in the TRAINING objective (val_loss "
                          "stays plain/unweighted for comparability) directly targets this. 0 "
                          "reproduces plain unweighted MSE.")
    ap.add_argument("--out-name", default="model_checkpoint.pt")
    args = ap.parse_args()

    site_dir = os.path.join(PROJ_DIR, "site3_gee_creek", "gnn_training", args.site)
    mesh_ckpt = os.path.join(site_dir, "checkpoints", "mesh_and_masks.pkl")
    scenario_dir = os.path.join(site_dir, "scenarios")

    print("=" * 70)
    print(f"MeshGraphKAN training on site3's own real mesh physics — {args.site}")
    print("=" * 70)

    with open(mesh_ckpt, "rb") as fh:
        mesh, gim, lm, gh, gng = pickle.load(fh)
    T = mesh["T"]
    n_edges = len(mesh["edges"]["i"])
    print(f"\n[1/4] Mesh: {T:,} triangles, {n_edges:,} edges")

    static_feats = build_static_features(mesh, gim, lm, gh, gng)
    edge_attr_np = build_edge_features(mesh)

    print(f"\n[2/4] Loading scenarios from {scenario_dir} …")
    scenarios = load_scenarios(scenario_dir)
    train_names = [n for n in scenarios if n not in HELD_OUT_SCENARIOS]
    val_names = [n for n in scenarios if n in HELD_OUT_SCENARIOS]
    print(f"  {len(scenarios)} scenarios loaded: {sorted(scenarios)}")
    print(f"  TRAIN ({len(train_names)}): {sorted(train_names)}")
    print(f"  HELD-OUT VALIDATION ({len(val_names)}): {sorted(val_names)}")

    # Normalize depth using TRAINING scenarios only (val stats must not leak into normalization
    # — real held-out validation, not just held-out samples from an already-seen distribution).
    all_train_depths = np.concatenate([
        np.concatenate([h.ravel() for h in scenarios[n]["frames_h"]]) for n in train_names
    ])
    depth_mean = float(all_train_depths.mean())
    depth_std = float(all_train_depths.std() + 1e-8)
    print(f"  depth normalization (train-only): mean={depth_mean:.5f} std={depth_std:.5f}")

    print(f"\n[3/4] Building sliding-window samples (n_time_steps={N_TIME_STEPS}) …")
    train_samples_clean = []
    for n in train_names:
        train_samples_clean.extend(build_samples(scenarios[n], depth_mean, depth_std))
    val_samples = []
    for n in val_names:
        val_samples.extend(build_samples(scenarios[n], depth_mean, depth_std))
    print(f"  {len(train_samples_clean)} training samples, {len(val_samples)} held-out validation samples")

    diff_std = float(np.concatenate([y.ravel() for _, y in train_samples_clean]).std())
    noise_std = args.noise_std_frac * diff_std
    rng = np.random.default_rng(0)
    print(f"  real single-step diff std (normalized): {diff_std:.5f}  ->  "
          f"training-noise std = {noise_std:.5f} ({args.noise_std_frac}x)")

    device = torch.device(args.device)
    ei_t = torch.as_tensor(mesh["edges"]["i"], dtype=torch.int64, device=device)
    ej_t = torch.as_tensor(mesh["edges"]["j"], dtype=torch.int64, device=device)
    edge_attr_t = torch.as_tensor(edge_attr_np, dtype=torch.float32, device=device)
    graph = Data(edge_index=torch.stack([ei_t, ej_t]))
    static_t = torch.as_tensor(static_feats, dtype=torch.float32, device=device)

    n_input_features = static_feats.shape[1] + train_samples_clean[0][0].shape[1]   # 8 static + 3 dynamic (rain+2 depth window) = 11
    model = MeshGraphKAN(input_dim_nodes=n_input_features, input_dim_edges=3, output_dim=1,
                          processor_size=6, hidden_dim_processor=64,
                          hidden_dim_node_encoder=64, hidden_dim_edge_encoder=64,
                          hidden_dim_node_decoder=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    criterion_unreduced = nn.MSELoss(reduction="none")
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n[4/4] Training on {device} — {n_params:,} params, {n_input_features} node features, "
          f"{len(train_samples_clean)} train samples, {len(val_samples)} val samples, {args.epochs} epochs, "
          f"fresh training-noise redrawn every epoch\n")

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Redraw training noise every epoch (not fixed once) -- same regularization spirit as
        # dropout/data-augmentation: a fixed noise draw would let the model overfit to that one
        # specific perturbation instead of learning a genuinely robust correction behavior.
        train_samples = []
        for n in train_names:
            train_samples.extend(build_samples(scenarios[n], depth_mean, depth_std,
                                                 noise_std=noise_std, rng=rng))
        model.train()
        total_loss = 0.0
        for x_dyn, y in train_samples:
            x_dyn_t = torch.as_tensor(x_dyn, device=device)
            x_t = torch.cat([static_t, x_dyn_t], dim=1)
            y_t = torch.as_tensor(y, device=device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x_t, edge_attr_t, graph)
            if args.loss_weight_alpha > 0:
                per_node_sq_err = criterion_unreduced(pred, y_t)
                weight = 1.0 + args.loss_weight_alpha * torch.abs(y_t) / diff_std
                loss = (weight * per_node_sq_err).mean() / weight.mean()
            else:
                loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        train_loss = total_loss / len(train_samples)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_dyn, y in val_samples:
                x_dyn_t = torch.as_tensor(x_dyn, device=device)
                x_t = torch.cat([static_t, x_dyn_t], dim=1)
                y_t = torch.as_tensor(y, device=device).unsqueeze(1)
                pred = model(x_t, edge_attr_t, graph)
                val_loss += float(criterion(pred, y_t).item())
        val_loss /= max(len(val_samples), 1)
        best_val_loss = min(best_val_loss, val_loss)

        print(f"  epoch {epoch+1:3d}/{args.epochs}  train_loss={train_loss:.6e}  "
              f"val_loss(held-out scenarios)={val_loss:.6e}")

    print(f"\nDone. Best held-out validation loss: {best_val_loss:.6e}")
    print("This IS a genuine generalization test (2 full scenarios never seen during training, "
          "not just held-out timesteps from a seen event) -- unlike the earlier proof-of-concept, "
          "which only confirmed training plumbing on ONE event. Still limited: only 6 training "
          "scenarios (all short synthetic bursts except one longer one), all from ONE crop "
          "location -- real generalization to the full site3 domain and to the real multi-day "
          "Ian event's very different temporal profile remains untested until the next step "
          "(benchmark + run this trained model at full site3 scale).")

    ckpt_path = os.path.join(site_dir, args.out_name)
    torch.save({
        "model_state": model.state_dict(),
        "depth_mean": depth_mean, "depth_std": depth_std,
        "n_input_features": n_input_features,
        "train_names": train_names, "val_names": val_names,
        "best_val_loss": best_val_loss,
        "noise_std_frac": args.noise_std_frac, "noise_std": noise_std,
        "loss_weight_alpha": args.loss_weight_alpha,
    }, ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
