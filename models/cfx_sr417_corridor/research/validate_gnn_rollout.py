"""
Real autoregressive rollout validation of the trained MeshGraphKAN surrogate
=============================================================================
train_mesh_gnn_site3.py's "best held-out validation loss 6.38e-5" is a SINGLE-STEP loss: at
every timestep, the model was fed the REAL solver depth window and only asked to predict the
next real diff. That's a legitimate held-out-SCENARIO test (2 full rain events never seen during
training), but it does NOT test whether small per-step errors compound when the model has to
feed its OWN predictions back in as input over a full event -- the actual way a surrogate would
be used to replace the solver. This script does that real autoregressive rollout, on the same
2 held-out scenarios, and reports physically meaningful metrics (depth RMSE over time, final-
frame error, and total-volume drift -- the GNN has no built-in mass-conservation constraint the
way the physics solver does, so volume drift is a real, otherwise-invisible failure mode worth
checking explicitly).

Usage:
    .venv/bin/python3 simulation/validate_gnn_rollout.py --site site3_crop_coarse
"""
import os, sys, argparse, pickle
import numpy as np
import torch
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)

from mesh_shallow_water import synthetic_rain_rate_ms  # noqa: E402
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN  # noqa: E402
from train_mesh_gnn_site3 import build_static_features, build_edge_features, N_TIME_STEPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3_crop_coarse")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--checkpoint-name", default="model_checkpoint.pt")
    ap.add_argument("--out-name", default="rollout_validation.json")
    args = ap.parse_args()

    site_dir = os.path.join(PROJ_DIR, "site3_gee_creek", "gnn_training", args.site)
    mesh_ckpt = os.path.join(site_dir, "checkpoints", "mesh_and_masks.pkl")
    scenario_dir = os.path.join(site_dir, "scenarios")
    model_ckpt_path = os.path.join(site_dir, args.checkpoint_name)

    print("=" * 70)
    print(f"Autoregressive rollout validation — {args.site}")
    print("=" * 70)

    device = torch.device(args.device)
    ckpt = torch.load(model_ckpt_path, map_location=device)
    depth_mean, depth_std = ckpt["depth_mean"], ckpt["depth_std"]
    val_names = ckpt["val_names"]
    print(f"\nLoaded model_checkpoint.pt — best single-step held-out val loss: {ckpt['best_val_loss']:.6e}")
    print(f"Held-out scenarios to roll out: {val_names}")

    with open(mesh_ckpt, "rb") as fh:
        mesh, gim, lm, gh, gng = pickle.load(fh)
    T = mesh["T"]
    area = mesh["area"].astype(np.float32)
    static_feats = build_static_features(mesh, gim, lm, gh, gng)
    edge_attr_np = build_edge_features(mesh)

    ei_t = torch.as_tensor(mesh["edges"]["i"], dtype=torch.int64, device=device)
    ej_t = torch.as_tensor(mesh["edges"]["j"], dtype=torch.int64, device=device)
    edge_attr_t = torch.as_tensor(edge_attr_np, dtype=torch.float32, device=device)
    graph = Data(edge_index=torch.stack([ei_t, ej_t]))
    static_t = torch.as_tensor(static_feats, dtype=torch.float32, device=device)

    model = MeshGraphKAN(input_dim_nodes=ckpt["n_input_features"], input_dim_edges=3, output_dim=1,
                          processor_size=6, hidden_dim_processor=64,
                          hidden_dim_node_encoder=64, hidden_dim_edge_encoder=64,
                          hidden_dim_node_decoder=64).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    summary = {}
    for name in val_names:
        f = os.path.join(scenario_dir, f"{name}.pkl")
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        frames_h_real = [h.astype(np.float32) for h in d["frames_h"]]
        frame_times = d["frame_times"]
        sc = d["scenario"]
        n_frames = len(frames_h_real)

        # seed the rollout window with the REAL first N_TIME_STEPS frames (normalized)
        window_norm = [(h - depth_mean) / depth_std for h in frames_h_real[:N_TIME_STEPS]]
        rollout_depths_real_units = list(frames_h_real[:N_TIME_STEPS])

        rmses, real_vols, pred_vols = [], [], []
        with torch.no_grad():
            for t in range(N_TIME_STEPS, n_frames):
                rain = synthetic_rain_rate_ms(frame_times[t - 1], sc["rain_duration_min"] * 60, sc["peak_mm_hr"])
                depth_window = np.stack(window_norm[-N_TIME_STEPS:], axis=1)
                x_dyn = np.hstack([np.full((T, 1), rain, dtype=np.float32), depth_window])
                x_t = torch.cat([static_t, torch.as_tensor(x_dyn, device=device)], dim=1)
                pred_diff_norm = model(x_t, edge_attr_t, graph).squeeze(1).cpu().numpy()

                next_norm = window_norm[-1] + pred_diff_norm
                window_norm.append(next_norm)
                pred_depth_real = np.clip(next_norm * depth_std + depth_mean, 0.0, None)
                rollout_depths_real_units.append(pred_depth_real)

                real_depth = frames_h_real[t]
                rmse = float(np.sqrt(np.mean((pred_depth_real - real_depth) ** 2)))
                rmses.append(rmse)
                real_vols.append(float(np.sum(real_depth * area)))
                pred_vols.append(float(np.sum(pred_depth_real * area)))

        final_real = frames_h_real[-1]
        final_pred = rollout_depths_real_units[-1]
        final_max_err = float(np.max(np.abs(final_pred - final_real)))
        vol_drift_pct = 100.0 * (pred_vols[-1] - real_vols[-1]) / max(real_vols[-1], 1e-9)

        n_roll = len(real_vols)
        n_clipped_frac = [float(np.mean((w * depth_std + depth_mean) < 0)) for w in window_norm[N_TIME_STEPS::max(n_roll // 5, 1)]]

        print(f"\n--- {name} (peak {sc['peak_mm_hr']}mm/hr, {n_frames} frames) ---")
        print(f"  rollout RMSE: first-step {rmses[0]:.6f} m -> last-step {rmses[-1]:.6f} m "
              f"(mean over rollout: {np.mean(rmses):.6f} m)")
        print(f"  final-frame max abs error: {final_max_err:.6f} m "
              f"(real max depth this frame: {final_real.max():.4f} m)")
        print(f"  final volume: real={real_vols[-1]:.2f} m3, predicted={pred_vols[-1]:.2f} m3, "
              f"drift={vol_drift_pct:+.2f}%")
        print(f"  volume-drift trajectory (5 evenly-spaced rollout steps):")
        for idx in range(0, n_roll, max(n_roll // 5, 1)):
            d = 100.0 * (pred_vols[idx] - real_vols[idx]) / max(real_vols[idx], 1e-9)
            print(f"    step {idx:4d}/{n_roll}: real={real_vols[idx]:9.2f} m3  pred={pred_vols[idx]:9.2f} m3  drift={d:+7.2f}%")
        print(f"  fraction of pre-clip predicted depths that were negative (5 samples over rollout): "
              f"{[round(f, 4) for f in n_clipped_frac]}")

        summary[name] = dict(
            n_frames=n_frames, rmse_first=rmses[0], rmse_last=rmses[-1], rmse_mean=float(np.mean(rmses)),
            final_max_abs_error_m=final_max_err, final_real_max_depth_m=float(final_real.max()),
            final_real_vol_m3=real_vols[-1], final_pred_vol_m3=pred_vols[-1], vol_drift_pct=vol_drift_pct,
        )

    out_path = os.path.join(site_dir, args.out_name)
    import json
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
