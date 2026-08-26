"""
MeshGraphKAN surrogate — proof-of-concept training on OUR OWN Ian data
=========================================================================
Per the 2026-07-23/24 strategic reflection: the highest-value use of
HydroGraphNet's architecture for THIS project isn't reproducing NVIDIA's literal model on the
White River (Indiana) reference dataset — that geography/physics doesn't transfer here at all —
it's training a fast learned surrogate of OUR OWN from-scratch solver to accelerate future
calibration sweeps (Manning's n, Horton infiltration, impervious buffers) against the real
ground truth this project has been building (hydrograph_ian.csv + the gauge/routing comparisons).

This script reuses the REAL, confirmed-working MeshGraphKAN model class (verified importable/
runnable on MPS earlier this session — see the .venv setup log) and the same node/edge feature
CONCEPTS HydroGraphNet's own HydroGraphDataset uses (static terrain attributes + a windowed
history of depth/volume + current forcing), but builds them directly from our own in-memory
solver output — NOT their WD_/V_/Pr_/US_InF_ text-file schema, which is tightly coupled to their
own White River dataset layout and would be pure translation overhead with zero benefit here.

Deliberately a coarse-grid, small-scale PROOF OF CONCEPT, not a production surrogate:
  - Runs flood_sim_ian.py's own solver at a coarse cell size (default 30m, vs. the project's
    production 5m) to get a small (~5,000-6,000 node) grid — close to HydroGraphNet's own
    reference-case scale (4,787 nodes), fast enough to train interactively on this machine.
  - Uses this AOI's real 4-connected grid adjacency for edges (not an approximate k-NN graph —
    a real regular grid actually HAS exact neighbor structure, unlike HydroGraphNet's own
    irregular unstructured-mesh case, so there's no reason to approximate it).
  - Slope/aspect are computed directly from the coarse z grid (simple centered-difference
    gradient); curvature is NOT computed (set to 0) — a real simplification, not a hidden one.
  - Training data is ONE real event (Hurricane Ian) sliced into many overlapping sliding-window
    samples in time — real temporal diversity, but still only one storm's spatial pattern. This
    is nowhere close to enough data for a genuinely generalizing model; the goal here is
    confirming the training PLUMBING works end-to-end (loss decreases, forward/backward pass
    correct), not claiming a validated surrogate.

Usage:
    python3 simulation/train_mesh_gnn.py --cell-size 30 --epochs 20
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)

from flood_sim_ian import (  # noqa: E402
    load_dem_for_sim, load_ian_hyetograph, load_spatial_horton, apply_impervious_mask,
    apply_nlcd_graded_impervious, run_sim, HORTON, MANNING_N,
)

from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN  # noqa: E402

N_TIME_STEPS = 2   # sliding-window length, matches HydroGraphNet's own default (conf/config.yaml)


def build_grid_graph(nrows, ncols, cell_size_m, z):
    """Real 4-connected grid adjacency (up/down/left/right) — exact neighbor structure for a
    regular grid, not an approximation the way k-NN would be on an irregular point set."""
    node_id = np.arange(nrows * ncols).reshape(nrows, ncols)
    edges_i, edges_j = [], []
    # Horizontal edges
    edges_i.append(node_id[:, :-1].ravel()); edges_j.append(node_id[:, 1:].ravel())
    edges_i.append(node_id[:, 1:].ravel());  edges_j.append(node_id[:, :-1].ravel())
    # Vertical edges
    edges_i.append(node_id[:-1, :].ravel()); edges_j.append(node_id[1:, :].ravel())
    edges_i.append(node_id[1:, :].ravel());  edges_j.append(node_id[:-1, :].ravel())
    ei = np.concatenate(edges_i)
    ej = np.concatenate(edges_j)

    xy = np.stack(np.meshgrid(np.arange(ncols) * cell_size_m,
                              np.arange(nrows) * cell_size_m), axis=-1).reshape(-1, 2)
    rel = xy[ei] - xy[ej]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    edge_attr = np.hstack([rel, dist]).astype(np.float32)   # 3 features, same convention as
                                                              # HydroGraphNet's create_edge_features
    return ei.astype(np.int64), ej.astype(np.int64), edge_attr


def compute_slope_aspect(z, cell_size_m):
    """Simple centered-difference gradient — real slope/aspect, not a placeholder, just a
    simpler computation than a full richdem terrain-analysis pass (not needed at this coarse
    resolution / for this proof-of-concept's purposes)."""
    dzdy, dzdx = np.gradient(z, cell_size_m)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)
    aspect = np.arctan2(dzdy, dzdx)
    return slope, aspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-size", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=20.0)
    ap.add_argument("--frame-interval", type=float, default=10.0, help="minutes")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    print("=" * 70)
    print("MeshGraphKAN proof-of-concept training — our own Hurricane Ian data")
    print("=" * 70)

    print(f"\n[1/4] Running solver at coarse {args.cell_size:.0f}m cell size for a small, "
          f"fast training grid …")
    z, profile, dx = load_dem_for_sim(args.cell_size)
    nrows, ncols = z.shape
    valid = np.isfinite(z)
    print(f"  Grid: {nrows}x{ncols} = {nrows*ncols} nodes ({valid.sum()} valid)")

    horton_arrays = load_spatial_horton(z.shape, profile["transform"], profile["crs"])
    if horton_arrays is not None:
        horton_arrays = apply_impervious_mask(horton_arrays, z.shape, profile["transform"], profile["crs"])
        horton_arrays = apply_nlcd_graded_impervious(horton_arrays, z.shape, profile["transform"], profile["crs"])

    rain_sim, hours, rain_mm = load_ian_hyetograph(args.dt)

    print(f"\n[2/4] Running Bates solver ({len(rain_sim)} steps) to generate training frames …")
    h_max, cum_infil, flooded_ha_ts, rain_ts, Pe_ts, mean_depth_ts, frame_data = run_sim(
        z, dx, rain_sim, args.dt, frame_interval_min=args.frame_interval,
        use_infiltration=horton_arrays is not None, horton_arrays=horton_arrays, verbose=False,
    )
    frames_h = frame_data["frames"]       # list of (nrows,ncols) depth snapshots
    frame_times_min = frame_data["times_min"]
    n_frames = len(frames_h)
    print(f"  {n_frames} frames saved @ {args.frame_interval:.0f}min interval")

    print(f"\n[3/4] Building graph + static/dynamic features …")
    ei, ej, edge_attr = build_grid_graph(nrows, ncols, dx, z)
    slope, aspect = compute_slope_aspect(z, dx)
    cell_area = dx * dx

    fc_arr = (horton_arrays["fc"] if horton_arrays is not None
              else np.full(z.shape, HORTON["fc"] / 1000 / 3600))

    def standardize(a):
        a = a.astype(np.float32).ravel()
        return (a - a.mean()) / (a.std() + 1e-8)

    static_feats = np.stack([
        standardize(np.tile(np.arange(ncols) * dx, (nrows, 1))),   # x
        standardize(np.tile((np.arange(nrows) * dx)[:, None], (1, ncols))),  # y
        standardize(np.full(z.shape, cell_area)),                  # area
        standardize(z),                                            # elevation
        standardize(slope),                                        # slope
        standardize(aspect),                                       # aspect
        standardize(np.zeros(z.shape)),                            # curvature (not computed
                                                                     # — see module docstring)
        standardize(np.full(z.shape, MANNING_N)),                  # manning
        standardize(np.zeros(z.shape)),                            # flow_accum (not computed
                                                                     # at this coarse res)
        standardize(fc_arr),                                       # infiltration capacity
    ], axis=1)   # (N, 10) — matches HydroGraphNet's own static-feature COUNT (xy+area+elev+
                 # slope+aspect+curv+manning+flow_accum+infiltration = 10, they then add 2 more
                 # "current-step forcing" columns per sample below, exactly like their scheme)

    depths_raw = [h.astype(np.float32).ravel() for h in frames_h]
    volumes_raw = [d * cell_area for d in depths_raw]
    # Normalize depth/volume to zero-mean/unit-std BEFORE computing target diffs, exactly like
    # HydroGraphNet's own HydroGraphDataset does (see its process()/__getitem__: dynamic_data
    # stores already-standardized water_depth/volume, and target_depth/target_volume are diffs
    # of THOSE normalized arrays, not raw ones). Fixes a real scaling bug found while testing
    # this script: volume (depth*cell_area, area~900m^2 at 30m cells) is orders of magnitude
    # larger than depth, so an unnormalized MSE loss was dominated entirely by the volume term
    # and the model wasn't converging (loss oscillated ~1.28-1.32 over 20 epochs, not decreasing).
    depth_all = np.concatenate(depths_raw)
    vol_all = np.concatenate(volumes_raw)
    depth_mean, depth_std = float(depth_all.mean()), float(depth_all.std() + 1e-8)
    vol_mean, vol_std = float(vol_all.mean()), float(vol_all.std() + 1e-8)
    depths = [(d - depth_mean) / depth_std for d in depths_raw]
    volumes = [(v - vol_mean) / vol_std for v in volumes_raw]
    rain_at_frame = np.interp(frame_times_min, np.arange(len(rain_ts)) * args.dt / 60, rain_ts)

    print(f"\n[4/4] Building sliding-window training samples (n_time_steps={N_TIME_STEPS}) …")
    samples = []
    for t in range(n_frames - N_TIME_STEPS - 1):
        depth_window = np.stack(depths[t:t + N_TIME_STEPS], axis=1)     # (N, n_time_steps)
        vol_window = np.stack(volumes[t:t + N_TIME_STEPS], axis=1)
        current_rain = np.full((depth_window.shape[0], 1), rain_at_frame[t + N_TIME_STEPS - 1],
                                dtype=np.float32)
        current_inflow = np.zeros_like(current_rain)   # no upstream river inflow for this
                                                          # closed AOI — see the module docstring
        x = np.hstack([static_feats, current_inflow, current_rain, depth_window, vol_window])
        target_depth = depths[t + N_TIME_STEPS] - depths[t + N_TIME_STEPS - 1]
        target_vol = volumes[t + N_TIME_STEPS] - volumes[t + N_TIME_STEPS - 1]
        y = np.stack([target_depth, target_vol], axis=1)
        samples.append((x.astype(np.float32), y.astype(np.float32)))
    print(f"  {len(samples)} training samples from this one event")

    device = torch.device(args.device)
    ei_t = torch.as_tensor(ei, dtype=torch.int64, device=device)
    ej_t = torch.as_tensor(ej, dtype=torch.int64, device=device)
    edge_attr_t = torch.as_tensor(edge_attr, dtype=torch.float32, device=device)
    graph = Data(edge_index=torch.stack([ei_t, ej_t]))

    n_input_features = samples[0][0].shape[1]
    model = MeshGraphKAN(input_dim_nodes=n_input_features, input_dim_edges=3, output_dim=2,
                          processor_size=6, hidden_dim_processor=64,
                          hidden_dim_node_encoder=64, hidden_dim_edge_encoder=64,
                          hidden_dim_node_decoder=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"\nTraining on {device} — {sum(p.numel() for p in model.parameters()):,} params, "
          f"{n_input_features} node features, {len(samples)} samples, {args.epochs} epochs\n")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for x, y in samples:
            x_t = torch.as_tensor(x, device=device)
            y_t = torch.as_tensor(y, device=device)
            optimizer.zero_grad()
            pred = model(x_t, edge_attr_t, graph)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / len(samples)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  avg_loss={avg_loss:.6e}")

    print("\nDone. This confirms the training PLUMBING works end-to-end (real forward/backward "
          "passes through the actual MeshGraphKAN architecture, on our own real solver output) — "
          "it is NOT a claim of a validated, generalizing surrogate: all training samples come "
          "from one single event's spatial pattern, far short of what a GNN normally needs to "
          "generalize across different storms.")


if __name__ == "__main__":
    main()
