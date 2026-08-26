"""
Benchmark ONE MeshGraphKAN forward+backward pass on the REAL site3_crop mesh graph
======================================================================================
The previous proof-of-concept (train_mesh_gnn.py) trained on a deliberately coarse ~5,000-6,000
node grid, specifically for training speed. site3_crop's real mesh is 710,302 triangles /
1,065,089 edges -- ~120x more nodes than that proof-of-concept. Rather than assume this needs
coarsening (more engineering, another real approximation), benchmark ONE real forward+backward
pass on the actual graph first -- same "measure before committing" discipline as every other
step (the GPU solver port, the training-scenario count, etc).

Usage:
    .venv/bin/python3 simulation/benchmark_gnn_forward.py
"""
import os, sys, time, pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN  # noqa: E402

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site3_crop_coarse")
    ap.add_argument("--inference-only", action="store_true",
                     help="forward pass only, no backward/optimizer step -- the real question "
                          "for a DEPLOYED surrogate (training tractability was already answered "
                          "separately); typically much cheaper than a full train step")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    MESH_CKPT = os.path.join(PROJ_DIR, "site3_gee_creek", "gnn_training", args.site,
                              "checkpoints", "mesh_and_masks.pkl")

    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"mesh checkpoint: {MESH_CKPT}")

    with open(MESH_CKPT, "rb") as fh:
        mesh, gim, lm, gh, gng = pickle.load(fh)
    T = mesh["T"]
    ei = mesh["edges"]["i"].astype(np.int64)
    ej = mesh["edges"]["j"].astype(np.int64)
    n_edges = len(ei)
    print(f"mesh: {T:,} nodes (triangles), {n_edges:,} edges")

    ei_t = torch.as_tensor(ei, device=device)
    ej_t = torch.as_tensor(ej, device=device)
    graph = Data(edge_index=torch.stack([ei_t, ej_t]))

    # Same feature-count convention as train_mesh_gnn.py: 10 static + 2 forcing + 2*N_TIME_STEPS
    # dynamic (depth+volume windows). N_TIME_STEPS=2 -> 10+2+4=16 node features, 3 edge features.
    n_input_features = 16
    n_edge_features = 3
    model = MeshGraphKAN(input_dim_nodes=n_input_features, input_dim_edges=n_edge_features,
                          output_dim=2, processor_size=6, hidden_dim_processor=64,
                          hidden_dim_node_encoder=64, hidden_dim_edge_encoder=64,
                          hidden_dim_node_decoder=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params")

    x = torch.randn(T, n_input_features, device=device)
    edge_attr = torch.randn(n_edges, n_edge_features, device=device)
    y = torch.randn(T, 2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    mode = "forward-only (inference)" if args.inference_only else "forward+backward (training)"
    print(f"\nWarming up (1 pass, not timed — first call includes graph/kernel compilation) — {mode} …")
    if args.inference_only:
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_attr, graph)
    else:
        optimizer.zero_grad()
        pred = model(x, edge_attr, graph)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()

    print(f"Timing 5 real passes ({mode}) …")
    times = []
    for i in range(5):
        t0 = time.time()
        if args.inference_only:
            with torch.no_grad():
                pred = model(x, edge_attr, graph)
        else:
            optimizer.zero_grad()
            pred = model(x, edge_attr, graph)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"  pass {i+1}: {dt:.3f}s")

    avg = sum(times) / len(times)
    print(f"\navg per {mode} pass: {avg:.3f}s")
    if args.inference_only:
        # A real rollout at this scale needs ~1 inference call per solver frame -- estimate
        # against the same frame-count/duration conventions used elsewhere in this project
        # (e.g. the real Ian run: 73 saved frames at 60-min intervals over 72h).
        for n_frames in [73, 300, 1000]:
            total_s = avg * n_frames
            print(f"  {n_frames} rollout frames: {total_s:.0f}s ({total_s/60:.1f} min)")
    else:
        for n_samples in [200, 500, 1200, 2000]:
            total_s = avg * n_samples
            print(f"  {n_samples} samples x 1 epoch: {total_s:.0f}s ({total_s/60:.1f} min)")
        print(f"  1200 samples x 20 epochs: {avg*1200*20:.0f}s ({avg*1200*20/3600:.2f} hours)")


if __name__ == "__main__":
    main()
