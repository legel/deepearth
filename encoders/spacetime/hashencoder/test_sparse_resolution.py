"""Sparse-path resolution-training gate.

The champion trains the absolute encoder through a DETACHED precomputed leaf + sparse Adam (so it never materializes
the dense ~200M-param embedding gradient). The detach also discards the per_level_scale gradient the autograd Function
would produce, so resolution would stay frozen unless the sparse step rebuilds it from dy_dx. This gate proves the wiring:

  (a) every sparse step's loss is finite;
  (b) after the first backward + sparse_hash_step, each absolute sub-encoder's per_level_scale.grad is non-None, nonzero;
  (c) per_level_scale VALUES change from step 0 to step 5 -- resolution is actually training.

Run: python -m deepearth.encoders.spacetime.hashencoder.test_sparse_resolution
"""
import torch
from deepearth.core.fusion import DeepEarth, Variable


def main():
    torch.manual_seed(0)
    dev = "cuda"
    variables = [Variable("a", "continuous", dim=8), Variable("b", "continuous", dim=8)]
    model = DeepEarth(variables, d_model=64, n_latents=8, n_layers=2, n_heads=4).to(dev)
    model.train()

    # Fixed coordinate set (lat, lon, elev, time) -> ECEF-normalized inside Earth4D
    N = 8192
    lat = torch.rand(N, device=dev) * 120 - 60
    lon = torch.rand(N, device=dev) * 240 - 120
    elev = torch.rand(N, device=dev) * 2000
    tt = torch.rand(N, device=dev)
    coords = torch.stack([lat, lon, elev, tt], dim=-1)

    model.enable_sparse_hash(coords, lr=1e-2)
    abs_encs = model._abs_encs
    hash_mods = [m for m in model.modules() if hasattr(m, "clamp_per_level_scale")]

    freq_params = [p for n, p in model.named_parameters() if "per_level_scale" in n]
    other = list(model.absolute_proj.parameters())
    opt = torch.optim.AdamW([{"params": other, "lr": 1e-3},
                             {"params": freq_params, "lr": 1e-1}], weight_decay=0.0)

    target = torch.randn(N, model.d_model, device=dev)
    pls0 = [en.per_level_scale.detach().clone() for en in abs_encs]

    B = 1024
    ok = True
    for step in range(5):
        idx = torch.randint(0, N, (B,), device=dev)
        flat = model.read_absolute_leaf(idx)                 # detached leaf; stashes dy_dx/inputs
        pos = model.absolute_proj(flat)                      # gradient path into the leaf
        loss = (pos - target[idx]).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.sparse_hash_step(flat, idx)                    # sparse Adam on embeddings + build per_level_scale grad

        finite = bool(torch.isfinite(loss))
        if not finite:
            ok = False
        if step == 0:
            for j, en in enumerate(abs_encs):
                gr = en.per_level_scale.grad
                gnorm = None if gr is None else gr.norm().item()
                good = gr is not None and gnorm is not None and gnorm > 0.0
                print(f"  step0 sub-encoder[{j}] per_level_scale.grad: "
                      f"{'set' if gr is not None else 'None'} norm={gnorm}  -> {'OK' if good else 'FAIL'}")
                if not good:
                    ok = False

        torch.nn.utils.clip_grad_norm_(freq_params, 2.0)
        opt.step()
        for m in hash_mods:
            m.clamp_per_level_scale()
        print(f"  step {step} loss {float(loss):.4f} finite={finite}")

    # (c) resolution values actually moved
    print("per_level_scale value drift (max |Δ| over 5 steps):")
    for j, (en, p0) in enumerate(zip(abs_encs, pls0)):
        d = (en.per_level_scale.detach() - p0).abs().max().item()
        moved = d > 0.0
        print(f"  sub-encoder[{j}]  max|Δ pls| = {d:.3e}  -> {'moved' if moved else 'FROZEN'}")
        if not moved:
            ok = False

    print(f"\nSPARSE-RESOLUTION GATE: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
