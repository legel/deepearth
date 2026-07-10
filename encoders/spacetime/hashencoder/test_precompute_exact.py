"""Non-compromising precompute gate.

The precomputed forward+backward must EXACTLY match the standard forward+backward at the
same resolution — for the output and for every trainable grad (embeddings, per_level_scale,
index_logits). If per_level_scale.grad differs, the precompute path has frozen the resolution
(a compromise) and the gate fails.

Run: python -m deepearth.encoders.spacetime.hashencoder.test_precompute_exact
"""
import torch
from .hashgrid import HashEncoder


def _grads(enc, inputs, target, precomputed):
    for p in (enc.embeddings, enc.per_level_scale, enc.index_logits):
        if p is not None and p.grad is not None:
            p.grad = None
    enc.train()
    if precomputed:
        enc.precompute(inputs)
        out = enc.forward_precomputed()
    else:
        out = enc(inputs)
    (out * target).sum().backward()
    g = {"out": out.detach().clone(),
         "emb": enc.embeddings.grad.detach().clone(),
         "pls": None if enc.per_level_scale.grad is None else enc.per_level_scale.grad.detach().clone(),
         "idx": None if (enc.index_logits is None or enc.index_logits.grad is None) else enc.index_logits.grad.detach().clone()}
    return g


def _rep(name, a, b):
    if b is None:
        print(f"  {name:5s}  FROZEN (precompute returned no grad)  [std norm {a.norm():.3e}]")
        return False
    d = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    print(f"  {name:5s}  max|Δ| {d:.3e}  allclose={ok}  [norm {a.norm():.3e}]")
    return ok


def main():
    torch.manual_seed(0)
    dev = "cuda"
    enc = HashEncoder(input_dim=3, num_levels=8, level_dim=2, base_resolution=16,
                      log2_hashmap_size=16, enable_learned_probing=True,
                      probing_range=4, index_codebook_size=256).to(dev)
    with torch.no_grad():
        enc.per_level_scale.add_(0.1 * torch.randn_like(enc.per_level_scale))
    B = 4096
    inputs = (torch.rand(B, 3, device=dev) * 1.8 - 0.9)
    target = torch.randn(B, enc.output_dim, device=dev)

    # --- Test A: same resolution (precompute at the resolution we forward at) ---
    std = _grads(enc, inputs, target, precomputed=False)
    pre = _grads(enc, inputs, target, precomputed=True)
    print("A) standard vs precomputed, SAME resolution:")
    okA = all([_rep("out", std["out"], pre["out"]), _rep("emb", std["emb"], pre["emb"]),
               _rep("pls", std["pls"], pre["pls"]), _rep("idx", std["idx"], pre["idx"])])

    # --- Test B: drift. Precompute at S, then move per_level_scale by a small delta (no cell
    # crossing), forward — the precomputed path must recompute continuous terms from the CURRENT
    # scale and still match the standard forward at the drifted scale. ---
    enc.precompute(inputs)                                    # cache discrete structure at S
    with torch.no_grad():
        enc.per_level_scale.add_(1e-3 * torch.randn_like(enc.per_level_scale))   # drift
    for p in (enc.embeddings, enc.per_level_scale, enc.index_logits):
        if p is not None:
            p.grad = None
    out_pre = enc.forward_precomputed(); (out_pre * target).sum().backward()
    driftpre = {"out": out_pre.detach().clone(), "emb": enc.embeddings.grad.clone(),
                "pls": None if enc.per_level_scale.grad is None else enc.per_level_scale.grad.clone(),
                "idx": None if enc.index_logits.grad is None else enc.index_logits.grad.clone()}
    driftstd = _grads(enc, inputs, target, precomputed=False)   # standard at the drifted scale
    print("\nB) drifted resolution (precompute at S, forward at S+δ):")
    okB = all([_rep("out", driftstd["out"], driftpre["out"]), _rep("emb", driftstd["emb"], driftpre["emb"]),
               _rep("pls", driftstd["pls"], driftpre["pls"]), _rep("idx", driftstd["idx"], driftpre["idx"])])

    allok = okA and okB
    print(f"\nGATE: {'PASS' if allok else 'FAIL'}  (A same-scale={okA}, B drift={okB})")
    return allok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
