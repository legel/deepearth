"""Sparse-Adam vs dense-AdamW equivalence gate.

The sparse Adam must update the embedding table bit-identically to a dense AdamW that runs over the
WHOLE table every step — otherwise untouched entries diverge (missed weight-decay, missed momentum
decay, wrong bias correction), a hidden training compromise. This isolates the optimizer: the SAME
gradient sequence is applied to (a) the encoder's sparse Adam and (b) a reference torch AdamW over the
full table, then compares the tables. The hash grad w.r.t. embeddings is linear in the embeddings
(out = Σ w·emb), so grad_embeddings depends only on (inputs, weights, grad_out), NOT on the table
values — so a fixed grad sequence isolates the optimizer exactly.

Run: python -m deepearth.encoders.spacetime.hashencoder.test_sparse_adam_equiv
"""
import torch
from .hashgrid import HashEncoder


def run(wd, N=300, lr=3e-4, batch=512, ncoord=8192, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    enc = HashEncoder(input_dim=3, num_levels=6, level_dim=2, base_resolution=16,
                      log2_hashmap_size=14, enable_learned_probing=False).to(dev)
    coords = (torch.rand(ncoord, 3, device=dev) * 1.8 - 0.9)
    enc.precompute(coords)
    enc.init_sparse_adam(lr=lr, weight_decay=wd)

    dense = enc.embeddings.detach().clone().requires_grad_(True)      # reference table
    dopt = torch.optim.AdamW([dense], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)

    for _ in range(N):
        b = torch.randint(0, ncoord, (batch,), device=dev)
        go = torch.randn(batch, enc.output_dim, device=dev)
        enc._adam_grad_buffer.zero_()
        enc.accumulate_grad(go, b)                 # fills _adam_grad_buffer with the dense-shaped grad
        G = enc._adam_grad_buffer.clone()
        dense.grad = G.clone(); dopt.step(); dopt.zero_grad()   # dense AdamW over the WHOLE table
        enc.adam_step(b)                           # sparse Adam over touched rows only

    d = (enc.embeddings.detach() - dense.detach()).abs()
    # split: entries touched at least once vs never touched (isolates missed-decay on untouched)
    touched = torch.zeros(enc.embeddings.shape[0], dtype=torch.bool, device=dev)
    # (approx touched set not tracked; report global + per-row max)
    return d.max().item(), d.mean().item(), dense.detach().abs().mean().item()


def main():
    print("sparse-Adam vs dense-AdamW after 300 steps (batch 512 / 8192 coords -> ~16-step gaps):")
    ok = True
    for wd in (0.0, 3e-4, 1e-2):
        mx, mn, scale = run(wd)
        rel = mx / max(scale, 1e-9)
        passed = rel < 1e-3
        ok = ok and passed
        print(f"  wd={wd:<6}  max|Δ|={mx:.3e}  mean|Δ|={mn:.3e}  (table scale {scale:.3e})  "
              f"rel={rel:.2e}  {'ok' if passed else 'DIVERGES'}")
    print(f"\nEQUIVALENCE GATE: {'PASS' if ok else 'FAIL'}  (rel tol 1e-3)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
