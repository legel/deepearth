#!/usr/bin/env python3
"""
Gradient Flow Diagnostic
========================

Detailed analysis of gradient flow through learned probing parameters.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from encoders.xyzt.hashencoder import HashEncoder


def test_gradient_flow():
    print("="*80)
    print("GRADIENT FLOW DIAGNOSTIC")
    print("="*80)
    print()

    # Create small encoder for detailed analysis
    encoder = HashEncoder(
        input_dim=3,
        num_levels=4,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=10,  # Smaller for easier analysis
        per_level_scale=2.0,
        enable_learned_probing=True,
        probing_range=4,
        index_codebook_size=128
    ).cuda()

    print(f"Encoder configuration:")
    print(f"  Num levels: {encoder.num_levels}")
    print(f"  N_f: {encoder.N_f}")
    print(f"  N_p: {encoder.N_p}")
    print(f"  N_c: {encoder.N_c}")
    print(f"  Embeddings shape: {encoder.embeddings.shape}")
    print(f"  Index logits shape: {encoder.index_logits.shape}")
    print()

    # Create sample inputs
    inputs = torch.rand(100, 3, device='cuda') * 2 - 1
    targets = torch.randn(100, encoder.output_dim, device='cuda')

    # Forward pass
    outputs = encoder(inputs, size=1.0)
    loss = torch.nn.functional.mse_loss(outputs, targets)

    print(f"Forward pass:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Outputs shape: {outputs.shape}")
    print(f"  Loss: {loss.item():.6f}")
    print()

    # Backward pass
    loss.backward()

    # Analyze gradients
    print("Gradient Analysis:")
    print("-" * 80)

    # Embeddings
    if encoder.embeddings.grad is not None:
        embed_grad = encoder.embeddings.grad
        print(f"Embeddings gradient:")
        print(f"  Shape: {embed_grad.shape}")
        print(f"  Norm: {embed_grad.norm().item():.10f}")
        print(f"  Mean: {embed_grad.mean().item():.10f}")
        print(f"  Std:  {embed_grad.std().item():.10f}")
        print(f"  Min:  {embed_grad.min().item():.10f}")
        print(f"  Max:  {embed_grad.max().item():.10f}")
        print(f"  Non-zero elements: {(embed_grad != 0).sum().item()} / {embed_grad.numel()}")
    else:
        print(f"Embeddings gradient: None (ERROR!)")
    print()

    # Index logits
    if encoder.index_logits.grad is not None:
        logit_grad = encoder.index_logits.grad
        print(f"Index logits gradient:")
        print(f"  Shape: {logit_grad.shape}")
        print(f"  Norm: {logit_grad.norm().item():.10f}")
        print(f"  Mean: {logit_grad.mean().item():.10f}")
        print(f"  Std:  {logit_grad.std().item():.10f}")
        print(f"  Min:  {logit_grad.min().item():.10f}")
        print(f"  Max:  {logit_grad.max().item():.10f}")
        print(f"  Non-zero elements: {(logit_grad != 0).sum().item()} / {logit_grad.numel()}")

        # Analyze per-level gradients
        print()
        print(f"  Per-level gradient norms:")
        for level in range(encoder.num_levels):
            level_grad_norm = logit_grad[level].norm().item()
            print(f"    Level {level}: {level_grad_norm:.10f}")
    else:
        print(f"Index logits gradient: None (ERROR!)")
    print()

    # Check requires_grad
    print("Parameter status:")
    print(f"  embeddings.requires_grad: {encoder.embeddings.requires_grad}")
    print(f"  index_logits.requires_grad: {encoder.index_logits.requires_grad}")
    print()

    print("="*80)
    if encoder.index_logits.grad is not None and encoder.index_logits.grad.norm().item() > 0:
        print("✓ GRADIENT FLOW IS WORKING CORRECTLY")
    else:
        print("✗ GRADIENT FLOW ISSUE DETECTED")
    print("="*80)


if __name__ == "__main__":
    test_gradient_flow()
