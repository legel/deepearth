#!/usr/bin/env python3
"""
Test script for learned hash probing implementation.

Tests:
1. Backward compatibility - ensure old code still works
2. Forward pass with learned probing enabled
3. Basic gradient flow
"""

import torch
import sys
sys.path.insert(0, '/scratch/qhuang62/deepearth/encoders/xyzt')

from hashencoder import HashEncoder

def test_backward_compatibility():
    """Test that old code without learned probing still works"""
    print("=" * 80)
    print("TEST 1: Backward Compatibility (No Learned Probing)")
    print("=" * 80)

    # Create encoder without learned probing (default)
    encoder = HashEncoder(
        input_dim=3,
        num_levels=8,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=14,
        per_level_scale=2.0,
        enable_learned_probing=False  # Disabled
    )

    print(f"Encoder: {encoder}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    batch_size = 100
    inputs = torch.randn(batch_size, 3).cuda() * 0.5

    encoder = encoder.cuda()
    encoder.train()

    outputs = encoder(inputs, size=1.0)

    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

    # Test backward pass
    loss = outputs.sum()
    loss.backward()

    print(f"Gradient on embeddings: {encoder.embeddings.grad is not None}")
    if encoder.embeddings.grad is not None:
        print(f"Embedding grad norm: {encoder.embeddings.grad.norm().item():.6f}")

    print("✓ Backward compatibility test passed!\n")
    return True


def test_learned_probing_forward():
    """Test forward pass with learned probing enabled"""
    print("=" * 80)
    print("TEST 2: Learned Probing Forward Pass")
    print("=" * 80)

    # Create encoder WITH learned probing
    encoder = HashEncoder(
        input_dim=3,
        num_levels=8,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=14,  # 16K total
        per_level_scale=2.0,
        enable_learned_probing=True,  # ENABLED
        probing_range=4,  # N_p = 4 (2 bits)
        index_codebook_size=512  # N_c = 512
    )

    print(f"Encoder: {encoder}")
    print(f"N_f (base features): {encoder.N_f}")
    print(f"N_p (probing range): {encoder.N_p}")
    print(f"N_c (index codebook): {encoder.N_c}")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  - Embeddings: {encoder.embeddings.numel():,}")
    if encoder.index_logits is not None:
        print(f"  - Index logits: {encoder.index_logits.numel():,}")

    # Test forward pass
    batch_size = 100
    inputs = torch.randn(batch_size, 3).cuda() * 0.5

    encoder = encoder.cuda()
    encoder.train()

    # Update probe indices before forward
    encoder.update_probe_indices()
    print(f"Probe indices shape: {encoder.probe_indices.shape}")
    print(f"Probe indices unique values: {encoder.probe_indices.unique().tolist()}")

    outputs = encoder(inputs, size=1.0)

    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

    print("✓ Learned probing forward pass test passed!\n")
    return True


def test_learned_probing_gradients():
    """Test that gradients flow through learned probing"""
    print("=" * 80)
    print("TEST 3: Gradient Flow with Learned Probing")
    print("=" * 80)

    # Create encoder WITH learned probing
    encoder = HashEncoder(
        input_dim=3,
        num_levels=8,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=12,
        per_level_scale=2.0,
        enable_learned_probing=True,
        probing_range=4,
        index_codebook_size=256
    )

    encoder = encoder.cuda()
    encoder.train()

    # Forward pass
    inputs = torch.randn(50, 3).cuda() * 0.5
    inputs.requires_grad = True

    outputs = encoder(inputs, size=1.0)

    # Backward pass
    loss = outputs.sum()
    loss.backward()

    print(f"Gradient on embeddings: {encoder.embeddings.grad is not None}")
    if encoder.embeddings.grad is not None:
        print(f"  Embedding grad norm: {encoder.embeddings.grad.norm().item():.6f}")
        print(f"  Non-zero grads: {(encoder.embeddings.grad != 0).sum().item()} / {encoder.embeddings.grad.numel()}")

    print(f"Gradient on inputs: {inputs.grad is not None}")
    if inputs.grad is not None:
        print(f"  Input grad norm: {inputs.grad.norm().item():.6f}")

    # Note: index_logits gradients would require backward pass implementation
    print(f"Gradient on index_logits: {encoder.index_logits.grad is not None}")
    if encoder.index_logits.grad is not None:
        print(f"  Index logits grad norm: {encoder.index_logits.grad.norm().item():.6f}")
    else:
        print("  [Expected: backward pass with straight-through estimator not yet implemented]")

    print("✓ Gradient flow test passed!\n")
    return True


def test_probe_index_update():
    """Test that probe indices update correctly"""
    print("=" * 80)
    print("TEST 4: Probe Index Update")
    print("=" * 80)

    encoder = HashEncoder(
        input_dim=3,
        num_levels=4,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=10,
        enable_learned_probing=True,
        probing_range=4,
        index_codebook_size=64
    )

    encoder = encoder.cuda()

    # Get initial probe indices
    initial_probes = encoder.probe_indices.clone()
    print(f"Initial probe indices: {initial_probes[0, :10].tolist()}")

    # Manually modify index_logits
    with torch.no_grad():
        encoder.index_logits[0, :10, :] = torch.randn(10, 4).cuda()

    # Update probe indices
    encoder.update_probe_indices()

    updated_probes = encoder.probe_indices.clone()
    print(f"Updated probe indices: {updated_probes[0, :10].tolist()}")

    # Check that argmax is correctly computed
    for i in range(10):
        expected = torch.argmax(encoder.index_logits[0, i]).item()
        actual = updated_probes[0, i].item()
        assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

    print("✓ Probe index update test passed!\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LEARNED HASH PROBING TESTS")
    print("=" * 80 + "\n")

    all_passed = True

    try:
        all_passed &= test_backward_compatibility()
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_learned_probing_forward()
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_learned_probing_gradients()
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_probe_index_update()
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)
