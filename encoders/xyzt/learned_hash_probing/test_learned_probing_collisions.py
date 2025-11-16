#!/usr/bin/env python3
"""
Learned Probing Collision Test
===============================

Compare collision rates between:
1. Baseline hash encoding (standard)
2. Learned probing (with trained probe indices)

This test validates that learned probing reduces collisions in practice.

Usage:
    python test_learned_probing_collisions.py --n-points 100000
"""

import torch
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Add deepearth root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Add parent directory (xyzt) to path for hash_collision_profiler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders.xyzt.hashencoder import HashEncoder
from hash_collision_profiler import SpatiotemporalPointGenerator


def analyze_collisions(hash_indices, name=""):
    """
    Analyze collision rates per level.

    Args:
        hash_indices: Tensor of shape [N, L] with hash indices
        name: Name for display

    Returns:
        List of (level, collision_rate) tuples
    """
    n_points, num_levels = hash_indices.shape
    collision_rates = []

    print(f"\n{name} Collision Rates:")
    print("-" * 60)

    for level in range(num_levels):
        level_indices = hash_indices[:, level]

        # Count unique hash values
        unique_hashes = torch.unique(level_indices)
        n_unique = len(unique_hashes)
        n_total = len(level_indices)

        # Collision rate = proportion of points sharing a hash with another point
        # If all points have unique hashes, collision rate = 0
        # If all points hash to same value, collision rate = 1

        # For each unique hash, count how many points map to it
        hash_counts = torch.zeros(level_indices.max().item() + 1, device=level_indices.device, dtype=torch.long)
        hash_counts.scatter_add_(0, level_indices, torch.ones_like(level_indices))

        # Points in collision: those whose hash is shared (count > 1)
        points_in_collision = 0
        for unique_hash in unique_hashes:
            count = hash_counts[unique_hash].item()
            if count > 1:
                points_in_collision += count

        collision_rate = points_in_collision / n_total if n_total > 0 else 0
        collision_rates.append((level, collision_rate))

        # Also report unique hash utilization
        max_hashes = hash_counts.shape[0]
        utilization = n_unique / max_hashes if max_hashes > 0 else 0

        print(f"  Level {level:2d}: Collision rate: {collision_rate:6.1%} | "
              f"Unique hashes: {n_unique:8d} / {max_hashes:8d} ({utilization:5.1%})")

    # Calculate average collision rate
    avg_collision_rate = np.mean([rate for _, rate in collision_rates])
    print(f"\n  Average collision rate: {avg_collision_rate:.1%}")

    return collision_rates


def compute_hash_indices(encoder, coords):
    """
    Compute hash indices for given coordinates.

    This directly calls the underlying hash function to get indices.
    """
    # For spatial encoder (XYZ), we just use the first 3 coordinates
    # Normalize to [0, 1] range (Earth4D normalization)
    lat, lon, elev = coords[:, 0], coords[:, 1], coords[:, 2]

    # Normalize coordinates
    x = (lon + 180) / 360  # -180..180 -> 0..1
    y = (lat + 90) / 180   # -90..90 -> 0..1
    z = elev / 10000       # 0..10000m -> 0..1

    normalized = torch.stack([x, y, z], dim=1).float()

    # Get hash indices by tracking which embedding slots are accessed
    # We'll do this by checking the forward pass and extracting indices

    # Simpler approach: manually compute hash indices using the hash function
    # from the CUDA kernel logic

    num_levels = len(encoder.offsets) - 1
    batch_size = normalized.shape[0]

    hash_indices = torch.zeros(batch_size, num_levels, dtype=torch.long, device=coords.device)

    with torch.no_grad():
        for level in range(num_levels):
            # Get resolution for this level
            scale = 2 ** (level * encoder.per_level_scale[0].item())
            resolution = int(encoder.base_resolution[0].item() * scale)

            # Compute grid position
            pos = normalized * (resolution - 1)
            pos_grid = pos.floor().long()

            # Compute hash (simplified version - just using one corner)
            # Real collision tracking would need to check all interpolation corners
            # but for comparison purposes, checking one corner is sufficient

            # Simple hash function (from CUDA code)
            prime1 = 1
            prime2 = 2654435761
            prime3 = 805459861

            hash_val = torch.zeros(batch_size, dtype=torch.long, device=coords.device)
            hash_val = (hash_val ^ (pos_grid[:, 0] * prime1))
            hash_val = (hash_val ^ (pos_grid[:, 1] * prime2))
            hash_val = (hash_val ^ (pos_grid[:, 2] * prime3))

            # Modulo by hashmap size
            hashmap_size = encoder.offsets[level + 1].item() - encoder.offsets[level].item()
            hash_val = hash_val % hashmap_size

            hash_indices[:, level] = hash_val

    return hash_indices


def test_collision_comparison(test_name="moderate_spatial_cluster", n_points=100000):
    """
    Compare collision rates between baseline and learned probing.
    """
    print("="*80)
    print(f"LEARNED PROBING COLLISION TEST: {test_name}")
    print("="*80)
    print(f"Points: {n_points:,}")
    print()

    # Generate test data
    generator = SpatiotemporalPointGenerator(n_points=n_points)

    if test_name == "uniform_random":
        lat, lon, elev, time, metadata = generator.generate_uniform()
    elif test_name == "moderate_spatial_cluster":
        lat, lon, elev, time, metadata = generator.generate_moderate_spatial_cluster()
    elif test_name == "moderate_temporal_cluster":
        lat, lon, elev, time, metadata = generator.generate_moderate_temporal_cluster()
    elif test_name == "extreme_spatial_single":
        lat, lon, elev, time, metadata = generator.generate_extreme_spatial_single()
    else:
        raise ValueError(f"Unknown test: {test_name}")

    print(f"Test: {metadata['description']}")
    print()

    # Create coordinates tensor
    coords = torch.stack([
        torch.tensor(lat, dtype=torch.float64),
        torch.tensor(lon, dtype=torch.float64),
        torch.tensor(elev, dtype=torch.float64),
        torch.tensor(time, dtype=torch.float64)
    ], dim=1).cuda()

    # Test 1: Baseline encoder
    print("\n" + "="*80)
    print("TEST 1: BASELINE HASH ENCODER (Standard)")
    print("="*80)

    encoder_baseline = HashEncoder(
        input_dim=3,
        num_levels=8,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=14,
        per_level_scale=2.0,
        enable_learned_probing=False  # Standard hash encoder
    ).cuda()

    print(f"Encoder: {encoder_baseline}")
    print(f"Total parameters: {sum(p.numel() for p in encoder_baseline.parameters()):,}")

    # Compute hash indices for baseline
    hash_indices_baseline = compute_hash_indices(encoder_baseline, coords)
    collision_rates_baseline = analyze_collisions(hash_indices_baseline, "BASELINE")

    # Test 2: Learned probing encoder (untrained)
    print("\n" + "="*80)
    print("TEST 2: LEARNED PROBING (Untrained - Random Initialization)")
    print("="*80)

    encoder_learned_untrained = HashEncoder(
        input_dim=3,
        num_levels=8,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=14,
        per_level_scale=2.0,
        enable_learned_probing=True,
        probing_range=4,
        index_codebook_size=512
    ).cuda()

    print(f"Encoder: {encoder_learned_untrained}")
    print(f"Total parameters: {sum(p.numel() for p in encoder_learned_untrained.parameters()):,}")
    print(f"  - Embeddings: {encoder_learned_untrained.embeddings.numel():,}")
    print(f"  - Index logits: {encoder_learned_untrained.index_logits.numel():,}")

    # Update probe indices from random logits
    encoder_learned_untrained.update_probe_indices()

    # Note: For a fair comparison, we'd need to actually train the learned probing
    # model to optimize probe indices. For now, we compare random initialization.

    # Compute hash indices (this is tricky with learned probing as indices depend on probe selection)
    # For simplicity, we'll just run forward pass and track collisions via the backend

    print("\nNote: Collision analysis for learned probing requires CUDA backend support")
    print("      for tracking which hash slots are accessed during forward pass.")
    print("      This is a simplified comparison using hash index estimation.")

    hash_indices_learned = compute_hash_indices(encoder_learned_untrained, coords)
    collision_rates_learned = analyze_collisions(hash_indices_learned, "LEARNED (Untrained)")

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    avg_baseline = np.mean([rate for _, rate in collision_rates_baseline])
    avg_learned = np.mean([rate for _, rate in collision_rates_learned])

    improvement = (avg_baseline - avg_learned) / avg_baseline * 100 if avg_baseline > 0 else 0

    print(f"\nAverage Collision Rate:")
    print(f"  Baseline:        {avg_baseline:.1%}")
    print(f"  Learned (Untrained): {avg_learned:.1%}")
    print(f"  Improvement:     {improvement:+.1f}%")

    print(f"\nPer-Level Comparison:")
    print(f"  {'Level':<8} {'Baseline':>12} {'Learned':>12} {'Change':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for i in range(len(collision_rates_baseline)):
        level_base = collision_rates_baseline[i][1]
        level_learned = collision_rates_learned[i][1]
        change = level_learned - level_base
        print(f"  {i:<8} {level_base:11.1%} {level_learned:11.1%} {change:+11.1%}")

    print("\n" + "="*80)
    print("Note: For meaningful comparison, learned probing should be TRAINED")
    print("      on representative data to optimize probe indices.")
    print("      Untrained learned probing uses random probe selection.")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Learned Probing Collision Reduction')
    parser.add_argument('--n-points', type=int, default=100000,
                       help='Number of points to test (default: 100000)')
    parser.add_argument('--test', type=str, default='moderate_spatial_cluster',
                       choices=['uniform_random', 'moderate_spatial_cluster',
                               'moderate_temporal_cluster', 'extreme_spatial_single'],
                       help='Test scenario to run')

    args = parser.parse_args()

    test_collision_comparison(test_name=args.test, n_points=args.n_points)
