#!/usr/bin/env python3
"""
Earth4D Comprehensive Test Suite
=================================

Tests Earth4D encoder at various resolutions and configurations, including:
- Memory profiling
- Resolution discrimination
- Training convergence
- Multi-scale performance

Usage:
    # Quick test with defaults (0.5m spatial, 1hr temporal)
    python earth4d_test.py

    # Custom resolution test
    python earth4d_test.py --spatial-levels 20 --temporal-levels 16 --iterations 100

    # Memory stress test
    python earth4d_test.py --mode memory --batch-size 10000

    # Resolution discrimination test
    python earth4d_test.py --mode discrimination
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import gc
import sys
import os
from typing import Dict, Optional, Tuple
from enum import Enum

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from earth4d import Earth4D
from earth_system_dataset import EarthSystemDataset, DataSplit
from high_res_earth_dataset import HighResEarthDataset


class TestMode(Enum):
    """Test modes available."""
    FULL = "full"
    QUICK = "quick"
    MEMORY = "memory"
    DISCRIMINATION = "discrimination"
    TRAINING = "training"


class Earth4DModel(nn.Module):
    """Earth4D model with MLP head for testing."""

    def __init__(self,
                 spatial_levels: int = 36,
                 temporal_levels: int = 20,
                 spatial_hashmap_size: int = 22,
                 temporal_hashmap_size: int = 18,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 verbose: bool = True):
        super().__init__()

        self.encoder = Earth4D(
            spatial_levels=spatial_levels,
            temporal_levels=temporal_levels,
            spatial_log2_hashmap_size=spatial_hashmap_size,
            temporal_log2_hashmap_size=temporal_hashmap_size,
            verbose=verbose
        )

        # Get output dimension
        encoder_dim = self.encoder.get_output_dim()

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.encoder(coords)
        output = self.head(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def profile_memory(device: str = 'cuda') -> Dict:
    """Profile current memory usage."""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}


def test_memory_scaling(args: argparse.Namespace) -> Dict:
    """Test memory requirements at different scales."""
    print("\n" + "="*80)
    print("MEMORY SCALING TEST")
    print("="*80)

    device = args.device
    results = {}

    # Test configurations
    configs = [
        ("Minimal", 16, 12, 19, 16, 100),  # 16 levels, small hashmap
        ("Standard", 20, 16, 22, 18, 1000),  # Moderate
        ("High-Res", 36, 20, 22, 18, 1000),  # Default high-res
        ("Maximum", 40, 24, 24, 20, 10000),  # Max that fits in L4
    ]

    for name, s_lvl, t_lvl, s_hash, t_hash, batch_size in configs:
        print(f"\n{name} Configuration:")
        print(f"  Spatial: {s_lvl} levels, 2^{s_hash} hashmap")
        print(f"  Temporal: {t_lvl} levels, 2^{t_hash} hashmap")
        print(f"  Batch size: {batch_size}")

        # Clear memory
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            # Create model
            mem_start = profile_memory(device)

            model = Earth4DModel(
                spatial_levels=s_lvl,
                temporal_levels=t_lvl,
                spatial_hashmap_size=s_hash,
                temporal_hashmap_size=t_hash,
                verbose=False
            ).to(device)

            mem_model = profile_memory(device)
            model_size = mem_model['allocated_mb'] - mem_start['allocated_mb']

            # Test forward pass
            coords = torch.randn(batch_size, 4, device=device)
            coords[:, 0] = coords[:, 0] * 90  # Lat
            coords[:, 1] = coords[:, 1] * 180  # Lon
            coords[:, 2] = coords[:, 2] * 100  # Elevation
            coords[:, 3] = torch.rand(batch_size, device=device)  # Time

            output = model(coords)
            mem_forward = profile_memory(device)
            forward_size = mem_forward['allocated_mb'] - mem_model['allocated_mb']

            # Test backward pass
            loss = output.mean()
            loss.backward()
            mem_backward = profile_memory(device)
            backward_size = mem_backward['allocated_mb'] - mem_forward['allocated_mb']

            # Test optimizer step
            optimizer = optim.Adam(model.parameters())
            optimizer.step()
            mem_optimizer = profile_memory(device)
            optimizer_size = mem_optimizer['allocated_mb'] - mem_backward['allocated_mb']

            total_params = sum(p.numel() for p in model.parameters())

            results[name] = {
                'params': total_params,
                'model_mb': model_size,
                'forward_mb': forward_size,
                'backward_mb': backward_size,
                'optimizer_mb': optimizer_size,
                'total_mb': mem_optimizer['max_allocated_mb'],
                'status': 'success'
            }

            print(f"  ✓ Model: {model_size:.1f} MB")
            print(f"  ✓ Forward: +{forward_size:.1f} MB")
            print(f"  ✓ Backward: +{backward_size:.1f} MB")
            print(f"  ✓ Optimizer: +{optimizer_size:.1f} MB")
            print(f"  ✓ Total: {mem_optimizer['max_allocated_mb']:.1f} MB")

        except Exception as e:
            results[name] = {'status': 'failed', 'error': str(e)}
            print(f"  ✗ Failed: {e}")

    return results


def test_resolution_discrimination(args: argparse.Namespace) -> Dict:
    """Test if model can discriminate between points at various distances."""
    print("\n" + "="*80)
    print("RESOLUTION DISCRIMINATION TEST")
    print("="*80)

    device = args.device

    # Create model
    print(f"\nCreating Earth4D Model:")
    model = Earth4DModel(
        spatial_levels=args.spatial_levels,
        temporal_levels=args.temporal_levels,
        spatial_hashmap_size=args.spatial_hashmap,
        temporal_hashmap_size=args.temporal_hashmap,
        verbose=True  # Show Earth4D resolution table
    ).to(device)

    model.eval()

    # Test distances
    test_distances = [
        ("1000 km", 9.0),
        ("100 km", 0.9),
        ("10 km", 0.09),
        ("1 km", 0.009),
        ("100 m", 0.0009),
        ("10 m", 0.00009),
        ("1 m", 0.000009),
    ]

    base_point = torch.tensor([[40.7128, -74.0060, 100, 0.5]], device=device)

    results = {}
    with torch.no_grad():
        base_output = model(base_point)
        base_features = model.encoder(base_point)

        print(f"\nBase point (NYC):")
        print(f"  Output: {base_output.item():.6f}")
        print(f"  Feature std: {base_features.std().item():.6f}")

        print(f"\nDistance tests:")
        print(f"{'Distance':<12} {'Output Δ':<15} {'Feature Δ':<15} {'Discriminates':<12}")
        print("-" * 60)

        for name, delta_deg in test_distances:
            test_point = base_point.clone()
            test_point[0, 0] += delta_deg

            test_output = model(test_point)
            test_features = model.encoder(test_point)

            output_diff = abs(test_output.item() - base_output.item())
            feature_diff = (test_features - base_features).abs().mean().item()
            discriminates = feature_diff > 1e-5

            results[name] = {
                'output_diff': output_diff,
                'feature_diff': feature_diff,
                'discriminates': discriminates
            }

            disc_str = "Yes" if discriminates else "No"
            print(f"{name:<12} {output_diff:<15.6f} {feature_diff:<15.6f} {disc_str:<12}")

    return results


def test_training_convergence(args: argparse.Namespace) -> Dict:
    """Test training convergence and generalization."""
    print("\n" + "="*80)
    print("TRAINING CONVERGENCE TEST")
    print("="*80)

    device = args.device

    # Create model
    print(f"\nCreating Earth4D Model:")
    model = Earth4DModel(
        spatial_levels=args.spatial_levels,
        temporal_levels=args.temporal_levels,
        spatial_hashmap_size=args.spatial_hashmap,
        temporal_hashmap_size=args.temporal_hashmap,
        verbose=True  # Show Earth4D resolution table
    ).to(device)

    # Create dataset
    print("\nCreating dataset...")
    if args.spatial_levels >= 30:  # High-res config
        dataset = HighResEarthDataset(
            num_samples=args.dataset_size,
            device=device,
            include_fine_scale=True
        )
    else:
        dataset = EarthSystemDataset(
            num_samples=args.dataset_size,
            device=device
        )

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"\nTraining for {args.iterations} iterations...")
    print(f"{'Iter':<8} {'Train Loss':<12} {'Train MAPE':<12} {'Val Loss':<12} {'Val MAPE':<12}")
    print("-" * 60)

    train_losses = []
    val_losses = []
    train_mapes = []
    val_mapes = []

    model.train()
    for i in range(args.iterations):
        # Training step
        coords, targets = dataset.get_batch(args.batch_size, DataSplit.TRAIN)
        outputs = model(coords)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Calculate MAPE
        with torch.no_grad():
            train_metrics = dataset.compute_metrics(outputs, targets)
            train_mapes.append(train_metrics['mape'])

        # Validation every 10 iterations
        if (i + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_coords, val_targets = dataset.get_batch(args.batch_size // 4, DataSplit.VAL)
                val_outputs = model(val_coords)
                val_loss = criterion(val_outputs, val_targets)
                val_metrics = dataset.compute_metrics(val_outputs, val_targets)

                val_losses.append(val_loss.item())
                val_mapes.append(val_metrics['mape'])

                print(f"{i+1:<8} {loss.item():<12.4f} {train_metrics['mape']:<12.2f} "
                      f"{val_loss.item():<12.4f} {val_metrics['mape']:<12.2f}")
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_coords, test_targets = dataset.get_batch(1000, DataSplit.TEST)
        test_outputs = model(test_coords)
        test_metrics = dataset.compute_metrics(test_outputs, test_targets)

    return {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1] if val_losses else None,
        'final_train_mape': train_mapes[-1],
        'final_val_mape': val_mapes[-1] if val_mapes else None,
        'test_mape': test_metrics['mape'],
        'test_rmse': test_metrics['rmse'],
        'test_r2': test_metrics['r2']
    }


def run_full_test(args: argparse.Namespace):
    """Run complete test suite."""
    print("="*80)
    print("EARTH4D COMPREHENSIVE TEST SUITE")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Spatial: {args.spatial_levels} levels, 2^{args.spatial_hashmap} hashmap")
    print(f"  Temporal: {args.temporal_levels} levels, 2^{args.temporal_hashmap} hashmap")
    print(f"  Device: {args.device}")

    results = {}

    # Memory test
    if args.mode in [TestMode.FULL, TestMode.MEMORY]:
        results['memory'] = test_memory_scaling(args)

    # Discrimination test
    if args.mode in [TestMode.FULL, TestMode.DISCRIMINATION]:
        results['discrimination'] = test_resolution_discrimination(args)

    # Training test
    if args.mode in [TestMode.FULL, TestMode.TRAINING]:
        results['training'] = test_training_convergence(args)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    if 'memory' in results:
        print("\nMemory Requirements:")
        for config, metrics in results['memory'].items():
            if metrics['status'] == 'success':
                print(f"  {config}: {metrics['total_mb']:.1f} MB peak")

    if 'discrimination' in results:
        print("\nResolution Discrimination:")
        can_discriminate = [k for k, v in results['discrimination'].items() if v['discriminates']]
        if can_discriminate:
            finest = can_discriminate[-1]
            print(f"  Finest discrimination: {finest}")
        else:
            print(f"  No discrimination capability detected")

    if 'training' in results:
        print("\nTraining Performance:")
        print(f"  Test MAPE: {results['training']['test_mape']:.2f}%")
        print(f"  Test R²: {results['training']['test_r2']:.3f}")

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Earth4D Comprehensive Test Suite")

    # Test mode
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['full', 'quick', 'memory', 'discrimination', 'training'],
                       help='Test mode to run')

    # Model configuration
    parser.add_argument('--spatial-levels', type=int, default=36,
                       help='Number of spatial levels (default: 36 for 0.5m)')
    parser.add_argument('--temporal-levels', type=int, default=20,
                       help='Number of temporal levels (default: 20 for 1hr)')
    parser.add_argument('--spatial-hashmap', type=int, default=22,
                       help='Log2 spatial hashmap size (default: 22 = 4M)')
    parser.add_argument('--temporal-hashmap', type=int, default=18,
                       help='Log2 temporal hashmap size (default: 18 = 256K)')

    # Training configuration
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Training iterations (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size (default: 1000)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--dataset-size', type=int, default=10000,
                       help='Dataset size (default: 10000)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Convert mode string to enum
    args.mode = TestMode(args.mode)

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Quick mode adjustments
    if args.mode == TestMode.QUICK:
        args.iterations = min(args.iterations, 20)
        args.dataset_size = min(args.dataset_size, 1000)
        args.batch_size = min(args.batch_size, 100)
        # For quick mode, just run training test
        args.mode = TestMode.TRAINING

    # Run tests
    run_full_test(args)


if __name__ == "__main__":
    main()
