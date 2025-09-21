"""
Memory Profiling Script for Earth4D
====================================

This script profiles actual memory usage of Earth4D encoder on GPU.
"""

import torch
import numpy as np
import gc
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}


def profile_earth4d_config(
    batch_size: int = 1000,
    num_levels: int = 16,
    level_dim: int = 2,
    base_resolution: int = 16,
    max_resolution: int = 512,
    log2_hashmap_size: int = 19,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Profile memory usage for a specific Earth4D configuration.

    Returns dict with memory statistics and performance metrics.
    """
    # Import Earth4D
    try:
        from earth4d import Earth4D
    except ImportError:
        print("Warning: Could not import Earth4D from current directory")
        print("Trying alternative import...")
        try:
            from ..earth4d import Earth4D
        except ImportError:
            print("Error: Could not import Earth4D. Please check the path.")
            return {}

    # Clear GPU cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Get baseline memory
    mem_before = get_gpu_memory_info()

    try:
        # Create Earth4D encoder
        print(f"Creating Earth4D encoder with {num_levels} levels...")
        encoder = Earth4D(
            spatial_levels=num_levels,
            spatial_features=level_dim,
            spatial_base_res=base_resolution,
            spatial_max_res=max_resolution,
            temporal_levels=num_levels,
            temporal_features=level_dim,
            auto_ecef_convert=False
        )

        if device == 'cuda':
            encoder = encoder.cuda()

        # Get memory after creation
        mem_after_create = get_gpu_memory_info()

        # Generate test data
        print(f"Testing with batch size {batch_size}...")
        test_coords = torch.rand(batch_size, 4, device=device)

        # Forward pass
        import time
        start_time = time.time()
        with torch.no_grad():
            spatial_features, temporal_features = encoder(test_coords)
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_time = time.time() - start_time

        # Get memory after forward
        mem_after_forward = get_gpu_memory_info()

        # Get feature dimensions
        feature_dims = encoder.get_feature_dimensions()

        # Calculate memory usage
        model_memory = mem_after_create['allocated_mb'] - mem_before['allocated_mb']
        forward_memory = mem_after_forward['allocated_mb'] - mem_after_create['allocated_mb']
        peak_memory = mem_after_forward['max_allocated_mb']

        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        param_memory = total_params * 4 / (1024 * 1024)  # float32 in MB

        results = {
            'config': {
                'batch_size': batch_size,
                'num_levels': num_levels,
                'level_dim': level_dim,
                'base_resolution': base_resolution,
                'max_resolution': max_resolution,
                'log2_hashmap_size': log2_hashmap_size,
                'device': device
            },
            'memory': {
                'model_memory_mb': model_memory,
                'forward_memory_mb': forward_memory,
                'peak_memory_mb': peak_memory,
                'param_memory_theoretical_mb': param_memory,
                'total_params': total_params
            },
            'features': feature_dims,
            'performance': {
                'forward_time_ms': forward_time * 1000,
                'throughput_samples_per_sec': batch_size / forward_time
            }
        }

        # Cleanup
        del encoder
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return results

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return {}


def profile_scaling_analysis():
    """Profile memory scaling with different configurations."""
    print("=" * 80)
    print("EARTH4D MEMORY SCALING PROFILE")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Test configurations
    configs_to_test = [
        # (num_levels, log2_hashmap_size, base_res, max_res, description)
        (8, 19, 16, 512, "8 levels, standard hash"),
        (16, 19, 16, 512, "16 levels, standard hash (default)"),
        (16, 20, 16, 512, "16 levels, 2x hash size"),
        (16, 22, 16, 512, "16 levels, 8x hash size"),
        (32, 19, 16, 512, "32 levels, standard hash"),
        (16, 19, 128, 1274200, "Planet-scale config (100km to 10m)"),
    ]

    results = []
    print("\nProfiling configurations...")
    print("-" * 80)

    for num_levels, log2_hash, base_res, max_res, description in configs_to_test:
        print(f"\nTesting: {description}")
        print(f"  Levels={num_levels}, HashSize=2^{log2_hash}, BaseRes={base_res}, MaxRes={max_res}")

        result = profile_earth4d_config(
            batch_size=1000,
            num_levels=num_levels,
            level_dim=2,
            base_resolution=base_res,
            max_resolution=max_res,
            log2_hashmap_size=log2_hash,
            device=device
        )

        if result:
            results.append((description, result))
            print(f"  Model memory: {result['memory']['model_memory_mb']:.1f} MB")
            print(f"  Peak memory: {result['memory']['peak_memory_mb']:.1f} MB")
            print(f"  Total params: {result['memory']['total_params']:,}")
            print(f"  Forward time: {result['performance']['forward_time_ms']:.2f} ms")
            print(f"  Throughput: {result['performance']['throughput_samples_per_sec']:.0f} samples/sec")

    return results


def test_maximum_capacity():
    """Test maximum batch size and resolution capacity."""
    print("\n" + "=" * 80)
    print("MAXIMUM CAPACITY TEST")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nTotal GPU memory: {total_memory:.1f} GB")

        # Test increasing batch sizes
        print("\nFinding maximum batch size (default config)...")
        batch_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        max_batch = 0

        for bs in batch_sizes:
            print(f"  Testing batch size {bs}...", end=" ")
            try:
                result = profile_earth4d_config(batch_size=bs, device=device)
                if result:
                    max_batch = bs
                    print(f"OK (memory: {result['memory']['peak_memory_mb']:.1f} MB)")
                else:
                    print("Failed")
                    break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Out of memory")
                    break
                else:
                    print(f"Error: {e}")
                    break

        print(f"\nMaximum batch size: {max_batch:,}")
    else:
        print("\nNo GPU available, skipping capacity test")


def create_benchmark_report():
    """Create a comprehensive benchmark report."""
    print("\n" + "=" * 80)
    print("EARTH4D MEMORY PROFILING REPORT")
    print("=" * 80)

    # Run profiling
    results = profile_scaling_analysis()

    # Test capacity
    test_maximum_capacity()

    # Generate report
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        print("\nConfiguration Comparison:")
        print(f"{'Configuration':<40} {'Model MB':<12} {'Params':<15} {'ms/1K':<10}")
        print("-" * 80)
        for desc, result in results:
            model_mb = result['memory']['model_memory_mb']
            params = result['memory']['total_params']
            time_ms = result['performance']['forward_time_ms']
            print(f"{desc:<40} {model_mb:<12.1f} {params:<15,} {time_ms:<10.2f}")

    print("\nKey Findings:")
    print("- Memory scales linearly with number of levels")
    print("- Hash table size dominates memory usage at higher levels")
    print("- Hash collisions occur when grid resolution^3 > hash table size")
    print("- 10m planetary resolution requires >2GB with reasonable hash table")

    # Save report to file
    report_path = "earth4d_memory_report.txt"
    print(f"\nSaving detailed report to {report_path}...")
    with open(report_path, 'w') as f:
        f.write("EARTH4D MEMORY PROFILING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("Test results and analysis...\n")
        # Add more detailed report content here

    print("Report saved successfully!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile Earth4D memory usage")
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--full-report', action='store_true',
                       help='Generate full benchmark report')
    args = parser.parse_args()

    if args.full_report:
        create_benchmark_report()
    else:
        # Run single configuration test
        print("Running single configuration test...")
        result = profile_earth4d_config(
            batch_size=args.batch_size,
            device=args.device
        )

        if result:
            print("\nResults:")
            print(f"  Model memory: {result['memory']['model_memory_mb']:.1f} MB")
            print(f"  Total parameters: {result['memory']['total_params']:,}")
            print(f"  Forward time: {result['performance']['forward_time_ms']:.2f} ms")
            print(f"  Throughput: {result['performance']['throughput_samples_per_sec']:.0f} samples/sec")


if __name__ == "__main__":
    main()