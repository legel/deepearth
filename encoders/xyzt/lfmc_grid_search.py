#!/usr/bin/env python3
"""
Quick grid search over learned hash probing hyperparameters.
Tests multiple configurations with configurable epochs.
"""

import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

def run_single_test(probing_range, entropy_weight, index_codebook_size, hashmap_size, test_id, total_tests, args, baseline_results_path=None):
    """Run a single training test with given hyperparameters."""

    print(f"\n{'='*80}")
    print(f"TEST {test_id}/{total_tests}")
    print(f"{'='*80}")
    print(f"N_p={probing_range}, N_c={index_codebook_size}, entropy_weight={entropy_weight}, hashmap_size=2^{hashmap_size}")
    if baseline_results_path:
        print(f"Reusing baseline from: {Path(baseline_results_path).parent.name}")
    else:
        print(f"Running baseline + learned")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}\n")

    # Build command
    output_dir = f"{args.output_dir}/test_{test_id:02d}_Np{probing_range}_Nc{index_codebook_size}_ent{entropy_weight}_hash{hashmap_size}"

    cmd = [
        'python', '-u', 'earth4d_to_lfmc_comparison.py',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--species-dim', str(args.species_dim),
        '--seed', str(args.seed),
        '--probing-range', str(probing_range),
        '--index-codebook-size', str(index_codebook_size),
        '--index-lr-multiplier', str(args.index_lr_multiplier),
        '--entropy-weight', str(entropy_weight),
        '--spatial-log2-hashmap-size', str(hashmap_size),
        '--temporal-log2-hashmap-size', str(hashmap_size),
        '--output-dir', output_dir,
    ]

    # Add entropy flag if weight > 0
    if entropy_weight > 0:
        cmd.append('--enable-entropy-loss')

    # Reuse baseline if available (skip baseline training)
    if baseline_results_path:
        cmd.extend(['--baseline-results-path', baseline_results_path])

    # Run with live output (don't capture)
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    # Parse results from comparison_summary.json
    summary_path = Path(output_dir) / 'comparison_summary.json'

    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            # Extract learned metrics
            learned = summary['learned']

            return {
                'test_id': test_id,
                'probing_range': probing_range,
                'index_codebook_size': index_codebook_size,
                'entropy_weight': entropy_weight,
                'hashmap_size': hashmap_size,
                'test_rmse': learned['test_rmse'],
                'test_mae': learned['test_mae'],
                'test_r2': learned['test_r2'],
                'training_time': learned['total_training_seconds'],
                'total_time': elapsed,
                'output_dir': output_dir
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"ERROR: Failed to parse results for test {test_id}: {e}")
            print(f"  File: {summary_path}")
            return None
    else:
        print(f"WARNING: No results found for test {test_id}")
        print(f"  Expected: {summary_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Grid search for learned hash probing hyperparameters')

    # Grid search parameters
    parser.add_argument('--probing-ranges', type=int, nargs='+', default=[2, 4, 8],
                       help='List of N_p values to test (default: 2 4 8)')
    parser.add_argument('--codebook-sizes', type=int, nargs='+', default=[512],
                       help='List of N_c values to test (default: 512)')
    parser.add_argument('--entropy-weights', type=float, nargs='+', default=[0.0, 0.01, 0.05],
                       help='List of entropy weights to test (default: 0.0 0.01 0.05)')
    parser.add_argument('--hashmap-sizes', type=int, nargs='+', default=[22],
                       help='List of log2 hashmap sizes to test (default: 22 for 2^22)')

    # Fixed training parameters
    parser.add_argument('--epochs', type=int, default=250,
                       help='Epochs per test (default: 250)')
    parser.add_argument('--batch-size', type=int, default=30000,
                       help='Batch size (default: 30000)')
    parser.add_argument('--lr', type=float, default=0.0125,
                       help='Learning rate (default: 0.0125)')
    parser.add_argument('--species-dim', type=int, default=768,
                       help='Species embedding dimension (default: 768)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--index-lr-multiplier', type=float, default=10.0,
                       help='LR multiplier for index_logits (default: 10.0)')
    parser.add_argument('--output-dir', type=str, default='./grid_search',
                       help='Output directory for all tests (default: ./grid_search)')

    # Baseline handling
    parser.add_argument('--baseline-results-path', type=str, default=None,
                       help='Path to existing baseline results JSON to skip baseline training (e.g., ./previous_run/test_01/comparison_summary.json)')

    args = parser.parse_args()

    # Calculate total tests
    total_tests = len(args.probing_ranges) * len(args.codebook_sizes) * len(args.entropy_weights) * len(args.hashmap_sizes)
    est_time_min = total_tests * args.epochs / 10  # Rough estimate: 10 epochs/min

    print("\n" + "="*80)
    print("LEARNED HASH PROBING HYPERPARAMETER GRID SEARCH")
    print("="*80)
    print(f"Total tests: {total_tests}")
    print(f"Epochs per test: {args.epochs}")
    print(f"Estimated total time: ~{est_time_min:.0f} minutes")
    print(f"\nGrid:")
    print(f"  N_p (probing_range): {args.probing_ranges}")
    print(f"  N_c (codebook_size): {args.codebook_sizes}")
    print(f"  Entropy weights: {args.entropy_weights}")
    print(f"  Hashmap sizes (log2): {args.hashmap_sizes}")
    print(f"\nFixed:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Species dim: {args.species_dim}")
    print(f"  Index LR multiplier: {args.index_lr_multiplier}")
    print(f"  Seed: {args.seed}")
    print("="*80)

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # Run all tests
    results = []
    test_id = 1
    baseline_results_path = args.baseline_results_path  # Use provided baseline or None (will be set after first test)

    # Print baseline status
    if baseline_results_path:
        print(f"\n✓ Using existing baseline from: {baseline_results_path}")
        print(f"  All tests will skip baseline training.\n")

    for probing_range in args.probing_ranges:
        for codebook_size in args.codebook_sizes:
            for entropy_weight in args.entropy_weights:
                for hashmap_size in args.hashmap_sizes:
                    result = run_single_test(probing_range, entropy_weight, codebook_size, hashmap_size,
                                            test_id, total_tests, args, baseline_results_path)
                    if result:
                        results.append(result)

                        # After first test, save baseline results path for reuse (if not already provided)
                        if test_id == 1 and not args.baseline_results_path:
                            baseline_results_path = str(Path(result['output_dir']) / 'comparison_summary.json')
                            print(f"\n✓ Baseline results saved. Subsequent tests will reuse baseline from test 1.\n")

                    test_id += 1

    # Sort by test R²
    results.sort(key=lambda x: x['test_r2'], reverse=True)

    # Print summary table
    print("\n\n" + "="*80)
    print("GRID SEARCH RESULTS (sorted by R²)")
    print("="*80)
    print(f"\n{'Rank':<6}{'N_p':<6}{'N_c':<8}{'Hash':<7}{'Entropy':<10}{'RMSE':<10}{'MAE':<10}{'R²':<10}{'Time(s)':<10}")
    print("-"*80)

    for rank, r in enumerate(results, 1):
        print(f"{rank:<6}{r['probing_range']:<6}{r['index_codebook_size']:<8}2^{r['hashmap_size']:<5}{r['entropy_weight']:<10.3f}"
              f"{r['test_rmse']:<10.2f}{r['test_mae']:<10.2f}"
              f"{r['test_r2']:<10.4f}{r['training_time']:<10.1f}")

    # Print best configuration
    best = results[0]
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"  N_p (probing_range): {best['probing_range']}")
    print(f"  N_c (codebook_size): {best['index_codebook_size']}")
    print(f"  Hashmap size: 2^{best['hashmap_size']}")
    print(f"  Entropy weight: {best['entropy_weight']}")
    print(f"\nPerformance:")
    print(f"  Test RMSE: {best['test_rmse']:.2f}pp")
    print(f"  Test MAE: {best['test_mae']:.2f}pp")
    print(f"  Test R²: {best['test_r2']:.4f}")
    print(f"  Training time: {best['training_time']:.1f}s")
    print(f"\nOutput: {best['output_dir']}")

    # Save results
    results_path = f"{args.output_dir}/grid_search_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'grid': {
                'probing_ranges': args.probing_ranges,
                'codebook_sizes': args.codebook_sizes,
                'entropy_weights': args.entropy_weights
            },
            'fixed': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'species_dim': args.species_dim,
                'seed': args.seed,
                'index_lr_multiplier': args.index_lr_multiplier
            },
            'results': results,
            'best': best,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
