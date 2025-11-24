#!/usr/bin/env python3
"""
Collision Rate Heatmap Visualization
=====================================

Visualizes hash collision rates across all grids, levels, and simulation scenarios.

Usage:
    python visualize_collision_rates.py --input hash_collision_tests --output collision_heatmap.png
    python visualize_collision_rates.py --input hash_collision_tests --output collision_heatmap.png --dpi 300

Features:
- 4-panel layout (XYZ, XYT, YZT, XZT grids)
- Rows: 10 simulation scenarios
- Columns: 24 hash levels (0-23)
- Viridis colormap: Dark (purple/blue) = HIGH collision, Light (yellow) = LOW collision
- Collision rates computed on-the-fly from hash indices

Author: Claude Code
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

# Scenario order and display names
SCENARIOS = [
    ('uniform_random', 'Uniform Random'),
    ('continental_sparse', 'Continental Sparse'),
    ('extreme_spatial_single', 'Extreme Spatial Single'),
    ('extreme_spatial_multi', 'Extreme Spatial Multi'),
    ('extreme_temporal_single', 'Extreme Temporal Single'),
    ('extreme_temporal_multi', 'Extreme Temporal Multi'),
    ('moderate_spatial_cluster', 'Moderate Spatial Cluster'),
    ('moderate_spatiotemporal', 'Moderate Spatiotemporal'),
    ('moderate_temporal_cluster', 'Moderate Temporal Cluster'),
    ('time_series', 'Time Series'),
]

GRIDS = ['xyz', 'xyt', 'yzt', 'xzt']
GRID_TITLES = ['XYZ Grid', 'XYT Grid', 'YZT Grid', 'XZT Grid']


def compute_collision_rate(hash_indices_1d):
    """
    Compute collision rate for a single level.

    Collision rate = (total_points - unique_hashes) / total_points

    Args:
        hash_indices_1d: 1D array of hash indices for a single level

    Returns:
        Collision rate as float [0, 1]
    """
    if len(hash_indices_1d) == 0:
        return 0.0

    unique_hashes = len(np.unique(hash_indices_1d))
    total_points = len(hash_indices_1d)
    collision_rate = (total_points - unique_hashes) / total_points

    return collision_rate


def load_collision_data(data_dir):
    """
    Load collision data for all scenarios and compute collision rates.

    Args:
        data_dir: Directory containing scenario subdirectories

    Returns:
        Dictionary: {scenario_name: {grid_name: [collision_rates_per_level]}}
    """
    data_dir = Path(data_dir)
    collision_data = {}

    print("Loading collision data...")
    for scenario_name, scenario_display in tqdm(SCENARIOS, desc="Scenarios"):
        scenario_dir = data_dir / scenario_name
        collision_file = scenario_dir / 'collision_data.pt'

        if not collision_file.exists():
            print(f"Warning: Missing data for {scenario_name}")
            continue

        # Load collision data
        data = torch.load(collision_file, map_location='cpu')
        hash_indices = data['hash_indices']

        # Compute collision rates for each grid and level
        scenario_data = {}
        for grid in GRIDS:
            if grid not in hash_indices:
                print(f"Warning: Missing grid {grid} for {scenario_name}")
                continue

            grid_data = hash_indices[grid].numpy()  # Shape: [n_points, n_levels]
            n_levels = grid_data.shape[1]

            # Compute collision rate for each level
            collision_rates = []
            for level in range(n_levels):
                level_indices = grid_data[:, level]
                collision_rate = compute_collision_rate(level_indices)
                collision_rates.append(collision_rate)

            scenario_data[grid] = collision_rates

        collision_data[scenario_name] = scenario_data

    return collision_data


def create_collision_heatmap(collision_data, output_path, title_suffix="", dpi=200):
    """
    Create a 4-panel heatmap showing collision rates across all scenarios and levels.

    Args:
        collision_data: Dictionary from load_collision_data()
        output_path: Path to save output image
        title_suffix: Optional suffix for title (e.g., "10K Points")
        dpi: Output resolution
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure with 2x2 grid layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    # Determine number of levels from first scenario
    first_scenario = next(iter(collision_data.values()))
    first_grid = next(iter(first_scenario.values()))
    n_levels = len(first_grid)

    # Process each grid
    for grid_idx, (grid, grid_title) in enumerate(zip(GRIDS, GRID_TITLES)):
        ax = axes[grid_idx]

        # Build collision rate matrix: [n_scenarios, n_levels]
        collision_matrix = []
        scenario_labels = []

        for scenario_name, scenario_display in SCENARIOS:
            if scenario_name in collision_data and grid in collision_data[scenario_name]:
                collision_rates = collision_data[scenario_name][grid]
                collision_matrix.append(collision_rates)
                scenario_labels.append(scenario_display)
            else:
                # Fill with NaN if data missing
                collision_matrix.append([np.nan] * n_levels)
                scenario_labels.append(scenario_display)

        collision_matrix = np.array(collision_matrix)  # Shape: [n_scenarios, n_levels]

        # Create heatmap with Viridis colormap
        # Viridis: Dark (purple/blue) = HIGH values, Light (yellow) = LOW values
        # Collision rates are [0, 1], so 0% = yellow (low), 100% = dark purple (high)
        im = ax.imshow(
            collision_matrix,
            aspect='auto',
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
            origin='upper',
            interpolation='nearest'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Collision Rate', fontsize=11, weight='bold')
        cbar.ax.tick_params(labelsize=9)

        # Format colorbar labels as percentages
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

        # Set title
        ax.set_title(grid_title, fontsize=14, weight='bold', pad=10)

        # Set axis labels
        ax.set_xlabel('Hash Level', fontsize=11, weight='bold')
        ax.set_ylabel('Simulation Scenario', fontsize=11, weight='bold')

        # Set x-axis ticks (levels)
        ax.set_xticks(range(0, n_levels, 2))  # Every 2 levels
        ax.set_xticklabels(range(0, n_levels, 2), fontsize=9)

        # Set y-axis ticks (scenarios)
        ax.set_yticks(range(len(scenario_labels)))
        ax.set_yticklabels(scenario_labels, fontsize=9)

        # Add grid for readability
        ax.set_xticks(np.arange(n_levels) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(scenario_labels)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # Main title
    n_points = None
    if collision_data:
        first_scenario_name = next(iter(collision_data.keys()))
        first_scenario_data = collision_data[first_scenario_name]
        first_grid_name = next(iter(first_scenario_data.keys()))
        # Try to infer point count from directory name or use title_suffix
        n_points = title_suffix

    main_title = f'Hash Collision Rates - {n_points}' if n_points else 'Hash Collision Rates'
    fig.suptitle(main_title, fontsize=16, weight='bold', y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {output_path}")

    plt.close(fig)


def create_summary_statistics(collision_data, output_path):
    """
    Create a summary table of collision statistics.

    Args:
        collision_data: Dictionary from load_collision_data()
        output_path: Path to save summary text file
    """
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        f.write("Hash Collision Rate Summary Statistics\n")
        f.write("=" * 80 + "\n\n")

        for scenario_name, scenario_display in SCENARIOS:
            if scenario_name not in collision_data:
                continue

            f.write(f"\n{scenario_display}\n")
            f.write("-" * 80 + "\n")

            scenario_data = collision_data[scenario_name]

            for grid in GRIDS:
                if grid not in scenario_data:
                    continue

                collision_rates = np.array(scenario_data[grid])

                f.write(f"\n  {grid.upper()} Grid:\n")
                f.write(f"    Mean collision rate: {collision_rates.mean():.1%}\n")
                f.write(f"    Min collision rate:  {collision_rates.min():.1%} (Level {collision_rates.argmin()})\n")
                f.write(f"    Max collision rate:  {collision_rates.max():.1%} (Level {collision_rates.argmax()})\n")
                f.write(f"    Std deviation:       {collision_rates.std():.1%}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("\nOverall Statistics (All Scenarios, All Grids)\n")
        f.write("-" * 80 + "\n")

        all_rates = []
        for scenario_data in collision_data.values():
            for grid_rates in scenario_data.values():
                all_rates.extend(grid_rates)

        all_rates = np.array(all_rates)
        f.write(f"Mean collision rate: {all_rates.mean():.1%}\n")
        f.write(f"Min collision rate:  {all_rates.min():.1%}\n")
        f.write(f"Max collision rate:  {all_rates.max():.1%}\n")
        f.write(f"Std deviation:       {all_rates.std():.1%}\n")

    print(f"✓ Saved summary statistics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize hash collision rates across scenarios and levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualize_collision_rates.py --input hash_collision_tests --output collision_heatmap.png

  # High resolution output
  python visualize_collision_rates.py --input hash_collision_tests --output collision_heatmap.png --dpi 300

  # Add title suffix
  python visualize_collision_rates.py --input hash_collision_tests_1M --output baseline_1M.png --title "1M Points"

  # Generate summary statistics
  python visualize_collision_rates.py --input hash_collision_tests --output heatmap.png --summary stats.txt
        """
    )

    parser.add_argument('--input', '-i', required=True, type=str,
                        help='Input directory containing scenario subdirectories with collision_data.pt files')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='Output path for heatmap image (PNG recommended)')
    parser.add_argument('--title', '-t', type=str, default='',
                        help='Optional title suffix (e.g., "10K Points Dataset")')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Output resolution in DPI (default: 200)')
    parser.add_argument('--summary', '-s', type=str, default=None,
                        help='Optional path to save summary statistics text file')

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    print("\n" + "=" * 80)
    print("HASH COLLISION RATE VISUALIZATION")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output path:     {args.output}")
    print(f"DPI:             {args.dpi}")
    print(f"Title suffix:    {args.title if args.title else '(none)'}")
    print("=" * 80 + "\n")

    # Load collision data
    collision_data = load_collision_data(input_dir)

    if not collision_data:
        print("Error: No collision data found!")
        return 1

    print(f"\nLoaded {len(collision_data)} scenarios")

    # Create heatmap
    print("\nGenerating heatmap...")
    create_collision_heatmap(
        collision_data,
        args.output,
        title_suffix=args.title,
        dpi=args.dpi
    )

    # Create summary statistics if requested
    if args.summary:
        print("\nGenerating summary statistics...")
        create_summary_statistics(collision_data, args.summary)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
