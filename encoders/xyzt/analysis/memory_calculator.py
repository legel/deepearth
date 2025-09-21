"""
Memory and Resolution Calculator for Earth4D
============================================

This script analyzes the memory requirements and effective resolutions
for the Earth4D multi-resolution hash encoder at planetary scale.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple

class Earth4DMemoryAnalyzer:
    """Analyzes memory usage and resolution capabilities of Earth4D encoder."""

    def __init__(self,
                 earth_radius: float = 6371000.0,  # meters
                 num_levels: int = 16,
                 level_dim: int = 2,
                 log2_hashmap_size: int = 19):
        """
        Initialize analyzer.

        Args:
            earth_radius: Earth radius in meters
            num_levels: Number of encoding levels
            level_dim: Features per level
            log2_hashmap_size: Log2 of hash table size
        """
        self.earth_radius = earth_radius
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.hashmap_size = 2 ** log2_hashmap_size

    def calculate_resolution_at_level(self,
                                     level: int,
                                     base_resolution: int = 16,
                                     per_level_scale: float = 2.0) -> Dict:
        """
        Calculate the effective spatial resolution at a given level.

        Returns dict with:
        - grid_resolution: Number of grid cells per dimension
        - meters_per_cell: Physical size of each cell in meters
        - total_cells: Total number of cells (before hashing)
        - params_used: Actual parameters used (after hash table limit)
        """
        # Grid resolution at this level
        grid_res = np.ceil(base_resolution * (per_level_scale ** level))

        # For normalized ECEF coordinates in [-1, 1]
        # The range spans ~2 * earth_radius (pole to pole)
        normalized_range = 2.0  # -1 to 1
        physical_range = 2 * self.earth_radius  # meters

        # Meters per grid cell
        meters_per_cell = physical_range / grid_res

        # Total cells and parameters
        total_cells = grid_res ** 3  # for 3D
        params_used = min(total_cells, self.hashmap_size)

        return {
            'level': level,
            'grid_resolution': int(grid_res),
            'meters_per_cell': meters_per_cell,
            'km_per_cell': meters_per_cell / 1000,
            'total_cells': int(total_cells),
            'params_used': int(params_used),
            'hash_collisions': total_cells > self.hashmap_size
        }

    def analyze_all_levels(self,
                          base_resolution: int = 16,
                          per_level_scale: float = 2.0) -> List[Dict]:
        """Analyze all encoding levels."""
        results = []
        for level in range(self.num_levels):
            results.append(self.calculate_resolution_at_level(
                level, base_resolution, per_level_scale
            ))
        return results

    def calculate_memory_usage(self,
                              base_resolution: int = 16,
                              per_level_scale: float = 2.0) -> Dict:
        """
        Calculate total memory usage for Earth4D.

        Returns dict with memory usage in different units.
        """
        # Get parameters per level
        levels = self.analyze_all_levels(base_resolution, per_level_scale)

        # Sum parameters across levels
        total_params_per_encoder = sum(l['params_used'] for l in levels)

        # Earth4D has 4 3D encoders (xyz, xyt, yzt, xzt)
        num_encoders = 4
        total_params = total_params_per_encoder * num_encoders * self.level_dim

        # Memory in different units (assuming float32)
        bytes_per_param = 4
        total_bytes = total_params * bytes_per_param

        return {
            'params_per_encoder': total_params_per_encoder,
            'total_params': total_params,
            'bytes': total_bytes,
            'megabytes': total_bytes / (1024 * 1024),
            'gigabytes': total_bytes / (1024 * 1024 * 1024),
            'num_encoders': num_encoders,
            'level_dim': self.level_dim
        }

    def find_config_for_target_resolutions(self,
                                          target_meters: List[float]) -> Dict:
        """
        Find configuration to achieve target resolutions.

        Args:
            target_meters: List of target resolutions in meters
                          e.g., [100000, 10000, 1000, 100, 10]

        Returns optimal configuration parameters.
        """
        target_meters = sorted(target_meters, reverse=True)
        num_targets = len(target_meters)

        # We need to find base_resolution and per_level_scale
        # such that we hit these targets across our levels

        # Calculate required grid sizes
        physical_range = 2 * self.earth_radius
        required_grids = [physical_range / t for t in target_meters]

        # If we use num_targets levels, find the scale factor
        if num_targets > 1:
            # Geometric mean of scale factors
            scale_ratios = [required_grids[i+1] / required_grids[i]
                           for i in range(num_targets-1)]
            per_level_scale = np.mean(scale_ratios)
        else:
            per_level_scale = 2.0

        base_resolution = required_grids[0]

        # Verify we can achieve targets
        achieved = []
        for i in range(num_targets):
            level_res = base_resolution * (per_level_scale ** i)
            achieved_meters = physical_range / level_res
            achieved.append(achieved_meters)

        return {
            'base_resolution': int(np.ceil(base_resolution)),
            'per_level_scale': per_level_scale,
            'num_levels_needed': num_targets,
            'target_meters': target_meters,
            'achieved_meters': achieved,
            'errors_meters': [abs(t - a) for t, a in zip(target_meters, achieved)],
            'memory': self.calculate_memory_usage(
                int(np.ceil(base_resolution)),
                per_level_scale
            )
        }

    def print_analysis(self,
                       base_resolution: int = 16,
                       per_level_scale: float = 2.0):
        """Print detailed analysis."""
        print("=" * 80)
        print("Earth4D Memory and Resolution Analysis")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Earth radius: {self.earth_radius/1000:.0f} km")
        print(f"  Hash table size: 2^{int(np.log2(self.hashmap_size))} = {self.hashmap_size:,}")
        print(f"  Number of levels: {self.num_levels}")
        print(f"  Features per level: {self.level_dim}")
        print(f"  Base resolution: {base_resolution}")
        print(f"  Per-level scale: {per_level_scale}")

        print(f"\n{'Level':<6} {'Grid Res':<12} {'Meters/Cell':<15} {'KM/Cell':<12} {'Total Cells':<15} {'Params Used':<15} {'Hash Collision':<15}")
        print("-" * 110)

        levels = self.analyze_all_levels(base_resolution, per_level_scale)
        for l in levels:
            collision = "Yes" if l['hash_collisions'] else "No"
            print(f"{l['level']:<6} {l['grid_resolution']:<12,} {l['meters_per_cell']:<15.1f} {l['km_per_cell']:<12.2f} {l['total_cells']:<15,} {l['params_used']:<15,} {collision:<15}")

        memory = self.calculate_memory_usage(base_resolution, per_level_scale)
        print(f"\nMemory Usage:")
        print(f"  Parameters per encoder: {memory['params_per_encoder']:,}")
        print(f"  Total parameters (4 encoders): {memory['total_params']:,}")
        print(f"  Memory usage: {memory['megabytes']:.1f} MB ({memory['gigabytes']:.3f} GB)")

    def analyze_planetary_targets(self):
        """Analyze configuration for planetary-scale targets."""
        print("\n" + "=" * 80)
        print("Planetary-Scale Resolution Requirements")
        print("=" * 80)

        # Target resolutions: 100km, 10km, 1km, 100m, 10m
        targets = [100000, 10000, 1000, 100, 10]  # meters

        print(f"\nTarget resolutions: {[f'{t/1000:.0f}km' if t >= 1000 else f'{t}m' for t in targets]}")

        config = self.find_config_for_target_resolutions(targets)

        print(f"\nOptimal configuration:")
        print(f"  Base resolution: {config['base_resolution']}")
        print(f"  Per-level scale: {config['per_level_scale']:.3f}")
        print(f"  Levels needed: {config['num_levels_needed']}")

        print(f"\nAchieved resolutions:")
        for i, (target, achieved, error) in enumerate(zip(
            config['target_meters'],
            config['achieved_meters'],
            config['errors_meters']
        )):
            t_str = f"{target/1000:.0f}km" if target >= 1000 else f"{target}m"
            a_str = f"{achieved/1000:.1f}km" if achieved >= 1000 else f"{achieved:.1f}m"
            e_str = f"{error/1000:.1f}km" if error >= 1000 else f"{error:.1f}m"
            print(f"  Level {i}: Target {t_str:<8} → Achieved {a_str:<10} (error: {e_str})")

        print(f"\nMemory requirement: {config['memory']['megabytes']:.1f} MB")

        # Check if we can fit 10m resolution
        print(f"\n10m Resolution Feasibility:")
        physical_range = 2 * self.earth_radius
        required_grid_10m = physical_range / 10  # ~1.27M cells per dimension
        total_cells_10m = required_grid_10m ** 3
        print(f"  Required grid resolution: {required_grid_10m:,.0f}")
        print(f"  Total cells needed: {total_cells_10m:.2e}")
        print(f"  Hash table size: {self.hashmap_size:,.0f}")
        print(f"  Collision ratio: {total_cells_10m/self.hashmap_size:,.1f}:1")

        # Calculate memory for different hash table sizes
        print(f"\nMemory scaling with hash table size (for 10m resolution):")
        for log2_size in [19, 20, 22, 24, 26, 28, 30]:
            hash_size = 2 ** log2_size
            params = min(total_cells_10m, hash_size)
            memory_mb = (params * 4 * self.level_dim * 4) / (1024 * 1024)  # 4 encoders
            collision_ratio = max(1, total_cells_10m / hash_size)
            print(f"  2^{log2_size} ({hash_size/1e6:.1f}M entries): {memory_mb:,.0f} MB, collision ratio {collision_ratio:.1f}:1")


def main():
    """Run the analysis."""
    analyzer = Earth4DMemoryAnalyzer()

    # Analyze default configuration
    print("\n" + "=" * 80)
    print("DEFAULT CONFIGURATION ANALYSIS")
    analyzer.print_analysis()

    # Analyze planetary-scale targets
    analyzer.analyze_planetary_targets()

    # Analyze with larger hash table
    print("\n" + "=" * 80)
    print("ANALYSIS WITH LARGER HASH TABLE (2^24)")
    analyzer_large = Earth4DMemoryAnalyzer(log2_hashmap_size=24)
    analyzer_large.print_analysis()

    # Memory scaling analysis
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 80)

    print("\nHow memory scales with configuration:")
    print(f"{'Levels':<8} {'Base Res':<10} {'Scale':<8} {'Hash Size':<12} {'Memory (MB)':<12}")
    print("-" * 60)

    for num_levels in [8, 16, 24, 32]:
        for log2_hash in [19, 22, 24]:
            analyzer = Earth4DMemoryAnalyzer(
                num_levels=num_levels,
                log2_hashmap_size=log2_hash
            )
            memory = analyzer.calculate_memory_usage()
            hash_size_str = f"2^{log2_hash}"
            print(f"{num_levels:<8} {16:<10} {2.0:<8.1f} {hash_size_str:<12} {memory['megabytes']:<12.1f}")


if __name__ == "__main__":
    main()