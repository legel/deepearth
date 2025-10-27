#!/usr/bin/env python3
"""
Hash Collision Profiler
========================

See hash collision rates across all indices for Earth4D for a given number of simulated points, over various distributions.
Examples:

python hash_collision_profiler.py --n-points 1000
python hash_collision_profiler.py --n-points 10000
python hash_collision_profiler.py --n-points 100000
python hash_collision_profiler.py --n-points 1000000

Statistical profiling of hash collisions in Earth4D spatiotemporal encoding.

Features:
- Synthetic test data generation for controlled collision analysis
- Complete collision tracking and analysis
- CSV/JSON export with test metadata

To do:
- Explore visualizations of input data vs. hash collisions from output files.

Author: Lance Legel, Qin Huang
License: MIT
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from collections import defaultdict
import datetime

# Add deepearth root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from encoders.xyzt.earth4d import Earth4D


class SpatiotemporalPointGenerator:
    """Generate synthetic (lat, lon, elev, time) coordinates for collision testing."""

    def __init__(self, n_points=1_000_000, seed=42):
        self.n_points = n_points
        self.seed = seed
        np.random.seed(seed)

    def generate_uniform(self):
        """Test 1: Uniform random across Earth surface and full time range."""
        lat = np.random.uniform(-90, 90, self.n_points).astype(np.float64)
        lon = np.random.uniform(-180, 180, self.n_points).astype(np.float64)
        elev = np.random.uniform(0, 5000, self.n_points).astype(np.float64)  # 0-5km elevation
        time = np.random.uniform(0, 1, self.n_points).astype(np.float64)

        metadata = {
            'test_name': 'uniform_random',
            'description': 'Uniform random distribution across Earth surface and time',
            'spatial_distribution': 'uniform_global',
            'temporal_distribution': 'uniform',
            'expected_collision_behavior': 'Low collisions due to natural sparsity'
        }

        return lat, lon, elev, time, metadata

    def generate_moderate_spatial_cluster(self, center_lat=37.7749, center_lon=-122.4194):
        """Test 2: 1M points in 10km × 10km region."""
        # 10km ≈ 0.09 degrees at equator
        lat_offset = np.random.uniform(-0.045, 0.045, self.n_points).astype(np.float64)
        lon_offset = np.random.uniform(-0.045, 0.045, self.n_points).astype(np.float64)

        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        elev = np.random.uniform(0, 100, self.n_points).astype(np.float64)
        time = np.random.uniform(0, 1, self.n_points).astype(np.float64)

        metadata = {
            'test_name': 'moderate_spatial_cluster',
            'description': '1M points in 10km × 10km region',
            'spatial_extent': '10km × 10km',
            'center': f'({center_lat}, {center_lon})',
            'density': '~10 points/m²',
            'expected_collision_behavior': 'Moderate spatial collisions at fine levels'
        }

        return lat, lon, elev, time, metadata

    def generate_moderate_temporal_cluster(self, n_locations=1000):
        """Test 3: 1000 locations × 1000 time samples each."""
        samples_per_location = self.n_points // n_locations

        # Generate 1000 random locations
        location_lats = np.random.uniform(-90, 90, n_locations).astype(np.float64)
        location_lons = np.random.uniform(-180, 180, n_locations).astype(np.float64)
        location_elevs = np.random.uniform(0, 3000, n_locations).astype(np.float64)

        # Repeat each location for all its time samples
        lat = np.repeat(location_lats, samples_per_location).astype(np.float64)
        lon = np.repeat(location_lons, samples_per_location).astype(np.float64)
        elev = np.repeat(location_elevs, samples_per_location).astype(np.float64)
        time = np.random.uniform(0, 1, self.n_points).astype(np.float64)

        metadata = {
            'test_name': 'moderate_temporal_cluster',
            'description': '1000 locations with 1000 time samples each',
            'n_locations': n_locations,
            'samples_per_location': samples_per_location,
            'expected_collision_behavior': 'Temporal grid stress test'
        }

        return lat, lon, elev, time, metadata

    def generate_moderate_spatiotemporal(self, center_lat=40.7128, center_lon=-74.0060):
        """Test 4: 1M points in 1km × 1km × 1 hour window."""
        # 1km ≈ 0.009 degrees
        lat_offset = np.random.uniform(-0.0045, 0.0045, self.n_points).astype(np.float64)
        lon_offset = np.random.uniform(-0.0045, 0.0045, self.n_points).astype(np.float64)

        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        elev = np.random.uniform(0, 50, self.n_points).astype(np.float64)
        # 1 hour = 1/24 of day, assuming time range spans years, use fraction
        time = np.random.uniform(0.5, 0.5 + 1/8760, self.n_points).astype(np.float64)  # 1 hour in year span

        metadata = {
            'test_name': 'moderate_spatiotemporal',
            'description': '1M points in 1km × 1km × 1 hour',
            'spatial_extent': '1km × 1km',
            'temporal_extent': '1 hour',
            'expected_collision_behavior': 'High collisions in space and time'
        }

        return lat, lon, elev, time, metadata

    def generate_extreme_spatial_single(self, center_lat=51.5074, center_lon=-0.1278):
        """Test 5: 1M points in 10m × 10m region."""
        # 10m ≈ 0.00009 degrees
        lat_offset = np.random.uniform(-0.000045, 0.000045, self.n_points).astype(np.float64)
        lon_offset = np.random.uniform(-0.000045, 0.000045, self.n_points).astype(np.float64)

        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        elev = np.random.uniform(0, 5, self.n_points).astype(np.float64)  # Within building height
        time = np.random.uniform(0, 1, self.n_points).astype(np.float64)

        metadata = {
            'test_name': 'extreme_spatial_single',
            'description': '1M points in 10m × 10m region (single cluster)',
            'spatial_extent': '10m × 10m',
            'density': '~10,000 points/m²',
            'expected_collision_behavior': 'Extreme spatial hash stress test'
        }

        return lat, lon, elev, time, metadata

    def generate_extreme_spatial_multi(self, n_clusters=10):
        """Test 6: 10 clusters × 100k points each, 10m × 10m per cluster."""
        points_per_cluster = self.n_points // n_clusters

        # Distribute clusters globally
        cluster_lats = np.random.uniform(-90, 90, n_clusters).astype(np.float64)
        cluster_lons = np.random.uniform(-180, 180, n_clusters).astype(np.float64)

        lat_list = []
        lon_list = []
        elev_list = []
        time_list = []

        for i in range(n_clusters):
            lat_offset = np.random.uniform(-0.000045, 0.000045, points_per_cluster).astype(np.float64)
            lon_offset = np.random.uniform(-0.000045, 0.000045, points_per_cluster).astype(np.float64)

            lat_list.append(cluster_lats[i] + lat_offset)
            lon_list.append(cluster_lons[i] + lon_offset)
            elev_list.append(np.random.uniform(0, 5, points_per_cluster).astype(np.float64))
            time_list.append(np.random.uniform(0, 1, points_per_cluster).astype(np.float64))

        lat = np.concatenate(lat_list).astype(np.float64)
        lon = np.concatenate(lon_list).astype(np.float64)
        elev = np.concatenate(elev_list).astype(np.float64)
        time = np.concatenate(time_list).astype(np.float64)

        metadata = {
            'test_name': 'extreme_spatial_multi',
            'description': f'{n_clusters} clusters, 100k points each, 10m × 10m per cluster',
            'n_clusters': n_clusters,
            'points_per_cluster': points_per_cluster,
            'spatial_extent_per_cluster': '10m × 10m',
            'expected_collision_behavior': 'Multiple spatial hotspots'
        }

        return lat, lon, elev, time, metadata

    def generate_extreme_temporal_single(self, center_time=0.5):
        """Test 7: 1M points within 1 hour time window, spatially distributed."""
        lat = np.random.uniform(-90, 90, self.n_points).astype(np.float64)
        lon = np.random.uniform(-180, 180, self.n_points).astype(np.float64)
        elev = np.random.uniform(0, 3000, self.n_points).astype(np.float64)
        # 1 hour in normalized time (0-1 spanning 200 years)
        time = np.random.uniform(center_time, center_time + 1/(200*365*24), self.n_points).astype(np.float64)

        metadata = {
            'test_name': 'extreme_temporal_single',
            'description': '1M points in 1 hour window (single cluster)',
            'temporal_extent': '1 hour',
            'temporal_center': center_time,
            'expected_collision_behavior': 'Extreme temporal hash stress test'
        }

        return lat, lon, elev, time, metadata

    def generate_extreme_temporal_multi(self, n_clusters=10):
        """Test 8: 10 clusters × 100k points each, 1 hour per cluster, spread 1900-2100."""
        points_per_cluster = self.n_points // n_clusters

        # Distribute time clusters across full range
        cluster_times = np.linspace(0.1, 0.9, n_clusters, dtype=np.float64)

        lat_list = []
        lon_list = []
        elev_list = []
        time_list = []

        for i in range(n_clusters):
            lat_list.append(np.random.uniform(-90, 90, points_per_cluster).astype(np.float64))
            lon_list.append(np.random.uniform(-180, 180, points_per_cluster).astype(np.float64))
            elev_list.append(np.random.uniform(0, 3000, points_per_cluster).astype(np.float64))
            # 1 hour window around cluster center
            time_list.append(np.random.uniform(
                cluster_times[i],
                cluster_times[i] + 1/(200*365*24),
                points_per_cluster
            ).astype(np.float64))

        lat = np.concatenate(lat_list).astype(np.float64)
        lon = np.concatenate(lon_list).astype(np.float64)
        elev = np.concatenate(elev_list).astype(np.float64)
        time = np.concatenate(time_list).astype(np.float64)

        metadata = {
            'test_name': 'extreme_temporal_multi',
            'description': f'{n_clusters} clusters, 100k points each, 1 hour per cluster',
            'n_clusters': n_clusters,
            'points_per_cluster': points_per_cluster,
            'temporal_extent_per_cluster': '1 hour',
            'expected_collision_behavior': 'Multiple temporal hotspots'
        }

        return lat, lon, elev, time, metadata

    def generate_continental_sparse(self):
        """Test 9: 1M points across North America, evenly distributed."""
        # North America approximate bounds
        lat = np.random.uniform(25, 72, self.n_points).astype(np.float64)  # ~25°N to 72°N
        lon = np.random.uniform(-170, -50, self.n_points).astype(np.float64)  # ~170°W to 50°W
        elev = np.random.uniform(0, 4000, self.n_points).astype(np.float64)
        time = np.random.uniform(0, 1, self.n_points).astype(np.float64)

        metadata = {
            'test_name': 'continental_sparse',
            'description': '1M points across North America (sparse coverage)',
            'spatial_extent': 'Continental (~20M km²)',
            'density': '~0.05 points/km²',
            'expected_collision_behavior': 'Very low collisions due to sparsity'
        }

        return lat, lon, elev, time, metadata

    def generate_time_series(self, n_locations=10000, n_times=100):
        """Test 10: 10,000 locations × 100 time points each."""
        # Generate random locations
        location_lats = np.random.uniform(-90, 90, n_locations).astype(np.float64)
        location_lons = np.random.uniform(-180, 180, n_locations).astype(np.float64)
        location_elevs = np.random.uniform(0, 3000, n_locations).astype(np.float64)

        # Generate regular time samples
        times = np.linspace(0, 1, n_times, dtype=np.float64)

        # Create all combinations
        lat_list = []
        lon_list = []
        elev_list = []
        time_list = []

        for i in range(n_locations):
            lat_list.append(np.full(n_times, location_lats[i], dtype=np.float64))
            lon_list.append(np.full(n_times, location_lons[i], dtype=np.float64))
            elev_list.append(np.full(n_times, location_elevs[i], dtype=np.float64))
            time_list.append(times)

        lat = np.concatenate(lat_list).astype(np.float64)
        lon = np.concatenate(lon_list).astype(np.float64)
        elev = np.concatenate(elev_list).astype(np.float64)
        time = np.concatenate(time_list).astype(np.float64)

        metadata = {
            'test_name': 'time_series',
            'description': f'{n_locations} locations with {n_times} regular time samples',
            'n_locations': n_locations,
            'n_times': n_times,
            'expected_collision_behavior': 'Balanced spatial and temporal distribution'
        }

        return lat, lon, elev, time, metadata


def profile_collisions(coords, test_metadata, output_base_dir, spatial_levels=24, temporal_levels=24,
                      spatial_log2_hashmap_size=22, temporal_log2_hashmap_size=22):
    """
    Profile hash collisions for given coordinates.

    Args:
        coords: Tensor of shape [N, 4] with [lat, lon, elev, time_normalized]
        test_metadata: Dict with test information
        output_base_dir: Base directory for outputs
        spatial_levels: Number of spatial levels
        temporal_levels: Number of temporal levels
        spatial_log2_hashmap_size: Spatial hash table size (log2)
        temporal_log2_hashmap_size: Temporal hash table size (log2)

    Returns:
        Summary dictionary
    """
    print("="*80)
    print(f"TEST: {test_metadata['test_name'].upper()}")
    print("="*80)
    print(f"Description: {test_metadata['description']}")
    print(f"Points: {len(coords):,}")
    print()

    # Print representative samples to reveal structure
    print("Representative Samples:")
    print("-" * 60)

    test_name = test_metadata['test_name']

    if 'uniform' in test_name or 'continental_sparse' in test_name:
        # Show geographic distribution
        lat, lon, elev, time = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        print(f"  Latitude:   [{lat.min():.10f}° to {lat.max():.10f}°]")
        print(f"  Longitude:  [{lon.min():.10f}° to {lon.max():.10f}°]")
        print(f"  Elevation:  [{elev.min():.6f}m to {elev.max():.6f}m]")
        print(f"  Time:       [{time.min():.12f} to {time.max():.12f}] (normalized)")
        print(f"\n  Random samples (float64 precision):")
        indices = torch.randperm(len(coords))[:3]
        for idx in indices:
            c = coords[idx]
            print(f"    ({c[0]:15.10f}°, {c[1]:16.10f}°, {c[2]:12.6f}m) @ t={c[3]:.12f}")

    elif 'spatial_single' in test_name or 'spatial_cluster' in test_name:
        # Show spatial clustering - compute range
        lat, lon, elev = coords[:, 0], coords[:, 1], coords[:, 2]
        lat_range = (lat.max() - lat.min()).item()
        lon_range = (lon.max() - lon.min()).item()

        # Approximate meters (at equator: 1° ≈ 111km)
        lat_meters = lat_range * 111000
        lon_meters = lon_range * 111000 * torch.cos(lat.mean() * 3.14159 / 180).item()

        print(f"  Center:     ({lat.mean():.12f}°, {lon.mean():.12f}°)")
        print(f"  Lat range:  {lat_range:.12f}° (~{lat_meters:.3f}m)")
        print(f"  Lon range:  {lon_range:.12f}° (~{lon_meters:.3f}m)")
        print(f"  Elevation:  [{elev.min():.6f}m to {elev.max():.6f}m]")
        print(f"\n  Sample points (float64 precision):")
        for i in range(min(3, len(coords))):
            c = coords[i]
            print(f"    ({c[0]:15.12f}°, {c[1]:16.12f}°, {c[2]:10.6f}m) @ t={c[3]:.12f}")

    elif 'spatial_multi' in test_name:
        # Show multiple cluster centers
        n_clusters = test_metadata.get('n_clusters', 10)
        points_per = len(coords) // n_clusters
        print(f"  {n_clusters} clusters × {points_per:,} points each\n")
        print(f"  Cluster centers (float64 precision):")
        for i in range(n_clusters):
            cluster_start = i * points_per
            cluster_end = (i + 1) * points_per
            cluster_coords = coords[cluster_start:cluster_end]
            lat_mean = cluster_coords[:, 0].mean().item()
            lon_mean = cluster_coords[:, 1].mean().item()
            print(f"    Cluster {i+1:2d}: ({lat_mean:15.10f}°, {lon_mean:16.10f}°)")

    elif 'temporal_single' in test_name or 'temporal_cluster' in test_name:
        # Show temporal clustering
        time = coords[:, 3]
        time_range = (time.max() - time.min()).item()

        # Convert to approximate time span (assuming 0-1 = 200 years)
        total_seconds = time_range * 200 * 365.25 * 24 * 3600
        if total_seconds < 3600:
            time_span = f"{total_seconds/60:.6f} minutes"
        elif total_seconds < 86400:
            time_span = f"{total_seconds/3600:.6f} hours"
        else:
            time_span = f"{total_seconds/86400:.6f} days"

        print(f"  Time range: {time_range:.15f} normalized (~{time_span})")
        print(f"  Time span:  [{time.min():.15f} to {time.max():.15f}]")
        print(f"  Spatial:    Global distribution")
        print(f"\n  Sample points (sorted by time, float64 precision):")
        sorted_idx = torch.argsort(time)[:5]
        for idx in sorted_idx:
            c = coords[idx]
            print(f"    ({c[0]:15.10f}°, {c[1]:16.10f}°) @ t={c[3]:.15f}")

    elif 'temporal_multi' in test_name:
        # Show multiple temporal clusters
        n_clusters = test_metadata.get('n_clusters', 10)
        points_per = len(coords) // n_clusters
        print(f"  {n_clusters} temporal clusters × {points_per:,} points each\n")
        print(f"  Temporal cluster centers (float64 precision):")
        for i in range(n_clusters):
            cluster_start = i * points_per
            cluster_end = (i + 1) * points_per
            time_mean = coords[cluster_start:cluster_end, 3].mean().item()
            # Convert to approximate year (0=1900, 1=2100)
            year = 1900 + time_mean * 200
            print(f"    Cluster {i+1:2d}: t={time_mean:.15f} (~year {year:.1f})")

    elif 'spatiotemporal' in test_name:
        # Show both dimensions
        lat, lon, elev, time = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        lat_range = (lat.max() - lat.min()).item() * 111000  # meters
        lon_range = (lon.max() - lon.min()).item() * 111000
        time_range = (time.max() - time.min()).item() * 200 * 365.25 * 24  # hours

        print(f"  Spatial extent: ~{lat_range:.6f}m × {lon_range:.6f}m")
        print(f"  Temporal span:  ~{time_range:.6f} hours")
        print(f"  Center: ({lat.mean():.12f}°, {lon.mean():.12f}°)")
        print(f"\n  Sample points (float64 precision):")
        for i in range(min(3, len(coords))):
            c = coords[i]
            print(f"    ({c[0]:15.12f}°, {c[1]:16.12f}°, {c[2]:10.6f}m) @ t={c[3]:.15f}")

    elif 'time_series' in test_name:
        # Show time series structure
        n_locations = test_metadata.get('n_locations', 10000)
        n_times = test_metadata.get('n_times', 100)

        print(f"  Structure: {n_locations:,} locations × {n_times} time steps")
        print(f"  Time steps: equally spaced from {coords[0, 3]:.12f} to {coords[-1, 3]:.12f}")
        print(f"\n  First 3 locations (showing temporal progression, float64 precision):")
        for loc_idx in range(min(3, n_locations)):
            # Get all times for this location
            start_idx = loc_idx * n_times
            first_coord = coords[start_idx]
            last_coord = coords[start_idx + n_times - 1]
            print(f"    Location {loc_idx+1}: ({first_coord[0]:15.10f}°, {first_coord[1]:16.10f}°)")
            print(f"      Time: {first_coord[3]:.15f} → {last_coord[3]:.15f} ({n_times} steps)")

    print()

    max_tracked = len(coords)
    # Use float64 precision throughout (CUDA kernel now supports double inputs)
    coords_gpu = coords.cuda()  # Keep float64

    # Initialize Earth4D with collision tracking
    print("Initializing Earth4D...")
    model = Earth4D(
        spatial_levels=spatial_levels,
        temporal_levels=temporal_levels,
        spatial_log2_hashmap_size=spatial_log2_hashmap_size,
        temporal_log2_hashmap_size=temporal_log2_hashmap_size,
        enable_collision_tracking=True,
        max_tracked_examples=max_tracked,
        verbose=True
    ).cuda()

    # Store time as simple indices for synthetic data
    model.datetime_strings = [str(i) for i in range(max_tracked)]

    print(f"Model output dimension: {model.get_output_dim()}")

    # Run forward pass to collect collision data
    print("\n" + "="*60)
    print("COLLECTING HASH COLLISION DATA (using float64 inputs)")
    print("="*60)

    with torch.no_grad():
        batch_size = 5000
        total_processed = 0

        for i in range(0, max_tracked, batch_size):
            end_idx = min(i + batch_size, max_tracked)
            batch = coords_gpu[i:end_idx]  # Use float64

            features = model(batch)

            total_processed += batch.shape[0]
            if (i // batch_size + 1) % 20 == 0 or end_idx == max_tracked:
                print(f"Processed batch {i//batch_size + 1}: {total_processed}/{max_tracked} samples")

        print(f"✓ Completed processing {total_processed} samples")
        print(f"✓ Tracked coordinates: {model.collision_tracking_data['coordinates']['count']}")

    # Export complete collision data
    print("\n" + "="*60)
    print("EXPORTING COLLISION PROFILING DATA")
    print("="*60)

    # Create test-specific output directory
    test_name = test_metadata['test_name']
    output_dir = os.path.join(output_base_dir, test_name)

    # Add test metadata to summary (use .pt format for fast GPU operations)
    summary = model.export_collision_data(output_dir, format='pt')
    summary['test_metadata'] = test_metadata

    # Update JSON metadata with test info
    import json
    json_path = Path(output_dir) / "earth4d_collision_metadata.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    metadata['test'] = test_metadata
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Collision rate analysis using GPU-accelerated operations
    print("\n" + "="*60)
    print("COLLISION RATE ANALYSIS (GPU-Accelerated)")
    print("="*60)

    # Load .pt file with float64 precision
    pt_path = Path(output_dir) / "collision_data.pt"
    data = torch.load(pt_path)

    # Move to GPU for fast processing
    normalized_coords = data['coordinates']['normalized'].cuda()  # float64
    n_points = normalized_coords.shape[0]

    print(f"Loaded {n_points:,} points with float64 precision")
    print(f"Coordinate precision: {normalized_coords.dtype}")
    print()

    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        hash_indices = data['hash_indices'][grid].cuda()  # [N, num_levels]
        num_levels = hash_indices.shape[1]

        print(f"{grid.upper()} Grid Collision Rates:")

        # Apply correct coordinate transformations on GPU
        if grid == 'xyz':
            # XYZ: (x, y, z) -> normalized to [0, 1]
            x_norm = (normalized_coords[:, 0] + 1.0) / 2.0
            y_norm = (normalized_coords[:, 1] + 1.0) / 2.0
            z_norm = (normalized_coords[:, 2] + 1.0) / 2.0
            coord_data = torch.stack([x_norm, y_norm, z_norm], dim=1)
        else:
            # Temporal grids: apply time scaling
            t_scaled = (normalized_coords[:, 3] * 2.0 - 1.0) * 0.9
            t_norm = (t_scaled + 1.0) / 2.0

            if grid == 'xyt':
                x_norm = (normalized_coords[:, 0] + 1.0) / 2.0
                y_norm = (normalized_coords[:, 1] + 1.0) / 2.0
                coord_data = torch.stack([x_norm, y_norm, t_norm], dim=1)
            elif grid == 'yzt':
                y_norm = (normalized_coords[:, 1] + 1.0) / 2.0
                z_norm = (normalized_coords[:, 2] + 1.0) / 2.0
                coord_data = torch.stack([y_norm, z_norm, t_norm], dim=1)
            elif grid == 'xzt':
                x_norm = (normalized_coords[:, 0] + 1.0) / 2.0
                z_norm = (normalized_coords[:, 2] + 1.0) / 2.0
                coord_data = torch.stack([x_norm, z_norm, t_norm], dim=1)

        # Quantize coordinates to integers for exact comparison (vectorized approach)
        # Use very high precision: 1e12 gives sub-micrometer precision
        PRECISION = 1e12
        coords_int = (coord_data * PRECISION).long()

        # Analyze each level with fully vectorized GPU operations
        collision_rates = []
        for level in range(num_levels):
            level_indices = hash_indices[:, level]

            # Create unique identifier combining hash index and coordinates
            # Stack: [hash_index, coord1, coord2, ...]
            n_dims = coords_int.shape[1]
            if n_dims == 3:
                combined = torch.stack([
                    level_indices,
                    coords_int[:, 0],
                    coords_int[:, 1],
                    coords_int[:, 2]
                ], dim=1)
            else:  # 2D coordinates
                combined = torch.stack([
                    level_indices,
                    coords_int[:, 0],
                    coords_int[:, 1]
                ], dim=1)

            # Fully vectorized collision detection (no Python loops)
            unique_pairs = torch.unique(combined, dim=0)

            # Count how many unique (hash, coord) pairs exist for each hash index
            # unique_pairs[:, 0] contains hash indices
            hash_in_pairs = unique_pairs[:, 0]
            unique_hash_vals, pair_counts_per_hash = torch.unique(hash_in_pairs, return_counts=True)

            # Hashes with >1 unique coordinates are collisions
            collision_hashes = unique_hash_vals[pair_counts_per_hash > 1]

            # Count how many original points map to collision hashes
            if len(collision_hashes) > 0:
                # Create mask for points that map to collision hashes
                collision_mask = torch.isin(level_indices, collision_hashes)
                n_collisions = collision_mask.sum().item()
            else:
                n_collisions = 0

            collision_rate = n_collisions / n_points if n_points > 0 else 0
            collision_rates.append((level, collision_rate))

            # Progress indicator
            if (level + 1) % 4 == 0 or level == 0:
                print(f"  Progress: Level {level + 1}/{num_levels} analyzed", end='\r')

        # Show all levels
        for level, collision_rate in collision_rates:
            print(f"  Level {level:2d}: {collision_rate:.1%}")

        print()

    # File size info
    pt_size = pt_path.stat().st_size / (1024 * 1024)
    json_size = json_path.stat().st_size / 1024

    print(f"Exported files:")
    print(f"  PT: {pt_path} ({pt_size:.2f} MB)")
    print(f"  JSON: {json_path} ({json_size:.2f} KB)")
    print()

    return summary


def run_synthetic_tests(output_dir="hash_collision_tests", n_points=1_000_000):
    """Run all synthetic collision tests."""
    print("\n" + "="*80)
    print("SYNTHETIC HASH COLLISION TEST SUITE")
    print("="*80)
    print(f"Running 10 tests with {n_points:,} points each")
    print()

    generator = SpatiotemporalPointGenerator(n_points=n_points)

    # Define all tests
    tests = [
        generator.generate_uniform,
        generator.generate_moderate_spatial_cluster,
        generator.generate_moderate_temporal_cluster,
        generator.generate_moderate_spatiotemporal,
        generator.generate_extreme_spatial_single,
        generator.generate_extreme_spatial_multi,
        generator.generate_extreme_temporal_single,
        generator.generate_extreme_temporal_multi,
        generator.generate_continental_sparse,
        generator.generate_time_series,
    ]

    summaries = []

    for i, test_func in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"RUNNING TEST {i}/10")
        print(f"{'='*80}\n")

        # Generate coordinates (float64 for maximum precision)
        lat, lon, elev, time, metadata = test_func()
        coords = torch.stack([
            torch.tensor(lat, dtype=torch.float64),
            torch.tensor(lon, dtype=torch.float64),
            torch.tensor(elev, dtype=torch.float64),
            torch.tensor(time, dtype=torch.float64)
        ], dim=1)

        # Profile
        summary = profile_collisions(coords, metadata, output_dir)
        summaries.append(summary)

        print(f"\n✅ Test {i}/10 completed: {metadata['test_name']}\n")

    print("\n" + "="*80)
    print("ALL SYNTHETIC TESTS COMPLETED")
    print("="*80)
    print(f"Results saved to: {output_dir}/")

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hash Collision Profiler')
    parser.add_argument('--n-points', type=int, default=1_000_000,
                       help='Number of points per test (default: 1M)')
    parser.add_argument('--output-dir', type=str, default='hash_collision_tests',
                       help='Output directory for test results')

    args = parser.parse_args()

    run_synthetic_tests(output_dir=args.output_dir, n_points=args.n_points)
