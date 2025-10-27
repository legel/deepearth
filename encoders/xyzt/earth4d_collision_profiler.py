#!/usr/bin/env python3
"""
Earth4D Hash Collision Profiler
===============================

Statistical profiling of hash collisions in Earth4D spatiotemporal encoding.

This profiler analyzes hash collision patterns across Earth4D's 4 grid spaces 
(xyz, xyt, yzt, xzt) using real-world planetary data. It provides comprehensive
data export for scientific analysis and visualization.

Features:
- Real-time collision tracking during CUDA hash encoding
- Complete coordinate preservation (original + normalized)
- Per-coordinate grid index export across all resolution levels  
- Professional CSV/JSON export for downstream analysis
- Production Earth4D configuration support

Author: Earth4D Team
License: MIT
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
# Add deepearth root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from encoders.xyzt.earth4d import Earth4D

def profile_earth4d_collisions():
    """Profile hash collisions in Earth4D with real LFMC data."""
    
    print("="*80)
    print("EARTH4D HASH COLLISION PROFILER")
    print("="*80)
    
    # Load real LFMC data (now located with the profiler)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lfmc_path = os.path.join(script_dir, "globe_lfmc_extracted.csv")
    print(f"Loading LFMC data from: {lfmc_path}")
    
    df = pd.read_csv(lfmc_path)
    print(f"Loaded {len(df)} LFMC samples")
    
    # Extract coordinates using actual LFMC column names
    lat = df['Latitude (WGS84, EPSG:4326)'].values
    lon = df['Longitude (WGS84, EPSG:4326)'].values
    elev = df['Elevation (m.a.s.l)'].values
    dates = df['Sampling date (YYYYMMDD)'].values
    
    # Create coordinate strings for deduplication
    coord_strings = [f"{lat[i]:.6f},{lon[i]:.6f},{elev[i]:.2f},{dates[i]}" for i in range(len(df))]
    
    # Find unique coordinates and their indices
    unique_coords, unique_indices = np.unique(coord_strings, return_index=True)
    print(f"Total samples: {len(df)}, Unique spatiotemporal coordinates: {len(unique_coords)}")
    
    # Keep only unique coordinates
    df_unique = df.iloc[unique_indices].copy()
    lat = torch.tensor(df_unique['Latitude (WGS84, EPSG:4326)'].values, dtype=torch.float32)
    lon = torch.tensor(df_unique['Longitude (WGS84, EPSG:4326)'].values, dtype=torch.float32)
    elev = torch.tensor(df_unique['Elevation (m.a.s.l)'].values, dtype=torch.float32)
    
    # Process time from sampling date - preserve original date format
    dates_unique = df_unique['Sampling date (YYYYMMDD)'].values
    min_date = dates_unique.min()
    max_date = dates_unique.max()
    time_normalized = torch.tensor((dates_unique - min_date) / (max_date - min_date), dtype=torch.float32)
    
    # Convert dates to datetime strings for export
    import datetime
    time_strings = []
    for date_int in dates_unique:
        date_str = str(date_int)
        if len(date_str) == 8:  # YYYYMMDD format
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            dt = datetime.datetime(year, month, day)
            time_strings.append(dt.strftime('%Y-%m-%d'))
        else:
            time_strings.append(str(date_int))  # fallback
    
    # Stack coordinates
    coords = torch.stack([lat, lon, elev, time_normalized], dim=1)
    print(f"Unique coordinate tensor shape: {coords.shape}")
    
    # Use all unique coordinates for profiling
    max_tracked = len(coords)
    coords_subset = coords.cuda()
    print(f"Processing {max_tracked} unique spatiotemporal coordinates")
    
    # Initialize Earth4D with collision tracking
    print("\nInitializing Earth4D...")
    model = Earth4D(
        spatial_levels=24,
        temporal_levels=24,
        spatial_log2_hashmap_size=22,
        temporal_log2_hashmap_size=22,
        enable_collision_tracking=True,
        max_tracked_examples=max_tracked,
        verbose=True
    ).cuda()
    
    # Store datetime strings for export
    model.datetime_strings = time_strings
    
    print(f"Model output dimension: {model.get_output_dim()}")
    
    # Run forward pass to collect collision data
    print("\n" + "="*60)
    print("COLLECTING HASH COLLISION DATA")
    print("="*60)
    
    with torch.no_grad():
        # Process in batches to demonstrate coordinate tracking
        batch_size = 5000
        total_processed = 0
        
        for i in range(0, max_tracked, batch_size):
            end_idx = min(i + batch_size, max_tracked)
            batch = coords_subset[i:end_idx]
            
            # Run forward pass (coordinates are automatically tracked)
            features = model(batch)
            
            total_processed += batch.shape[0]
            print(f"Processed batch {i//batch_size + 1}: {total_processed}/{max_tracked} samples")
        
        print(f"✓ Completed processing {total_processed} samples")
        print(f"✓ Tracked coordinates: {model.collision_tracking_data['coordinates']['count']}")
    
    # Export complete collision data
    print("\n" + "="*60)
    print("EXPORTING COLLISION PROFILING DATA")
    print("="*60)
    
    # Set output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "earth4d_collision_profiling")
    summary = model.export_collision_data(output_dir)
    
    # Verify the exported CSV format
    print("\n" + "="*60)
    print("VERIFYING EXPORTED DATA FORMAT")
    print("="*60)
    
    csv_path = Path(output_dir) / "earth4d_collision_data.csv"
    exported_df = pd.read_csv(csv_path)
    
    print(f"Exported CSV shape: {exported_df.shape}")
    print(f"CSV columns ({len(exported_df.columns)}):")
    
    # Show coordinate columns
    coord_cols = ['latitude', 'longitude', 'elevation_m', 'time_original', 
                  'x_normalized', 'y_normalized', 'z_normalized', 'time_normalized']
    print("  Coordinate columns:")
    for col in coord_cols:
        if col in exported_df.columns:
            print(f"    ✓ {col}")
        else:
            print(f"    ✗ {col} (missing)")
    
    # Show sample grid index columns
    print("  Sample grid index columns:")
    grid_cols = [col for col in exported_df.columns if 'level' in col and 'index' in col]
    for col in sorted(grid_cols)[:10]:  # Show first 10
        print(f"    ✓ {col}")
    if len(grid_cols) > 10:
        print(f"    ... and {len(grid_cols) - 10} more grid index columns")
    
    # Show sample collision flag columns
    collision_cols = [col for col in exported_df.columns if 'collision' in col]
    print(f"  Collision flag columns: {len(collision_cols)}")
    for col in sorted(collision_cols)[:5]:  # Show first 5
        print(f"    ✓ {col}")
    if len(collision_cols) > 5:
        print(f"    ... and {len(collision_cols) - 5} more collision columns")
    
    # Verify data integrity
    print("\n" + "="*60)
    print("DATA INTEGRITY VERIFICATION")
    print("="*60)
    
    # Check coordinate ranges
    print("Coordinate ranges:")
    for col in coord_cols:
        if col in exported_df.columns:
            if col == 'time_original':
                # Handle datetime strings
                if isinstance(exported_df[col].iloc[0], str):
                    print(f"  {col}: [{exported_df[col].iloc[0]} to {exported_df[col].iloc[-1]}] (datetime strings)")
                else:
                    print(f"  {col}: [{exported_df[col].min():.4f}, {exported_df[col].max():.4f}]")
            else:
                min_val = exported_df[col].min()
                max_val = exported_df[col].max()
                print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")
    
    # Check grid indices are reasonable
    print("\nGrid index sample check (hash table indices):")
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        level_0_cols = [col for col in exported_df.columns if f"{grid}_level_00_index" in col]
        if level_0_cols:
            col = level_0_cols[0]
            min_val = exported_df[col].min()
            max_val = exported_df[col].max()
            print(f"  {col}: [{min_val}, {max_val}]")
        else:
            print(f"  ⚠️  {grid}_level_00_index column not found")
    
    # Scientific analysis with collision detection
    print("\n" + "="*60)
    print("COLLISION RATE ANALYSIS")
    print("="*60)
    print("Analyzing collisions (different coordinates → same hash index)")
    print()

    # Calculate collision statistics by grid and level
    from collections import defaultdict

    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        index_cols = [col for col in exported_df.columns if f"{grid}_level_" in col and "_index" in col]
        index_cols = sorted(index_cols, key=lambda x: int(x.split('_level_')[1].split('_')[0]))

        if index_cols:
            print(f"{grid.upper()} Grid Collision Rates:")

            collision_rates = []

            # Get coordinates with SAME transformations as during encoding
            # Must match what CUDA kernel receives after ALL transformations
            if grid == 'xyz':
                # Spatial grid: apply HashEncoder normalization (coord + 1) / 2
                coord_data = (exported_df[['x_normalized', 'y_normalized', 'z_normalized']].values + 1.0) / 2.0
            else:
                # Temporal grids: apply BOTH transformations
                # 1. Time scaling: t_scaled = (time * 2 - 1) * 0.9
                # 2. HashEncoder normalization: (coord + 1) / 2
                t_scaled = (exported_df['time_normalized'].values * 2.0 - 1.0) * 0.9
                t_normalized = (t_scaled + 1.0) / 2.0

                if grid == 'xyt':
                    spatial_normalized = (exported_df[['x_normalized', 'y_normalized']].values + 1.0) / 2.0
                    coord_data = np.column_stack([spatial_normalized, t_normalized])
                elif grid == 'yzt':
                    spatial_normalized = (exported_df[['y_normalized', 'z_normalized']].values + 1.0) / 2.0
                    coord_data = np.column_stack([spatial_normalized, t_normalized])
                elif grid == 'xzt':
                    spatial_normalized = (exported_df[['x_normalized', 'z_normalized']].values + 1.0) / 2.0
                    coord_data = np.column_stack([spatial_normalized, t_normalized])

            for index_col in index_cols:
                level = int(index_col.split('_level_')[1].split('_')[0])

                # Get hash indices
                hash_indices = exported_df[index_col].values

                # Group by hash index and count unique coordinates
                hash_to_coords = defaultdict(list)
                for i in range(len(exported_df)):
                    hash_idx = hash_indices[i]
                    coord_tuple = tuple(coord_data[i])
                    hash_to_coords[hash_idx].append(coord_tuple)

                # Count collisions
                collisions = 0
                for hash_idx, coord_list in hash_to_coords.items():
                    unique_coords = set(coord_list)
                    if len(unique_coords) > 1:
                        # Collision: multiple different coordinates map to same index
                        collisions += len(coord_list)

                # Calculate rate
                total_coords = len(exported_df)
                collision_rate = collisions / total_coords if total_coords > 0 else 0

                collision_rates.append((level, collision_rate))

            # Show all levels
            for level, collision_rate in collision_rates:
                print(f"  Level {level:2d}: {collision_rate:.1%}")

            print()
    
    # File size information
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    json_path = Path(output_dir) / "earth4d_collision_metadata.json"
    json_size = json_path.stat().st_size / 1024  # KB
    
    print(f"\nExported files:")
    print(f"  CSV: {csv_path} ({csv_size:.2f} MB)")
    print(f"  JSON: {json_path} ({json_size:.2f} KB)")
    
    print("\n" + "="*80)
    print("EARTH4D COLLISION PROFILING COMPLETED SUCCESSFULLY")
    print("="*80)
    print("✅ Original coordinates tracked and exported")
    print("✅ Complete CSV with grid indices per coordinate")
    print("✅ Professional format: 8 coord columns + grid-level columns")
    print("✅ Collision analysis integrated")
    print("✅ Scientific analysis ready data export")

    print(f"\nProfiling data ready for analysis at: {output_dir}")

    return summary

if __name__ == "__main__":
    profile_earth4d_collisions()
