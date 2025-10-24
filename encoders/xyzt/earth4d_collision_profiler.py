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
    lat = torch.tensor(df['Latitude (WGS84, EPSG:4326)'].values, dtype=torch.float32)
    lon = torch.tensor(df['Longitude (WGS84, EPSG:4326)'].values, dtype=torch.float32)
    elev = torch.tensor(df['Elevation (m.a.s.l)'].values, dtype=torch.float32)
    
    # Process time from sampling date
    dates = df['Sampling date (YYYYMMDD)'].values
    min_date = dates.min()
    max_date = dates.max()
    time = torch.tensor((dates - min_date) / (max_date - min_date), dtype=torch.float32)
    
    # Stack coordinates
    coords = torch.stack([lat, lon, elev, time], dim=1)
    print(f"Coordinate tensor shape: {coords.shape}")
    
    # Use complete dataset for production profiling
    #max_tracked = min(1000, len(coords))  # Smaller set for testing
    max_tracked = len(coords)          # Process complete LFMC dataset
    coords_subset = coords[:max_tracked].cuda()
    print(f"Using {max_tracked} samples for hash collision profiling")
    
    # Initialize Earth4D with production configuration and collision tracking
    print("\nInitializing Earth4D with production configuration...")
    model = Earth4D(
        spatial_levels=24,           # Production: 24 levels  
        temporal_levels=19,          # Production: 19 levels
        spatial_log2_hashmap_size=23,  # Production: 8M entries
        temporal_log2_hashmap_size=18, # Production: 256K entries
        enable_collision_tracking=True,
        max_tracked_examples=max_tracked,
        verbose=False  # Disable verbose for cleaner output
    ).cuda()
    
    print(f"Model output dimension: {model.get_output_dim()}")
    
    # Run forward pass to collect collision data
    print("\n" + "="*60)
    print("COLLECTING HASH COLLISION DATA")
    print("="*60)
    
    with torch.no_grad():
        # Process in batches to demonstrate coordinate tracking
        batch_size = 100
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
    grid_cols = [col for col in exported_df.columns if 'level' in col and 'dim' in col]
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
            min_val = exported_df[col].min()
            max_val = exported_df[col].max()
            print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")
    
    # Check grid indices are reasonable
    print("\nGrid index sample check:")
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        level_0_cols = [col for col in exported_df.columns if f"{grid}_level_00_dim" in col]
        if level_0_cols:
            col = level_0_cols[0]
            min_val = exported_df[col].min()
            max_val = exported_df[col].max()
            print(f"  {col}: [{min_val}, {max_val}]")
    
    # Scientific analysis preview
    print("\n" + "="*60)
    print("SCIENTIFIC ANALYSIS PREVIEW")
    print("="*60)
    
    # Calculate collision statistics by grid and level
    for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
        collision_cols = [col for col in exported_df.columns if f"{grid}_level" in col and "collision" in col]
        if collision_cols:
            collision_rates = []
            for col in sorted(collision_cols):
                level = int(col.split('_level_')[1].split('_')[0])
                rate = exported_df[col].mean()
                collision_rates.append((level, rate))
            
            print(f"\n{grid.upper()} Grid Collision Rates:")
            for level, rate in collision_rates[:5]:  # Show first 5 levels
                print(f"  Level {level:2d}: {rate:.1%}")
            if len(collision_rates) > 10:
                print("  ...")
                for level, rate in collision_rates[-5:]:  # Show last 5 levels
                    print(f"  Level {level:2d}: {rate:.1%}")
    
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
    print("✅ Scientific analysis ready data export")
    print(f"\nProfiling data ready for analysis at: {output_dir}")
    
    return summary

if __name__ == "__main__":
    profile_earth4d_collisions()