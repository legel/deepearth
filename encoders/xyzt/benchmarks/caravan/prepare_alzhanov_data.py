#!/usr/bin/env python3
"""
Prepare Caravan dataset matching Alzhanov et al. (2025) setup exactly.

Extracts:
- 150 Caravan basins (randomly selected by Alzhanov et al.)
- Uba River basin (ubakz_99999999)

Time splits:
- Pre-training: 1995-2009
- Validation: 2010-2011
- Test: 2012-2020
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_basin_list(basin_list_file):
    """Load basin IDs from text file."""
    with open(basin_list_file, 'r') as f:
        basins = [line.strip() for line in f if line.strip()]
    return basins

def prepare_alzhanov_dataset():
    """Prepare dataset matching Alzhanov et al. exact setup."""

    print("="*80)
    print("Preparing Caravan Dataset - Alzhanov et al. (2025) Setup")
    print("="*80)

    # Paths
    data_dir = Path(__file__).parent / "data"
    full_data_path = data_dir / "caravan_full.csv"
    basin_list_path = Path(__file__).parent / "basin_lists" / "alzhanov_150_basins.txt"
    output_path = data_dir / "caravan_alzhanov_150plus1.csv"

    # Load basin list
    print(f"\nLoading basin list from: {basin_list_path}")
    basins = load_basin_list(basin_list_path)
    print(f"  Found {len(basins)} basins")
    print(f"  First 5: {basins[:5]}")
    print(f"  Last basin (Uba River): {basins[-1]}")

    # Load full Caravan data
    print(f"\nLoading full Caravan data from: {full_data_path}")
    print("  This may take a few minutes...")
    df = pd.read_csv(full_data_path)
    print(f"  Loaded {len(df):,} rows, {len(df['gauge_id'].unique())} unique basins")
    print(f"  Columns: {list(df.columns)}")

    # Filter to selected basins
    print(f"\nFiltering to {len(basins)} selected basins...")
    df_filtered = df[df['gauge_id'].isin(basins)].copy()
    print(f"  Filtered to {len(df_filtered):,} rows")
    print(f"  Unique basins found: {df_filtered['gauge_id'].nunique()}")

    # Check which basins are missing
    found_basins = set(df_filtered['gauge_id'].unique())
    requested_basins = set(basins)
    missing_basins = requested_basins - found_basins
    if missing_basins:
        print(f"\n  WARNING: {len(missing_basins)} basins not found in data:")
        for basin in sorted(missing_basins):
            print(f"    - {basin}")

    # Convert date column to datetime
    print("\nParsing dates...")
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])

    # Filter to time range 1995-2020 (matching Alzhanov)
    print("\nFiltering to 1995-2020 time range...")
    df_filtered = df_filtered[
        (df_filtered['date'] >= '1995-01-01') &
        (df_filtered['date'] <= '2020-12-31')
    ].copy()
    print(f"  Filtered to {len(df_filtered):,} rows (1995-2020)")

    # Add split labels
    print("\nAdding train/val/test split labels...")
    def assign_split(date):
        year = date.year
        if year < 2010:
            return 'train'
        elif year < 2012:
            return 'val'
        else:
            return 'test'

    df_filtered['split'] = df_filtered['date'].apply(assign_split)

    # Statistics by split
    print("\nDataset statistics by split:")
    for split in ['train', 'val', 'test']:
        split_data = df_filtered[df_filtered['split'] == split]
        print(f"  {split:5s}: {len(split_data):,} rows, "
              f"{split_data['gauge_id'].nunique()} basins, "
              f"{split_data['date'].min().year}-{split_data['date'].max().year}")

    # Statistics by basin
    print("\nSample basin statistics:")
    basin_stats = df_filtered.groupby('gauge_id').agg({
        'date': ['min', 'max', 'count'],
        'streamflow_mm_per_day': ['mean', 'median', 'std']
    })
    print(f"\n  First 10 basins:")
    print(basin_stats.head(10))

    # Check for Uba River specifically
    if 'ubakz_99999999' in df_filtered['gauge_id'].values:
        print("\nUba River (ubakz_99999999) statistics:")
        uba_data = df_filtered[df_filtered['gauge_id'] == 'ubakz_99999999']
        print(f"  Total observations: {len(uba_data):,}")
        print(f"  Date range: {uba_data['date'].min()} to {uba_data['date'].max()}")
        print(f"  Train: {len(uba_data[uba_data['split']=='train']):,} rows")
        print(f"  Val:   {len(uba_data[uba_data['split']=='val']):,} rows")
        print(f"  Test:  {len(uba_data[uba_data['split']=='test']):,} rows")
        print(f"  Mean streamflow: {uba_data['streamflow_mm_per_day'].mean():.3f} mm/day")
        print(f"  Median streamflow: {uba_data['streamflow_mm_per_day'].median():.3f} mm/day")
    else:
        print("\n  WARNING: Uba River (ubakz_99999999) NOT FOUND in dataset!")

    # Save to CSV
    print(f"\nSaving to: {output_path}")
    df_filtered.to_csv(output_path, index=False)
    print(f"  Saved {len(df_filtered):,} rows")

    # File size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)

    return df_filtered

if __name__ == "__main__":
    df = prepare_alzhanov_dataset()
