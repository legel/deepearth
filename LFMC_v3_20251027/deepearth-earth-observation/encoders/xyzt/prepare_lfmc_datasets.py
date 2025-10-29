#!/usr/bin/env python3
"""
LFMC Dataset Standardization Script
====================================

Converts raw CSV files to standardized Parquet format with optional .pt caching.

This script implements Phase 1 of the LFMC Modular Dataset Architecture Plan.

Usage:
    python prepare_lfmc_datasets.py --input-dir ./data --output-dir ./data --create-cache

Input Files (CSV):
    - globe_lfmc_extracted.csv           (Base LFMC data)
    - globe_lfmc_with_aef_embeddings.csv (AlphaEarth features)
    - globe_lfmc_extracted_with_daymet.csv (Weather data)

Output Files (Parquet):
    - lfmc_base.parquet      (~2 MB)
    - lfmc_aef.parquet       (~35-40 MB)
    - lfmc_daymet.parquet    (~5 MB)

Optional Cache Files (.pt):
    - cache/lfmc_base.pt
    - cache/lfmc_aef.pt
    - cache/lfmc_daymet.pt

Author: Claude Code
Date: 2025-10-28
Version: 1.0
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Try to import torch (optional, only needed for .pt caching)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. .pt caching will be disabled.")


class LFMCDataStandardizer:
    """Standardizes LFMC datasets from CSV to Parquet format."""

    def __init__(self, input_dir: str = "./data", output_dir: str = "./data", verbose: bool = True, apply_filters: bool = True):
        """
        Initialize the standardizer.

        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory for output Parquet files
            verbose: Print progress messages
            apply_filters: Apply quality filters (default: True)
                - Base: Remove invalid LFMC values, missing coords, bad dates
                - AEF: Remove failed extractions
                - Daymet: No filtering
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.apply_filters = apply_filters

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        self.stats = {}

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)

    def create_sample_id(self, lat: pd.Series, lon: pd.Series, date: pd.Series, species: pd.Series) -> pd.Series:
        """
        Create unique sample IDs based on lat, lon, date, species, and sequence number.

        Uses MD5 hash of concatenated values plus a sequence number for duplicates.
        This ensures truly unique IDs even when multiple measurements exist at the
        same location/date/species combination.

        Args:
            lat: Latitude values
            lon: Longitude values
            date: Date strings (YYYYMMDD format)
            species: Species names

        Returns:
            Series of unique int64 sample IDs
        """
        self.log("Generating unique sample IDs with sequence numbers...")

        # Create composite key: "lat_lon_date_species"
        composite_keys = (
            lat.astype(str) + "_" +
            lon.astype(str) + "_" +
            date.astype(str) + "_" +
            species.astype(str)
        )

        # Add sequence number for duplicates within each group
        # This handles cases where multiple measurements exist at same location/date/species
        sequence_nums = composite_keys.groupby(composite_keys).cumcount()

        # Create final composite key with sequence: "lat_lon_date_species_seq"
        final_keys = composite_keys + "_" + sequence_nums.astype(str)

        # Hash to int64
        sample_ids = final_keys.apply(
            lambda x: int(hashlib.md5(x.encode()).hexdigest()[:16], 16) % (2**63)
        )

        # Check for uniqueness
        n_unique = sample_ids.nunique()
        n_total = len(sample_ids)
        n_duplicated_locations = (sequence_nums > 0).sum()

        if n_unique < n_total:
            self.log(f"  WARNING: {n_total - n_unique} duplicate sample IDs still detected!")
        else:
            self.log(f"  [OK] All {n_total} sample IDs are unique")
            if n_duplicated_locations > 0:
                self.log(f"  [INFO] {n_duplicated_locations} samples had duplicate locations (sequence numbers added)")

        return sample_ids

    def standardize_base_lfmc(self, csv_path: str = None) -> pd.DataFrame:
        """
        Standardize the base LFMC dataset.

        Input columns (from globe_lfmc_extracted.csv):
            - "Latitude (WGS84, EPSG:4326)"
            - "Longitude (WGS84, EPSG:4326)"
            - "Elevation (m.a.s.l)"
            - "Sampling date (YYYYMMDD)"
            - "Sampling time (24h format)"
            - "LFMC value (%)"
            - "Species collected"

        Output columns (standardized):
            - sample_id (int64)
            - lat (float32)
            - lon (float32)
            - elevation_m (float32)
            - date (datetime64[ns])
            - lfmc_percent (float32)
            - species (string)

        Args:
            csv_path: Path to input CSV file (default: globe_lfmc_extracted.csv)

        Returns:
            Standardized DataFrame
        """
        if csv_path is None:
            csv_path = self.input_dir / "globe_lfmc_extracted.csv"

        self.log("\n" + "="*60)
        self.log("STANDARDIZING BASE LFMC DATASET")
        self.log("="*60)
        self.log(f"Reading: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        self.log(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Show original columns
        self.log(f"  Original columns: {list(df.columns)}")

        # Rename columns to standardized names
        column_mapping = {
            "Latitude (WGS84, EPSG:4326)": "lat",
            "Longitude (WGS84, EPSG:4326)": "lon",
            "Elevation (m.a.s.l)": "elevation_m",
            "Sampling date (YYYYMMDD)": "date_str",
            "Sampling time (24h format)": "time_str",
            "LFMC value (%)": "lfmc_percent",
            "Species collected": "species"
        }

        df = df.rename(columns=column_mapping)

        # Filter valid data (optional)
        if self.apply_filters:
            n_before = len(df)
            df = df[
                (df['lfmc_percent'] >= 0) &
                (df['lfmc_percent'] <= 600) &
                df['lat'].notna() &
                df['lon'].notna() &
                df['elevation_m'].notna()
            ].copy()
            n_after = len(df)

            if n_before > n_after:
                self.log(f"  Filtered {n_before - n_after:,} invalid rows ({100*(n_before-n_after)/n_before:.1f}%)")
        else:
            self.log(f"  Filtering disabled (--no-filter flag used)")

        # Convert date strings to datetime
        self.log("  Converting dates...")
        df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce')

        # Drop rows with invalid dates (optional)
        if self.apply_filters:
            n_before = len(df)
            df = df[df['date'].notna()].copy()
            n_after = len(df)
            if n_before > n_after:
                self.log(f"  Dropped {n_before - n_after:,} rows with invalid dates")

        # Create sample_id
        df['sample_id'] = self.create_sample_id(
            df['lat'], df['lon'], df['date_str'], df['species']
        )

        # Select final columns in order
        final_df = df[[
            'sample_id',
            'lat',
            'lon',
            'elevation_m',
            'date',
            'lfmc_percent',
            'species'
        ]].copy()

        # Convert to appropriate dtypes
        final_df['sample_id'] = final_df['sample_id'].astype('int64')
        final_df['lat'] = final_df['lat'].astype('float32')
        final_df['lon'] = final_df['lon'].astype('float32')
        final_df['elevation_m'] = final_df['elevation_m'].astype('float32')
        final_df['lfmc_percent'] = final_df['lfmc_percent'].astype('float32')
        final_df['species'] = final_df['species'].astype('string')

        # Statistics
        self.stats['base_lfmc'] = {
            'n_samples': len(final_df),
            'n_species': final_df['species'].nunique(),
            'date_range': (final_df['date'].min(), final_df['date'].max()),
            'lfmc_range': (final_df['lfmc_percent'].min(), final_df['lfmc_percent'].max()),
            'lat_range': (final_df['lat'].min(), final_df['lat'].max()),
            'lon_range': (final_df['lon'].min(), final_df['lon'].max())
        }

        self.log(f"\n  Statistics:")
        self.log(f"    Samples: {len(final_df):,}")
        self.log(f"    Species: {final_df['species'].nunique():,}")
        self.log(f"    Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        self.log(f"    LFMC range: {final_df['lfmc_percent'].min():.1f}% to {final_df['lfmc_percent'].max():.1f}%")
        self.log(f"    Lat range: {final_df['lat'].min():.2f} to {final_df['lat'].max():.2f}")
        self.log(f"    Lon range: {final_df['lon'].min():.2f} to {final_df['lon'].max():.2f}")

        self.log(f"\n  Final columns: {list(final_df.columns)}")
        self.log(f"  Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return final_df

    def standardize_aef_embeddings(self, csv_path: str = None, base_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Standardize the AlphaEarth Features (AEF) dataset.

        Extracts only the 64-dimensional embeddings and matches sample_ids from base dataset.

        Input columns (from globe_lfmc_with_aef_embeddings.csv):
            - latitude, longitude, Elevation (m.a.s.l), date_str, Species collected
            - aef_extraction_success
            - aef_00, aef_01, ..., aef_63

        Output columns:
            - sample_id (int64)
            - aef_00 through aef_63 (float32)

        Args:
            csv_path: Path to input CSV file (default: globe_lfmc_with_aef_embeddings.csv)
            base_df: Base LFMC DataFrame with sample_ids (for consistency checking)

        Returns:
            Standardized DataFrame with embeddings only
        """
        if csv_path is None:
            csv_path = self.input_dir / "globe_lfmc_with_aef_embeddings.csv"

        self.log("\n" + "="*60)
        self.log("STANDARDIZING AEF EMBEDDINGS DATASET")
        self.log("="*60)
        self.log(f"Reading: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        self.log(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Filter by AEF extraction success (optional)
        if self.apply_filters and 'aef_extraction_success' in df.columns:
            n_before = len(df)
            df = df[df['aef_extraction_success'] == True].copy()
            n_after = len(df)
            self.log(f"  Filtered by AEF extraction success: {n_before:,} -> {n_after:,} ({100*n_after/n_before:.1f}%)")
        elif not self.apply_filters:
            self.log(f"  Filtering disabled (keeping all {len(df):,} samples including failed extractions)")

        # Create sample_id using same method as base
        # Note: AEF file has different column names
        date_col = 'date_str' if 'date_str' in df.columns else 'Sampling date (YYYYMMDD)'
        species_col = 'Species collected'

        df['sample_id'] = self.create_sample_id(
            df['latitude'],
            df['longitude'],
            df[date_col],
            df[species_col]
        )

        # Extract AEF embedding columns
        aef_columns = [f'aef_{i:02d}' for i in range(64)]

        # Check if all AEF columns exist
        missing_cols = [col for col in aef_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing AEF columns: {missing_cols[:5]}... ({len(missing_cols)} total)")

        self.log(f"  Found all 64 AEF embedding columns")

        # Select sample_id + AEF columns
        final_df = df[['sample_id'] + aef_columns].copy()

        # Convert to float32
        for col in aef_columns:
            final_df[col] = final_df[col].astype('float32')

        final_df['sample_id'] = final_df['sample_id'].astype('int64')

        # Check for matches with base dataset if provided
        if base_df is not None:
            base_ids = set(base_df['sample_id'])
            aef_ids = set(final_df['sample_id'])

            matched = len(base_ids & aef_ids)
            base_only = len(base_ids - aef_ids)
            aef_only = len(aef_ids - base_ids)

            self.log(f"\n  Sample ID matching:")
            self.log(f"    Base LFMC samples: {len(base_ids):,}")
            self.log(f"    AEF samples: {len(aef_ids):,}")
            self.log(f"    Matched: {matched:,} ({100*matched/len(base_ids):.1f}% of base)")
            if base_only > 0:
                self.log(f"    Base only (no AEF): {base_only:,}")
            if aef_only > 0:
                self.log(f"    AEF only (no base): {aef_only:,}")

        # Statistics
        self.stats['aef'] = {
            'n_samples': len(final_df),
            'n_dimensions': 64,
            'embedding_range': (final_df[aef_columns].values.min(), final_df[aef_columns].values.max())
        }

        self.log(f"\n  Statistics:")
        self.log(f"    Samples: {len(final_df):,}")
        self.log(f"    Embedding dimensions: 64")
        self.log(f"    Value range: {final_df[aef_columns].values.min():.4f} to {final_df[aef_columns].values.max():.4f}")
        self.log(f"    Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return final_df

    def standardize_daymet(self, csv_path: str = None, base_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Standardize the Daymet weather dataset.

        Extracts only the weather features and matches sample_ids from base dataset.

        Input columns (from globe_lfmc_extracted_with_daymet.csv):
            - Base LFMC columns (same as base dataset)
            - prcp_d_minus2, prcp_d_minus1, prcp_d0
            - tmin_d_minus2, tmin_d_minus1, tmin_d0
            - tmax_d_minus2, tmax_d_minus1, tmax_d0
            - srad_d_minus2, srad_d_minus1, srad_d0
            - vp_d_minus2, vp_d_minus1, vp_d0
            - dayl_d_minus2, dayl_d_minus1, dayl_d0
            - swe_d_minus2, swe_d_minus1, swe_d0
            - daymet_start, daymet_end

        Output columns:
            - sample_id (int64)
            - 22 weather feature columns (float32)

        Args:
            csv_path: Path to input CSV file (default: globe_lfmc_extracted_with_daymet.csv)
            base_df: Base LFMC DataFrame with sample_ids (for consistency checking)

        Returns:
            Standardized DataFrame with weather features only
        """
        if csv_path is None:
            csv_path = self.input_dir / "globe_lfmc_extracted_with_daymet.csv"

        self.log("\n" + "="*60)
        self.log("STANDARDIZING DAYMET WEATHER DATASET")
        self.log("="*60)
        self.log(f"Reading: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        self.log(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Create sample_id using same method as base
        date_col = 'Sampling date (YYYYMMDD)'
        species_col = 'Species collected'
        lat_col = "Latitude (WGS84, EPSG:4326)"
        lon_col = "Longitude (WGS84, EPSG:4326)"

        df['sample_id'] = self.create_sample_id(
            df[lat_col],
            df[lon_col],
            df[date_col],
            df[species_col]
        )

        # Define Daymet feature columns
        daymet_columns = [
            'prcp_d_minus2', 'prcp_d_minus1', 'prcp_d0',
            'tmin_d_minus2', 'tmin_d_minus1', 'tmin_d0',
            'tmax_d_minus2', 'tmax_d_minus1', 'tmax_d0',
            'srad_d_minus2', 'srad_d_minus1', 'srad_d0',
            'vp_d_minus2', 'vp_d_minus1', 'vp_d0',
            'dayl_d_minus2', 'dayl_d_minus1', 'dayl_d0',
            'swe_d_minus2', 'swe_d_minus1', 'swe_d0',
            'daymet_start', 'daymet_end'
        ]

        # Check if all Daymet columns exist
        missing_cols = [col for col in daymet_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing Daymet columns: {missing_cols}")

        self.log(f"  Found all {len(daymet_columns)} Daymet feature columns")

        # Select sample_id + Daymet columns
        final_df = df[['sample_id'] + daymet_columns].copy()

        # Convert numeric columns to float32 (dates stay as strings for now)
        numeric_columns = daymet_columns[:-2]  # Exclude daymet_start and daymet_end
        for col in numeric_columns:
            final_df[col] = final_df[col].astype('float32')

        # Convert date columns to datetime
        final_df['daymet_start'] = pd.to_datetime(final_df['daymet_start'], errors='coerce')
        final_df['daymet_end'] = pd.to_datetime(final_df['daymet_end'], errors='coerce')

        final_df['sample_id'] = final_df['sample_id'].astype('int64')

        # Check for matches with base dataset if provided
        if base_df is not None:
            base_ids = set(base_df['sample_id'])
            daymet_ids = set(final_df['sample_id'])

            matched = len(base_ids & daymet_ids)
            base_only = len(base_ids - daymet_ids)
            daymet_only = len(daymet_ids - base_ids)

            self.log(f"\n  Sample ID matching:")
            self.log(f"    Base LFMC samples: {len(base_ids):,}")
            self.log(f"    Daymet samples: {len(daymet_ids):,}")
            self.log(f"    Matched: {matched:,} ({100*matched/len(base_ids):.1f}% of base)")
            if base_only > 0:
                self.log(f"    Base only (no Daymet): {base_only:,}")
            if daymet_only > 0:
                self.log(f"    Daymet only (no base): {daymet_only:,}")

        # Statistics
        self.stats['daymet'] = {
            'n_samples': len(final_df),
            'n_features': len(numeric_columns)
        }

        self.log(f"\n  Statistics:")
        self.log(f"    Samples: {len(final_df):,}")
        self.log(f"    Weather features: {len(numeric_columns)}")
        self.log(f"    Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return final_df

    def save_parquet(self, df: pd.DataFrame, filename: str) -> Tuple[str, float, float]:
        """
        Save DataFrame as Parquet file with compression.

        Args:
            df: DataFrame to save
            filename: Output filename (e.g., 'lfmc_base.parquet')

        Returns:
            Tuple of (output_path, file_size_mb, compression_ratio)
        """
        output_path = self.output_dir / filename

        self.log(f"\nSaving to Parquet: {output_path}")

        # Get memory size before saving
        mem_size_mb = df.memory_usage(deep=True).sum() / 1024**2

        # Save with compression
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',  # Fast compression with good ratio
            index=False
        )

        # Get file size
        file_size_mb = output_path.stat().st_size / 1024**2
        compression_ratio = (1 - file_size_mb / mem_size_mb) * 100 if mem_size_mb > 0 else 0

        self.log(f"  File size: {file_size_mb:.2f} MB")
        self.log(f"  Memory size: {mem_size_mb:.2f} MB")
        self.log(f"  Compression: {compression_ratio:.1f}% reduction")
        self.log(f"  OK Saved successfully")

        return str(output_path), file_size_mb, compression_ratio

    def create_pt_cache(self, df: pd.DataFrame, filename: str, cache_dir: str = "cache"):
        """
        Create PyTorch .pt cache file for fast loading.

        Converts DataFrame to tensors and saves with metadata.

        Args:
            df: DataFrame to cache
            filename: Output filename (e.g., 'lfmc_base.pt')
            cache_dir: Cache directory name
        """
        if not TORCH_AVAILABLE:
            self.log(f"\nSkipping PyTorch cache (torch not available): {filename}")
            return None

        cache_path = self.output_dir / cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)

        output_path = cache_path / filename

        self.log(f"\nCreating PyTorch cache: {output_path}")

        # Convert DataFrame to dictionary of tensors
        cache_data = {
            'sample_id': torch.from_numpy(df['sample_id'].values),
            'columns': list(df.columns)
        }

        # Add numeric columns as tensors
        for col in df.columns:
            if col == 'sample_id':
                continue

            if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                cache_data[col] = torch.from_numpy(df[col].values)
            elif df[col].dtype == 'datetime64[ns]':
                # Convert datetime to Unix timestamp
                cache_data[col] = torch.from_numpy(df[col].astype('int64').values / 1e9)  # seconds since epoch
            # Skip string columns for now (species)

        # Add metadata
        cache_data['metadata'] = {
            'n_samples': len(df),
            'created': pd.Timestamp.now().isoformat(),
            'source': 'prepare_lfmc_datasets.py'
        }

        # Save
        torch.save(cache_data, output_path)

        file_size_mb = output_path.stat().st_size / 1024**2
        self.log(f"  File size: {file_size_mb:.2f} MB")
        self.log(f"  OK Cache created successfully")

        return str(output_path)

    def run_full_standardization(self, create_cache: bool = True) -> Dict:
        """
        Run complete standardization pipeline for all datasets.

        Args:
            create_cache: Whether to create .pt cache files

        Returns:
            Dictionary with paths and statistics
        """
        results = {}

        self.log("\n" + "="*60)
        self.log("LFMC DATASET STANDARDIZATION PIPELINE")
        self.log("="*60)
        self.log(f"Input directory: {self.input_dir}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Create cache: {create_cache}")

        # Step 1: Standardize base LFMC
        base_df = self.standardize_base_lfmc()
        base_path, base_size, base_comp = self.save_parquet(base_df, 'lfmc_base.parquet')
        results['base'] = {
            'parquet_path': base_path,
            'parquet_size_mb': base_size,
            'compression_ratio': base_comp,
            'n_samples': len(base_df)
        }

        if create_cache:
            base_cache = self.create_pt_cache(base_df, 'lfmc_base.pt')
            results['base']['cache_path'] = base_cache

        # Step 2: Standardize AEF embeddings
        aef_df = self.standardize_aef_embeddings(base_df=base_df)
        aef_path, aef_size, aef_comp = self.save_parquet(aef_df, 'lfmc_aef.parquet')
        results['aef'] = {
            'parquet_path': aef_path,
            'parquet_size_mb': aef_size,
            'compression_ratio': aef_comp,
            'n_samples': len(aef_df)
        }

        if create_cache:
            aef_cache = self.create_pt_cache(aef_df, 'lfmc_aef.pt')
            results['aef']['cache_path'] = aef_cache

        # Step 3: Standardize Daymet weather
        daymet_df = self.standardize_daymet(base_df=base_df)
        daymet_path, daymet_size, daymet_comp = self.save_parquet(daymet_df, 'lfmc_daymet.parquet')
        results['daymet'] = {
            'parquet_path': daymet_path,
            'parquet_size_mb': daymet_size,
            'compression_ratio': daymet_comp,
            'n_samples': len(daymet_df)
        }

        if create_cache:
            daymet_cache = self.create_pt_cache(daymet_df, 'lfmc_daymet.pt')
            results['daymet']['cache_path'] = daymet_cache

        # Summary
        self.log("\n" + "="*60)
        self.log("STANDARDIZATION COMPLETE")
        self.log("="*60)

        total_parquet_size = base_size + aef_size + daymet_size
        self.log(f"\nTotal Parquet size: {total_parquet_size:.2f} MB")
        self.log(f"  Base LFMC:  {base_size:.2f} MB ({len(base_df):,} samples)")
        self.log(f"  AEF:        {aef_size:.2f} MB ({len(aef_df):,} samples)")
        self.log(f"  Daymet:     {daymet_size:.2f} MB ({len(daymet_df):,} samples)")

        self.log(f"\nFiles saved to: {self.output_dir}")
        self.log("  OK lfmc_base.parquet")
        self.log("  OK lfmc_aef.parquet")
        self.log("  OK lfmc_daymet.parquet")

        if create_cache:
            self.log(f"\nCache files saved to: {self.output_dir / 'cache'}")
            self.log("  OK lfmc_base.pt")
            self.log("  OK lfmc_aef.pt")
            self.log("  OK lfmc_daymet.pt")

        results['summary'] = {
            'total_parquet_size_mb': total_parquet_size,
            'avg_compression_ratio': (base_comp + aef_comp + daymet_comp) / 3
        }

        return results


def main():
    """Main entry point for the standardization script."""
    parser = argparse.ArgumentParser(
        description='Standardize LFMC datasets from CSV to Parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standardize with default paths and create cache
  python prepare_lfmc_datasets.py --create-cache

  # Custom input/output directories
  python prepare_lfmc_datasets.py --input-dir ./raw_data --output-dir ./processed_data

  # Parquet only (no cache)
  python prepare_lfmc_datasets.py --no-cache

  # Keep all 90,002 rows (disable quality filtering)
  python prepare_lfmc_datasets.py --no-filter
"""
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='./data',
        help='Directory containing input CSV files (default: ./data)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Directory for output Parquet files (default: ./data)'
    )

    parser.add_argument(
        '--create-cache',
        action='store_true',
        default=False,
        help='Create PyTorch .pt cache files for faster loading'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        default=False,
        help='Do not create cache files (Parquet only)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Suppress progress messages'
    )

    parser.add_argument(
        '--no-filter',
        action='store_true',
        default=False,
        help='Disable quality filtering (keep all 90,002 rows including invalid/failed samples)'
    )

    args = parser.parse_args()

    # Handle cache flags
    create_cache = args.create_cache and not args.no_cache

    # Create standardizer
    standardizer = LFMCDataStandardizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        apply_filters=not args.no_filter
    )

    # Run standardization
    try:
        results = standardizer.run_full_standardization(create_cache=create_cache)

        print("\n[SUCCESS] Standardization completed successfully!")
        print(f"Output files saved to: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Error during standardization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
