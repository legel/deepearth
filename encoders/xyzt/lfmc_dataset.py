#!/usr/bin/env python3
"""
LFMC Modular Dataset Loader
============================

Easy-to-use interface for loading LFMC datasets with optional features.

This module implements Phase 2 of the LFMC Modular Dataset Architecture Plan.

Features:
- Selective loading (base, +AEF, +Daymet, +SOLUS)
- Automatic caching with .pt files for fast repeated loads
- Auto-download missing files from URLs
- PyTorch Dataset integration
- Train/val/test splitting utilities

Usage:
    # Load base LFMC only
    dataset = LFMCDataset()

    # Load base + AlphaEarth features
    dataset = LFMCDataset(use_aef=True)

    # Load all features with auto-download
    dataset = LFMCDataset(use_aef=True, use_daymet=True, auto_download=True)

    # Convert to PyTorch Dataset
    torch_dataset = dataset.to_torch_dataset()

Author: Claude Code
Date: 2025-10-28
Version: 1.0
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try to import torch (optional)
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object  # Dummy class for type hints


class LFMCDataset:
    """
    Modular LFMC dataset loader with selective feature loading.

    Attributes:
        data (pd.DataFrame): Merged dataset with selected features
        base_df (pd.DataFrame): Base LFMC data
        aef_df (pd.DataFrame): AlphaEarth features (if loaded)
        daymet_df (pd.DataFrame): Daymet weather features (if loaded)
        n_samples (int): Total number of samples
        feature_columns (List[str]): List of feature column names
    """

    def __init__(
        self,
        data_dir: str = "./data",
        use_aef: bool = False,
        use_daymet: bool = False,
        use_solus: bool = False,
        auto_download: bool = False,
        use_cache: Union[bool, str] = 'auto',
        cache_dir: str = "./data/cache",
        urls_file: str = "./dataset_urls.json",
        verbose: bool = True
    ):
        """
        Initialize the LFMC dataset loader.

        Args:
            data_dir: Directory containing Parquet files
            use_aef: Include AlphaEarth Features (64D embeddings)
            use_daymet: Include Daymet weather features (22 features)
            use_solus: Include SOLUS soil features (119 features) [FUTURE]
            auto_download: Auto-download missing files from URLs
            use_cache: Cache strategy ('auto', True, False)
                - 'auto': Use cache if exists and newer than Parquet
                - True: Always use cache (create if missing)
                - False: Never use cache, always load from Parquet
            cache_dir: Directory for .pt cache files
            urls_file: Path to dataset_urls.json with download URLs
            verbose: Print loading progress
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.urls_file = Path(urls_file)
        self.verbose = verbose

        self.use_aef = use_aef
        self.use_daymet = use_daymet
        self.use_solus = use_solus
        self.use_cache = use_cache

        # Create cache directory if needed
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check and download missing files
        if auto_download:
            self._check_and_download_files()

        # Load datasets
        self._load_datasets()

        # Merge selected features
        self._merge_features()

        # Log summary
        if self.verbose:
            self._print_summary()

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)

    def _check_and_download_files(self):
        """Check for missing files and download from URLs if available."""
        self.log("Checking for missing files...")

        # Check which files are needed
        required_files = ['lfmc_base.parquet']
        if self.use_aef:
            required_files.append('lfmc_aef.parquet')
        if self.use_daymet:
            required_files.append('lfmc_daymet.parquet')
        if self.use_solus:
            required_files.append('lfmc_solus.parquet')

        # Check which are missing
        missing_files = []
        for filename in required_files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                missing_files.append(filename)

        if not missing_files:
            self.log("  [OK] All required files present")
            return

        self.log(f"  Missing files: {', '.join(missing_files)}")

        # Try to load URLs
        if not self.urls_file.exists():
            raise FileNotFoundError(
                f"Missing files and no URLs file found at {self.urls_file}. "
                f"Please run upload script or manually place files in {self.data_dir}"
            )

        with open(self.urls_file, 'r') as f:
            urls = json.load(f)

        # Download missing files
        for filename in missing_files:
            # Handle both flat dict and nested dict formats
            if 'files' in urls and filename in urls['files']:
                url = urls['files'][filename]['url']
            elif filename in urls:
                url = urls[filename]
            else:
                raise ValueError(f"No URL found for {filename} in {self.urls_file}")

            output_path = self.data_dir / filename

            self.log(f"  Downloading {filename}...")
            self.log(f"    From: {url}")

            try:
                urllib.request.urlretrieve(url, output_path)
                size_mb = output_path.stat().st_size / 1024**2
                self.log(f"    [OK] Downloaded ({size_mb:.2f} MB)")
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename}: {e}")

    def _should_use_cache(self, cache_path: Path, parquet_path: Path) -> bool:
        """Determine if cache should be used based on strategy and freshness."""
        if self.use_cache is False:
            return False

        if not cache_path.exists():
            return False

        if self.use_cache is True:
            return True

        # Auto mode: use cache if newer than Parquet
        if self.use_cache == 'auto':
            cache_mtime = cache_path.stat().st_mtime
            parquet_mtime = parquet_path.stat().st_mtime
            return cache_mtime > parquet_mtime

        return False

    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame:
        """Load dataset from .pt cache file."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available, cannot load .pt cache files")

        self.log(f"    Loading from cache: {cache_path.name}")

        cache_data = torch.load(cache_path, weights_only=False)

        # Reconstruct DataFrame from cached tensors
        df_dict = {}
        for col in cache_data['columns']:
            if col in cache_data:
                if isinstance(cache_data[col], torch.Tensor):
                    df_dict[col] = cache_data[col].numpy()

        # Handle sample_id separately (always included)
        df_dict['sample_id'] = cache_data['sample_id'].numpy()

        df = pd.DataFrame(df_dict)

        # Restore string columns from metadata (e.g., species)
        if 'string_columns' in cache_data:
            for col_name, col_values in cache_data['string_columns'].items():
                df[col_name] = col_values

        return df

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save DataFrame to .pt cache file."""
        if not TORCH_AVAILABLE:
            self.log(f"    [SKIP] PyTorch not available, cannot create cache")
            return

        self.log(f"    Creating cache: {cache_path.name}")

        # Convert DataFrame to dictionary of tensors
        cache_data = {
            'sample_id': torch.from_numpy(df['sample_id'].values),
            'columns': list(df.columns)
        }

        # Store string columns separately (e.g., species, date strings)
        string_columns = {}

        # Add numeric columns as tensors
        for col in df.columns:
            if col == 'sample_id':
                continue

            if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                cache_data[col] = torch.from_numpy(df[col].values)
            elif df[col].dtype == 'datetime64[ns]':
                # Convert datetime to Unix timestamp (float64)
                cache_data[col] = torch.from_numpy(df[col].astype('int64').values / 1e9)
            elif df[col].dtype == 'object' or df[col].dtype == 'string':
                # Store string columns in metadata
                string_columns[col] = df[col].tolist()

        # Add string columns to cache if any exist
        if string_columns:
            cache_data['string_columns'] = string_columns

        # Add metadata
        cache_data['metadata'] = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Exclude sample_id
            'created': pd.Timestamp.now().isoformat(),
            'source': 'lfmc_dataset.py'
        }

        torch.save(cache_data, cache_path)
        size_mb = cache_path.stat().st_size / 1024**2
        self.log(f"    [OK] Cache created ({size_mb:.2f} MB)")

    def _load_datasets(self):
        """Load all requested datasets (Parquet or cache)."""
        self.log("\nLoading datasets...")

        # Load base LFMC (always required)
        base_parquet = self.data_dir / 'lfmc_base.parquet'
        base_cache = self.cache_dir / 'lfmc_base.pt'

        self.log("  Base LFMC:")
        if self._should_use_cache(base_cache, base_parquet):
            self.base_df = self._load_from_cache(base_cache)
        else:
            self.log(f"    Loading from Parquet: {base_parquet.name}")
            self.base_df = pd.read_parquet(base_parquet)
            if self.use_cache:
                self._save_to_cache(self.base_df, base_cache)

        self.log(f"    [OK] Loaded {len(self.base_df):,} samples")

        # Load AlphaEarth Features
        self.aef_df = None
        if self.use_aef:
            aef_parquet = self.data_dir / 'lfmc_aef.parquet'
            aef_cache = self.cache_dir / 'lfmc_aef.pt'

            self.log("  AlphaEarth Features (AEF):")
            if self._should_use_cache(aef_cache, aef_parquet):
                self.aef_df = self._load_from_cache(aef_cache)
            else:
                self.log(f"    Loading from Parquet: {aef_parquet.name}")
                self.aef_df = pd.read_parquet(aef_parquet)
                if self.use_cache:
                    self._save_to_cache(self.aef_df, aef_cache)

            self.log(f"    [OK] Loaded {len(self.aef_df):,} samples, 64 dimensions")

        # Load Daymet Weather
        self.daymet_df = None
        if self.use_daymet:
            daymet_parquet = self.data_dir / 'lfmc_daymet.parquet'
            daymet_cache = self.cache_dir / 'lfmc_daymet.pt'

            self.log("  Daymet Weather:")
            if self._should_use_cache(daymet_cache, daymet_parquet):
                self.daymet_df = self._load_from_cache(daymet_cache)
            else:
                self.log(f"    Loading from Parquet: {daymet_parquet.name}")
                self.daymet_df = pd.read_parquet(daymet_parquet)
                if self.use_cache:
                    self._save_to_cache(self.daymet_df, daymet_cache)

            n_weather_features = len(self.daymet_df.columns) - 1  # Exclude sample_id
            self.log(f"    [OK] Loaded {len(self.daymet_df):,} samples, {n_weather_features} features")

        # SOLUS soil data (future)
        self.solus_df = None
        if self.use_solus:
            self.log("  SOLUS Soil Data:")
            self.log("    [SKIP] SOLUS integration not yet implemented (Phase 5)")

    def _merge_features(self):
        """Merge selected feature datasets."""
        self.log("\nMerging features...")

        # Start with base
        self.data = self.base_df.copy()
        self.log(f"  Base: {len(self.data):,} samples, {len(self.data.columns)} columns")

        # Merge AEF
        if self.aef_df is not None:
            n_before = len(self.data)
            self.data = self.data.merge(self.aef_df, on='sample_id', how='inner')
            self.log(f"  + AEF: {len(self.data):,} samples, {len(self.data.columns)} columns")
            if len(self.data) < n_before:
                self.log(f"    [INFO] Lost {n_before - len(self.data)} samples without AEF data")

        # Merge Daymet
        if self.daymet_df is not None:
            n_before = len(self.data)
            self.data = self.data.merge(self.daymet_df, on='sample_id', how='inner')
            self.log(f"  + Daymet: {len(self.data):,} samples, {len(self.data.columns)} columns")
            if len(self.data) < n_before:
                self.log(f"    [INFO] Lost {n_before - len(self.data)} samples without Daymet data")

        # Merge SOLUS (future)
        if self.solus_df is not None:
            n_before = len(self.data)
            self.data = self.data.merge(self.solus_df, on='sample_id', how='inner')
            self.log(f"  + SOLUS: {len(self.data):,} samples, {len(self.data.columns)} columns")

        # Store feature column names (exclude metadata and datetime columns)
        metadata_cols = ['sample_id', 'lat', 'lon', 'elevation_m', 'date', 'lfmc_percent', 'species']
        self.feature_columns = [
            col for col in self.data.columns
            if col not in metadata_cols and self.data[col].dtype != 'datetime64[ns]'
        ]

        self.n_samples = len(self.data)

    def _print_summary(self):
        """Print dataset summary."""
        self.log("\n" + "="*60)
        self.log("DATASET SUMMARY")
        self.log("="*60)
        self.log(f"Total samples: {self.n_samples:,}")
        self.log(f"Total features: {len(self.feature_columns)}")

        if self.use_aef:
            aef_cols = [c for c in self.feature_columns if c.startswith('aef_')]
            self.log(f"  - AlphaEarth embeddings: {len(aef_cols)} dimensions")

        if self.use_daymet:
            daymet_cols = [c for c in self.feature_columns if any(x in c for x in ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl', 'swe'])]
            self.log(f"  - Daymet weather: {len(daymet_cols)} features")

        self.log(f"\nLFMC range: {self.data['lfmc_percent'].min():.1f}% to {self.data['lfmc_percent'].max():.1f}%")
        self.log(f"Species: {self.data['species'].nunique()} unique")
        self.log(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        self.log(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        self.log("="*60)

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'n_samples': self.n_samples,
            'n_features': len(self.feature_columns),
            'n_species': self.data['species'].nunique(),
            'lfmc_min': float(self.data['lfmc_percent'].min()),
            'lfmc_max': float(self.data['lfmc_percent'].max()),
            'lfmc_mean': float(self.data['lfmc_percent'].mean()),
            'lfmc_std': float(self.data['lfmc_percent'].std()),
            'date_min': str(self.data['date'].min()),
            'date_max': str(self.data['date'].max()),
            'lat_range': (float(self.data['lat'].min()), float(self.data['lat'].max())),
            'lon_range': (float(self.data['lon'].min()), float(self.data['lon'].max())),
            'has_aef': self.use_aef,
            'has_daymet': self.use_daymet,
            'has_solus': self.use_solus
        }
        return stats

    def create_splits(
        self,
        temporal_frac: float = 0.05,
        spatial_frac: float = 0.05,
        random_frac: float = 0.05,
        random_seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Create train/test splits similar to earth4d-aef_to_lfmc.py.

        Returns indices for:
        - train: Remaining samples after holdouts
        - temporal: Latest 5% of samples by date
        - spatial: 5% of samples in spatial clusters
        - random: Random 5% of remaining samples

        Args:
            temporal_frac: Fraction for temporal holdout (default: 0.05)
            spatial_frac: Fraction for spatial holdout (default: 0.05)
            random_frac: Fraction for random holdout (default: 0.05)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with split names and indices arrays
        """
        np.random.seed(random_seed)

        n = len(self.data)
        all_idx = np.arange(n)
        used = np.zeros(n, dtype=bool)

        # 1. Temporal: last N% by date
        n_temp = int(n * temporal_frac)
        sorted_idx = np.argsort(self.data['date'].values)
        temp_idx = sorted_idx[-n_temp:]
        used[temp_idx] = True

        # 2. Spatial: create clusters
        n_spat = int(n * spatial_frac)
        available = all_idx[~used]

        # Random spatial clusters (simplified version)
        # In production, use proper spatial clustering
        spat_idx = np.random.choice(available, size=n_spat, replace=False)
        used[spat_idx] = True

        # 3. Random
        n_rand = int(n * random_frac)
        available = all_idx[~used]
        rand_idx = np.random.choice(available, size=n_rand, replace=False)
        used[rand_idx] = True

        # 4. Train: everything else
        train_idx = all_idx[~used]

        self.log(f"\nCreated splits:")
        self.log(f"  Train:    {len(train_idx):,} samples ({100*len(train_idx)/n:.1f}%)")
        self.log(f"  Temporal: {len(temp_idx):,} samples ({100*len(temp_idx)/n:.1f}%)")
        self.log(f"  Spatial:  {len(spat_idx):,} samples ({100*len(spat_idx)/n:.1f}%)")
        self.log(f"  Random:   {len(rand_idx):,} samples ({100*len(rand_idx)/n:.1f}%)")

        return {
            'train': train_idx,
            'temporal': temp_idx,
            'spatial': spat_idx,
            'random': rand_idx
        }

    def to_torch_dataset(
        self,
        indices: Optional[np.ndarray] = None,
        return_species: bool = True,
        normalize: bool = True
    ) -> 'LFMCTorchDataset':
        """
        Convert to PyTorch Dataset.

        Args:
            indices: Subset of indices to use (None = all)
            return_species: Include species information
            normalize: Normalize features (recommended)

        Returns:
            LFMCTorchDataset instance
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")

        return LFMCTorchDataset(
            self.data,
            feature_columns=self.feature_columns,
            indices=indices,
            return_species=return_species,
            normalize=normalize
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __repr__(self) -> str:
        """String representation."""
        features_str = []
        if self.use_aef:
            features_str.append("AEF")
        if self.use_daymet:
            features_str.append("Daymet")
        if self.use_solus:
            features_str.append("SOLUS")

        features = "+".join(features_str) if features_str else "none"

        return (
            f"LFMCDataset("
            f"samples={self.n_samples:,}, "
            f"features={len(self.feature_columns)}, "
            f"modules=[{features}])"
        )


class LFMCTorchDataset(TorchDataset):
    """PyTorch Dataset wrapper for LFMC data."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        indices: Optional[np.ndarray] = None,
        return_species: bool = True,
        normalize: bool = True
    ):
        """
        Initialize PyTorch Dataset.

        Args:
            data: Full DataFrame
            feature_columns: List of feature column names
            indices: Subset of indices to use
            return_species: Include species embedding indices
            normalize: Normalize features
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        self.data = data if indices is None else data.iloc[indices]
        self.feature_columns = feature_columns
        self.return_species = return_species

        # Extract arrays
        self.coords = torch.tensor(
            self.data[['lat', 'lon', 'elevation_m']].values,
            dtype=torch.float32
        )

        # Add normalized time (convert date to float)
        dates = pd.to_datetime(self.data['date'])
        time_float = (dates - dates.min()).dt.total_seconds() / (dates.max() - dates.min()).total_seconds()
        self.coords = torch.cat([
            self.coords,
            torch.tensor(time_float.values, dtype=torch.float32).unsqueeze(1)
        ], dim=1)

        # Features (exclude datetime columns - they're metadata, not features)
        feature_arrays = []
        for col in feature_columns:
            if col in self.data.columns:
                # Skip datetime columns (e.g., daymet_start, daymet_end)
                if self.data[col].dtype == 'datetime64[ns]':
                    continue
                feature_arrays.append(self.data[col].values.reshape(-1, 1))

        if feature_arrays:
            self.features = torch.tensor(
                np.concatenate(feature_arrays, axis=1),
                dtype=torch.float32
            )

            # Normalize features
            if normalize:
                self.feature_mean = self.features.mean(dim=0, keepdim=True)
                self.feature_std = self.features.std(dim=0, keepdim=True) + 1e-8
                self.features = (self.features - self.feature_mean) / self.feature_std
        else:
            self.features = None

        # Targets
        self.targets = torch.tensor(
            self.data['lfmc_percent'].values,
            dtype=torch.float32
        )

        # Species
        if return_species:
            unique_species = self.data['species'].unique()
            self.species_to_idx = {s: i for i, s in enumerate(unique_species)}
            self.species_indices = torch.tensor(
                [self.species_to_idx[s] for s in self.data['species']],
                dtype=torch.long
            )
            self.n_species = len(unique_species)
        else:
            self.species_indices = None
            self.n_species = 0

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single sample.

        Returns:
            Tuple of (coords, features, species_idx, target)
            or (coords, features, target) if return_species=False
        """
        coords = self.coords[idx]
        target = self.targets[idx]

        if self.features is not None:
            features = self.features[idx]
        else:
            features = torch.tensor([], dtype=torch.float32)

        if self.return_species:
            species_idx = self.species_indices[idx]
            return coords, features, species_idx, target
        else:
            return coords, features, target


def main():
    """Example usage and testing."""
    print("\n" + "="*60)
    print("LFMC Dataset Loader - Example Usage")
    print("="*60)

    # Example 1: Load base only
    print("\nExample 1: Base LFMC only")
    dataset1 = LFMCDataset()
    print(dataset1)

    # Example 2: Load base + AEF
    print("\nExample 2: Base + AlphaEarth Features")
    dataset2 = LFMCDataset(use_aef=True, use_cache=True)
    print(dataset2)

    # Example 3: Load all features
    print("\nExample 3: All features (Base + AEF + Daymet)")
    dataset3 = LFMCDataset(use_aef=True, use_daymet=True, use_cache=True)
    print(dataset3)

    # Example 4: Create splits
    print("\nExample 4: Create train/test splits")
    splits = dataset3.create_splits()

    # Example 5: Convert to PyTorch (if available)
    if TORCH_AVAILABLE:
        print("\nExample 5: Convert to PyTorch Dataset")
        torch_dataset = dataset3.to_torch_dataset(indices=splits['train'])
        print(f"  PyTorch dataset: {len(torch_dataset)} samples")

        # Test loading a batch
        sample = torch_dataset[0]
        print(f"  Sample shape: coords={sample[0].shape}, features={sample[1].shape}, target={sample[2]}")
    else:
        print("\nExample 5: PyTorch not available, skipping conversion")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
