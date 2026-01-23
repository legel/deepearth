"""
Caravan dataset loading and preprocessing for Earth4D.

Implements dataset compatible with Earth4D training pipeline.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from .constants import (
    MAX_LOG_STREAMFLOW,
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    YEAR_MIN,
    YEAR_MAX,
)


class CaravanDataset:
    """
    Caravan streamflow dataset with everything on GPU.

    Implements TrainableDataset protocol with get_batch_data() method.
    Loads processed Caravan CSV and creates temporal train/test splits.
    """

    def __init__(
        self,
        data_path: str,
        device: str = 'cuda',
        use_temporal_split: bool = True
    ):
        """
        Load Caravan dataset.

        Args:
            data_path: Path to processed Caravan CSV file
            device: Device for GPU tensors
            use_temporal_split: If True, split by time (train: pre-2016, test: 2016+)
        """
        print(f"Loading Caravan data from {data_path}...", flush=True)
        df = pd.read_csv(data_path)

        print(f"Loaded {len(df):,} daily streamflow observations", flush=True)

        # Expected columns: gauge_id, lat, lon, elev, date, streamflow_mm_per_day
        required_cols = ['gauge_id', 'lat', 'lon', 'elev', 'date', 'streamflow_mm_per_day']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter valid streamflow values (>= 0, not NaN)
        df = df[
            (df['streamflow_mm_per_day'] >= 0) &
            df['streamflow_mm_per_day'].notna() &
            df['lat'].notna() &
            df['lon'].notna() &
            df['elev'].notna()
        ].copy()

        print(f"After filtering: {len(df):,} valid observations", flush=True)

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        # Normalize time to [0, 1] based on date
        date_floats = df['year'] + (df['month'] - 1) / 12.0 + df['day'] / 365.0
        time_norm = (date_floats - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
        time_norm = np.clip(time_norm, 0, 1)

        # BASIN ENCODING (analogous to species in LFMC)
        unique_basins = df['gauge_id'].unique()
        self.basin_to_idx = {basin: idx for idx, basin in enumerate(unique_basins)}
        self.idx_to_basin = {idx: basin for basin, idx in self.basin_to_idx.items()}
        self.n_basins = len(unique_basins)

        basin_indices = np.array([self.basin_to_idx[b] for b in df['gauge_id'].values])

        print(f"\nBasin Statistics:", flush=True)
        print(f"  Unique basins: {self.n_basins}", flush=True)

        # Show basin sample sizes
        basin_counts = df['gauge_id'].value_counts()
        print(f"  Observations per basin:", flush=True)
        print(f"    Min: {basin_counts.min():,}", flush=True)
        print(f"    Median: {basin_counts.median():.0f}", flush=True)
        print(f"    Max: {basin_counts.max():,}", flush=True)

        # Log-transform streamflow (common in hydrology)
        # log(Q + 1) to handle zero values
        streamflow_log = np.log(df['streamflow_mm_per_day'].values + 1.0)

        # Normalize to [0, 1]
        streamflow_norm = streamflow_log / MAX_LOG_STREAMFLOW
        streamflow_norm = np.clip(streamflow_norm, 0, 1)

        print(f"\nStreamflow Statistics (original mm/day):", flush=True)
        print(f"  Min: {df['streamflow_mm_per_day'].min():.3f}", flush=True)
        print(f"  Median: {df['streamflow_mm_per_day'].median():.3f}", flush=True)
        print(f"  Max: {df['streamflow_mm_per_day'].max():.3f}", flush=True)
        print(f"\nStreamflow Statistics (log-transformed):", flush=True)
        print(f"  Min: {streamflow_log.min():.3f}", flush=True)
        print(f"  Median: {np.median(streamflow_log):.3f}", flush=True)
        print(f"  Max: {streamflow_log.max():.3f}", flush=True)

        # Transfer to GPU
        n = len(df)
        self.coords = torch.tensor(
            np.column_stack([
                df['lat'].values,
                df['lon'].values,
                df['elev'].values,
                time_norm
            ]),
            dtype=torch.float32, device=device
        )
        self.targets = torch.tensor(streamflow_norm, dtype=torch.float32, device=device)
        self.basin_idx = torch.tensor(basin_indices, dtype=torch.long, device=device)
        self.years = torch.tensor(df['year'].values, dtype=torch.long, device=device)

        # Store raw streamflow for evaluation
        self.streamflow_raw = torch.tensor(
            df['streamflow_mm_per_day'].values,
            dtype=torch.float32, device=device
        )

        # Create temporal splits if requested
        self.has_splits = use_temporal_split
        if use_temporal_split:
            # Check if CSV has pre-defined 'split' column (for Alzhanov benchmark)
            if 'split' in df.columns:
                print(f"\nUsing pre-defined splits from 'split' column", flush=True)
                train_mask = df['split'] == 'train'
                val_mask = df['split'] == 'val'
                test_mask = df['split'] == 'test'

                self.train_indices = torch.tensor(
                    np.where(train_mask)[0], dtype=torch.long, device=device
                )
                self.val_indices = torch.tensor(
                    np.where(val_mask)[0], dtype=torch.long, device=device
                )
                self.test_indices = torch.tensor(
                    np.where(test_mask)[0], dtype=torch.long, device=device
                )

                # Get year ranges for reporting
                train_years = df[train_mask]['year'].unique() if len(self.train_indices) > 0 else []
                val_years = df[val_mask]['year'].unique() if len(self.val_indices) > 0 else []
                test_years = df[test_mask]['year'].unique() if len(self.test_indices) > 0 else []

                print(f"\nTemporal Split:", flush=True)
                print(f"  Train: {len(self.train_indices):,} samples (years {train_years.min()}-{train_years.max()})" if len(train_years) > 0 else "  Train: 0 samples", flush=True)
                if len(self.val_indices) > 0:
                    print(f"  Val: {len(self.val_indices):,} samples (years {val_years.min()}-{val_years.max()})", flush=True)
                print(f"  Test: {len(self.test_indices):,} samples (years {test_years.min()}-{test_years.max()})" if len(test_years) > 0 else "  Test: 0 samples", flush=True)
            else:
                # Default: use hardcoded year cutoff
                print(f"\nUsing default temporal split (year < {TEST_START_YEAR} for train)", flush=True)
                train_mask = df['year'] < TEST_START_YEAR
                test_mask = df['year'] >= TEST_START_YEAR

                self.train_indices = torch.tensor(
                    np.where(train_mask)[0], dtype=torch.long, device=device
                )
                self.test_indices = torch.tensor(
                    np.where(test_mask)[0], dtype=torch.long, device=device
                )
                self.val_indices = None

                print(f"\nTemporal Split:", flush=True)
                print(f"  Train: {len(self.train_indices):,} samples (years < {TEST_START_YEAR})", flush=True)
                print(f"  Test: {len(self.test_indices):,} samples (years >= {TEST_START_YEAR})", flush=True)

            # If no test data, fall back to random split
            if len(self.test_indices) == 0:
                print(f"  WARNING: No test data, falling back to random 80/20 split", flush=True)
                n = len(df)
                indices = torch.randperm(n, device=device)
                split_idx = int(0.8 * n)
                self.train_indices = indices[:split_idx]
                self.test_indices = indices[split_idx:]
                self.val_indices = None
                print(f"  Random Split: {len(self.train_indices):,} train, {len(self.test_indices):,} test", flush=True)
            else:
                # Check basin coverage
                train_basins = set(df.iloc[self.train_indices.cpu().numpy()]['gauge_id'].unique())
                test_basins = set(df.iloc[self.test_indices.cpu().numpy()]['gauge_id'].unique())

                print(f"\n  Train basins: {len(train_basins)}", flush=True)
                print(f"  Test basins: {len(test_basins)} ({100*len(test_basins & train_basins)/len(test_basins):.1f}% overlap with train)", flush=True)

                # Check for test basins NOT in training
                test_novel = test_basins - train_basins
                if test_novel:
                    print(f"  WARNING: {len(test_novel)} basins in test NOT in training", flush=True)

        else:
            self.train_indices = None
            self.test_indices = None
            self.val_indices = None

        self.n = n
        self.device = device
        self.df = df

        print(f"\nGPU dataset ready: {n:,} samples", flush=True)

    def get_batch_data(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get batch data for training.

        Implements TrainableDataset protocol.

        Args:
            indices: (B,) tensor of sample indices

        Returns:
            Dictionary with coords, targets, basin_idx
        """
        return {
            'coords': self.coords[indices],
            'targets': self.targets[indices],
            'basin_idx': self.basin_idx[indices],
            'streamflow_raw': self.streamflow_raw[indices]
        }


def get_temporal_splits(dataset: CaravanDataset) -> Dict[str, torch.Tensor]:
    """
    Return temporal train/test splits.

    Args:
        dataset: CaravanDataset instance

    Returns:
        Dictionary with 'train' and 'test' index tensors
    """
    if not dataset.has_splits or dataset.train_indices is None:
        raise ValueError("Dataset does not have temporal splits enabled")

    return {
        'train': dataset.train_indices,
        'test': dataset.test_indices
    }


def compute_streamflow_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    batch_data: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute streamflow prediction metrics.

    Metrics computed on original scale (mm/day) after denormalization.

    Args:
        predictions: Model predictions (normalized [0, 1])
        targets: Ground truth targets (normalized [0, 1])
        batch_data: Dictionary containing 'streamflow_raw' for reference

    Returns:
        Dictionary with MAE, RMSE, R2, NSE
    """
    # Denormalize from [0, 1] back to log scale
    pred_log = predictions * MAX_LOG_STREAMFLOW
    target_log = targets * MAX_LOG_STREAMFLOW

    # Convert back to mm/day: exp(log(Q+1)) - 1
    pred_mm = torch.exp(pred_log) - 1.0
    target_mm = torch.exp(target_log) - 1.0

    # Clip to valid range
    pred_mm = torch.clamp(pred_mm, min=0.0)
    target_mm = torch.clamp(target_mm, min=0.0)

    if len(pred_mm) == 0:
        return {'mae': 0, 'rmse': 0, 'r2': 0, 'nse': 0, 'n_samples': 0}

    # Compute metrics
    errors = pred_mm - target_mm
    mse = (errors ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = torch.abs(errors).mean()

    # R²
    ss_res = (errors ** 2).sum()
    ss_tot = ((target_mm - target_mm.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0, device=pred_mm.device)

    # Nash-Sutcliffe Efficiency (common in hydrology, equivalent to R² for regression)
    nse = r2

    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item(),
        'nse': nse.item(),
        'n_samples': len(pred_mm)
    }
