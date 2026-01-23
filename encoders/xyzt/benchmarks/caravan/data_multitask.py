"""
Caravan dataset for multi-task learning.

Enhanced version that loads multiple meteorological variables as prediction targets:
- Primary: streamflow
- Auxiliary: precipitation, temperature
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .constants import (
    MAX_LOG_STREAMFLOW,
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    YEAR_MIN,
    YEAR_MAX,
)


class CaravanDatasetMultitask:
    """
    Caravan dataset for multi-task learning.

    Loads coordinates and multiple meteorological targets for multi-task prediction.
    """

    def __init__(
        self,
        data_path: str,
        device: str = 'cuda',
        use_temporal_split: bool = True,
    ):
        """
        Load Caravan dataset with multiple prediction targets.

        Args:
            data_path: Path to CSV with meteorological columns
            device: Device for tensors
            use_temporal_split: Use temporal train/test split
        """
        print(f"Loading Caravan multi-task data from {data_path}...", flush=True)
        df = pd.read_csv(data_path)

        print(f"Loaded {len(df):,} daily observations", flush=True)

        # Check required columns
        required_cols = ['gauge_id', 'lat', 'lon', 'elev', 'date',
                        'streamflow_mm_per_day', 'precipitation_mm_per_day', 'temperature_2m_mean']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Filter valid data (all targets not NaN)
        valid_mask = (df['streamflow_mm_per_day'] >= 0) & df['streamflow_mm_per_day'].notna()
        valid_mask &= df['precipitation_mm_per_day'].notna()
        valid_mask &= df['temperature_2m_mean'].notna()

        df = df[valid_mask].copy()
        print(f"After filtering: {len(df):,} valid observations", flush=True)

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        # Normalize time
        date_floats = df['year'] + (df['month'] - 1) / 12.0 + df['day'] / 365.0
        time_norm = (date_floats - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
        time_norm = np.clip(time_norm, 0, 1)

        # Basin encoding
        unique_basins = df['gauge_id'].unique()
        self.basin_to_idx = {basin: idx for idx, basin in enumerate(unique_basins)}
        self.idx_to_basin = {idx: basin for basin, idx in self.basin_to_idx.items()}
        self.n_basins = len(unique_basins)

        basin_indices = np.array([self.basin_to_idx[b] for b in df['gauge_id'].values])

        print(f"\nBasin Statistics:", flush=True)
        print(f"  Unique basins: {self.n_basins}", flush=True)

        # Normalize streamflow (log transform)
        streamflow_log = np.log(df['streamflow_mm_per_day'].values + 1.0)
        streamflow_norm = streamflow_log / MAX_LOG_STREAMFLOW
        streamflow_norm = np.clip(streamflow_norm, 0, 1)

        # Normalize precipitation (log transform, similar to streamflow)
        precip_log = np.log(df['precipitation_mm_per_day'].values + 1.0)
        # Normalize by typical max precipitation (~50 mm/day)
        precip_norm = precip_log / np.log(51.0)
        precip_norm = np.clip(precip_norm, 0, 1)

        # Normalize temperature (z-score)
        temp_values = df['temperature_2m_mean'].values.astype(np.float32)
        self.temp_mean = temp_values.mean()
        self.temp_std = temp_values.std()
        temp_norm = (temp_values - self.temp_mean) / (self.temp_std + 1e-8)

        print(f"\nStreamflow statistics:", flush=True)
        print(f"  Min: {df['streamflow_mm_per_day'].min():.3f} mm/day", flush=True)
        print(f"  Median: {df['streamflow_mm_per_day'].median():.3f} mm/day", flush=True)
        print(f"  Max: {df['streamflow_mm_per_day'].max():.3f} mm/day", flush=True)

        print(f"\nPrecipitation statistics:", flush=True)
        print(f"  Min: {df['precipitation_mm_per_day'].min():.3f} mm/day", flush=True)
        print(f"  Median: {df['precipitation_mm_per_day'].median():.3f} mm/day", flush=True)
        print(f"  Max: {df['precipitation_mm_per_day'].max():.3f} mm/day", flush=True)

        print(f"\nTemperature statistics:", flush=True)
        print(f"  Min: {df['temperature_2m_mean'].min():.3f} °C", flush=True)
        print(f"  Median: {df['temperature_2m_mean'].median():.3f} °C", flush=True)
        print(f"  Max: {df['temperature_2m_mean'].max():.3f} °C", flush=True)
        print(f"  Mean: {self.temp_mean:.3f} °C, Std: {self.temp_std:.3f} °C", flush=True)

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

        # Multiple targets for multi-task learning
        self.targets_streamflow = torch.tensor(streamflow_norm, dtype=torch.float32, device=device)
        self.targets_precipitation = torch.tensor(precip_norm, dtype=torch.float32, device=device)
        self.targets_temperature = torch.tensor(temp_norm, dtype=torch.float32, device=device)

        self.basin_idx = torch.tensor(basin_indices, dtype=torch.long, device=device)
        self.years = torch.tensor(df['year'].values, dtype=torch.long, device=device)

        # Store raw values for evaluation
        self.streamflow_raw = torch.tensor(
            df['streamflow_mm_per_day'].values,
            dtype=torch.float32, device=device
        )

        # Create temporal splits
        self.has_splits = use_temporal_split
        if use_temporal_split:
            # Check for pre-defined split column
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

                # Report year ranges
                train_years = df[train_mask]['year'].unique() if len(self.train_indices) > 0 else []
                val_years = df[val_mask]['year'].unique() if len(self.val_indices) > 0 else []
                test_years = df[test_mask]['year'].unique() if len(self.test_indices) > 0 else []

                print(f"\nTemporal Split:", flush=True)
                if len(train_years) > 0:
                    print(f"  Train: {len(self.train_indices):,} samples (years {train_years.min()}-{train_years.max()})", flush=True)
                if len(self.val_indices) > 0:
                    print(f"  Val: {len(self.val_indices):,} samples (years {val_years.min()}-{val_years.max()})", flush=True)
                if len(test_years) > 0:
                    print(f"  Test: {len(self.test_indices):,} samples (years {test_years.min()}-{test_years.max()})", flush=True)
            else:
                # Default temporal split
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

            # Check basin coverage
            if len(self.test_indices) > 0:
                train_basins = set(df.iloc[self.train_indices.cpu().numpy()]['gauge_id'].unique())
                test_basins = set(df.iloc[self.test_indices.cpu().numpy()]['gauge_id'].unique())
                print(f"\n  Train basins: {len(train_basins)}", flush=True)
                print(f"  Test basins: {len(test_basins)} ({100*len(test_basins & train_basins)/len(test_basins):.1f}% overlap with train)", flush=True)
        else:
            self.train_indices = None
            self.test_indices = None
            self.val_indices = None

        self.n = n
        self.device = device
        self.df = df

        print(f"\nGPU dataset ready: {n:,} samples with 3 prediction targets", flush=True)

    def get_batch_data(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get batch data for multi-task learning.

        Args:
            indices: (B,) tensor of sample indices

        Returns:
            Dictionary with coords, targets (dict), basin_idx, streamflow_raw
        """
        return {
            'coords': self.coords[indices],
            'targets': {
                'streamflow': self.targets_streamflow[indices],
                'precipitation': self.targets_precipitation[indices],
                'temperature': self.targets_temperature[indices],
            },
            'basin_idx': self.basin_idx[indices],
            'streamflow_raw': self.streamflow_raw[indices]
        }
