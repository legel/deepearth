"""
Data loader for Earth4D with meteorological input features.

Loads:
- Coordinates (x, y, z, t)
- Precipitation, temperature, snow (input features)
- Streamflow (target)
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Literal


class CaravanDatasetInputs:
    """
    Caravan dataset with meteorological input features.

    Args:
        csv_path: Path to CSV with columns: gauge_id, date, streamflow_mm_per_day,
                  lat, lon, elev, split, precipitation_mm_per_day, temperature_2m_mean,
                  snow_depth_water_equivalent_mean
        coordinate_system: 'ecef' or 'latlon'
        device: torch device
        normalize: Whether to normalize features
    """

    def __init__(
        self,
        csv_path: str,
        coordinate_system: Literal['ecef', 'latlon'] = 'ecef',
        device: str = 'cuda',
        normalize: bool = True,
    ):
        print(f"Loading Caravan input features data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Total rows: {len(df):,}")
        print(f"  Basins: {df['gauge_id'].nunique()}")

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])

        # Create basin index mapping
        unique_basins = sorted(df['gauge_id'].unique())
        self.basin_to_idx = {basin: idx for idx, basin in enumerate(unique_basins)}
        self.idx_to_basin = {idx: basin for basin, idx in self.basin_to_idx.items()}
        self.num_basins = len(unique_basins)

        # Add basin index column
        df['basin_idx'] = df['gauge_id'].map(self.basin_to_idx)

        # Split data
        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'test'].copy()

        print(f"  Train: {len(train_df):,} rows")
        print(f"  Test: {len(test_df):,} rows")

        # Convert coordinates
        self.coordinate_system = coordinate_system
        if coordinate_system == 'ecef':
            print("  Converting to ECEF coordinates...")
            train_coords = self._latlon_to_ecef(
                train_df['lat'].values,
                train_df['lon'].values,
                train_df['elev'].values,
            )
            test_coords = self._latlon_to_ecef(
                test_df['lat'].values,
                test_df['lon'].values,
                test_df['elev'].values,
            )
        else:
            train_coords = np.stack([
                train_df['lat'].values,
                train_df['lon'].values,
                train_df['elev'].values,
            ], axis=1)
            test_coords = np.stack([
                test_df['lat'].values,
                test_df['lon'].values,
                test_df['elev'].values,
            ], axis=1)

        # Add time coordinate (days since epoch)
        train_time = (train_df['date'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / 86400.0
        test_time = (test_df['date'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / 86400.0

        train_coords = np.concatenate([train_coords, train_time.values.reshape(-1, 1)], axis=1)
        test_coords = np.concatenate([test_coords, test_time.values.reshape(-1, 1)], axis=1)

        # Normalize coordinates
        coord_mean = train_coords.mean(axis=0)
        coord_std = train_coords.std(axis=0)
        train_coords = (train_coords - coord_mean) / coord_std
        test_coords = (test_coords - coord_mean) / coord_std

        # Extract features and targets
        feature_cols = ['precipitation_mm_per_day', 'temperature_2m_mean', 'snow_depth_water_equivalent_mean']

        train_precip = train_df['precipitation_mm_per_day'].values
        train_temp = train_df['temperature_2m_mean'].values
        train_snow = train_df['snow_depth_water_equivalent_mean'].values
        train_streamflow = train_df['streamflow_mm_per_day'].values
        train_basin_idx = train_df['basin_idx'].values

        test_precip = test_df['precipitation_mm_per_day'].values
        test_temp = test_df['temperature_2m_mean'].values
        test_snow = test_df['snow_depth_water_equivalent_mean'].values
        test_streamflow = test_df['streamflow_mm_per_day'].values
        test_basin_idx = test_df['basin_idx'].values

        # Normalize features using train statistics
        if normalize:
            print("  Normalizing features...")
            # Precipitation: log-normalization
            train_precip_log = np.log(train_precip + 1.0)
            self.precip_max_log = train_precip_log.max()
            train_precip_norm = train_precip_log / self.precip_max_log
            test_precip_log = np.log(test_precip + 1.0)
            test_precip_norm = test_precip_log / self.precip_max_log

            # Temperature: standardization
            self.temp_mean = train_temp.mean()
            self.temp_std = train_temp.std()
            train_temp_norm = (train_temp - self.temp_mean) / self.temp_std
            test_temp_norm = (test_temp - self.temp_mean) / self.temp_std

            # Snow: log-normalization
            train_snow_log = np.log(train_snow + 1.0)
            self.snow_max_log = train_snow_log.max()
            train_snow_norm = train_snow_log / self.snow_max_log
            test_snow_log = np.log(test_snow + 1.0)
            test_snow_norm = test_snow_log / self.snow_max_log

            # Streamflow: log-normalization (target)
            train_streamflow_log = np.log(train_streamflow + 1.0)
            self.streamflow_max_log = train_streamflow_log.max()
            train_streamflow_norm = train_streamflow_log / self.streamflow_max_log
            test_streamflow_log = np.log(test_streamflow + 1.0)
            test_streamflow_norm = test_streamflow_log / self.streamflow_max_log

            print(f"    Precipitation: max_log={self.precip_max_log:.2f}")
            print(f"    Temperature: mean={self.temp_mean:.2f}°C, std={self.temp_std:.2f}°C")
            print(f"    Snow: max_log={self.snow_max_log:.2f}")
            print(f"    Streamflow: max_log={self.streamflow_max_log:.2f}")

        else:
            train_precip_norm = train_precip
            test_precip_norm = test_precip
            train_temp_norm = train_temp
            test_temp_norm = test_temp
            train_snow_norm = train_snow
            test_snow_norm = test_snow
            train_streamflow_norm = train_streamflow
            test_streamflow_norm = test_streamflow

        # Convert to tensors
        self.train_coords = torch.tensor(train_coords, dtype=torch.float32, device=device)
        self.train_precip = torch.tensor(train_precip_norm, dtype=torch.float32, device=device)
        self.train_temp = torch.tensor(train_temp_norm, dtype=torch.float32, device=device)
        self.train_snow = torch.tensor(train_snow_norm, dtype=torch.float32, device=device)
        self.train_streamflow = torch.tensor(train_streamflow_norm, dtype=torch.float32, device=device)
        self.train_streamflow_raw = torch.tensor(train_streamflow, dtype=torch.float32, device=device)
        self.train_basin_idx = torch.tensor(train_basin_idx, dtype=torch.long, device=device)

        self.test_coords = torch.tensor(test_coords, dtype=torch.float32, device=device)
        self.test_precip = torch.tensor(test_precip_norm, dtype=torch.float32, device=device)
        self.test_temp = torch.tensor(test_temp_norm, dtype=torch.float32, device=device)
        self.test_snow = torch.tensor(test_snow_norm, dtype=torch.float32, device=device)
        self.test_streamflow = torch.tensor(test_streamflow_norm, dtype=torch.float32, device=device)
        self.test_streamflow_raw = torch.tensor(test_streamflow, dtype=torch.float32, device=device)
        self.test_basin_idx = torch.tensor(test_basin_idx, dtype=torch.long, device=device)

        self.train_size = len(self.train_coords)
        self.test_size = len(self.test_coords)

        print(f"  Data loaded to {device}")

    def _latlon_to_ecef(self, lat: np.ndarray, lon: np.ndarray, elev: np.ndarray) -> np.ndarray:
        """Convert lat/lon/elevation to ECEF coordinates."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # WGS84 parameters
        a = 6378137.0  # semi-major axis
        f = 1 / 298.257223563  # flattening
        e2 = 2 * f - f ** 2  # eccentricity squared

        N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

        x = (N + elev) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + elev) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + elev) * np.sin(lat_rad)

        return np.stack([x, y, z], axis=1)

    def get_batch_data(self, indices: torch.Tensor, split: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Get batch data for given indices.

        Args:
            indices: (batch_size,) indices
            split: 'train' or 'test'

        Returns:
            Dictionary with coords, precipitation, temperature, snow, basin_idx,
            streamflow (normalized), streamflow_raw
        """
        if split == 'train':
            return {
                'coords': self.train_coords[indices],
                'precipitation': self.train_precip[indices],
                'temperature': self.train_temp[indices],
                'snow': self.train_snow[indices],
                'basin_idx': self.train_basin_idx[indices],
                'streamflow': self.train_streamflow[indices],
                'streamflow_raw': self.train_streamflow_raw[indices],
            }
        else:
            return {
                'coords': self.test_coords[indices],
                'precipitation': self.test_precip[indices],
                'temperature': self.test_temp[indices],
                'snow': self.test_snow[indices],
                'basin_idx': self.test_basin_idx[indices],
                'streamflow': self.test_streamflow[indices],
                'streamflow_raw': self.test_streamflow_raw[indices],
            }
