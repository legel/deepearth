#!/usr/bin/env python3
"""
Earth System Science inspired dataset for testing Earth4D.

Generates realistic phenomena with proper train/test splits and holdout regions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum


class DataSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SPATIAL_HOLDOUT = "spatial_holdout"
    TEMPORAL_HOLDOUT = "temporal_holdout"


class EarthSystemDataset:
    """
    Dataset simulating Earth System Science phenomena with multiple scales.

    Phenomena modeled:
    1. Temperature field with latitude gradient and seasonal cycles
    2. Pressure systems with traveling waves
    3. Ocean currents with mesoscale eddies
    4. Land-sea contrasts
    5. Urban heat islands
    """

    def __init__(self,
                 num_samples: int = 10000,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 spatial_holdout_regions: Optional[List[Tuple[float, float, float, float]]] = None,
                 temporal_holdout_range: Optional[Tuple[float, float]] = None,
                 device: str = 'cuda',
                 seed: int = 42):
        """
        Initialize Earth System dataset.

        Args:
            num_samples: Total number of samples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            spatial_holdout_regions: List of (lat_min, lat_max, lon_min, lon_max) for holdout
            temporal_holdout_range: (t_min, t_max) for temporal holdout
            device: Device to use
            seed: Random seed for reproducibility
        """
        self.device = device
        self.num_samples = num_samples

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default spatial holdout: North America
        if spatial_holdout_regions is None:
            spatial_holdout_regions = [
                (30.0, 50.0, -125.0, -70.0),  # Continental US
            ]
        self.spatial_holdout_regions = spatial_holdout_regions

        # Default temporal holdout: Summer months (0.4-0.7 of year)
        if temporal_holdout_range is None:
            temporal_holdout_range = (0.4, 0.7)
        self.temporal_holdout_range = temporal_holdout_range

        # Generate all coordinates
        self.all_coords = self._generate_global_coordinates(num_samples)

        # Generate targets (ground truth phenomena)
        self.all_targets = self._generate_earth_system_targets(self.all_coords)

        # Create splits
        self._create_splits(train_ratio, val_ratio, test_ratio)

        # Compute normalization statistics from training set only
        self._compute_normalization_stats()

    def _generate_global_coordinates(self, n: int) -> torch.Tensor:
        """Generate realistic global sampling patterns."""
        coords_list = []

        # 1. Ocean grid points (40%) - regular sampling over oceans
        n_ocean = int(n * 0.4)
        lat_ocean = torch.rand(n_ocean) * 180 - 90
        lon_ocean = torch.rand(n_ocean) * 360 - 180
        # Ocean surface with small variations
        elev_ocean = torch.randn(n_ocean) * 10 - 50  # Around -50m
        time_ocean = torch.rand(n_ocean)

        # 2. Land stations (30%) - concentrated in populated areas
        n_land = int(n * 0.3)
        # Major population centers
        land_centers = [
            (40.0, -100.0),  # North America
            (50.0, 10.0),    # Europe
            (30.0, 120.0),   # East Asia
            (-25.0, 135.0),  # Australia
            (-10.0, -60.0),  # South America
            (0.0, 20.0),     # Africa
        ]

        lat_land = []
        lon_land = []
        for i in range(n_land):
            center = land_centers[i % len(land_centers)]
            lat_land.append(center[0] + torch.randn(1) * 15)
            lon_land.append(center[1] + torch.randn(1) * 20)

        lat_land = torch.cat(lat_land)
        lon_land = torch.cat(lon_land)
        elev_land = torch.abs(torch.randn(n_land)) * 500  # 0-500m elevation
        time_land = torch.rand(n_land)

        # 3. Polar regions (15%)
        n_polar = int(n * 0.15)
        lat_polar = torch.cat([
            torch.randn(n_polar//2) * 5 + 75,   # Arctic
            torch.randn(n_polar//2) * 5 - 75    # Antarctic
        ])
        lon_polar = torch.rand(n_polar) * 360 - 180
        elev_polar = torch.randn(n_polar) * 200
        time_polar = torch.rand(n_polar)

        # 4. Tropical belt (15%)
        n_tropical = n - n_ocean - n_land - n_polar
        lat_tropical = torch.randn(n_tropical) * 15  # ±15 degrees from equator
        lon_tropical = torch.rand(n_tropical) * 360 - 180
        elev_tropical = torch.randn(n_tropical) * 100
        time_tropical = torch.rand(n_tropical)

        # Combine all coordinates
        lat = torch.cat([lat_ocean, lat_land, lat_polar, lat_tropical])
        lon = torch.cat([lon_ocean, lon_land, lon_polar, lon_tropical])
        elev = torch.cat([elev_ocean, elev_land, elev_polar, elev_tropical])
        time = torch.cat([time_ocean, time_land, time_polar, time_tropical])

        # Clip to valid ranges
        lat = torch.clamp(lat, -90, 90)
        lon = torch.clamp(lon, -180, 180)

        coords = torch.stack([lat, lon, elev, time], dim=-1)
        return coords.to(self.device)

    def _generate_earth_system_targets(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Generate physically-inspired target values.

        Simulates temperature in Celsius with realistic patterns.
        """
        lat = coords[:, 0]
        lon = coords[:, 1]
        elev = coords[:, 2]
        time = coords[:, 3]

        # Base temperature (Celsius)
        temperature = torch.zeros_like(lat)

        # 1. Latitude gradient (strongest effect)
        # Equator ~30°C, poles ~-30°C
        temperature += 30 * torch.cos(torch.deg2rad(lat))

        # 2. Seasonal variation (stronger at high latitudes)
        seasonal_amp = 10 * (torch.abs(lat) / 90)  # 0 at equator, 10 at poles
        temperature += seasonal_amp * torch.sin(2 * np.pi * time)

        # 3. Land-sea contrast (simplified)
        # Assume land is warmer in summer, cooler in winter
        is_land = elev > 0
        land_effect = 5 * torch.sin(2 * np.pi * time)
        temperature += is_land.float() * land_effect

        # 4. Elevation lapse rate
        temperature -= 0.0065 * torch.clamp(elev, min=0)  # -6.5°C per km

        # 5. Traveling weather systems (Rossby waves)
        # Wave number 3-5 in mid-latitudes
        wave_lat_factor = torch.exp(-((lat - 45) ** 2) / (2 * 20**2))  # Peak at 45°N
        wave_lat_factor += torch.exp(-((lat + 45) ** 2) / (2 * 20**2))  # Peak at 45°S
        wave_phase = torch.deg2rad(lon) * 4 - 2 * np.pi * time * 20  # Eastward propagation
        temperature += 3 * wave_lat_factor * torch.sin(wave_phase)

        # 6. Mesoscale eddies (ocean)
        is_ocean = elev < 0
        eddy_scale = 2  # degrees
        eddy_pattern = torch.sin(torch.deg2rad(lat) * 180/eddy_scale) * \
                      torch.cos(torch.deg2rad(lon) * 360/eddy_scale)
        temperature += is_ocean.float() * 2 * eddy_pattern

        # 7. Urban heat islands (sparse hot spots)
        # Randomly place some urban centers
        torch.manual_seed(42)  # Consistent urban locations
        n_cities = 100
        city_lats = torch.rand(n_cities) * 120 - 60  # Most cities in mid-latitudes
        city_lons = torch.rand(n_cities) * 360 - 180

        urban_heat = torch.zeros_like(temperature)
        for city_lat, city_lon in zip(city_lats, city_lons):
            dist_sq = (lat - city_lat)**2 + (lon - city_lon)**2
            urban_heat += 3 * torch.exp(-dist_sq / (2 * 0.1**2))  # 3°C peak, 0.1° radius

        temperature += urban_heat

        # 8. Diurnal cycle (smaller scale)
        local_time = (time * 24 + lon / 15) % 24  # Local solar time
        diurnal = 2 * torch.sin(2 * np.pi * (local_time - 14) / 24)  # Peak at 2pm
        temperature += diurnal

        # 9. Random weather variability
        temperature += torch.randn_like(temperature) * 1.0

        return temperature.to(self.device)

    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Create data splits with spatial and temporal holdouts."""
        n = self.num_samples
        indices = torch.arange(n)

        # Identify spatial holdout samples (ensure on same device)
        spatial_holdout_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        for region in self.spatial_holdout_regions:
            lat_min, lat_max, lon_min, lon_max = region
            mask = (self.all_coords[:, 0] >= lat_min) & (self.all_coords[:, 0] <= lat_max) & \
                   (self.all_coords[:, 1] >= lon_min) & (self.all_coords[:, 1] <= lon_max)
            spatial_holdout_mask = spatial_holdout_mask | mask  # Use | instead of |= to avoid device issues

        # Identify temporal holdout samples
        t_min, t_max = self.temporal_holdout_range
        temporal_holdout_mask = (self.all_coords[:, 3] >= t_min) & (self.all_coords[:, 3] <= t_max)

        # Get indices for different splits (move masks to CPU for indexing)
        spatial_holdout_mask_cpu = spatial_holdout_mask.cpu()
        temporal_holdout_mask_cpu = temporal_holdout_mask.cpu()

        self.spatial_holdout_indices = indices[spatial_holdout_mask_cpu]
        self.temporal_holdout_indices = indices[temporal_holdout_mask_cpu & ~spatial_holdout_mask_cpu]

        # Remaining indices for train/val/test
        remaining_mask = ~(spatial_holdout_mask_cpu | temporal_holdout_mask_cpu)
        remaining_indices = indices[remaining_mask]

        # Shuffle and split remaining
        perm = torch.randperm(len(remaining_indices))
        remaining_indices = remaining_indices[perm]

        n_remaining = len(remaining_indices)
        n_train = int(n_remaining * train_ratio / (train_ratio + val_ratio + test_ratio))
        n_val = int(n_remaining * val_ratio / (train_ratio + val_ratio + test_ratio))

        self.train_indices = remaining_indices[:n_train]
        self.val_indices = remaining_indices[n_train:n_train + n_val]
        self.test_indices = remaining_indices[n_train + n_val:]

        print(f"Dataset splits:")
        print(f"  Train: {len(self.train_indices)} samples")
        print(f"  Val: {len(self.val_indices)} samples")
        print(f"  Test: {len(self.test_indices)} samples")
        print(f"  Spatial holdout: {len(self.spatial_holdout_indices)} samples")
        print(f"  Temporal holdout: {len(self.temporal_holdout_indices)} samples")

    def _compute_normalization_stats(self):
        """Compute mean and std from training set only."""
        train_targets = self.all_targets[self.train_indices]
        self.target_mean = train_targets.mean()
        self.target_std = train_targets.std()

        print(f"Target statistics (from train set):")
        print(f"  Mean: {self.target_mean:.2f}")
        print(f"  Std: {self.target_std:.2f}")
        print(f"  Range: [{train_targets.min():.2f}, {train_targets.max():.2f}]")

    def get_batch(self, batch_size: int, split: DataSplit = DataSplit.TRAIN) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch from specified split."""
        if split == DataSplit.TRAIN:
            indices = self.train_indices
        elif split == DataSplit.VAL:
            indices = self.val_indices
        elif split == DataSplit.TEST:
            indices = self.test_indices
        elif split == DataSplit.SPATIAL_HOLDOUT:
            indices = self.spatial_holdout_indices
        elif split == DataSplit.TEMPORAL_HOLDOUT:
            indices = self.temporal_holdout_indices
        else:
            raise ValueError(f"Unknown split: {split}")

        if len(indices) == 0:
            raise ValueError(f"No samples in {split} split")

        # Sample with replacement if batch_size > split size
        batch_indices = indices[torch.randint(0, len(indices), (batch_size,))]

        coords = self.all_coords[batch_indices]
        targets = self.all_targets[batch_indices]

        return coords, targets

    def get_normalized_batch(self, batch_size: int, split: DataSplit = DataSplit.TRAIN) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a normalized batch from specified split."""
        coords, targets = self.get_batch(batch_size, split)
        # Normalize targets using training statistics
        targets_norm = (targets - self.target_mean) / (self.target_std + 1e-6)
        return coords, targets_norm

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                       normalized: bool = False) -> Dict[str, float]:
        """
        Compute various error metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            normalized: Whether inputs are normalized

        Returns:
            Dictionary of metrics
        """
        if normalized:
            # Denormalize for metric computation
            predictions = predictions * self.target_std + self.target_mean
            targets = targets * self.target_std + self.target_mean

        with torch.no_grad():
            mse = ((predictions - targets) ** 2).mean()
            mae = torch.abs(predictions - targets).mean()

            # Avoid division by zero in relative error
            mask = torch.abs(targets) > 0.1  # Avoid near-zero temperatures
            if mask.any():
                relative_error = torch.abs(predictions[mask] - targets[mask]) / torch.abs(targets[mask])
                mape = relative_error.mean() * 100
            else:
                mape = torch.tensor(0.0)

            # R-squared
            ss_tot = ((targets - targets.mean()) ** 2).sum()
            ss_res = ((targets - predictions) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)

        return {
            'mse': mse.item(),
            'rmse': torch.sqrt(mse).item(),
            'mae': mae.item(),
            'mape': mape.item(),
            'r2': r2.item()
        }