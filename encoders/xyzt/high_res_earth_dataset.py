#!/usr/bin/env python3
"""
High-resolution Earth System Science dataset with sub-meter phenomena.

Simulates various Earth observation data with fine-scale features:
- Urban microclimate variations (building shadows, heat islands)
- Agricultural field boundaries and crop patterns
- Coastal dynamics with wave patterns
- Forest canopy structure
- Infrastructure effects (roads, pipelines)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum

# Import DataSplit from earth_system_dataset if available, otherwise define it
try:
    from earth_system_dataset import DataSplit
except ImportError:
    class DataSplit(Enum):
        """Dataset split types."""
        TRAIN = "train"
        VAL = "val"
        TEST = "test"
        SPATIAL_HOLDOUT = "spatial_holdout"
        TEMPORAL_HOLDOUT = "temporal_holdout"


class HighResEarthDataset:
    """
    High-resolution dataset with sub-meter phenomena.

    Features multiple scales of variation:
    - Continental: Climate zones (1000km)
    - Regional: Weather systems (100km)
    - Local: Urban/rural differences (1km)
    - Fine: Building-level (10m)
    - Ultra-fine: Surface materials (1m)
    """

    def __init__(self,
                 num_samples: int = 100000,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 include_fine_scale: bool = True,
                 spatial_holdout_regions: Optional[List[Tuple[float, float, float, float]]] = None,
                 temporal_holdout_range: Optional[Tuple[float, float]] = None,
                 device: str = 'cuda',
                 seed: int = 42):
        """
        Initialize high-resolution dataset.

        Args:
            num_samples: Total number of samples (default 100,000)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            include_fine_scale: Whether to include sub-meter phenomena
            spatial_holdout_regions: List of (lat_min, lat_max, lon_min, lon_max)
            temporal_holdout_range: (t_min, t_max) for temporal holdout
            device: Device to use
            seed: Random seed
        """
        self.device = device
        self.num_samples = num_samples
        self.include_fine_scale = include_fine_scale

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default spatial holdout: European cities for urban microclimate
        if spatial_holdout_regions is None:
            spatial_holdout_regions = [
                (48.0, 52.0, 2.0, 10.0),   # Paris to Berlin region
            ]
        self.spatial_holdout_regions = spatial_holdout_regions

        # Default temporal holdout: Peak summer (for max heat island effect)
        if temporal_holdout_range is None:
            temporal_holdout_range = (0.5, 0.6)  # Mid-summer
        self.temporal_holdout_range = temporal_holdout_range

        print(f"Generating {num_samples:,} high-resolution samples...")

        # Generate coordinates with focus on areas with fine-scale phenomena
        self.all_coords = self._generate_targeted_coordinates(num_samples)

        # Generate multi-scale targets
        self.all_targets = self._generate_multiscale_targets(self.all_coords)

        # Create splits
        self._create_splits(train_ratio, val_ratio, test_ratio)

        # Compute normalization statistics
        self._compute_normalization_stats()

    def _generate_targeted_coordinates(self, n: int) -> torch.Tensor:
        """
        Generate coordinates focused on areas with fine-scale phenomena.
        """
        coords_list = []

        # 1. Urban centers (40%) - for building-level microclimate
        n_urban = int(n * 0.4)
        urban_centers = [
            # Major cities with known heat island effects
            (40.7128, -74.0060, "New York"),
            (51.5074, -0.1278, "London"),
            (35.6762, 139.6503, "Tokyo"),
            (48.8566, 2.3522, "Paris"),
            (34.0522, -118.2437, "Los Angeles"),
            (41.8781, -87.6298, "Chicago"),
            (37.7749, -122.4194, "San Francisco"),
            (52.5200, 13.4050, "Berlin"),
            (55.7558, 37.6173, "Moscow"),
            (39.9042, 116.4074, "Beijing"),
            (-23.5505, -46.6333, "São Paulo"),
            (19.4326, -99.1332, "Mexico City"),
        ]

        urban_coords = []
        for i in range(n_urban):
            city = urban_centers[i % len(urban_centers)]
            # Dense sampling within 0.1 degrees (~10km) of city center
            # This gives ~10m resolution when many samples are in same area
            lat = city[0] + np.random.randn() * 0.01  # ~1km spread
            lon = city[1] + np.random.randn() * 0.01

            # Add micro-variations for building-level detail
            if self.include_fine_scale:
                lat += np.random.randn() * 0.0001  # ~10m variations
                lon += np.random.randn() * 0.0001

            elev = np.random.uniform(0, 300)  # Urban elevations
            time = np.random.uniform(0, 1)

            urban_coords.append([lat, lon, elev, time])

        # 2. Agricultural areas (25%) - for field boundaries
        n_agri = int(n * 0.25)
        agri_regions = [
            (40.0, -95.0, "US Midwest"),
            (49.0, 7.0, "Rhine Valley"),
            (22.0, 78.0, "Indian farmland"),
            (-15.0, -47.0, "Brazilian farmland"),
        ]

        agri_coords = []
        for i in range(n_agri):
            region = agri_regions[i % len(agri_regions)]
            # Agricultural fields with regular patterns
            base_lat = region[0] + np.random.uniform(-5, 5)
            base_lon = region[1] + np.random.uniform(-5, 5)

            if self.include_fine_scale:
                # Add field boundary effects (100m scale)
                field_x = np.random.randint(0, 100)
                field_y = np.random.randint(0, 100)
                lat = base_lat + field_x * 0.001  # ~100m grid
                lon = base_lon + field_y * 0.001
                # Add within-field variation
                lat += np.random.uniform(-0.0005, 0.0005)  # ~50m
                lon += np.random.uniform(-0.0005, 0.0005)
            else:
                lat = base_lat
                lon = base_lon

            elev = np.random.uniform(-10, 200)
            time = np.random.uniform(0, 1)

            agri_coords.append([lat, lon, elev, time])

        # 3. Coastal zones (20%) - for land-sea transitions
        n_coastal = int(n * 0.2)
        coastal_coords = []
        for i in range(n_coastal):
            # Sample along coastlines
            angle = np.random.uniform(0, 2*np.pi)
            # Major coastal cities/regions
            if i % 4 == 0:  # US East Coast
                lat = np.random.uniform(25, 45)
                lon = -75 + np.random.randn() * 2
            elif i % 4 == 1:  # Mediterranean
                lat = np.random.uniform(35, 45)
                lon = np.random.uniform(-5, 35)
            elif i % 4 == 2:  # Japan
                lat = np.random.uniform(30, 45)
                lon = 140 + np.random.randn() * 3
            else:  # California
                lat = np.random.uniform(32, 42)
                lon = -122 + np.random.randn() * 2

            # Add coastal gradient
            if self.include_fine_scale:
                # Distance from shore (can be negative for ocean)
                dist_from_shore = np.random.uniform(-0.01, 0.01)  # ~1km each side
                lat += dist_from_shore * np.cos(angle)
                lon += dist_from_shore * np.sin(angle)

            elev = np.random.uniform(-50, 100)
            time = np.random.uniform(0, 1)

            coastal_coords.append([lat, lon, elev, time])

        # 4. Mountainous terrain (15%) - for topographic effects
        n_mountain = n - n_urban - n_agri - n_coastal
        mountain_regions = [
            (46.0, 8.0, "Alps"),
            (40.0, -106.0, "Rockies"),
            (27.0, 87.0, "Himalayas"),
            (-32.0, -70.0, "Andes"),
        ]

        mountain_coords = []
        for i in range(n_mountain):
            region = mountain_regions[i % len(mountain_regions)]
            lat = region[0] + np.random.uniform(-3, 3)
            lon = region[1] + np.random.uniform(-3, 3)

            if self.include_fine_scale:
                # Add ridge/valley structure
                lat += np.sin(lon * 100) * 0.001  # Sinuous mountain ridges
                lon += np.cos(lat * 100) * 0.001

            # Higher elevations with more variation
            elev = np.random.uniform(500, 4000)
            time = np.random.uniform(0, 1)

            mountain_coords.append([lat, lon, elev, time])

        # Combine all coordinates
        all_coords = urban_coords + agri_coords + coastal_coords + mountain_coords
        coords_tensor = torch.tensor(all_coords, dtype=torch.float32)

        # Shuffle
        perm = torch.randperm(len(coords_tensor))
        coords_tensor = coords_tensor[perm]

        return coords_tensor.to(self.device)

    def _generate_multiscale_targets(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Generate target values with phenomena at multiple scales.

        Simulates surface temperature with:
        - Large scale: Climate zones
        - Regional: Weather patterns
        - Local: Urban heat islands
        - Fine: Building shadows
        - Ultra-fine: Surface materials
        """
        lat = coords[:, 0]
        lon = coords[:, 1]
        elev = coords[:, 2]
        time = coords[:, 3]

        # Initialize temperature
        temperature = torch.zeros_like(lat)

        # 1. Continental scale - base climate
        temperature += 30 * torch.cos(torch.deg2rad(lat))

        # 2. Seasonal variation
        seasonal_amp = 10 * (torch.abs(lat) / 90)
        temperature += seasonal_amp * torch.sin(2 * np.pi * time)

        # 3. Elevation effect
        temperature -= 0.0065 * torch.clamp(elev, min=0)

        # 4. Regional weather systems (100km scale)
        weather_pattern = 5 * torch.sin(torch.deg2rad(lon) * 3) * \
                         torch.cos(torch.deg2rad(lat) * 2 - 4 * np.pi * time)
        temperature += weather_pattern

        # 5. Diurnal cycle
        local_hour = (time * 24 + lon / 15) % 24
        diurnal = 3 * torch.sin(2 * np.pi * (local_hour - 14) / 24)
        temperature += diurnal

        if self.include_fine_scale:
            # 6. Urban heat islands (1km scale)
            # Check if in urban area (simplified - near round lat/lon)
            urban_mask = self._detect_urban_areas(coords)
            urban_heat = 4.0 * urban_mask  # 4°C warmer in city centers

            # Urban canyon effects (100m scale)
            street_pattern = torch.sin(lat * 1000) * torch.cos(lon * 1000)
            urban_heat += urban_mask * street_pattern * 1.5

            temperature += urban_heat

            # 7. Building-level shadows (10m scale)
            # Create building grid pattern
            building_x = torch.floor(lat * 10000) % 10  # ~10m buildings
            building_y = torch.floor(lon * 10000) % 10
            in_shadow = ((building_x < 3) & (building_y < 3)).float()

            # Shadows are cooler, but only during day and in urban areas
            is_day = ((local_hour > 6) & (local_hour < 18)).float()
            shadow_cooling = -2.0 * in_shadow * is_day * urban_mask
            temperature += shadow_cooling

            # 8. Surface material effects (1m scale)
            # Different materials (concrete, asphalt, grass, water)
            material_x = torch.floor(lat * 100000) % 4  # ~1m patches
            material_y = torch.floor(lon * 100000) % 4

            # Asphalt is hottest, grass coolest
            material_type = (material_x + material_y) % 4
            material_effect = torch.where(
                material_type == 0,  # Asphalt
                torch.tensor(2.0, device=self.device),
                torch.where(
                    material_type == 1,  # Concrete
                    torch.tensor(1.0, device=self.device),
                    torch.where(
                        material_type == 2,  # Grass
                        torch.tensor(-1.0, device=self.device),
                        torch.tensor(0.0, device=self.device)  # Water/other
                    )
                )
            )

            # Material effects are stronger during day in urban areas
            material_effect *= is_day * urban_mask * 0.5
            temperature += material_effect

            # 9. Agricultural field patterns (100m scale)
            # Different crops have different temperatures
            field_pattern = torch.sin(lat * 100) * torch.cos(lon * 100)
            agri_mask = self._detect_agricultural_areas(coords)
            temperature += agri_mask * field_pattern * 2.0

            # 10. Coastal gradients (10m scale near shore)
            coastal_mask = self._detect_coastal_areas(coords)
            # Sharp temperature gradient at coastline
            distance_from_shore = (elev + 50) / 100  # Normalized distance
            coastal_gradient = torch.tanh(distance_from_shore * 10) * 3
            temperature += coastal_mask * coastal_gradient

        # Add fine-scale noise
        temperature += torch.randn_like(temperature) * 0.5

        return temperature.to(self.device)

    def _detect_urban_areas(self, coords: torch.Tensor) -> torch.Tensor:
        """Detect if coordinates are in urban areas."""
        urban_mask = torch.zeros(len(coords), device=self.device)

        urban_centers = [
            (40.7128, -74.0060),  # NYC
            (51.5074, -0.1278),   # London
            (35.6762, 139.6503),  # Tokyo
            (48.8566, 2.3522),    # Paris
            (34.0522, -118.2437), # LA
            (41.8781, -87.6298),  # Chicago
        ]

        for center_lat, center_lon in urban_centers:
            dist_sq = (coords[:, 0] - center_lat)**2 + (coords[:, 1] - center_lon)**2
            # Urban influence decreases with distance
            urban_influence = torch.exp(-dist_sq / (2 * 0.01**2))  # ~1km radius
            urban_mask = torch.maximum(urban_mask, urban_influence)

        return urban_mask

    def _detect_agricultural_areas(self, coords: torch.Tensor) -> torch.Tensor:
        """Detect if coordinates are in agricultural areas."""
        agri_mask = torch.zeros(len(coords), device=self.device)

        # Simple heuristic: flat areas (low elevation) away from cities
        is_flat = (coords[:, 2] < 500) & (coords[:, 2] > -10)

        # Not in urban areas
        urban_mask = self._detect_urban_areas(coords)

        # Mid-latitudes (where most agriculture is)
        mid_lat = (torch.abs(coords[:, 0]) > 20) & (torch.abs(coords[:, 0]) < 60)

        agri_mask = is_flat.float() * (1 - urban_mask) * mid_lat.float()

        return agri_mask

    def _detect_coastal_areas(self, coords: torch.Tensor) -> torch.Tensor:
        """Detect if coordinates are near coastlines."""
        # Simple heuristic: low elevation near specific longitudes
        near_sea_level = torch.abs(coords[:, 2]) < 100

        # Near major coastlines (simplified)
        west_coast = torch.abs(coords[:, 1] + 122) < 5  # US West Coast
        east_coast = torch.abs(coords[:, 1] + 75) < 5   # US East Coast
        med_coast = (coords[:, 0] > 35) & (coords[:, 0] < 45) & \
                    (coords[:, 1] > -5) & (coords[:, 1] < 35)  # Mediterranean

        coastal_mask = near_sea_level.float() * (west_coast.float() + east_coast.float() + med_coast.float())
        coastal_mask = torch.clamp(coastal_mask, 0, 1)

        return coastal_mask

    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Create data splits with spatial and temporal holdouts."""
        n = self.num_samples
        indices = torch.arange(n)

        # Identify spatial holdout samples
        spatial_holdout_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        for region in self.spatial_holdout_regions:
            lat_min, lat_max, lon_min, lon_max = region
            mask = (self.all_coords[:, 0] >= lat_min) & (self.all_coords[:, 0] <= lat_max) & \
                   (self.all_coords[:, 1] >= lon_min) & (self.all_coords[:, 1] <= lon_max)
            spatial_holdout_mask = spatial_holdout_mask | mask

        # Identify temporal holdout samples
        t_min, t_max = self.temporal_holdout_range
        temporal_holdout_mask = (self.all_coords[:, 3] >= t_min) & (self.all_coords[:, 3] <= t_max)

        # Get indices for different splits
        spatial_holdout_mask_cpu = spatial_holdout_mask.cpu()
        temporal_holdout_mask_cpu = temporal_holdout_mask.cpu()

        self.spatial_holdout_indices = indices[spatial_holdout_mask_cpu]
        self.temporal_holdout_indices = indices[temporal_holdout_mask_cpu & ~spatial_holdout_mask_cpu]

        # Remaining indices for train/val/test
        remaining_mask = ~(spatial_holdout_mask_cpu | temporal_holdout_mask_cpu)
        remaining_indices = indices[remaining_mask]

        # Shuffle and split
        perm = torch.randperm(len(remaining_indices))
        remaining_indices = remaining_indices[perm]

        n_remaining = len(remaining_indices)
        n_train = int(n_remaining * train_ratio / (train_ratio + val_ratio + test_ratio))
        n_val = int(n_remaining * val_ratio / (train_ratio + val_ratio + test_ratio))

        self.train_indices = remaining_indices[:n_train]
        self.val_indices = remaining_indices[n_train:n_train + n_val]
        self.test_indices = remaining_indices[n_train + n_val:]

        print(f"Dataset splits:")
        print(f"  Train: {len(self.train_indices):,} samples")
        print(f"  Val: {len(self.val_indices):,} samples")
        print(f"  Test: {len(self.test_indices):,} samples")
        print(f"  Spatial holdout: {len(self.spatial_holdout_indices):,} samples")
        print(f"  Temporal holdout: {len(self.temporal_holdout_indices):,} samples")

    def _compute_normalization_stats(self):
        """Compute mean and std from training set only."""
        train_targets = self.all_targets[self.train_indices]
        self.target_mean = train_targets.mean()
        self.target_std = train_targets.std()

        print(f"Target statistics (from train set):")
        print(f"  Mean: {self.target_mean:.2f}°C")
        print(f"  Std: {self.target_std:.2f}°C")
        print(f"  Range: [{train_targets.min():.2f}, {train_targets.max():.2f}]°C")

        # Analyze scale of variations
        if self.include_fine_scale:
            print(f"\nMulti-scale phenomena included:")
            print(f"  - Continental: Climate zones (~30°C range)")
            print(f"  - Regional: Weather systems (~5°C)")
            print(f"  - Urban: Heat islands (~4°C)")
            print(f"  - Building: Shadow effects (~2°C)")
            print(f"  - Surface: Material variations (~2°C)")

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

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                       normalized: bool = False) -> Dict[str, float]:
        """Compute various error metrics."""
        if normalized:
            predictions = predictions * self.target_std + self.target_mean
            targets = targets * self.target_std + self.target_mean

        with torch.no_grad():
            mse = ((predictions - targets) ** 2).mean()
            mae = torch.abs(predictions - targets).mean()

            # MAPE with better handling of small values
            # Use 5°C as reference to avoid division issues
            reference_temp = 5.0
            mape = (torch.abs(predictions - targets) / (torch.abs(targets) + reference_temp)).mean() * 100

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