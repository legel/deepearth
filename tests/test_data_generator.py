"""
Test data generation utilities for DeepEarth
Creates synthetic but realistic Earth observation data for testing
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import h5py
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json


class SyntheticEarthDataGenerator:
    """Generate synthetic Earth observation data for testing"""
    
    def __init__(
        self,
        spatial_bounds: Tuple[float, float, float, float] = (-122.5, 37.5, -122.0, 38.0),
        temporal_range: Tuple[datetime, datetime] = None,
        spatial_resolution: float = 10.0,
        temporal_resolution: str = 'daily'
    ):
        self.spatial_bounds = spatial_bounds  # (min_lon, min_lat, max_lon, max_lat)
        
        if temporal_range is None:
            # Default to one year of data
            self.temporal_range = (
                datetime(2023, 1, 1),
                datetime(2023, 12, 31)
            )
        else:
            self.temporal_range = temporal_range
        
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        
        # Create coordinate grids
        self._create_coordinate_grids()
    
    def _create_coordinate_grids(self):
        """Create spatial and temporal coordinate grids"""
        # Spatial grid
        lon_range = np.linspace(self.spatial_bounds[0], self.spatial_bounds[2], 50)
        lat_range = np.linspace(self.spatial_bounds[1], self.spatial_bounds[3], 50)
        self.lon_grid, self.lat_grid = np.meshgrid(lon_range, lat_range)
        
        # Temporal grid
        if self.temporal_resolution == 'daily':
            time_delta = timedelta(days=1)
        elif self.temporal_resolution == 'hourly':
            time_delta = timedelta(hours=1)
        else:
            time_delta = timedelta(days=1)
        
        current_time = self.temporal_range[0]
        self.time_points = []
        while current_time <= self.temporal_range[1]:
            self.time_points.append(current_time)
            current_time += time_delta
    
    def generate_batch(
        self,
        batch_size: int,
        modalities: List[str] = ['vision', 'language', 'weather', 'species']
    ) -> Dict[str, torch.Tensor]:
        """Generate a batch of synthetic data"""
        batch = {}
        
        # Generate spatiotemporal coordinates
        batch['xyzt'] = self._generate_coordinates(batch_size)
        
        # Generate modality data
        if 'vision' in modalities:
            batch['images'] = self._generate_images(batch_size)
        
        if 'language' in modalities:
            batch['input_ids'], batch['attention_mask'] = self._generate_text(batch_size)
        
        # Additional modalities
        batch['modalities'] = {}
        
        if 'weather' in modalities:
            batch['modalities']['weather'] = self._generate_weather(batch_size)
        
        if 'species' in modalities:
            batch['modalities']['species'] = self._generate_species(batch_size)
        
        if 'soil' in modalities:
            batch['modalities']['soil'] = self._generate_soil(batch_size)
        
        # Add metadata
        batch['metadata'] = self._generate_metadata(batch_size)
        
        return batch
    
    def _generate_coordinates(self, batch_size: int) -> torch.Tensor:
        """Generate random spatiotemporal coordinates"""
        # Random spatial coordinates within bounds
        lon = torch.rand(batch_size) * (self.spatial_bounds[2] - self.spatial_bounds[0]) + self.spatial_bounds[0]
        lat = torch.rand(batch_size) * (self.spatial_bounds[3] - self.spatial_bounds[1]) + self.spatial_bounds[1]
        
        # Random elevation (normalized)
        z = torch.rand(batch_size) * 0.1  # Small variation
        
        # Random temporal coordinates (normalized to [0, 1])
        t = torch.rand(batch_size)
        
        # Normalize spatial coordinates
        lon_norm = lon / 180.0
        lat_norm = lat / 90.0
        
        return torch.stack([lon_norm, lat_norm, z, t], dim=1)
    
    def _generate_images(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic satellite imagery"""
        # Create synthetic RGB images with spatial patterns
        images = torch.zeros(batch_size, 3, 224, 224)
        
        for i in range(batch_size):
            # Base pattern (e.g., terrain)
            x = torch.linspace(-1, 1, 224)
            y = torch.linspace(-1, 1, 224)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            # Red channel - elevation-like pattern
            images[i, 0] = torch.sin(3 * xx) * torch.cos(3 * yy) * 0.5 + 0.5
            
            # Green channel - vegetation-like pattern
            images[i, 1] = torch.exp(-((xx**2 + yy**2) / 0.5)) * 0.8 + 0.2
            
            # Blue channel - water-like pattern
            images[i, 2] = torch.sigmoid(5 * (xx + yy)) * 0.6 + 0.2
            
            # Add noise
            images[i] += torch.randn(3, 224, 224) * 0.1
        
        # Clamp to valid range
        images = torch.clamp(images, 0, 1)
        
        return images
    
    def _generate_text(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic text descriptions"""
        # Vocabulary of Earth observation terms
        vocab = [
            "forest", "urban", "water", "agriculture", "mountain",
            "temperature", "precipitation", "vegetation", "soil",
            "climate", "weather", "seasonal", "change", "pattern"
        ]
        
        # Generate random sequences
        seq_length = 32
        input_ids = torch.randint(0, len(vocab), (batch_size, seq_length))
        
        # Random attention masks (simulate variable length)
        lengths = torch.randint(10, seq_length, (batch_size,))
        attention_mask = torch.zeros(batch_size, seq_length)
        
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        
        return input_ids, attention_mask.long()
    
    def _generate_weather(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic weather data"""
        # Temperature, humidity, pressure, wind_u, wind_v
        weather = torch.zeros(batch_size, 5)
        
        # Temperature (Celsius) - seasonal variation
        base_temp = 15.0
        weather[:, 0] = torch.randn(batch_size) * 10 + base_temp
        
        # Humidity (0-1)
        weather[:, 1] = torch.rand(batch_size) * 0.8 + 0.2
        
        # Pressure (normalized)
        weather[:, 2] = torch.randn(batch_size) * 0.1 + 1.0
        
        # Wind components
        weather[:, 3] = torch.randn(batch_size) * 5  # u component
        weather[:, 4] = torch.randn(batch_size) * 5  # v component
        
        return weather
    
    def _generate_species(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic species observations"""
        # Binary presence/absence for 64 species
        # Use spatial correlation - nearby locations have similar species
        species = torch.zeros(batch_size, 64)
        
        # Create species communities (groups that occur together)
        num_communities = 8
        for i in range(num_communities):
            # Species in this community
            community_species = torch.randint(0, 64, (8,))
            
            # Samples that belong to this community
            community_samples = torch.rand(batch_size) > 0.7
            
            # Set presence
            for j in range(batch_size):
                if community_samples[j]:
                    species[j, community_species] = 1
        
        # Add random occurrences
        random_presence = torch.rand(batch_size, 64) > 0.95
        species = torch.maximum(species, random_presence.float())
        
        return species
    
    def _generate_soil(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic soil properties"""
        # pH, organic matter, sand%, clay%, moisture, etc.
        soil = torch.zeros(batch_size, 10)
        
        # pH (4-9)
        soil[:, 0] = torch.rand(batch_size) * 5 + 4
        
        # Organic matter (%)
        soil[:, 1] = torch.rand(batch_size) * 10
        
        # Texture components (sum to 100%)
        sand = torch.rand(batch_size) * 80 + 10
        clay = torch.rand(batch_size) * (90 - sand)
        silt = 100 - sand - clay
        
        soil[:, 2] = sand / 100
        soil[:, 3] = clay / 100
        soil[:, 4] = silt / 100
        
        # Other properties
        soil[:, 5:] = torch.rand(batch_size, 5)
        
        return soil
    
    def _generate_metadata(self, batch_size: int) -> Dict:
        """Generate metadata for the batch"""
        metadata = {
            'timestamps': [],
            'locations': [],
            'data_sources': []
        }
        
        for i in range(batch_size):
            # Random timestamp
            time_idx = np.random.randint(0, len(self.time_points))
            metadata['timestamps'].append(self.time_points[time_idx])
            
            # Random location
            lon_idx = np.random.randint(0, self.lon_grid.shape[1])
            lat_idx = np.random.randint(0, self.lat_grid.shape[0])
            metadata['locations'].append(
                (self.lon_grid[lat_idx, lon_idx], self.lat_grid[lat_idx, lon_idx])
            )
            
            # Data source
            sources = ['sentinel2', 'landsat8', 'modis', 'synthetic']
            metadata['data_sources'].append(np.random.choice(sources))
        
        return metadata


class TestDataset(torch.utils.data.Dataset):
    """Test dataset that generates data on the fly"""
    
    def __init__(
        self,
        num_samples: int = 1000,
        modalities: List[str] = ['vision', 'language', 'weather'],
        generator: Optional[SyntheticEarthDataGenerator] = None
    ):
        self.num_samples = num_samples
        self.modalities = modalities
        
        if generator is None:
            self.generator = SyntheticEarthDataGenerator()
        else:
            self.generator = generator
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate single sample
        batch = self.generator.generate_batch(1, self.modalities)
        
        # Remove batch dimension
        sample = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value[0]
            elif isinstance(value, dict):
                sample[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        sample[key][k] = v[0]
                    else:
                        sample[key][k] = v
            else:
                sample[key] = value
        
        # Add index
        sample['idx'] = idx
        
        return sample


def create_test_hdf5_dataset(
    filepath: str,
    num_samples: int = 1000,
    modalities: List[str] = ['vision', 'weather']
) -> None:
    """Create HDF5 dataset for testing"""
    generator = SyntheticEarthDataGenerator()
    
    with h5py.File(filepath, 'w') as f:
        # Create groups
        coords_group = f.create_group('coordinates')
        vision_group = f.create_group('vision') if 'vision' in modalities else None
        weather_group = f.create_group('weather') if 'weather' in modalities else None
        species_group = f.create_group('species') if 'species' in modalities else None
        
        # Generate and save data
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generating sample {i}/{num_samples}")
            
            # Generate single sample
            batch = generator.generate_batch(1, modalities)
            
            # Save coordinates
            coords_group.create_dataset(f'sample_{i}', data=batch['xyzt'][0].numpy())
            
            # Save modalities
            if vision_group is not None:
                vision_group.create_dataset(f'sample_{i}', data=batch['images'][0].numpy())
            
            if weather_group is not None:
                weather_group.create_dataset(
                    f'sample_{i}', 
                    data=batch['modalities']['weather'][0].numpy()
                )
            
            if species_group is not None:
                species_group.create_dataset(
                    f'sample_{i}',
                    data=batch['modalities']['species'][0].numpy()
                )
        
        # Save metadata
        f.attrs['num_samples'] = num_samples
        f.attrs['spatial_bounds'] = generator.spatial_bounds
        f.attrs['modalities'] = modalities


def create_test_batch_for_model(
    batch_size: int = 8,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """Create a test batch ready for model input"""
    generator = SyntheticEarthDataGenerator()
    batch = generator.generate_batch(
        batch_size,
        modalities=['vision', 'language', 'weather', 'species', 'soil']
    )
    
    # Move to device
    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        else:
            return x
    
    return to_device(batch)


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = SyntheticEarthDataGenerator(
        spatial_bounds=(-122.5, 37.5, -122.0, 38.0),  # San Francisco Bay Area
        temporal_range=(datetime(2023, 1, 1), datetime(2023, 12, 31))
    )
    
    # Generate a batch
    batch = generator.generate_batch(batch_size=4)
    
    print("Generated batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
    
    # Create test dataset
    dataset = TestDataset(num_samples=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        if i == 0:
            print("\nDataLoader batch:")
            print(f"  xyzt: {batch['xyzt'].shape}")
            print(f"  images: {batch['images'].shape}")
        break
    
    # Create HDF5 dataset
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        create_test_hdf5_dataset(tmp.name, num_samples=100)
        print(f"\nCreated HDF5 dataset at {tmp.name}")
