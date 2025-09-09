# Earth4D: Grid4D Encoder for Planetary (X,Y,Z,T) Deep Learning

Earth4D provides a Grid4D-based spatiotemporal encoder for planetary-scale deep learning tasks involving latitude, longitude, elevation, and time coordinates.

## Overview

Earth4D transforms your Grid4D LFMC prediction model into a general-purpose encoder for any spatiotemporal prediction task. It uses decomposed hash encoding with separate spatial (xyz) and temporal (xyt, yzt, xzt) projections for efficient 4D representation learning.

## Key Features

- **Multi-resolution hash encoding** for scalable feature extraction
- **Configurable spatial and temporal resolution hierarchies**
- **Optional automatic ECEF coordinate conversion** (lat/lon/elevation → normalized ECEF)
- **Configurable multi-resolution scales** in meters and seconds
- **Designed for planetary-scale spatiotemporal modeling**

## Installation

```bash
# Clone the deepearth repository
git clone https://github.com/legel/deepearth
cd deepearth

# Install dependencies
pip install torch torchvision
pip install numpy

# Install hash encoder (adjust based on your setup)
pip install hashencoder  # or your specific hash encoding library
```

## Usage

### Basic Usage with Normalized Coordinates

```python
from deepearth.encoders.xyzt import Earth4D

# Create basic encoder
encoder = Earth4D()

# Input: normalized coordinates (batch_size, 4) -> (x, y, z, t) in [0, 1]
coordinates = torch.rand(100, 4)  

# Encode
spatial_features, temporal_features = encoder(coordinates)
print(f"Spatial features: {spatial_features.shape}")    # (100, 32)
print(f"Temporal features: {temporal_features.shape}")  # (100, 96)
```

### Advanced Usage with Raw Geographic Coordinates

```python
from deepearth.encoders.xyzt import create_earth4d_with_auto_conversion

# Create encoder with automatic ECEF conversion
encoder = create_earth4d_with_auto_conversion()

# Input: raw geographic coordinates (lat, lon, elevation, time)
# lat/lon in degrees, elevation in meters, time in seconds since epoch
geo_coords = torch.tensor([
    [37.7749, -122.4194, 50.0, 1640995200.0],  # San Francisco
    [40.7128, -74.0060, 100.0, 1640995260.0],  # New York
    [51.5074, -0.1278, 25.0, 1640995320.0],    # London
])

# Encoder automatically converts to ECEF and normalizes
spatial_features, temporal_features = encoder(geo_coords)
```

### Custom Multi-Resolution Scales

```python
from deepearth.encoders.xyzt import create_earth4d_with_physical_scales

# Define scales in physical units
spatial_scales_meters = [16, 32, 64, 128, 256, 512]      # meters
temporal_scales_seconds = [3600, 86400, 604800, 2592000] # hour, day, week, month

# Create encoder with custom scales
encoder = create_earth4d_with_physical_scales(
    spatial_scales_meters=spatial_scales_meters,
    temporal_scales_seconds=temporal_scales_seconds
)

coordinates = torch.rand(50, 4)
features = encoder(coordinates)
```

### Integration with Your Models

```python
import torch.nn as nn
from deepearth.encoders.xyzt import Earth4D

class MyPlanetaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Earth4D encoder for spatiotemporal features
        self.earth4d = Earth4D(
            spatial_levels=16,
            temporal_levels=16,
            auto_ecef_convert=True  # Handle raw lat/lon/elevation
        )
        
        # Your prediction head
        feature_dim = self.earth4d.get_feature_dimensions()['total']
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Your target variable
        )
    
    def forward(self, coordinates):
        # coordinates: (batch, 4) -> (lat, lon, elevation, time)
        spatial_feat, temporal_feat = self.earth4d(coordinates)
        combined_feat = torch.cat([spatial_feat, temporal_feat], dim=-1)
        prediction = self.predictor(combined_feat)
        return prediction

# Usage
model = MyPlanetaryModel()
geo_coords = torch.tensor([[37.7749, -122.4194, 50.0, 1640995200.0]])
prediction = model(geo_coords)
```

## Configuration Options

### Earth4D Parameters

- `spatial_levels`: Number of spatial hash encoding levels (default: 16)
- `spatial_features`: Features per spatial level (default: 2)
- `spatial_base_res`: Base spatial resolution (default: 16)
- `spatial_max_res`: Maximum spatial resolution (default: 512)
- `temporal_levels`: Number of temporal hash encoding levels (default: 16)
- `temporal_features`: Features per temporal level (default: 2)
- `auto_ecef_convert`: Enable automatic coordinate conversion (default: False)
- `spatial_scales_meters`: Custom spatial scales in meters (optional)
- `temporal_scales_seconds`: Custom temporal scales in seconds (optional)

### Feature Dimensions

```python
encoder = Earth4D()
dims = encoder.get_feature_dimensions()
print(dims)
# {'spatial': 32, 'temporal': 96, 'total': 128}
```

## Architecture Details

Earth4D uses the Grid4D decomposed encoding strategy:

1. **Spatial Encoding**: 3D hash encoding for (x, y, z) coordinates
2. **Temporal Projections**: Three 3D hash encodings for:
   - (x, y, t) - XY plane over time
   - (y, z, t) - YZ plane over time  
   - (x, z, t) - XZ plane over time
3. **Feature Fusion**: Separate spatial and temporal feature vectors

This decomposition enables efficient learning of spatiotemporal patterns while maintaining computational efficiency.

## Performance Tips

1. **Batch Processing**: Use larger batch sizes for better GPU utilization
2. **Coordinate Normalization**: Ensure coordinates are properly normalized to [0, 1]
3. **Memory Usage**: Adjust hash map sizes (`spatial_hashmap`, `temporal_hashmap`) based on available memory
4. **Resolution Tuning**: Start with default resolutions and adjust based on your data scale

## Examples

See the [examples](examples/) directory for complete working examples:
- LFMC prediction with Earth4D
- Climate modeling with multi-scale features
- Species distribution modeling with raw coordinates

## Citation

If you use Earth4D in your research, please cite:

```bibtex
@software{earth4d2024,
  title={Earth4D: Grid4D Encoder for Planetary Deep Learning},
  author={Grid4D LFMC Team},
  year={2024},
  url={https://github.com/legel/deepearth}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions and support:
- Create an issue on GitHub
- Check the [DeepEarth documentation](https://deepearth.readthedocs.io)
- Join the DeepEarth community discussions