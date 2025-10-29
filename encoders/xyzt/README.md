# Earth4D: Multi-Resolution 4D Spacetime Encoder for DeepEarth

Earth4D is a pioneering 4D spatiotemporal encoder that enables planetary-scale deep learning on Earth observation data. Built on NVIDIA's multi-resolution hash encoding architecture and extended to 4D spacetime, Earth4D efficiently encodes latitude, longitude, elevation, and time into learnable features at multiple scales - from sub-meter spatial resolution to microsecond temporal precision.

## 🌍 Core Innovation

Earth4D is the foundation of DeepEarth's ability to process and learn from the entire planet's observational data across space and time. By using decomposed hash encoding with separate spatial (xyz) and temporal (xyt, yzt, xzt) projections, it achieves:

- **Planetary Coverage**: Multi-resolution encoding from continental scale to sub-meter precision
- **Temporal Dynamics**: Flexible temporal encoding from years to sub-second precision
- **Memory Efficiency**: Configurable hash table sizes to balance memory and collision rates
- **GPU Acceleration**: Custom CUDA kernels for real-time encoding at scale

## 🚀 Quick Start

### Installation

```bash
# Clone DeepEarth repository
git clone https://github.com/deepearth/deepearth.git
cd deepearth/encoders/xyzt

# Install dependencies
bash install.sh
```

### Basic Usage

```python
from earth4d import Earth4D
import torch

# Check device availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder = Earth4D(
    spatial_levels=24,
    temporal_levels=24,
    spatial_log2_hashmap_size=22,
    temporal_log2_hashmap_size=22,
    verbose=True
).to(device)

# Example coordinates: [lat, lon, elev_m, time_norm]
coords = torch.tensor([
    [37.7749, -122.4194, 50.0, 0.5],   # San Francisco
    [40.7128, -74.0060, 100.0, 0.7],   # New York
    [-33.8688, 151.2093, 20.0, 0.3],   # Sydney
], device=device)

features = encoder(coords)
print(f"\nInput shape: {coords.shape}")
print(f"Output shape: {features.shape}")
```

## 📊 Resolution Scale Table

### Spatial Encoder (XYZ)

| Level | Grid Resolution | Meters/Cell |
|-------|----------------|-------------|
| 0 | 32 | 398.2km |
| 1 | 64 | 199.1km |
| 2 | 128 | 99.5km |
| 3 | 256 | 49.8km |
| 4 | 512 | 24.9km |
| 5 | 1024 | 12.4km |
| 6 | 2048 | 6.2km |
| 7 | 4096 | 3.1km |
| 8 | 8192 | 1.6km |
| 9 | 16384 | 777.7m |
| 10 | 32768 | 388.9m |
| 11 | 65536 | 194.4m |
| 12 | 131072 | 97.21m |
| 13 | 262144 | 48.61m |
| 14 | 524288 | 24.30m |
| 15 | 1048576 | 12.15m |
| 16 | 2097152 | 6.076m |
| 17 | 4194304 | 3.038m |
| 18 | 8388608 | 1.519m |
| 19 | 16777216 | 0.7595m |
| 20 | 33554432 | 0.3797m |
| 21 | 67108864 | 0.1899m |
| 22 | 134217728 | 0.0949m |

### Temporal Encoders (XYT, YZT, XZT)

| Level | Grid Resolution | Seconds/Cell |
|-------|----------------|--------------|
| 0 | 32 | 986175.0 |
| 1 | 64 | 493087.5 |
| 2 | 128 | 246543.8 |
| 3 | 256 | 123271.9 |
| 4 | 512 | 61635.9 |
| 5 | 1024 | 30818.0 |
| 6 | 2048 | 15409.0 |
| 7 | 4096 | 7704.5 |
| 8 | 8192 | 3852.2 |
| 9 | 16384 | 1926.1 |
| 10 | 32768 | 963.1 |
| 11 | 65536 | 481.5 |
| 12 | 131072 | 240.8 |
| 13 | 262144 | 120.4 |
| 14 | 524288 | 60.2 |
| 15 | 1048576 | 30.1 |
| 16 | 2097152 | 15.0 |
| 17 | 4194304 | 7.5 |
| 18 | 8388608 | 3.8 |
| 19 | 16777216 | 1.9 |
| 20 | 33554432 | 0.9 |
| 21 | 67108864 | 0.5 |
| 22 | 134217728 | 0.2 |

## 🔬 Research Applications

Earth4D enables breakthrough research in:

- **Climate Modeling**: Multi-scale climate dynamics from global to local
- **Weather Prediction**: High-resolution nowcasting with temporal continuity
- **Earth Observation**: Fusion of satellite, aerial, and ground sensors
- **Urban Planning**: Building-level environmental modeling
- **Agriculture**: Precision crop monitoring at plant scale
- **Disaster Response**: Real-time multi-scale hazard assessment

## 📚 Technical Papers

Earth4D builds on:
- [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/) (Müller et al., 2022)
- [Grid4D](https://github.com/JiaweiXu8/Grid4D) (4D extension)

*Earth4D: Encoding the entire planet across space and time, one hash at a time.*