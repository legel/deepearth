# Earth4D: Multi-Resolution 4D Spacetime Encoder with Learned Hash Probing

Earth4D is a pioneering 4D spatiotemporal encoder that enables planetary-scale deep learning on Earth observation data. Built on NVIDIA's [multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/) architecture, extended to 4D spacetime, and enhanced with [learned hash probing](https://research.nvidia.com/labs/toronto-ai/compact-ngp/) (Takikawa et al., 2023), Earth4D efficiently encodes latitude, longitude, elevation, and time into learnable features at multiple scales - from sub-meter spatial resolution to microsecond temporal precision.

## 🌍 Core Innovation

Earth4D combines decomposed hash encoding with learned hash probing for superior accuracy. Using separate spatial (xyz) and temporal (xyt, yzt, xzt) projections with learned probe selection, it achieves:

- **State-of-the-Art Accuracy**: +26% R² improvement on [Globe-LFMC 2.0](https://www.nature.com/articles/s41597-024-03159-6) benchmark (matches pretrained foundation model performance using only coordinates)
- **Extreme Scalability**: 5M to 724M parameters (99% compression) with 4× training speedup and 15× memory reduction while maintaining strong performance
- **Coordinate-Only Learning**: Learns from (x,y,z,t) + species embeddings without satellite imagery, weather, or topography
- **Planetary Coverage**: Multi-resolution encoding from continental scale to sub-meter precision
- **Temporal Dynamics**: Flexible temporal encoding from years to sub-second precision
- **Automatic Optimization**: [Entropy-regularized](https://github.com/legel/deepearth/blob/9b34f6728974a9fcd2a8e4b517fa0637f39dde8a/encoders/xyzt/earth4d.py#L976) probe learning (optimal weight=0.5) with automatic tuning
- **GPU Acceleration**: Custom CUDA kernels with learned probe selection

## 🏆 Benchmark Performance

**Globe-LFMC 2.0** (Live Fuel Moisture Content Prediction, AI2 official train/test split):

### Full-Scale Performance

| Configuration | MAE (pp) | R² | Training Time | vs. Pretrained Galileo |
|--------------|----------|-----|---------------|----------------------|
| **Earth4D + Learned Probing (N_p=32)** | **12.7** | **0.735** | 1049s (2500 epochs) | **Matches** (12.6pp, 0.72) |
| Earth4D (baseline) | 16.5 | 0.583 | 612s (2500 epochs) | - |
| **Improvement** | **-23.3%** | **+26.1%** | **+71.3%** | - |

*Earth4D with learned hash probing matches Allen Institute for AI's Galileo foundation model (pretrained on large-scale multimodal Earth observation data) using only (x,y,z,t) coordinates and learnable species embeddings - no satellite imagery, weather data, or topography required.*

### Extreme Compression (Edge/Mobile Deployment)

| Configuration | Parameters | GPU Memory | Training Speed | MAE (pp) | R² | vs. Baseline |
|--------------|-----------|------------|----------------|----------|-----|--------------|
| **Compressed (2^14 hash)** | **5.1M** | **850MB** | **4× faster** | **15.0** | **0.668** | **+14.7% R²** |
| Baseline (2^22 hash) | 724M | 12GB+ | 1× | 16.6 | 0.582 | - |
| **Reduction** | **-99.3%** | **-93%** | **+300%** | **-9.7%** | **+14.7%** | - |

*99% parameter reduction (724M→5.1M) with 4× training speedup enables edge deployment while still outperforming baseline. Demonstrates Earth4D's scalability from mobile devices to datacenter-scale parallel computing.*

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

# Learned hash probing enabled by default with optimal settings (N_p=32, entropy=0.5)
encoder = Earth4D(
    spatial_levels=24,
    temporal_levels=24,
    spatial_log2_hashmap_size=22,
    temporal_log2_hashmap_size=22,
    verbose=True
    # enable_learned_probing=True (default),
    # probing_range=32 (default, must be power-of-2),
    # probe_entropy_weight=0.5 (default, automatic entropy regularization)
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

# Disable learned probing if needed (baseline mode)
encoder_baseline = Earth4D(enable_learned_probing=False).to(device)
```

## 📊 Resolution Scale Table

### Spatial Encoder (XYZ)

| Level | Grid Resolution | Meters/Cell |
|-------|----------------|-------------|
| 1 | 32 | 398.2km |
| 2 | 64 | 199.1km |
| 3 | 128 | 99.5km |
| 4 | 256 | 49.8km |
| 5 | 512 | 24.9km |
| 6 | 1024 | 12.4km |
| 7 | 2048 | 6.2km |
| 8 | 4096 | 3.1km |
| 9 | 8192 | 1.6km |
| 10 | 16384 | 777.7m |
| 11 | 32768 | 388.9m |
| 12 | 65536 | 194.4m |
| 13 | 131072 | 97.21m |
| 14 | 262144 | 48.61m |
| 15 | 524288 | 24.30m |
| 16 | 1048576 | 12.15m |
| 17 | 2097152 | 6.076m |
| 18 | 4194304 | 3.038m |
| 19 | 8388608 | 1.519m |
| 20 | 16777216 | 0.7595m |
| 21 | 33554432 | 0.3797m |
| 22 | 67108864 | 0.1899m |
| 23 | 134217728 | 0.0949m |

### Temporal Encoders (XYT, YZT, XZT)

| Level | Grid Resolution | Seconds/Cell |
|-------|----------------|--------------|
| 1 | 32 | 986175.0 |
| 2 | 64 | 493087.5 |
| 3 | 128 | 246543.8 |
| 4 | 256 | 123271.9 |
| 5 | 512 | 61635.9 |
| 6 | 1024 | 30818.0 |
| 7 | 2048 | 15409.0 |
| 8 | 4096 | 7704.5 |
| 9 | 8192 | 3852.2 |
| 10 | 16384 | 1926.1 |
| 11 | 32768 | 963.1 |
| 12 | 65536 | 481.5 |
| 13 | 131072 | 240.8 |
| 14 | 262144 | 120.4 |
| 15 | 524288 | 60.2 |
| 16 | 1048576 | 30.1 |
| 17 | 2097152 | 15.0 |
| 18 | 4194304 | 7.5 |
| 19 | 8388608 | 3.8 |
| 20 | 16777216 | 1.9 |
| 21 | 33554432 | 0.9 |
| 22 | 67108864 | 0.5 |
| 23 | 134217728 | 0.2 |

## 🔬 Research Applications

Earth4D enables breakthrough research in:

- **Climate Modeling**: Multi-scale climate dynamics from global to local
- **Weather Prediction**: High-resolution nowcasting with temporal continuity
- **Earth Observation**: Fusion of satellite, aerial, and ground sensors
- **Urban Planning**: Building-level environmental modeling
- **Agriculture**: Precision crop monitoring at plant scale
- **Disaster Response**: Real-time multi-scale hazard assessment

## 📚 Key Technical Foundations

Earth4D builds on:
- [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/) (Müller et al., 2022)
- [Compact Neural Graphics Primitives with Learned Hash Probing](https://research.nvidia.com/labs/toronto-ai/compact-ngp/) (Takikawa et al., 2023)
- [Grid4D](https://github.com/JiaweiXu8/Grid4D) (4D extension)

*Earth4D: Encoding the entire planet across space and time, one hash at a time.*