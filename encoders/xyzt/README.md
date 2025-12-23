# Earth4D: Multi-Resolution 4D Space-Time Positional Encoder

Earth4D is a planetary-scale 4D (_x_, _y_, _z_, _t_) space-time positional encoder for Earth observation data. 

Built on NVIDIA's [multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/) architecture, extended to 4D space-time, and enhanced with [learned hash probing](https://research.nvidia.com/labs/toronto-ai/compact-ngp/) (Takikawa et al., 2023), Earth4D efficiently encodes (**latitude**, **longitude**, **elevation**, **time**) into learnable features at multiple scales—from sub-meter spatial resolution to sub-second temporal precision.

## Core Innovation

Earth4D combines decomposed hash encoding with learned hash probing for state-of-the-art accuracy. Using separate spatial (xyz) and spatio-temporal (xyt, yzt, xzt) grids with learned probe selection, it achieves:

- **State-of-the-Art Accuracy**: Surpasses pre-trained foundation models on ecological forecasting benchmarks using only coordinates
- **Learned Hash Probing**: 25% MAE reduction and 28% R² improvement over baseline hash encoding
- **Planetary Coverage**: Multi-resolution encoding from continental scale to sub-meter precision
- **Temporal Dynamics**: Flexible temporal encoding from years to sub-second precision
- **GPU Acceleration**: Custom CUDA kernels with learned probe selection, parallelizable across levels and spatio-temporal boundaries

## Benchmark Performance

**Globe-LFMC 2.0** (Live Fuel Moisture Content Prediction, AI2 official train/test split: 76,467/13,297):

### State-of-the-Art Results

| Model | Data Inputs | MAE (pp) | RMSE (pp) | R² |
|-------|-------------|----------|-----------|-----|
| **Earth4D** (Learned Hashing) | (x,y,z,t) + Species Name | **12.1** | 19.9 | **0.755** |
| Galileo (Pre-Trained) | (x,y,z,t) + Species + Remote Sensing | 12.6 | **18.9** | 0.72 |

*Earth4D surpasses Allen Institute for AI's Galileo foundation model in MAE and R², without access to pre-trained data or weights from Sentinel-2 optical imagery, Sentinel-1 SAR, VIIRS night lights, ERA-5 weather, TerraClimate soil/water, and SRTM topography. Earth4D only inputs (x,y,z,t) coordinates and learns species embeddings from scratch.*

### Micro Earth4D

| Configuration | Parameters | GPU Memory | Training Speed | MAE (pp) | R² |  
|---------------|------------|------------|----------------|----------|----|
| **Micro** | **5.1M** | **850MB** | **4× faster** | **15.0** | **0.668** |


## Quick Start

### Installation

```bash
# Clone DeepEarth repository
git clone https://github.com/legel/deepearth.git
cd deepearth/encoders/xyzt

# Install dependencies
bash install.sh
```

### Run LFMC Benchmark

```bash
# Train on Globe-LFMC 2.0 benchmark (achieves SoTA in ~5 hours)
python -m benchmarks.lfmc.train --epochs 10000 --output-dir ./outputs
```

See [benchmarks/lfmc/README.md](benchmarks/lfmc/README.md) for full benchmark documentation.

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
print(f"Output shape: {features.shape}")  # [3, 192]

# Disable learned probing if needed (baseline mode)
encoder_baseline = Earth4D(enable_learned_probing=False).to(device)
```

## Architecture Details

Earth4D outputs a **192-dimensional feature vector** per (x,y,z,t) coordinate:
- 4 grids (xyz, xyt, yzt, xzt)
- 24 levels per grid
- 2D feature per level
- Total: 4 × 24 × 2 = 192 dimensions

Default configuration requires **724M trainable parameters** (~11 GB GPU memory during training). Each level stores up to 2²² entries. The architecture is parallelizable across levels and spatio-temporal boundaries.

## Resolution Scale Table

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
| 24 | 268435456 | 0.0475m |

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
| 24 | 268435456 | 0.1 |

## Research Applications

Earth4D enables research in:

- **Climate Modeling**: Multi-scale climate dynamics from global to local
- **Ecological Forecasting**: Vegetation moisture, phenology, species distributions
- **Weather Prediction**: High-resolution nowcasting with temporal continuity
- **Earth Observation**: Fusion of satellite, aerial, and ground sensors
- **Urban Planning**: Building-level environmental modeling
- **Agriculture**: Precision crop monitoring at plant scale
- **Disaster Response**: Real-time multi-scale hazard assessment
- **Subsurface Modeling**: Geological spatial reconstruction

## Project Structure

```
encoders/xyzt/
├── earth4d.py          # Main Earth4D encoder module
├── training.py         # Generic training infrastructure with Protocol classes
├── coordinates.py      # Coordinate transformation utilities
├── sorting.py          # Spatiotemporal sorting for cache locality
├── hashencoder/        # CUDA hash encoding kernels
│   ├── hashgrid.py     # PyTorch interface
│   └── src/            # CUDA source files
└── benchmarks/
    └── lfmc/           # Globe-LFMC 2.0 benchmark
        ├── train.py    # Training script
        ├── model.py    # SpeciesAwareLFMCModel
        ├── data.py     # Dataset and data loading
        └── README.md   # Benchmark documentation
```

## Key Technical Foundations

Earth4D builds on:
- [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/) (Müller et al., 2022)
- [Compact Neural Graphics Primitives with Learned Hash Probing](https://research.nvidia.com/labs/toronto-ai/compact-ngp/) (Takikawa et al., 2023)
- [Grid4D](https://github.com/JiaweiXu8/Grid4D) (Jiawei et al., 2024)

## Citation

```bibtex
@inproceedings{legel2026deepearth,
  title={Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding},
  author={Legel, Lance and Huang, Qin and Voelker, Brandon and Neamati, Daniel and Johnson, Patrick Alan and Bastani, Favyen and Rose, Jeff and Hennessy, James Ryan and Guralnick, Robert and Soltis, Douglas and Soltis, Pamela and Wang, Shaowen},
  booktitle={2026 World Modeling Workshop at Mila - Quebec AI Institute},
  year={2026}
}
```

*Earth4D: Encoding the entire planet across space and time, one hash at a time.*
