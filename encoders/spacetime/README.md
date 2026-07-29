# Earth4D: Multi-Resolution 4D Space-Time Positional Encoder

Earth4D is a planetary-scale 4D (_x_, _y_, _z_, _t_) space-time positional encoder for Earth observation data. 

Built on NVIDIA's [multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/) architecture, extended to 4D space-time, and enhanced with [learned hash probing](https://research.nvidia.com/labs/toronto-ai/compact-ngp/) (Takikawa et al., 2023), Earth4D efficiently encodes (**latitude**, **longitude**, **elevation**, **time**) into learnable features at multiple scales—from sub-meter spatial resolution to sub-second temporal precision.

## Core Innovation

Earth4D combines decomposed hash encoding with learned hash probing. Using separate spatial (xyz) and spatio-temporal (xyt, yzt, xzt) grids with learned probe selection, it targets:

- **Testable ecological representation**: to be evaluated against matched generic encoders on future and held-site data
- **Learned Hash Probing**: a historical exploratory run reported lower error than baseline hashing; confirmation is pending
- **Planetary Coverage**: Multi-resolution encoding from continental scale to sub-meter precision
- **Temporal Dynamics**: Flexible temporal encoding from years to sub-second precision
- **GPU Acceleration**: Custom CUDA kernels with learned probe selection, parallelizable across levels and spatio-temporal boundaries

## Benchmark Performance

**Globe-LFMC 2.0** (Live Fuel Moisture Content Prediction, AI2 official train/test split: 76,467/13,297):

### Historical exploratory result — not confirmed

| Model | Data Inputs | MAE (pp) | RMSE (pp) | R² |
|-------|-------------|----------|-----------|-----|
| Earth4D (historical learned-hashing run) | (x,y,z,t) + Species | 11.7 | 18.7 | 0.783 |
| Galileo (Pre-Trained) | (x,y,z,t) + Species + Remote Sensing | 12.6 | 18.9 | 0.72 |

These numbers are retained for provenance, not as a scientific headline. The historical search selected
configurations by test R², evaluated test during training, and fit geographic range on all coordinates. The
runnable LFMC harness is also absent from this checkout. Earth4D has not yet passed the preregistered temporal
and held-site gate in `autoresearch/programs/spacetime/program.md`.

## Quick Start

### Installation

```bash
# Clone DeepEarth repository
git clone https://github.com/legel/deepearth.git
cd deepearth/encoders/spacetime

# Install dependencies
bash install.sh
```

### Run the LFMC data/split gate

```bash
# From the repository root: pinned real data, strict split audit, train-only baselines
python3 -m autoresearch.programs.spacetime.science_gate \
  --download --json-out data/lfmc/earth4d_science_gate_dev.json
```

This command does not train Earth4D; its artifact explicitly records `earth4d_evaluated=false`.

### Basic Usage

```python
from earth4d import Earth4D
import torch

# Check device availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Historical learned-probing defaults (not yet confirmed optimal under the current gate)
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

## Coordinate System

Earth4D supports two coordinate systems for mapping (latitude, longitude, elevation, time) to the internal (x, y, z, t) representation:

### Geographic Mode (Default)

The geographic coordinate system directly maps latitude, longitude, and elevation to the hash grid dimensions:

| Dimension | Mapping | Range |
|-----------|---------|-------|
| x | Latitude | -90° to +90° |
| y | Longitude | -180° to +180° |
| z | Elevation | meters above MSL |
| t | Time | normalized [0, 1] |

**Key benefit**: Points at the same latitude share x-coordinate values across the globe. This enables **ecological prior transfer** between regions with similar latitudes.

For example, San Francisco (37.8°N, -122.4°W) and the Amalfi Coast (37.8°N, 14.5°E) share the same x-value in the **xzt grid** (latitude, elevation, time), allowing the model to learn shared patterns between Mediterranean climate regions despite being on different continents.

**Grid semantics in geographic mode:**
- **xyz** (lat, lon, elev): Pure spatial features for location
- **xyt** (lat, lon, time): Surface dynamics over time
- **yzt** (lon, elev, time): Continental altitude-time patterns
- **xzt** (lat, elev, time): **Enables ecological prior transfer across longitudes**

### ECEF Mode (Legacy)

The legacy Earth-Centered Earth-Fixed (ECEF) coordinate system transforms lat/lon/elev to Cartesian coordinates centered at Earth's center using the WGS84 ellipsoid:

```python
encoder = Earth4D(coordinate_system='ecef')  # Legacy mode
```

In ECEF mode, the latitude relationship is destroyed—points at the same latitude but different longitudes have completely different (x, y, z) coordinates.

### Range Configuration

**Global coverage** (default for geographic mode):
```python
encoder = Earth4D()  # Full Earth coverage by default
```

**Fit to training data** (recommended for regional datasets):
```python
encoder = Earth4D()
encoder.fit_range(train_coords, buffer_fraction=0.25)
# Allocates 25% buffer for generalization beyond training distribution
# Warns if test data exceeds fitted range
```

**Custom regional range**:
```python
from coordinates import GeoAdaptiveRange

mediterranean = GeoAdaptiveRange(
    lat_min=30.0, lat_max=50.0,
    lon_min=-10.0, lon_max=40.0,
    elev_min=0.0, elev_max=3000.0
)
encoder = Earth4D(geo_range=mediterranean)
```

### Elevation Semantics

Elevation is measured as **meters above Mean Sea Level (MSL)**, not relative to local terrain. This design choice:
- Enables learning of altitude-dependent ecological patterns (temperature, pressure, vegetation zones)
- Points at the same elevation share z-values regardless of underlying terrain
- Mountain peaks at identical elevations (e.g., two 14,000ft peaks) will share hash cells in the xzt grid

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
encoders/spacetime/
├── earth4d.py          # Main Earth4D encoder module
├── training.py         # Generic training infrastructure with Protocol classes
├── coordinates.py      # Coordinate transformation utilities
├── sorting.py          # Spatiotemporal sorting for cache locality
├── hashencoder/        # CUDA hash encoding kernels
│   ├── hashgrid.py     # PyTorch interface
│   └── src/            # CUDA source files
└── ../../autoresearch/programs/spacetime/
    └── science_gate.py # Pinned Globe-LFMC split/baseline audit
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
