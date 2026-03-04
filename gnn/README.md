# GNN Weather Forecasting with Multi-Scale Spatiotemporal Nodes

Graph Neural Network weather forecasting using multi-scale spatiotemporal node embeddings powered by hash-based positional encoding.

## Overview

This module provides a `MultiScaleSpatioTemporalNode` embedder that integrates multi-scale coordinate encoding into GNN-based weather models. It follows the encode-process-decode architecture (GraphCast, Neural-LAM) and provides spatiotemporally-aware node representations.

**Supported Encoders:**
- **Earth4D**: Absolute coordinates (lat, lon, elev, time) - Tested
- **Energy4D**: Relative coordinates (Δx, Δy, Δz, Δt) - Not yet trained/tested

**Key Features:**
- Multi-scale hash encoding (24 spatial + 24 temporal levels)
- Flexible coordinate systems (geographic or ECEF)
- Adaptive range fitting for regional domains
- ~724M parameters for comprehensive spatiotemporal encoding
- Drop-in replacement for standard MLP encoders

**Note:** Learned probing is disabled by default due to PyTorch autograd in-place operation errors during gradient computation.

## Quick Start

### Installation

Ensure you have:
- Python 3.10+
- PyTorch 2.7+
- CUDA 11.8+
- Earth4D encoder (in `encoders/xyzt/`)

### Basic Usage

```python
from gnn.multiscale_spatiotemporal_node import MultiScaleSpatioTemporalNode

# Initialize embedder with Earth4D (absolute coordinates)
embedder = MultiScaleSpatioTemporalNode(
    input_dim=13,              # Number of input features per node
    hidden_dim=64,             # Output dimension for GNN
    encoder_type="earth4d",    # Use Earth4D (absolute coordinates)
    spatial_levels=24,         # Multi-scale spatial encoding
    temporal_levels=24,        # Multi-scale temporal encoding
    coordinate_system="geographic",
    use_adaptive_range=True,
).cuda()

# Or use Energy4D (relative coordinates - not yet trained/tested)
# embedder = MultiScaleSpatioTemporalNode(
#     input_dim=13,
#     hidden_dim=64,
#     encoder_type="energy4d",   # Use Energy4D (relative coordinates)
#     ...
# )

# Encode nodes
node_embeddings = embedder(
    node_features,  # (batch, nodes, input_dim)
    coordinates,    # (nodes, 4) - [lat, lon, elev, time]
)  # Returns: (batch, nodes, hidden_dim)
```

### Training

```bash
# Configure environment
module load gcc-11.2.0-gcc-8.5.0 cuda-11.8.0-gcc-11.2.0
source activate earth4d

# Train model
bash train_gnn_weather.sh
```

Edit configuration in `train_gnn_weather.sh`:
- `SPATIAL_LEVELS`: Number of spatial resolution levels (default: 24)
- `TEMPORAL_LEVELS`: Number of temporal resolution levels (default: 24)
- `COORDINATE_SYSTEM`: "geographic" or "ecef"
- `ADAPTIVE_RANGE`: true/false for regional domain optimization

### Evaluation

```bash
# Evaluate trained model
bash evaluate_gnn_weather.sh <path_to_checkpoint>

# Example
bash evaluate_gnn_weather.sh saved_models/my_experiment/last.ckpt
```

## Architecture

```
Node Features (B, N, D_in) + Coordinates (N, 4)
    ↓
Earth4D Multi-Scale Hash Encoding → (B, N, 192)
    ↓
Concatenate with Node Features
    ↓
Projection MLP → (B, N, D_hidden)
    ↓
GNN Processor (message passing)
    ↓
Decoder → Predictions
```

**Earth4D Output Dimensions:**
- Spatial encoder (XYZ): 48 dimensions
- Spatiotemporal XYT: 48 dimensions
- Spatiotemporal YZT: 48 dimensions
- Spatiotemporal XZT: 48 dimensions
- **Total**: 192 dimensions

## Benchmarks

### DANRA Dataset (Danish Reanalysis, April 2022)

Comparison of baseline MLP vs multi-scale spatiotemporal nodes:

| Configuration | Parameters | Train Loss | Test Loss | Generalization |
|---------------|------------|------------|-----------|----------------|
| **Baseline MLP** | 211K | 0.678 | **10.84** | 16.0× gap |
| **Multi-Scale Nodes** | 724M | **0.346** | 14.35 | 41.5× gap |

**Key Finding:** With only 24 training timesteps, the multi-scale embedder overfits (32% worse test performance). Performance expected to improve significantly with larger datasets (1000+ timesteps).

**See** [`BENCHMARK_DANRA.md`](BENCHMARK_DANRA.md) for detailed analysis.

### Recommendations

**Small Datasets (< 100 timesteps):**
- Use baseline MLP or reduce levels to 6-12
- Strong regularization required
- Consider frozen weights from pretraining

**Large Datasets (> 1000 timesteps):**
- Full 24-level configuration recommended
- Multi-scale encoding captures hierarchical dynamics
- Expected to outperform standard encoders

## Configuration

### MultiScaleSpatioTemporalNode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | required | Input feature dimension per node |
| `hidden_dim` | int | required | Output dimension for GNN |
| `spatial_levels` | int | 24 | Spatial resolution levels |
| `temporal_levels` | int | 24 | Temporal resolution levels |
| `features_per_level` | int | 2 | Features per hash table level |
| `coordinate_system` | str | "geographic" | "geographic" or "ecef" |
| `use_adaptive_range` | bool | True | Fit range to training data |
| `resolution_mode` | str | "balanced" | Resolution distribution |
| `verbose` | bool | False | Print configuration |

### Coordinate Systems

**Geographic (Recommended):**
- x = latitude (-90° to +90°)
- y = longitude (-180° to +180°)
- z = elevation (meters)
- Preserves latitude relationships

**ECEF:**
- Earth-Centered Earth-Fixed Cartesian coordinates
- Uniform 3D Euclidean space
- May benefit global models

## Files

```
gnn/
├── multiscale_spatiotemporal_node.py   # Main embedder class
├── train_gnn_weather.sh                # Training script
├── evaluate_gnn_weather.sh             # Evaluation script
├── BENCHMARK_DANRA.md                  # Benchmark results
└── README.md                           # This file
```

## Integration Example

```python
import torch
import torch.nn as nn
from gnn.multiscale_spatiotemporal_node import MultiScaleSpatioTemporalNode

class WeatherGNN(nn.Module):
    def __init__(self, use_multiscale=True):
        super().__init__()

        # Node embedder
        if use_multiscale:
            self.node_embedder = MultiScaleSpatioTemporalNode(
                input_dim=13,
                hidden_dim=64,
                spatial_levels=24,
                temporal_levels=24,
            )
        else:
            self.node_embedder = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )

        # GNN processor
        self.processor = MessagePassingGNN(hidden_dim=64, num_layers=4)

        # Decoder
        self.decoder = nn.Linear(64, 13)

    def forward(self, features, coordinates=None):
        # Encode
        if isinstance(self.node_embedder, MultiScaleSpatioTemporalNode):
            embeddings = self.node_embedder(features, coordinates)
        else:
            embeddings = self.node_embedder(features)

        # Process
        processed = self.processor(embeddings)

        # Decode
        predictions = self.decoder(processed)

        return predictions
```

## Requirements

- Python 3.10+
- PyTorch 2.7+
- PyTorch Lightning 2.4+
- CUDA 11.8+
- Earth4D encoder (`encoders.xyzt.earth4d`)

## Performance

**Memory:**
- Earth4D parameters: ~724M (~2.8 GB)
- Training memory: ~11 GB (with gradients)
- Recommended: 16GB+ GPU VRAM

**Speed:**
- Hash encoding: 1-2ms per batch
- Overall overhead: ~10-15% vs baseline MLP

## Citation

If you use this work in your research, please cite:

```bibtex
@article{lam2023graphcast,
  title={GraphCast: Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023}
}
```

## Future Work

- **Energy4D**: Implemented but not yet trained or tested. Uses relative coordinates (Δx, Δy, Δz, Δt) instead of absolute, expected to improve generalization.
- **Hierarchical mixing**: Multi-scale synthesis with local receptive fields
- **Large dataset validation**: Test on MEPS, ERA5 (years of data)
- **Mixed precision**: FP16 training for reduced memory

## Implementation Notes

**Learned Probing:** Disabled by default (`enable_learned_probing=False`) due to PyTorch autograd in-place operation errors during gradient computation.

## Support

For questions or issues:
1. Review this README
2. Check [`BENCHMARK_DANRA.md`](BENCHMARK_DANRA.md)
3. Examine [`multiscale_spatiotemporal_node.py`](multiscale_spatiotemporal_node.py)
4. Open an issue on the repository

---

**Status:** Production-ready for research and experimentation
**Last Updated:** March 2026
