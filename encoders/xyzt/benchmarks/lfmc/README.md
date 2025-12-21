# Earth4D LFMC Benchmark

Species-aware Live Fuel Moisture Content (LFMC) prediction using Earth4D positional encoding.

## Overview

This benchmark predicts LFMC values (vegetation moisture content as a percentage of dry mass) across the Continental United States (CONUS) using:

- **Earth4D positional encoding**: Multi-scale hash-based spatiotemporal features (192D output)
- **Species embeddings**: Learned embeddings for 180+ plant species (randomly initialized, no prior knowledge)
- **Globe-LFMC 2.0 dataset**: Global ecological forecasting benchmark (Yebra et al., 2024)

## Quick Start

```bash
# Run training with fused Adam optimizer (achieves SoTA in ~5 hours)
python -m benchmarks.lfmc.train --epochs 10000 --batch-size 256 --fused-adam --weight-decay 0.001 --lr 0.00025 --output-dir ./outputs
```

## Dataset

The benchmark uses Globe-LFMC 2.0 (Yebra et al., 2024) with Allen Institute for AI's official train/test split:

- **Source**: Field measurements across diverse plant species, geographic regions, and temporal periods
- **Train samples**: 76,467
- **Test samples**: 13,297
- **Time range**: 2017-2023

The dataset is automatically downloaded on first run from [allenai/lfmc](https://github.com/allenai/lfmc).

## Model Architecture

```
SpeciesAwareLFMCModel
├── Earth4D (spatiotemporal encoding) → 192D
│   ├── XYZ encoder (pure spatial)
│   ├── XYT encoder (x-y-time)
│   ├── YZT encoder (y-z-time)
│   └── XZT encoder (x-z-time)
├── Species Embeddings (N_species x 768)
└── MLP Head (input_dim → 256 → 128 → 1)
```

## Results

Earth4D achieves state-of-the-art performance on the Globe-LFMC 2.0 benchmark:

| Model | Data Inputs | MAE (pp) | RMSE (pp) | R² |
|-------|-------------|----------|-----------|-----|
| **Earth4D (Learned Hashing)** | (x,y,z,t) + Species Name | **12.1** | **19.9** | **0.755** |
| Galileo (Pre-Trained) | (x,y,z,t) + Species Type + Remote Sensing | 12.6 | 18.9 | 0.72 |

Earth4D surpasses the pre-trained Galileo foundation model (Johnson et al., 2025; Tseng et al., 2025) without satellite imagery, weather data, or topography.

### Training Performance

With the fused Adam optimizer and precomputed hash indices:

| Metric | Training | Test |
|--------|----------|------|
| RMSE | 6.1pp | 19.9pp |
| MAE | 1.8pp | 12.1pp |
| R² | 0.977 | 0.755 |

- **Training speed**: ~1.8s/epoch (9x faster than standard training)
- **Total training time**: ~5 hours for 10,000 epochs

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 2500 | Number of training epochs |
| `--batch-size` | 1024 | Batch size (256 recommended for fused Adam) |
| `--lr` | 0.001 | Learning rate (0.00025 recommended for fused Adam) |
| `--fused-adam` | False | Use sparse fused Adam optimizer (9x faster) |
| `--weight-decay` | 0.001 | AdamW weight decay |
| `--species-dim` | 768 | Species embedding dimension |
| `--seed` | 0 | Random seed |
| `--use-adaptive-range` | False | Fit range to data extent |
| `--no-precomputed` | False | Disable precomputed hash indices |
| `--output-dir` | ./outputs | Output directory |

## Outputs

Training produces:

```
outputs/
├── final_model.pt              # Trained model checkpoint
├── training_metrics_*.csv      # Per-epoch metrics
├── test_predictions.csv        # All test predictions
├── error_histogram_metrics.json
├── error_distribution_histogram.png
├── geospatial_error_map_epoch_*.png
├── temporal_predictions_epoch_*.png
└── combined_scientific_figure_epoch_*.png
```

## Metrics

Metrics are reported in LFMC percentage points (pp):

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## Python API

```python
from benchmarks.lfmc import (
    FullyGPUDataset,
    SpeciesAwareLFMCModel,
    get_ai2_splits,
    compute_lfmc_metrics
)

# Load dataset
dataset = FullyGPUDataset(device='cuda')
splits = get_ai2_splits(dataset)

# Create model
model = SpeciesAwareLFMCModel(
    n_species=dataset.n_species,
    species_dim=768
).to('cuda')

# Get batch data (TrainableDataset protocol)
batch_data = dataset.get_batch_data(splits['train'][:1000])
# Returns: {'coords', 'targets', 'species_idx', 'is_degenerate'}

# Forward pass (TrainableModel protocol)
predictions = model(batch_data)
```

## File Structure

```
benchmarks/lfmc/
├── __init__.py         # Package exports
├── README.md           # This file
├── constants.py        # LFMC constants (MAX_LFMC_VALUE=302)
├── data.py             # FullyGPUDataset, Globe-LFMC splits
├── model.py            # SpeciesAwareLFMCModel
├── train.py            # Training script
├── utils.py            # EMA tracking, CSV export
└── visualization.py    # Geospatial/temporal plots
```

## References

- Yebra, M., et al. (2024). "Globe-LFMC 2.0, an enhanced and updated dataset for live fuel moisture content research." *Scientific Data*, 11(1):332.
- Johnson, P.A., et al. (2025). "High-resolution live fuel moisture content (LFMC) maps for wildfire risk from multimodal earth observation data." arXiv:2506.20132.
- Tseng, G., et al. (2025). "Galileo: Learning global & local features of many remote sensing modalities." *ICML 2025*.
- Allen Institute for AI: [allenai/lfmc](https://github.com/allenai/lfmc)
