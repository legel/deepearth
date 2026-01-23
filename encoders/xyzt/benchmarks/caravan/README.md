# Earth4D Caravan Benchmark

Basin-aware streamflow prediction using Earth4D positional encoding.

## Overview

This benchmark predicts daily streamflow (discharge in mm/day) across 12,000+ global river basins using:

- **Earth4D positional encoding**: Multi-scale hash-based spatiotemporal features (192D output)
- **Basin embeddings**: Learned embeddings for 12,000+ basins (randomly initialized, no prior knowledge)
- **Caravan dataset**: Global hydrology benchmark (Kratzert et al., 2023)

## Quick Start

```bash
# Run training (fused Adam recommended for 10x speedup)
python -m benchmarks.caravan.train --epochs 500 --fused-adam --output-dir ./outputs
```

## Dataset

The benchmark uses the Caravan v1.6 global hydrology dataset:

- **Source**: Daily streamflow observations from 12,079 basins across 7 regions
- **Train samples**: ~108M observations (pre-2016)
- **Test samples**: ~27M observations (2016+)
- **Time range**: 1950-2022
- **Regions**: CAMELS (US), CAMELS-AUS, CAMELS-GB, CAMELS-BR, CAMELS-CL, LAMAH (Europe), HYSETS (Canada)

The full dataset (9.3 GB CSV, 135M observations) is extracted from the 29 GB Caravan archive using `data_extraction.py`.

### Data Extraction

To extract the full dataset:

```bash
# Option 1: Direct execution
python data_extraction.py

# Option 2: SLURM job (recommended for HPC)
sbatch data_extraction.slurm
```

This processes the 29 GB archive and creates `data/caravan_full.csv` with:
- 135,039,157 observations
- 12,079 basins (filtered to ≥10 years of data)
- Columns: gauge_id, date, streamflow_mm_per_day, lat, lon, elev

## Model Architecture

```
StreamflowModel
├── Earth4D (spatiotemporal encoding) → 192D
│   ├── XYZ encoder (pure spatial)
│   ├── XYT encoder (x-y-time)
│   ├── YZT encoder (y-z-time)
│   └── XZT encoder (x-z-time)
├── Basin Embeddings (N_basins x 256)
└── MLP Head (input_dim → 256 → 128 → 1)
```

## Results

**Preliminary results** (5-basin test, 50 epochs):

| Model | Test NSE | Test R² | Test MAE (mm/day) |
|-------|----------|---------|-------------------|
| **Earth4D (Preliminary)** | 0.578 | 0.578 | 1.122 |
| LSTM (Literature) | 0.56-0.82 | - | - |

*Full results pending 500-epoch training on complete dataset.*

### Training Performance (Preliminary)

| Metric | Training | Test |
|--------|----------|------|
| NSE | 0.714 | 0.578 |
| R² | 0.714 | 0.578 |
| MAE | 0.659 mm/day | 1.122 mm/day |

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 500 | Number of training epochs |
| `--batch-size` | 512 | Batch size |
| `--lr` | 0.0008 | Learning rate (matches LFMC) |
| `--weight-decay` | 0.001 | Weight decay |
| `--basin-dim` | 256 | Basin embedding dimension |
| `--seed` | 0 | Random seed |
| `--output-dir` | ./outputs | Output directory |
| `--fused-adam` | - | Enable fused Adam (10x speedup, recommended) |
| `--no-precomputed` | - | Disable index caching (memory efficiency) |

## Outputs

Training produces:

```
outputs/
├── final_model.pt              # Trained model checkpoint
├── metrics_*.csv               # Per-epoch metrics
└── ...                         # Additional outputs
```

## Metrics

Metrics are reported in mm/day (millimeters per day):

- **NSE**: Nash-Sutcliffe Efficiency (standard in hydrology)
- **R²**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## Python API

```python
from benchmarks.caravan import (
    CaravanDataset,
    StreamflowModel,
    get_temporal_splits,
    compute_streamflow_metrics
)

# Load dataset
dataset = CaravanDataset('data/caravan_full.csv', device='cuda')
splits = get_temporal_splits(dataset)

# Create model
model = StreamflowModel(
    n_basins=dataset.n_basins,
    basin_dim=256
).to('cuda')

# Get batch data (TrainableDataset protocol)
batch_data = dataset.get_batch_data(splits['train'][:1000])
# Returns: {'coords', 'targets', 'basin_idx', 'is_degenerate'}

# Forward pass (TrainableModel protocol)
predictions = model(batch_data)
```

## File Structure

```
benchmarks/caravan/
├── __init__.py         # Package exports
├── README.md           # This file
├── RESULTS.md          # Detailed results and baselines
├── constants.py        # Hydrology constants
├── data.py             # CaravanDataset class
├── model.py            # StreamflowModel
├── train.py            # Training script
├── utils.py            # Metrics, logging
├── data_extraction.py  # Extract full dataset from archive
└── data_extraction.slurm  # SLURM job for extraction
```

## Comparison to LFMC Benchmark

| Aspect | LFMC | Caravan |
|--------|------|---------|
| **Domain** | Ecology (vegetation moisture) | Hydrology (streamflow) |
| **Samples** | ~90K observations | ~135M observations |
| **Entities** | 180+ species | 12,079 basins |
| **Embedding** | Species (768D) | Basin (256D) |
| **Target** | LFMC percentage | Streamflow mm/day (log-transformed) |
| **Train/Test** | AI2 official split | Temporal (pre-2016 / 2016+) |
| **Primary Metric** | MAE (pp), R² | NSE, R² |

## Research Question

**Can Earth4D predict streamflow from (x,y,z,t) coordinates alone?**

This tests whether spatiotemporal patterns encode sufficient information for streamflow prediction without:
- Physical routing models
- Explicit basin attributes (area, soil type, etc.)
- Weather forcing data
- Satellite observations

The basin embeddings capture catchment-specific characteristics analogous to species embeddings in LFMC.

## References

- **DeepEarth Paper**: Legel et al. 2026, "Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding"
- **Caravan Dataset**: Kratzert et al. 2023, "Caravan: A global community dataset for large-sample hydrology." *Scientific Data*, https://doi.org/10.1038/s41597-023-01975-w
- **LFMC Benchmark**: Globe-LFMC 2.0, Yebra et al., 2024

## Baseline Comparisons

See `RESULTS.md` for detailed comparisons with:
- LSTM models (Kratzert et al., 2019)
- Ensemble LSTM (Kratzert et al., 2019)
- Other deep learning approaches

---

**Status**: Full training in progress
**Last Updated**: 2026-01-19
**Version**: 2.0
