# LFMC Modular Dataset - Complete Usage Guide

**Version**: 1.0
**Date**: 2025-10-28
**Status**: Production Ready ✅

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Training Integration](#training-integration)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

---

## Quick Start

### 5-Minute Tutorial

```python
from lfmc_dataset import LFMCDataset
from torch.utils.data import DataLoader

# 1. Load dataset with all features
dataset = LFMCDataset(use_aef=True, use_daymet=True, use_cache=True)
print(dataset)  # LFMCDataset(samples=89,504, features=85, modules=[AEF+Daymet])

# 2. Get statistics
stats = dataset.get_statistics()
print(f"LFMC range: {stats['lfmc_min']:.1f}% - {stats['lfmc_max']:.1f}%")
print(f"Species: {stats['n_species']}")

# 3. Create train/test splits
splits = dataset.create_splits()
print(f"Training samples: {len(splits['train']):,}")

# 4. Convert to PyTorch and train
train_data = dataset.to_torch_dataset(indices=splits['train'])
loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 5. Start training!
for coords, features, species_idx, lfmc_target in loader:
    # Your model here
    pass
```

---

## Installation

### Requirements

```bash
# Core requirements (always needed)
pip install pandas numpy pyarrow

# For .pt caching and training (highly recommended)
pip install torch

# For Google Cloud downloads (if using auto-download)
pip install google-cloud-storage
```

### Verify Installation

```bash
cd /path/to/LFMC_v3_20251027
python test_lfmc_dataset.py
```

Expected output: `Results: 8/8 tests passed (100%)`

---

## Basic Usage

### 1. Loading Datasets

#### Load Base LFMC Only
```python
from lfmc_dataset import LFMCDataset

dataset = LFMCDataset()
# Loads: 89,961 samples with 7 columns
# Columns: sample_id, lat, lon, elevation_m, date, lfmc_percent, species
```

#### Load Base + AlphaEarth Features
```python
dataset = LFMCDataset(use_aef=True)
# Loads: 89,504 samples with 71 columns
# Adds: 64-dimensional AlphaEarth embeddings (aef_00 to aef_63)
```

#### Load Base + Daymet Weather
```python
dataset = LFMCDataset(use_daymet=True)
# Loads: 89,961 samples with 28 columns
# Adds: 21 weather features (precipitation, temperature, radiation, etc.)
```

#### Load All Features
```python
dataset = LFMCDataset(use_aef=True, use_daymet=True)
# Loads: 89,504 samples with 92 columns
# Total features: 85 (64 AEF + 21 Daymet)
```

### 2. Accessing Data

#### As Pandas DataFrame
```python
dataset = LFMCDataset(use_aef=True)

# Access the full DataFrame
df = dataset.data
print(df.head())

# Access specific columns
latitudes = df['lat'].values
lfmc_values = df['lfmc_percent'].values
aef_embeddings = df[[f'aef_{i:02d}' for i in range(64)]].values
```

#### Get Feature Columns
```python
# Get list of all feature column names (excludes metadata)
features = dataset.feature_columns
print(f"Total features: {len(features)}")
# Output: Total features: 85

# Separate by type
aef_cols = [c for c in features if c.startswith('aef_')]
daymet_cols = [c for c in features if any(x in c for x in ['prcp', 'tmin', 'tmax'])]
```

### 3. Statistics and Metadata

```python
dataset = LFMCDataset(use_aef=True, use_daymet=True)

stats = dataset.get_statistics()

print(f"Samples: {stats['n_samples']:,}")
print(f"Features: {stats['n_features']}")
print(f"Species: {stats['n_species']}")
print(f"LFMC range: {stats['lfmc_min']:.1f}% - {stats['lfmc_max']:.1f}%")
print(f"LFMC mean: {stats['lfmc_mean']:.1f}% ± {stats['lfmc_std']:.1f}%")
print(f"Date range: {stats['date_min']} to {stats['date_max']}")
print(f"Lat range: {stats['lat_range']}")
print(f"Lon range: {stats['lon_range']}")
```

---

## Advanced Features

### 1. Caching for Speed

#### Enable Caching
```python
import time

# First load: reads Parquet, creates cache (~0.12s)
t0 = time.time()
dataset1 = LFMCDataset(use_aef=True, use_cache=True)
print(f"First load: {time.time()-t0:.2f}s")

# Second load: uses cache (~0.08s, 1.5x faster!)
t0 = time.time()
dataset2 = LFMCDataset(use_aef=True, use_cache=True)
print(f"Cached load: {time.time()-t0:.2f}s")
```

#### Cache Strategies
```python
# Auto mode: use cache if newer than Parquet (default)
dataset = LFMCDataset(use_aef=True, use_cache='auto')

# Always use cache
dataset = LFMCDataset(use_aef=True, use_cache=True)

# Never use cache (always read Parquet)
dataset = LFMCDataset(use_aef=True, use_cache=False)
```

**Cache Location**: `./data/cache/lfmc_*.pt`

**When to clear cache**:
- After updating Parquet files
- After changing filtering settings
- If cache is corrupted

```bash
# Clear cache
rm -rf data/cache/*.pt
```

### 2. Train/Test Splitting

#### Default Splits (5% each for holdouts)
```python
dataset = LFMCDataset(use_aef=True, use_daymet=True)

splits = dataset.create_splits()
# Returns dict with keys: 'train', 'temporal', 'spatial', 'random'

print(f"Train: {len(splits['train']):,} samples")
print(f"Temporal holdout: {len(splits['temporal']):,} samples")
print(f"Spatial holdout: {len(splits['spatial']):,} samples")
print(f"Random holdout: {len(splits['random']):,} samples")
```

#### Custom Split Fractions
```python
splits = dataset.create_splits(
    temporal_frac=0.10,  # 10% temporal holdout
    spatial_frac=0.10,   # 10% spatial holdout
    random_frac=0.10,    # 10% random holdout
    random_seed=42       # For reproducibility
)
# Result: 70% training, 30% holdouts
```

#### Understanding Split Types

**Temporal Holdout**: Latest samples by date (tests future prediction)
- Use case: "How well does the model generalize to future dates?"

**Spatial Holdout**: Random spatial clusters (tests geographic generalization)
- Use case: "How well does the model generalize to new locations?"

**Random Holdout**: Random samples (tests overall performance)
- Use case: "Standard validation set"

### 3. PyTorch Integration

#### Basic PyTorch Dataset
```python
import torch
from torch.utils.data import DataLoader

dataset = LFMCDataset(use_aef=True, use_daymet=True)
splits = dataset.create_splits()

# Convert to PyTorch Dataset
train_dataset = dataset.to_torch_dataset(indices=splits['train'])
val_dataset = dataset.to_torch_dataset(indices=splits['random'])

print(f"Train: {len(train_dataset):,} samples")
print(f"Val: {len(val_dataset):,} samples")
```

#### DataLoader Configuration
```python
# Training loader (shuffled, with workers)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Validation loader (not shuffled, larger batches)
val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
```

#### Understanding Data Format
```python
# Each sample returns 4 items:
coords, features, species_idx, target = train_dataset[0]

print(f"Coords shape: {coords.shape}")        # (4,) = [lat, lon, elev, time]
print(f"Features shape: {features.shape}")    # (85,) = 64 AEF + 21 Daymet
print(f"Species idx: {species_idx}")          # int (0 to n_species-1)
print(f"Target (LFMC): {target}")             # float (% moisture)
```

#### Batch Format
```python
for batch in train_loader:
    coords, features, species_idx, targets = batch

    print(f"Batch coords: {coords.shape}")      # (batch_size, 4)
    print(f"Batch features: {features.shape}")  # (batch_size, 85)
    print(f"Batch species: {species_idx.shape}")# (batch_size,)
    print(f"Batch targets: {targets.shape}")    # (batch_size,)
    break
```

### 4. Auto-Download from URLs

```python
# If files are missing and URLs are available
dataset = LFMCDataset(
    use_aef=True,
    use_daymet=True,
    auto_download=True,  # Download missing files
    urls_file='./data/dataset_urls.json'
)
```

**Requirements**:
- `dataset_urls.json` must exist with download URLs
- Internet connection
- `google-cloud-storage` package (if using GCS)

---

## Training Integration

### Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lfmc_dataset import LFMCDataset

# 1. Load dataset
dataset = LFMCDataset(use_aef=True, use_daymet=True, use_cache=True)
splits = dataset.create_splits(random_seed=42)

# 2. Create PyTorch datasets
train_data = dataset.to_torch_dataset(indices=splits['train'])
val_data = dataset.to_torch_dataset(indices=splits['random'])

# 3. Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)

# 4. Define model (example)
class LFMCModel(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.coord_encoder = nn.Linear(4, 32)  # lat, lon, elev, time
        self.feature_encoder = nn.Linear(n_features, 128)
        self.species_embed = nn.Embedding(n_species, 32)
        self.predictor = nn.Sequential(
            nn.Linear(32 + 128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, coords, features, species_idx):
        coord_enc = self.coord_encoder(coords)
        feat_enc = self.feature_encoder(features)
        species_enc = self.species_embed(species_idx)
        combined = torch.cat([coord_enc, feat_enc, species_enc], dim=1)
        return self.predictor(combined).squeeze()

# 5. Initialize model
n_features = len(dataset.feature_columns)
n_species = train_data.n_species
model = LFMCModel(n_features, n_species).cuda()

# 6. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    model.train()
    train_loss = 0.0

    for coords, features, species_idx, targets in train_loader:
        coords = coords.cuda()
        features = features.cuda()
        species_idx = species_idx.cuda()
        targets = targets.cuda()

        # Forward pass
        predictions = model(coords, features, species_idx)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for coords, features, species_idx, targets in val_loader:
            coords = coords.cuda()
            features = features.cuda()
            species_idx = species_idx.cuda()
            targets = targets.cuda()

            predictions = model(coords, features, species_idx)
            loss = criterion(predictions, targets)
            val_loss += loss.item()

    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
          f"Val Loss={val_loss/len(val_loader):.4f}")
```

### Integration with Earth4D Encoder

```python
from lfmc_dataset import LFMCDataset
from earth4d import Earth4D  # Your existing encoder

# Load dataset
dataset = LFMCDataset(use_aef=True, use_daymet=True)
splits = dataset.create_splits()

# Initialize Earth4D encoder
earth4d = Earth4D(
    spatial_levels=8,
    temporal_levels=4,
    features_per_level=2
).cuda()

# Training with Earth4D
train_data = dataset.to_torch_dataset(indices=splits['train'])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

for coords, features, species_idx, targets in train_loader:
    coords = coords.cuda()
    features = features.cuda()

    # Earth4D encoding
    spatiotemporal_encoding = earth4d(coords)

    # Combine with other features
    combined = torch.cat([spatiotemporal_encoding, features], dim=1)

    # Your model prediction...
```

---

## Performance Optimization

### 1. Memory Optimization

#### Load Only What You Need
```python
# If you only need AEF features
dataset = LFMCDataset(use_aef=True, use_daymet=False)  # Saves ~20% memory

# If you only need base data
dataset = LFMCDataset()  # Minimal memory footprint
```

#### Use Caching Wisely
```python
# Cache is useful for repeated experiments
dataset = LFMCDataset(use_aef=True, use_daymet=True, use_cache=True)

# But uses disk space (~100-200 MB for cache files)
# Clear cache if disk space is limited
```

### 2. Loading Speed Optimization

#### Benchmark: Parquet vs Cache
```python
import time

# Without cache (slower, reads from Parquet)
t0 = time.time()
dataset1 = LFMCDataset(use_aef=True, use_daymet=True, use_cache=False)
parquet_time = time.time() - t0
print(f"Parquet load: {parquet_time:.2f}s")

# With cache (faster, reads from .pt)
t0 = time.time()
dataset2 = LFMCDataset(use_aef=True, use_daymet=True, use_cache=True)
cache_time = time.time() - t0
print(f"Cache load: {cache_time:.2f}s")
print(f"Speedup: {parquet_time/cache_time:.1f}x")
```

**Expected results**:
- Parquet load: ~0.4-0.5s
- Cached load: ~0.08-0.12s
- Speedup: 3-4x

### 3. Training Speed Optimization

#### DataLoader Settings
```python
# Optimal settings for training
train_loader = DataLoader(
    train_dataset,
    batch_size=64,           # Adjust based on GPU memory
    shuffle=True,            # Important for training
    num_workers=4,           # Parallel data loading (adjust based on CPU cores)
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True  # Keep workers alive between epochs
)
```

#### Batch Size Selection
```python
# Small GPU (8 GB): batch_size=32-64
# Medium GPU (16 GB): batch_size=128-256
# Large GPU (40 GB): batch_size=512-1024

# Test maximum batch size
import torch
batch_sizes = [32, 64, 128, 256]
for bs in batch_sizes:
    try:
        loader = DataLoader(train_dataset, batch_size=bs)
        batch = next(iter(loader))
        coords, features, species_idx, targets = batch
        # Try forward pass with your model
        print(f"Batch size {bs}: OK")
    except RuntimeError as e:
        print(f"Batch size {bs}: Too large (OOM)")
        break
```

---

## Troubleshooting

### Common Issues

#### 1. FileNotFoundError: Missing Parquet files

**Problem**:
```
FileNotFoundError: data/lfmc_base.parquet not found
```

**Solution**:
```bash
# Run standardization first
python prepare_lfmc_datasets.py --no-cache

# Or enable auto-download
python -c "from lfmc_dataset import LFMCDataset; LFMCDataset(auto_download=True)"
```

#### 2. PyTorch Not Available

**Problem**:
```
RuntimeError: PyTorch not available, cannot load .pt cache files
```

**Solution**:
```bash
# Install PyTorch
pip install torch

# Or disable caching
dataset = LFMCDataset(use_cache=False)
```

#### 3. Cache Files Corrupted

**Problem**:
```
Error loading cache: unexpected data format
```

**Solution**:
```bash
# Delete cache and regenerate
rm -rf data/cache/*.pt
python -c "from lfmc_dataset import LFMCDataset; LFMCDataset(use_cache=True)"
```

#### 4. Out of Memory (OOM)

**Problem**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```python
# Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=32)  # Instead of 64

# Or use fewer workers
train_loader = DataLoader(train_dataset, num_workers=2)  # Instead of 4

# Or use CPU
model = model.cpu()  # Train on CPU instead
```

#### 5. Slow Data Loading

**Problem**: Training is slow, GPU utilization is low

**Solution**:
```python
# Increase num_workers
train_loader = DataLoader(train_dataset, num_workers=8)  # More parallel workers

# Enable caching
dataset = LFMCDataset(use_aef=True, use_cache=True)

# Use pin_memory
train_loader = DataLoader(train_dataset, pin_memory=True)
```

---

## API Reference

### LFMCDataset Class

#### Constructor
```python
LFMCDataset(
    data_dir='./data',           # Directory with Parquet files
    use_aef=False,               # Include AlphaEarth features
    use_daymet=False,            # Include Daymet weather
    use_solus=False,             # Include SOLUS soil (future)
    auto_download=False,         # Auto-download missing files
    use_cache='auto',            # Cache strategy: 'auto', True, False
    cache_dir='./data/cache',    # Cache directory
    urls_file='./data/dataset_urls.json',  # URL mapping file
    verbose=True                 # Print progress messages
)
```

#### Attributes
```python
dataset.data              # pd.DataFrame: Full merged dataset
dataset.base_df           # pd.DataFrame: Base LFMC data
dataset.aef_df            # pd.DataFrame: AEF features (if loaded)
dataset.daymet_df         # pd.DataFrame: Daymet weather (if loaded)
dataset.n_samples         # int: Number of samples
dataset.feature_columns   # List[str]: Feature column names
```

#### Methods

**get_statistics()**
```python
stats = dataset.get_statistics()
# Returns dict with: n_samples, n_features, n_species, lfmc_min, lfmc_max,
# lfmc_mean, lfmc_std, date_min, date_max, lat_range, lon_range, etc.
```

**create_splits()**
```python
splits = dataset.create_splits(
    temporal_frac=0.05,    # Fraction for temporal holdout
    spatial_frac=0.05,     # Fraction for spatial holdout
    random_frac=0.05,      # Fraction for random holdout
    random_seed=42         # Random seed for reproducibility
)
# Returns dict with keys: 'train', 'temporal', 'spatial', 'random'
# Values are numpy arrays of indices
```

**to_torch_dataset()**
```python
torch_dataset = dataset.to_torch_dataset(
    indices=None,          # Optional subset of indices
    return_species=True,   # Include species information
    normalize=True         # Normalize features (recommended)
)
# Returns LFMCTorchDataset instance
```

**__len__()**
```python
n = len(dataset)  # Returns number of samples
```

**__repr__()**
```python
print(dataset)  # Human-readable representation
# Output: LFMCDataset(samples=89,504, features=85, modules=[AEF+Daymet])
```

---

### LFMCTorchDataset Class

#### Attributes
```python
torch_dataset.data          # pd.DataFrame: Subset of data
torch_dataset.coords        # torch.Tensor: (N, 4) lat/lon/elev/time
torch_dataset.features      # torch.Tensor: (N, n_features) normalized features
torch_dataset.targets       # torch.Tensor: (N,) LFMC values
torch_dataset.species_indices  # torch.Tensor: (N,) species indices
torch_dataset.n_species     # int: Number of unique species
```

#### Methods

**__len__()**
```python
n = len(torch_dataset)  # Returns number of samples
```

**__getitem__(idx)**
```python
coords, features, species_idx, target = torch_dataset[0]
# Returns tuple of tensors for one sample
```

---

## Best Practices

### 1. Always Use Caching for Experiments
```python
# Good: Fast repeated loading
dataset = LFMCDataset(use_aef=True, use_daymet=True, use_cache=True)

# Less optimal: Slow repeated loading
dataset = LFMCDataset(use_aef=True, use_daymet=True, use_cache=False)
```

### 2. Use Consistent Random Seeds
```python
# For reproducibility
splits = dataset.create_splits(random_seed=42)

# Document your seed
CONFIG = {'random_seed': 42, 'batch_size': 64, ...}
```

### 3. Monitor Memory Usage
```python
import psutil
import os

# Before loading
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024**2

dataset = LFMCDataset(use_aef=True, use_daymet=True)

# After loading
mem_after = process.memory_info().rss / 1024**2
print(f"Memory used: {mem_after - mem_before:.1f} MB")
```

### 4. Validate Data Before Training
```python
dataset = LFMCDataset(use_aef=True, use_daymet=True)

# Check for NaN values
assert not dataset.data[dataset.feature_columns].isna().any().any(), "Found NaN values!"

# Check LFMC range
assert dataset.data['lfmc_percent'].min() >= 0, "Negative LFMC values!"
assert dataset.data['lfmc_percent'].max() <= 600, "Suspicious LFMC values!"

print("✓ Data validation passed")
```

### 5. Log Your Configuration
```python
import json
from datetime import datetime

# Save configuration
config = {
    'date': datetime.now().isoformat(),
    'dataset': {
        'use_aef': True,
        'use_daymet': True,
        'n_samples': len(dataset),
        'n_features': len(dataset.feature_columns)
    },
    'splits': {
        'temporal_frac': 0.05,
        'spatial_frac': 0.05,
        'random_frac': 0.05,
        'random_seed': 42
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 100
    }
}

with open('experiment_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

---

## Support and Contribution

### Getting Help

1. **Check test results**: `python test_lfmc_dataset.py`
2. **Read error messages**: Most errors are self-explanatory
3. **Check documentation**: This guide covers most use cases
4. **Review examples**: See training example above

### Reporting Issues

When reporting issues, include:
1. Python version: `python --version`
2. PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Test results: `python test_lfmc_dataset.py`
4. Minimal reproducible example
5. Full error traceback

---

## Appendix

### File Structure
```
LFMC_v3_20251027/
├── data/
│   ├── lfmc_base.parquet           # Base LFMC data (1.17 MB)
│   ├── lfmc_aef.parquet            # AlphaEarth features (1.67 MB)
│   ├── lfmc_daymet.parquet         # Daymet weather (4.45 MB)
│   ├── dataset_urls.json           # Download URLs (future)
│   └── cache/                      # .pt cache files
│       ├── lfmc_base.pt
│       ├── lfmc_aef.pt
│       └── lfmc_daymet.pt
├── prepare_lfmc_datasets.py        # Data standardization script
├── lfmc_dataset.py                 # Modular dataset loader
├── test_lfmc_dataset.py            # Test suite
├── LFMC_DATASET_PLAN.md            # Architecture documentation
└── USAGE_GUIDE.md                  # This file
```

### Column Reference

**Base LFMC Columns**:
- `sample_id`: Unique identifier (int64)
- `lat`: Latitude in degrees (float32)
- `lon`: Longitude in degrees (float32)
- `elevation_m`: Elevation in meters (float32)
- `date`: Sampling date (datetime64)
- `lfmc_percent`: LFMC value in % (float32)
- `species`: Species name (string)

**AlphaEarth Feature Columns**:
- `aef_00` to `aef_63`: 64-dimensional embeddings (float32)

**Daymet Weather Columns**:
- `prcp_d_minus2`, `prcp_d_minus1`, `prcp_d0`: Precipitation (mm)
- `tmin_d_minus2`, `tmin_d_minus1`, `tmin_d0`: Min temperature (°C)
- `tmax_d_minus2`, `tmax_d_minus1`, `tmax_d0`: Max temperature (°C)
- `srad_d_minus2`, `srad_d_minus1`, `srad_d0`: Solar radiation (W/m²)
- `vp_d_minus2`, `vp_d_minus1`, `vp_d0`: Vapor pressure (Pa)
- `dayl_d_minus2`, `dayl_d_minus1`, `dayl_d0`: Day length (seconds)
- `swe_d_minus2`, `swe_d_minus1`, `swe_d0`: Snow water equivalent (kg/m²)

---

**Last Updated**: 2025-10-28
**Version**: 1.0
**Status**: ✅ Production Ready
