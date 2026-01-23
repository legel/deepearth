# Scripts Reference

Quick reference guide for all scripts in the Caravan benchmark.

---

## 📦 Core Components

### Data Loaders

**`data.py`** - Base data loader
- Loads coordinates (x,y,z,t) + streamflow
- Handles ECEF/lat-lon coordinate conversion
- Used by: baseline and Alzhanov experiments

**`data_multitask.py`** - Multi-task data loader
- Loads coordinates + streamflow + precipitation + temperature
- Returns 3 targets: Q, P, T
- Used by: multi-task learning experiments

**`data_inputs.py`** ⭐ NEW - Input features data loader
- Loads coordinates + P + T + Snow (as inputs) + streamflow (as target)
- Normalizes each feature appropriately
- Used by: input features experiments

---

### Models

**`model.py`** - Baseline model
- Architecture: `(x,y,z,t) → Earth4D + Basin Embedding → streamflow`
- Single output: streamflow
- Parameters: ~807M
- Used by: baseline and Alzhanov experiments

**`model_multitask.py`** - Multi-task model
- Architecture: `(x,y,z,t) → Earth4D + Basin Embedding → Shared Trunk → 3 heads`
- Three outputs: streamflow, precipitation, temperature
- Shared representation learning
- Used by: multi-task experiments

**`model_inputs.py`** ⭐ NEW - Multi-modal fusion model
- Architecture:
  ```
  (x,y,z,t) → Earth4D → 192D
  P → MLP1 → 32D
  T → MLP2 → 32D
  Snow → MLP3 → 32D
  Basin → Embedding → 256D
  Concat all (544D) → MLP → streamflow
  ```
- Separate MLPs for each input feature (avoids mixing raw + deep embeddings)
- Used by: input features experiments

---

### Training Scripts

**`train.py`** - Baseline training
- Trains: coordinates → streamflow
- Loss: MSE on streamflow
- Command:
  ```bash
  python -m benchmarks.caravan.train --epochs 50 --coordinate-system ecef
  ```

**`train_multitask.py`** - Multi-task training
- Trains: coordinates → (Q, P, T)
- Loss: Weighted sum of 3 MSE losses
- Supports loss balancing via weights
- Command:
  ```bash
  python -m benchmarks.caravan.train_multitask \
      --epochs 50 \
      --weight-streamflow 1.5 \
      --weight-precipitation 0.6 \
      --weight-temperature 0.11
  ```

**`train_inputs.py`** ⭐ NEW - Input features training
- Trains: (coordinates + P + T + Snow) → streamflow
- Loss: MSE on streamflow only
- Multi-modal fusion architecture
- Command:
  ```bash
  python -m benchmarks.caravan.train_inputs \
      --data-path benchmarks/caravan/data/caravan_alzhanov_147basins_inputs.csv \
      --epochs 50 \
      --batch-size 4096 \
      --feature-dim 32
  ```

---

## 🔧 Data Preparation Scripts

**`prepare_alzhanov_data.py`** - Create Alzhanov 147 basin dataset
- Extracts 147 CAMELS basins from Caravan
- Adds Uba River data from NetCDF
- Creates train/test splits
- Output: `caravan_alzhanov_147basins_with_uba.csv` (83MB)
- Run once to create base dataset

**`prep_data_multitask.py`** - Prepare multi-task dataset
- Adds precipitation and temperature from Caravan CSVs
- Input: base Alzhanov dataset
- Output: `caravan_alzhanov_147basins_multitask.csv` (94MB)
- For: multi-task experiments

**`prep_data_inputs.py`** ⭐ NEW - Prepare input features dataset
- Adds precipitation, temperature, snow from Caravan CSVs
- Input: base Alzhanov dataset
- Output: `caravan_alzhanov_147basins_inputs.csv` (99MB)
- For: input features experiments

---

## 🛠️ Utilities

**`constants.py`** - Constants and configuration
- `MAX_LOG_STREAMFLOW` - For log-normalization
- Default paths
- Basin lists

**`utils.py`** - Utility functions
- `MetricsEMA` - Exponential moving average for metrics
- `print_sample_predictions` - Display sample predictions
- Helper functions

---

## 📝 Usage Workflows

### Workflow 1: Baseline (Coordinates Only)

```bash
# 1. Prepare data (run once)
python prepare_alzhanov_data.py

# 2. Train baseline model
python -m benchmarks.caravan.train \
    --epochs 50 \
    --batch-size 4096 \
    --coordinate-system ecef \
    --output-dir outputs/baseline_ecef
```

### Workflow 2: Multi-task Learning

```bash
# 1. Prepare data (run once)
python prepare_alzhanov_data.py
python prep_data_multitask.py

# 2. Train multi-task model with balanced weights
python -m benchmarks.caravan.train_multitask \
    --data-path benchmarks/caravan/data/caravan_alzhanov_147basins_multitask.csv \
    --epochs 50 \
    --weight-streamflow 1.5 \
    --weight-precipitation 0.6 \
    --weight-temperature 0.11 \
    --output-dir outputs/multitask_balanced
```

### Workflow 3: Input Features (Multi-modal Fusion) ⭐

```bash
# 1. Prepare data (run once)
python prepare_alzhanov_data.py
python prep_data_inputs.py

# 2. Train input features model
python -m benchmarks.caravan.train_inputs \
    --data-path benchmarks/caravan/data/caravan_alzhanov_147basins_inputs.csv \
    --epochs 50 \
    --batch-size 4096 \
    --basin-dim 256 \
    --feature-dim 32 \
    --coordinate-system ecef \
    --output-dir outputs/inputs_ecef
```

---

## 📂 File Organization

### Files to Push to Git
- All `.py` files listed above
- `EXPERIMENTS_AND_RESULTS.md` - Comprehensive experiments summary
- `SCRIPTS_REFERENCE.md` - This file
- `README.md` - Main overview

### Files NOT in Git (Too Large)
- `data/*.csv` (80-99 MB each)
- `outputs/` (model checkpoints, multi-GB)
- `*.log`, `*.slurm` (HPC-specific)

### Data Files (Recreate Locally)
Users must download Caravan dataset and run preparation scripts:
1. `prepare_alzhanov_data.py` → creates base dataset
2. `prep_data_inputs.py` or `prep_data_multitask.py` → adds features

---

## 🔑 Key Points

**For Baseline:**
- Use: `data.py`, `model.py`, `train.py`
- Input: coordinates only
- Best for: comparing against coordinate-based approaches

**For Multi-task:**
- Use: `data_multitask.py`, `model_multitask.py`, `train_multitask.py`
- Input: coordinates only
- Output: predict Q, P, T simultaneously
- Best for: learning shared representations

**For Input Features:** ⭐ Recommended
- Use: `data_inputs.py`, `model_inputs.py`, `train_inputs.py`
- Input: coordinates + meteorology (P, T, Snow)
- Output: streamflow only
- Best for: leveraging meteorological inputs like traditional hydrology models

---

## 🚀 Quick Start

**I just cloned the repo, what do I do?**

1. Download prerequisites:
   - Caravan dataset: https://github.com/kratzert/Caravan
   - Place in: `benchmarks/caravan/data/Caravan-csv/`

2. Prepare data:
   ```bash
   cd benchmarks/caravan
   python prepare_alzhanov_data.py
   python prep_data_inputs.py  # For input features approach
   ```

3. Train a model:
   ```bash
   # Option A: Baseline
   python -m benchmarks.caravan.train --epochs 50

   # Option B: Input features (recommended)
   python -m benchmarks.caravan.train_inputs --epochs 50
   ```

4. Check results:
   - Models saved in: `outputs/`
   - Metrics CSV: `outputs/*/metrics_*.csv`

---

## 📊 See Also

- **`EXPERIMENTS_AND_RESULTS.md`** - Detailed results and comparisons
- **`README.md`** - Project overview
