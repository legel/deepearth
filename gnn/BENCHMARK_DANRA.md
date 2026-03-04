# Benchmark: Earth4D vs Baseline on DANRA Dataset

**Date**: March 3, 2026
**Dataset**: DANRA (Danish Reanalysis, April 2022)
**Test Period**: April 7-10, 2022 (3 days)
**Hardware**: NVIDIA A100-SXM4-80GB
**Framework**: PyTorch 2.7 + PyTorch Lightning

---

## Executive Summary

Comprehensive comparison of baseline MLP encoder vs Earth4D multi-scale spatiotemporal encoder on a limited-area weather forecasting task using the DANRA dataset.

**Key Finding**: On small datasets (24 training timesteps), the baseline MLP significantly outperforms Earth4D due to severe overfitting from the parameter imbalance. Earth4D requires substantially larger datasets to realize its potential.

---

## Dataset Configuration

### DANRA (Cropped)
- **Source**: Danish Reanalysis
- **Domain**: Denmark (54.6°N - 56.7°N, 10.4°E - 14.7°E)
- **Resolution**: ~2.5 km grid spacing
- **Grid Points**: 7,680 (64 × 120 spatial grid)
- **Temporal Resolution**: 3-hour intervals
- **Variables**: 13 features (u100m, v100m, r2m, t2m, swavr0m, lsm, orography, etc.)

### Data Splits
- **Train**: April 1-4, 2022 (24 timesteps)
- **Validation**: April 4-7, 2022 (24 timesteps)
- **Test**: April 7-10, 2022 (24 timesteps)

---

## Model Configurations

### Baseline (Standard MLP Encoder)
- **Total Parameters**: 211,081 (211K)
- **Grid Embedder**: 2-layer MLP
- **Embedder Parameters**: 5,200
- **Input Features**: 13 weather variables
- **Architecture**: MLP(13 → 64 → 64)

### Earth4D (Multi-Scale Hash Encoder)
- **Total Parameters**: 723,942,193 (724M)
- **Grid Embedder**: Earth4D hash-based encoder
- **Embedder Parameters**: 723,736,000 (99.97% of total)
- **Input Features**: 13 weather variables + (lat, lon, elev, time)
- **Spatial Levels**: 24 (multi-scale hash encoding)
- **Temporal Levels**: 24
- **Output Dimension**: 192 (48 spatial + 144 spatiotemporal)
- **Learned Probing**: Disabled for gradient stability

### Shared Configuration
- **GNN Architecture**: GraphCast-style encode-process-decode
- **Graph**: Multiscale mesh (8,409 nodes: 7,680 grid + 729 mesh)
- **Hidden Dimension**: 64
- **Processor Layers**: 4
- **Epochs**: 50
- **Batch Size**: 4
- **Learning Rate**: 1e-3
- **Seed**: 42 (for reproducibility)

---

## Test Set Performance

| Metric | Baseline (211K) | Earth4D (724M) | Difference |
|--------|----------------|----------------|------------|
| **Mean Test Loss** | **10.84** | **14.35** | **+32.4% worse** |
| Test Loss (1-step) | 3.20 | 4.23 | +32.2% worse |
| Test Loss (2-step) | 7.49 | 9.91 | +32.3% worse |
| Test Loss (3-step) | 10.38 | 13.65 | +31.5% worse |
| Test Loss (5-step) | 14.74 | 18.10 | +22.8% worse |
| Test Loss (10-step) | 8.91 | 12.50 | +40.3% worse |

**Result**: Baseline MLP outperforms Earth4D by 32.4% on average across all autoregressive rollout steps.

---

## Training vs Test Performance

| Model | Final Train Loss | Test Loss | Train/Test Gap | Interpretation |
|-------|------------------|-----------|----------------|----------------|
| **Baseline** | 0.678 | 10.84 | 16.0× | Reasonable generalization |
| **Earth4D** | 0.346 | 14.35 | **41.5×** | **Severe overfitting** |

- Earth4D achieved **49% lower training loss** but **32% higher test loss**
- The 41.5× train/test gap indicates memorization rather than learning

---

## Analysis

### Why Earth4D Underperformed

**1. Parameter-to-Data Ratio Mismatch**
- Earth4D: 724M parameters
- Training samples: 24 timesteps
- **Ratio**: 30.2 million parameters per training sample
- Extreme case of "more parameters than data" problem

**2. Overfitting Evidence**
- Training loss: 0.346 (excellent)
- Test loss: 14.35 (poor)
- Gap ratio: 41.5× (severe overfitting)
- Baseline gap: 16.0× (more reasonable given dataset size)

**3. Insufficient Regularization**
- No dropout applied to Earth4D encoder
- No weight decay mentioned in configuration
- Hash-based encoding alone doesn't provide sufficient inductive bias
- Small dataset cannot constrain 724M parameters

### When Earth4D Should Excel

Earth4D's multi-scale spatiotemporal encoding is designed for:

**Ideal Use Cases**:
- Large datasets (months to years of continuous data)
- Transfer learning (pretrain on large dataset, fine-tune on small)
- Global models with diverse spatial scales
- Datasets with 1000+ training timesteps

**Expected Benefits**:
- Capture synoptic-scale patterns (1000+ km)
- Resolve mesoscale features (10-100 km)
- Encode fine-scale details (<10 km)
- Provide spatial inductive bias through multi-resolution encoding

---

## Recommendations

### For Small Datasets (< 100 timesteps)

**Use baseline MLP**:
- Simpler architecture
- Parameter-efficient
- Better generalization
- Faster training

**If using Earth4D**:
- Reduce levels to 6-8 (not 24) to decrease parameters
- Add strong regularization (dropout ≥ 0.5)
- Consider freezing Earth4D weights after pretraining
- Use larger batch sizes to increase effective dataset size

### For Large Datasets (> 1000 timesteps)

**Earth4D recommended**:
- Full 24-level configuration
- Multi-scale encoding captures hierarchical patterns
- Expected to outperform standard MLP
- Suitable for operational weather forecasting

**Example Datasets**:
- MEPS (MetCoOp Ensemble Prediction System): Years of data
- ERA5 global reanalysis: Decades of data
- HRRR (High-Resolution Rapid Refresh): Continental-scale, high frequency

---

## Training Efficiency

| Model | Training Time | Checkpoint Size | Memory Usage |
|-------|---------------|----------------|--------------|
| **Baseline** | ~9 minutes (50 epochs) | 2.6 MB | ~8 GB |
| **Earth4D** | ~20 minutes (50 epochs) | 8.1 GB | ~18 GB |

- Earth4D requires 2.2× longer to train
- Checkpoint size is 3,115× larger
- Memory footprint is 2.25× higher

---

## Checkpoints

### Baseline
- **Location**: `saved_models/train-graph_lam-4x64-03_03_13-4227/`
- **Epoch 50 checkpoint**: `last.ckpt` (2.6 MB)

### Earth4D
- **Location**: `saved_models/train-graph_lam-4x64-03_03_12-6560/`
- **Epoch 50 checkpoint**: `last.ckpt` (8.1 GB)

---

## Lessons Learned

1. **Parameter efficiency matters**: 211K parameters outperformed 724M parameters
2. **Dataset size is critical**: 24 timesteps insufficient for 724M parameter model
3. **Training loss is misleading**: Always evaluate on held-out test data
4. **Inductive bias is valuable**: Simple MLP provided better implicit regularization
5. **Experimental validation is essential**: Hypotheses must be empirically tested

---

## Conclusion

This benchmark demonstrates that Earth4D's multi-scale spatiotemporal encoder, while theoretically compelling, **requires large datasets** to avoid overfitting. On the small DANRA dataset (24 training timesteps), the baseline MLP encoder provides superior generalization.

**For practitioners**:
- Small datasets: Use baseline MLP
- Large datasets: Earth4D likely to excel
- Always validate on held-out test data
- Match model capacity to dataset size

---

## Experimental Details

### Evaluation Protocol
- **Test set**: Held-out 3 days (April 7-10, 2022)
- **Metrics**: Mean squared error across autoregressive rollout steps (1, 2, 3, 5, 10)
- **Seed**: Fixed at 42 for reproducibility
- **Hardware**: Single NVIDIA A100-SXM4-80GB GPU

### Software Stack
- **PyTorch**: 2.7
- **PyTorch Lightning**: 2.4.0
- **CUDA**: 11.8
- **Python**: 3.10

### Data Source
- **Repository**: https://github.com/mllam/mllam-testdata
- **Version**: 2025-02-05
- **License**: Open data (DANRA dataset)

---

**Benchmark Date**: March 3, 2026
**Test Logs**: `experiments/neural-lam-comparison/{baseline,earth4d}/results/test_evaluation.log`
**Training Logs**: `experiments/neural-lam-comparison/{baseline,earth4d}/results/training_fresh.log`
