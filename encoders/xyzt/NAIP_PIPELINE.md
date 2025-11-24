# NAIP RGB Reconstruction Pipeline

## Overview

This pipeline trains Earth4D models to reconstruct aerial RGB imagery from spatiotemporal coordinates: **(x, y, z, t) → (r, g, b)**

The task uses paired LiDAR elevation data (from USGS 3DEP) and aerial RGB imagery (from USDA NAIP) to learn continuous representations of real-world geospatial data.

---

## Data Sources

### Two Separate Data Types Downloaded:

1. **CHM (Canopy Height Model)** - LiDAR-derived elevation
   - Source: USGS 3DEP LiDAR processed by NTSG
   - URL: http://rangeland.ntsg.umt.edu/data/rap/chm-naip/chm/15/
   - Format: Single-band 256×256 GeoTIFF
   - Provides: **z (elevation)** component

2. **NAIP (National Agriculture Imagery Program)** - Aerial RGB imagery
   - Source: USDA aerial photography
   - URL: http://rangeland.ntsg.umt.edu/data/rap/chm-naip/naip/15/
   - Format: 5-band 427×427 GeoTIFF (R, G, B, NIR, mask)
   - Provides: **RGB** component

### Data Coordinate Components:
- **x (latitude)**: From GeoTIFF corner coordinates
- **y (longitude)**: From GeoTIFF corner coordinates
- **z (elevation)**: From LiDAR-derived CHM files
- **t (timestamp)**: From filename date (format: YYYYMMDD)
- **RGB**: From NAIP aerial imagery

---

## Pipeline Stages

### 1. Data Download

**Script:** `download_naip_all.py`

**What it does:**
- Downloads paired CHM + NAIP tar files for UTM Zone 15
- Zone range: 316-450 (135 zones total)
- Total size: 268 GB
- Features: Parallel downloads (8 workers), checkpoint/resume capability, progress tracking

**Command:**
```bash
python3 download_naip_all.py
```

**Output:**
- `data/naip_all/chm/15/*.tar` - Elevation data
- `data/naip_all/naip/15/*.tar` - RGB imagery data

**⚠️ IMPORTANT NOTE:**
The team lead requested 5 specific target locations:
1. ASU campus (Tempe, AZ) - 33.42°N, 111.93°W - **UTM Zone 12**
2. Boulder, CO (Flatirons) - 40.01°N, 105.27°W - **UTM Zone 13**
3. NYC downtown (High Line) - 40.75°N, 74.00°W - **UTM Zone 18**
4. Stanford University - 37.43°N, 122.17°W - **UTM Zone 10**
5. South Beach Miami - 25.76°N, 80.13°W - **UTM Zone 17**

**WE DID NOT DOWNLOAD THESE LOCATIONS.** We downloaded UTM Zone 15 (Houston/Texas area) which does not contain any of the 5 target regions. The `download_naip_all.py` script is hardcoded to Zone 15 and would need modification to download the correct zones (10, 12, 13, 17, 18).

---

### 2. Data Parsing

**Script:** `parse_naip_rgb_elevation.py`

**What it does:**
- Pairs CHM (elevation) with NAIP (RGB) files
- Resamples NAIP from 427×427 to 256×256 to match CHM resolution
- Extracts coordinates from GeoTIFF metadata
- Normalizes RGB values to [0, 1]

**Key function:**
```python
def parse_paired_geotiffs(chm_path: str, naip_path: str):
    """
    Returns: (N, 7) tensor [lat, lon, elevation, timestamp, r, g, b]
    - Reads CHM: 256×256 elevation values
    - Reads NAIP: 427×427 RGB, resamples to 256×256
    - Each chip: 65,536 points (256×256 grid)
    """
```

**Command:**
```bash
python3 parse_naip_rgb_elevation.py \
    --chm-dir data/naip_all/chm/15 \
    --naip-dir data/naip_all/naip/15 \
    --output data/houston_316_full_xyztrgb.pt \
    --zones 316
```

**Output:**
- `data/houston_316_full_xyztrgb.pt` (194 MB)
  - 7,274,496 points total
  - 111 chips from zone 316 (Houston area)
  - Format: (latitude, longitude, elevation, timestamp, R, G, B)

---

### 3. Model Training

**Script:** `train_naip_rgb.py`

**What it does:**
- Trains Earth4D encoder + MLP head for RGB reconstruction
- Compares baseline vs learned probing
- 80/20 train/validation split
- Evaluates on held-out test data

**Model architecture:**
```python
class RGBReconstructionModel(nn.Module):
    def __init__(self, enable_learned_probing=False, probing_range=4):
        self.earth4d = Earth4D(
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
        )
        # earth4d outputs 192-dim features
        self.rgb_head = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # R, G, B
            nn.Sigmoid()
        )
```

**Training data:**
- Train: 5,819,597 points (89 chips, 80%)
- Validation: 1,454,899 points (22 chips, 20%)

**Commands:**

Baseline model:
```bash
module load gcc-11.2.0-gcc-8.5.0 cuda-11.8.0-gcc-11.2.0
source activate earth4d
python3 train_naip_rgb.py \
    --data data/houston_316_full_xyztrgb.pt \
    --output-dir naip_training_baseline \
    --epochs 25 \
    --batch-size 8192 \
    --device cuda
```

Learned probing model:
```bash
python3 train_naip_rgb.py \
    --data data/houston_316_full_xyztrgb.pt \
    --output-dir naip_training_learned \
    --learned-probing \
    --probing-range 32 \
    --epochs 25 \
    --batch-size 8192 \
    --device cuda
```

**Training results:**

| Model | Parameters | Val Loss | Test MAE | Improvement |
|-------|-----------|----------|----------|-------------|
| Baseline | 724M | 0.002090 | 0.0292 | - |
| Learned Probing | 807M | 0.001710 | 0.0281 | **18.2% better loss** |

**Output:**
- `naip_training_baseline/best_model.pt` (8.1 GB)
- `naip_training_learned/best_model.pt` (9.1 GB)

---

### 4. Visualization

**Script:** `visualize_proper_chip.py`

**What it does:**
- Loads trained models
- Selects ONE chip from validation set (chip #93)
- Reconstructs RGB for all 65,536 points in that chip
- Compares: Ground Truth vs Baseline vs Learned Probing

**Key code:**
```python
# Use metadata to identify exact chip boundaries
chip_idx = n_chips_train + 5  # 6th validation chip
start_idx = chip_idx * 65536  # Each chip = 256×256 = 65,536 points
end_idx = start_idx + 65536

coords_chip = data[start_idx:end_idx, :4]  # (x,y,z,t)
rgb_chip = data[start_idx:end_idx, 4:]     # (r,g,b)

# Reshape to proper 256×256 spatial grid
rgb_true_grid = rgb_chip.reshape(256, 256, 3).numpy()
rgb_baseline_grid = rgb_baseline.reshape(256, 256, 3).numpy()
rgb_learned_grid = rgb_learned.reshape(256, 256, 3).numpy()
```

**Command:**
```bash
python3 visualize_proper_chip.py
```

**Visualized data:**
- **Single chip** from validation set
- Chip #93 metadata:
  - Location: Coastal wetlands near Houston
  - Date: 2018-01-23
  - Size: 65,536 points (256×256 grid)

**Output:**
- `visuals/rgb_reconstruction_proper.png`
  - Shows: Ground Truth | Baseline Reconstruction | Learned Probing Reconstruction
  - Chip MSE: Baseline 0.003086 → Learned 0.002867 (**7.1% improvement**)

---

## What We Visualized

### `visuals/naip_chip_5.png` - Raw Input Data
- **Purpose:** Show original data quality
- **Data:** Training chip #5 from zone 316
- **Date:** 2019-01-25
- **Location:** Coastal beach area near Houston
- **Shows:** Original NAIP RGB + LiDAR elevation side-by-side

### `visuals/rgb_reconstruction_proper.png` - Model Reconstructions
- **Purpose:** Demonstrate learned geospatial patterns
- **Data:** Validation chip #93 (held-out during training)
- **Date:** 2018-01-23
- **Location:** Coastal wetlands near Houston
- **Size:** 65,536 points (ONE complete 256×256 chip)
- **Shows:** Ground Truth | Baseline | Learned Probing
- **Result:** Learned probing achieves 7% lower MSE on this chip

**Important:** These are two different chips from the Houston dataset. Both visualizations show proper aerial imagery structure with clear geographic features (coastlines, vegetation, water).

---

## Key Scripts Reference

1. **`download_naip_all.py`** (310 lines)
   - Bulk download with checkpoint/resume
   - Parallel downloads (8 workers)
   - ⚠️ Hardcoded to Zone 15 only

2. **`parse_naip_rgb_elevation.py`** (380 lines)
   - Pairs CHM + NAIP files
   - Resamples NAIP 427×427 → 256×256
   - Outputs (x,y,z,t,r,g,b) tensors

3. **`train_naip_rgb.py`** (363 lines)
   - Earth4D + MLP head
   - Baseline vs learned probing comparison
   - Coordinate normalization
   - 80/20 train/val split

4. **`visualize_proper_chip.py`** (113 lines)
   - Uses metadata to identify chip boundaries
   - Reconstructs single validation chip
   - Computes per-chip MSE

5. **`find_target_zones.py`** (166 lines)
   - Identifies UTM zones for target locations
   - Revealed we downloaded wrong geographic region

---

## Results Summary

### ✅ Achievements:
1. Built complete pipeline: download → parse → train → visualize
2. Successfully paired LiDAR elevation with aerial RGB data
3. Trained two Earth4D models on 7.3M real geospatial points
4. Learned probing shows **18% validation loss improvement**
5. Generated clear aerial imagery reconstructions

### ⚠️ Current Issues:
1. **Wrong geographic data:** Downloaded Houston (Zone 15) instead of 5 target locations (Zones 10,12,13,17,18)
2. **Collision profiling pending:** Haven't extracted collision rates from trained models yet
3. **Limited scale:** Only processed 111 chips from single zone (7.3M points) - team lead wanted 1-10M point range from target locations

### 📋 Next Steps:
1. Profile hash collision rates for both models
2. Download correct UTM zones for 5 target locations
3. Retrain on target regions data
4. Scale up to requested 10×10 to 100×100 chip clusters

---

## Data Flow Summary

```
Raw Data Sources:
├── CHM tar files (LiDAR elevation) → z component
└── NAIP tar files (Aerial RGB) → RGB components

↓ parse_naip_rgb_elevation.py

Parsed Dataset:
└── houston_316_full_xyztrgb.pt
    └── 7,274,496 points × 7 features (x,y,z,t,r,g,b)

↓ train_naip_rgb.py (80/20 split)

Training:
├── Train: 5,819,597 points (89 chips)
└── Validation: 1,454,899 points (22 chips)

↓ Model learns: (x,y,z,t) → (r,g,b)

Checkpoints:
├── naip_training_baseline/best_model.pt (724M params)
└── naip_training_learned/best_model.pt (807M params)

↓ visualize_proper_chip.py

Visualization:
└── Single chip reconstruction (65,536 points)
    └── rgb_reconstruction_proper.png
```

---

## Environment Setup

Required modules:
```bash
module load gcc-11.2.0-gcc-8.5.0 cuda-11.8.0-gcc-11.2.0
source activate earth4d
```

Required Python packages:
- torch
- numpy
- matplotlib
- rasterio
- tqdm
- requests
