# Earth4D Collision Profiler

Complete hash collision profiling system for Earth4D spatiotemporal encoding, providing comprehensive analysis of hash distribution patterns across planetary-scale coordinate data.

## Overview

The Earth4D Collision Profiler analyzes hash collision behavior in NVIDIA's 3D multi-resolution hash encoding when applied to real-world spatiotemporal coordinates. It tracks the exact 1D hash table indices that coordinates map to across all resolution levels and grid spaces, enabling scientific analysis of hash distribution uniformity and collision patterns.

## Features

✅ **Complete Coordinate Processing**
- Processes unique (latitude, longitude, elevation, time) coordinates only
- Automatic deduplication (e.g., 90K → 41K unique coordinates)
- Preserves original and normalized coordinate formats

✅ **Accurate Hash Index Tracking**  
- Returns actual 1D hash table indices (0 to 8,388,607 for spatial, 0 to 262,143 for temporal)
- Tracks indices across all 4 grid spaces: xyz, xyt, yzt, xzt
- Real-time collision detection (stride > hashmap_size)

✅ **Professional Data Export**
- CSV format with datetime strings (YYYY-MM-DD) 
- Single hash index column per level instead of theoretical 3D coordinates
- Complete metadata export for reproducible analysis
- Ready for scientific publication

✅ **Production Configuration**
- 24 spatial levels, 19 temporal levels
- 8.3M spatial hash table, 262K temporal hash tables
- Configurable tracking limits (default: 1M coordinates)

## Quick Start

### 1. Basic Usage

```python
from encoders.xyzt.earth4d import Earth4D

# Initialize with collision tracking enabled
model = Earth4D(
    enable_collision_tracking=True,
    max_tracked_examples=100000  # Track up to 100K coordinates
)

# Process your coordinates (lat, lon, elev, time)
features = model(coordinates)

# Export collision analysis
summary = model.export_collision_data("collision_results/")
```

### 2. Run Complete LFMC Analysis

```bash
# Ensure CUDA environment is loaded (ASU Sol HPC)
module load cuda-11.7.0-gcc-11.2.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run collision profiler
python earth4d_collision_profiler.py
```

### 3. Expected Output

```
================================================================================
EARTH4D HASH COLLISION PROFILER
================================================================================
Loading LFMC data from: ./globe_lfmc_extracted.csv
Loaded 90002 LFMC samples
Total samples: 90002, Unique spatiotemporal coordinates: 41261
Processing 41261 unique spatiotemporal coordinates

...

✅ Completed processing 41261 samples
✅ Tracked coordinates: 41261

Collision Summary:
  xyz: 45.8% overall, 80.0% fine resolution
  xyt: 63.2% overall, 60.0% fine resolution
  yzt: 63.2% overall, 60.0% fine resolution
  xzt: 63.2% overall, 60.0% fine resolution
```

## Output Format

### CSV Structure (`earth4d_collision_data.csv`)

| Column Type | Example | Description |
|-------------|---------|-------------|
| Coordinates | `latitude`, `longitude`, `elevation_m` | Original coordinate values |
| Time | `time_original` | Datetime string (YYYY-MM-DD) |
| Normalized | `x_normalized`, `y_normalized`, `z_normalized`, `time_normalized` | Normalized coordinates |
| Hash Indices | `xyz_level_00_index`, `xyz_level_01_index`, ... | 1D hash table slot numbers (0 to max_size-1) |
| Collisions | `xyz_level_00_collision`, `xyz_level_01_collision`, ... | Boolean flags for hash collisions |

### Key Improvements from Feedback

- **Row Count**: Substantially fewer rows (41K vs 90K) due to coordinate deduplication
- **Time Format**: Proper datetime strings instead of normalized float values  
- **Index Format**: Single 1D hash index per level instead of 3D theoretical coordinates
- **Index Ranges**: All indices within actual allocated hash table sizes

## Scientific Applications

### Hash Distribution Analysis
```python
import pandas as pd
import numpy as np

# Load collision data
df = pd.read_csv('earth4d_collision_profiling/earth4d_collision_data.csv')

# Analyze hash distribution uniformity
for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
    level_23_indices = df[f'{grid}_level_23_index']
    
    # Check uniformity (should be roughly uniform across hash table)
    hist, bins = np.histogram(level_23_indices, bins=100)
    uniformity_score = 1.0 - np.std(hist) / np.mean(hist)
    print(f'{grid} Level 23 uniformity: {uniformity_score:.3f}')
```

### Collision Pattern Analysis
```python
# Analyze collision rates by resolution level
collision_rates = {}
for grid in ['xyz', 'xyt', 'yzt', 'xzt']:
    rates = []
    for level in range(24 if grid == 'xyz' else 19):
        col = f'{grid}_level_{level:02d}_collision'
        if col in df.columns:
            rate = df[col].mean()
            rates.append(rate)
    collision_rates[grid] = rates

# Plot collision trends across resolution levels
import matplotlib.pyplot as plt
for grid, rates in collision_rates.items():
    plt.plot(rates, label=grid)
plt.xlabel('Resolution Level')
plt.ylabel('Collision Rate')
plt.legend()
plt.title('Hash Collision Rates by Resolution Level')
plt.show()
```

## Technical Details

### Hash Table Sizes
- **Spatial (xyz)**: 2^23 = 8,388,608 slots
- **Temporal (xyt, yzt, xzt)**: 2^18 = 262,144 slots

### Index Computation
Each `level_index` value represents the actual slot number where that coordinate's features are stored:

```cpp
// CUDA computation (simplified)
uint32_t index = 0;
for (d = 0; d < D && stride <= hashmap_size; d++) {
    index += pos_grid[d] * stride;
    stride *= resolution[d];
}
if (stride > hashmap_size) {
    index = fast_hash(pos_grid);  // Hash collision occurs
}
uint32_t hash_table_index = index % hashmap_size;  // Final 1D slot
```

### Memory Usage
- **Tracking overhead**: ~552 bytes per coordinate (4 grids × 23 levels × 3 axes × 2 bytes)
- **1M coordinates**: ~552 MB additional memory
- **Configurable**: Adjust `max_tracked_examples` based on available memory

## Configuration Options

### Earth4D Parameters
```python
model = Earth4D(
    # Core configuration
    spatial_levels=24,              # Production: 24 levels
    temporal_levels=19,             # Production: 19 levels  
    spatial_log2_hashmap_size=23,   # 8.3M entries
    temporal_log2_hashmap_size=18,  # 262K entries
    
    # Collision tracking
    enable_collision_tracking=True,
    max_tracked_examples=1000000,   # Track up to 1M coordinates
    verbose=True
)
```

### Profiler Script Options
Edit `earth4d_collision_profiler.py` to:
- Change LFMC data path: `lfmc_path = "your_data.csv"`
- Adjust batch size: `batch_size = 100`
- Modify output directory: `output_dir = "custom_results/"`

## Files

### Core Implementation
- `earth4d.py` - Earth4D class with collision tracking
- `hashencoder/src/hashencoder.cu` - CUDA tracking implementation
- `hashencoder/hashgrid.py` - PyTorch autograd functions

### Analysis Tools  
- `earth4d_collision_profiler.py` - Main profiling script
- `earth4d_collision_profiling/` - Output directory
  - `earth4d_collision_data.csv` - Complete collision data
  - `earth4d_collision_metadata.json` - Configuration and statistics

### Data Requirements
- `globe_lfmc_extracted.csv` - LFMC dataset (90K samples)
  - Required columns: `Latitude (WGS84, EPSG:4326)`, `Longitude (WGS84, EPSG:4326)`, `Elevation (m.a.s.l)`, `Sampling date (YYYYMMDD)`

## Troubleshooting

### CUDA Compilation Issues
```bash
# Ensure CUDA environment is properly set
module load cuda-11.7.0-gcc-11.2.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
cd hashencoder && python setup.py build_ext --inplace
```

### Memory Issues
- Reduce `max_tracked_examples` for large datasets
- Use smaller `batch_size` in profiler script
- Monitor GPU memory with `nvidia-smi`

### Index Range Validation
Verify indices are within expected ranges:
- Spatial indices: 0 to 8,388,607
- Temporal indices: 0 to 262,143
- All collision flags should be boolean

## Citation

If you use this collision profiler in your research, please cite:

```bibtex
@software{earth4d_collision_profiler,
  title={Earth4D Hash Collision Profiler},
  author={Earth4D Team},
  year={2024},
  url={https://github.com/qhuang62/deepearth}
}
```

## License

MIT License - see main repository for details.