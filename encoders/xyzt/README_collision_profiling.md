# Earth4D Hash Collision Profiling

Statistical profiling of hash collisions in Earth4D spatiotemporal encoding.

## Quick Start

```bash
cd /home/qhuang62/deepearth/encoders/xyzt
python earth4d_collision_profiler.py
```

**Required files:**
- `globe_lfmc_extracted.csv` - Globe-LFMC dataset (90K+ samples)
- Production CUDA-compiled `hashencoder` module

## Features

- **Real-time tracking**: Captures grid indices during CUDA hash encoding
- **Complete coordinate preservation**: Original (lat, lon, elev, time) + normalized coordinates
- **Professional export**: CSV with 332 columns including all grid indices per coordinate
- **Scientific analysis ready**: JSON metadata for reproducible research
- **Production configuration**: 24 spatial levels, 19 temporal levels

## Output Structure

```
earth4d_collision_profiling/
├── earth4d_collision_data.csv      # Complete dataset (90K+ samples, ~91MB)
└── earth4d_collision_metadata.json # Configuration and analysis metadata
```

## CSV Format

**Coordinate columns (8):**
- `latitude`, `longitude`, `elevation_m`, `time_original`
- `x_normalized`, `y_normalized`, `z_normalized`, `time_normalized`

**Grid index columns (324):**
- `{grid}_level_{level:02d}_dim_{dim}` (e.g., `xyz_level_23_dim_0`)
- `{grid}_level_{level:02d}_collision` (e.g., `xyz_level_23_collision`)
- For grids: `xyz`, `xyt`, `yzt`, `xzt`

## Analysis Results

The profiler processes the complete Globe-LFMC dataset (90,002 samples) and exports:

- Grid indices for every coordinate at every resolution level
- Collision flags indicating hash collisions vs. direct lookup
- Statistical analysis of collision patterns across grid spaces
- Complete metadata for scientific reproducibility

## Implementation

Built on NVIDIA's 3D multi-resolution hash encoding with real-time CUDA collision tracking integrated into Earth4D's production hash encoding kernels.