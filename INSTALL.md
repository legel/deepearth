# Earth4D Installation Guide

This guide provides instructions for installing Earth4D (Grid4D encoder for planetary X,Y,Z,T deep learning) on various systems with GPU/CUDA support.

## Quick Start for University Supercomputing Platforms (e.g. ASU Sol)

```bash
git clone <this-repository>
cd <repository-name>
chmod +x install.sh
./install.sh
```

## System Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability >= 7.5
- Minimum 4GB GPU memory recommended

### Software
- Linux operating system
- Python 3.8+
- CUDA 11.7+ or 12.x
- PyTorch with CUDA support
- C++17 compatible compiler

## Detailed Installation Instructions

### 1. CUDA Setup

#### On HPC Systems (with module system)
```bash
# Try one of these CUDA versions (compatibility with your PyTorch)
module load cuda-12.1.1-gcc-12.1.0
# OR
module load cuda-11.8.0-gcc-12.1.0
```

#### On Personal Systems
Install CUDA from NVIDIA's website and set environment variables:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. Python Environment Setup

Install PyTorch with CUDA support:
```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install additional dependencies:
```bash
pip install numpy cachetools ninja
```

### 3. Build Earth4D

#### Automated Installation (Recommended)
```bash
chmod +x install.sh
./install.sh
```

#### Manual Installation
```bash
cd encoders/xyzt/hashencoder
python setup.py build_ext --inplace
cd ../../..
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Version Mismatch
**Error**: `The detected CUDA version (X.X) mismatches the version that was used to compile PyTorch (Y.Y)`

**Solution**: 
- Install PyTorch version matching your CUDA installation
- Or set `export TORCH_CUDA_VERSION_CHECK=0` to bypass check (use with caution)

#### 2. nvcc Not Found
**Error**: `nvcc not found in PATH`

**Solution**:
```bash
# On HPC systems
module load cuda-12.1.1-gcc-12.1.0

# On personal systems
export PATH=/usr/local/cuda/bin:$PATH
```

#### 3. CUDA_HOME Not Set
**Error**: `CUDA_HOME environment variable is not set`

**Solution**:
```bash
# Find CUDA installation
which nvcc
# If output is /usr/local/cuda/bin/nvcc, then:
export CUDA_HOME=/usr/local/cuda

# On HPC systems, usually:
export CUDA_HOME=$CUDA_ROOT
```

#### 4. Compiler Issues
**Error**: Various C++ compilation errors

**Solution**:
- Ensure C++17 compatible compiler (GCC 7+)
- On HPC systems: `module load gcc/12.1.0`
- Update CUDA flags if needed

#### 5. Permission Errors
**Error**: Permission denied during build

**Solution**:
```bash
# Ensure write permissions
chmod -R u+w encoders/xyzt/hashencoder/
```

## Verification

Test your installation:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from encoders.xyzt import Earth4D
print('Import successful!')

import torch
if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA available')
else:
    device = 'cpu'
    print('CUDA not available - CPU only')

# Create encoder and test data
encoder = Earth4D().to(device)
coords = torch.rand(5, 4).to(device)

# Test encoding
spatial_feat, temporal_feat = encoder(coords)
print(f'Success! Spatial: {spatial_feat.shape}, Temporal: {temporal_feat.shape}')
"
```

## Usage Examples

### Basic Usage
```python
import torch
from encoders.xyzt import Earth4D

# Create encoder
encoder = Earth4D().cuda()  # Move to GPU

# Normalized coordinates (x, y, z, t) in [0, 1]
coords = torch.rand(100, 4).cuda()

# Encode
spatial_features, temporal_features = encoder(coords)
print(f"Spatial: {spatial_features.shape}")    # (100, 32)
print(f"Temporal: {temporal_features.shape}")  # (100, 96)
```

### Advanced Usage with Geographic Coordinates
```python
import torch
from encoders.xyzt import create_earth4d_with_auto_conversion

# Create encoder with automatic ECEF conversion
encoder = create_earth4d_with_auto_conversion().cuda()

# Raw geographic coordinates (lat, lon, elevation_m, time_seconds)
geo_coords = torch.tensor([
    [37.7749, -122.4194, 50.0, 1640995200.0],  # San Francisco
    [40.7128, -74.0060, 100.0, 1640995260.0],  # New York
]).cuda()

# Automatically converts and encodes
spatial_feat, temporal_feat = encoder(geo_coords)
```

### Custom Scales
```python
from encoders.xyzt import create_earth4d_with_physical_scales

# Define physical scales
encoder = create_earth4d_with_physical_scales(
    spatial_scales_meters=[16, 32, 64, 128, 256, 512],
    temporal_scales_seconds=[3600, 86400, 604800]  # hour, day, week
).cuda()
```

## Performance Notes

1. **GPU Memory**: Each encoder level uses ~2MB GPU memory
2. **Batch Size**: Larger batches (100-1000 samples) provide better GPU utilization  
3. **Precision**: Uses mixed precision (FP16) by default for memory efficiency
4. **Compute Capability**: Optimized for modern GPUs (Compute >= 7.5)

## System-Specific Notes

### University Supercomputing Platforms (e.g. ASU Sol)
- Use module system for CUDA: `module load cuda-12.1.1-gcc-12.1.0`
- GPU nodes have Tesla V100 or A100 GPUs
- Use SLURM job scheduler for GPU jobs

### Personal Workstations
- Ensure NVIDIA drivers are up to date
- Install CUDA toolkit from NVIDIA website
- May need to adjust compute capability flags

### Cloud Platforms (AWS, GCP, Azure)
- Use GPU instances (p3, V100, A100)
- CUDA usually pre-installed
- May need conda/pip environment setup

## Contributing

When contributing to this repository:
1. Test on multiple CUDA versions when possible
2. Update this installation guide for new dependencies
3. Include system requirements in PR descriptions

## Support

For installation issues:
1. Check this troubleshooting guide
2. Verify CUDA/PyTorch compatibility
3. Open GitHub issue with system details:
   - CUDA version (`nvcc --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - GPU model (`nvidia-smi`)
   - Error messages and full stack traces