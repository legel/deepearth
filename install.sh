#!/bin/bash

set -e  # Exit on any error

echo "Earth4D Installation Script for University Supercomputing Platforms"
echo "====================================================================="

# Configuration
REQUIRED_CUDA_MAJOR=11
REQUIRED_CUDA_MINOR=7
HASHENCODER_DIR="encoders/xyzt/hashencoder"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Sol supercomputer
check_environment() {
    print_status "Checking environment..."
    
    if [[ $(hostname) == *"sol"* ]]; then
        print_status "Detected ASU Sol supercomputer environment"
    else
        print_warning "Not on ASU Sol - some module commands may not work on this system"
    fi
}

# CUDA Detection and Setup
setup_cuda() {
    print_status "Setting up CUDA environment..."
    
    # Check if CUDA module is loaded or CUDA is available
    if ! command -v nvcc &> /dev/null; then
        print_warning "nvcc not found in PATH"
        print_status "Attempting to load CUDA module..."
        
        if command -v module &> /dev/null; then
            # Try loading CUDA modules on HPC systems - try CUDA 12.x first to match PyTorch
            CUDA_VERSIONS=("cuda-12.1.1-gcc-12.1.0" "cuda-12.0.1-gcc-12.1.0" "cuda-12.2.1-gcc-12.1.0" "cuda-12.3.0-gcc-12.1.0" "cuda-11.8.0-gcc-12.1.0" "cuda-11.7.0-gcc-11.2.0")
            CUDA_LOADED=false
            
            for cuda_ver in "${CUDA_VERSIONS[@]}"; do
                if module load "$cuda_ver" 2>/dev/null; then
                    print_status "Successfully loaded module: $cuda_ver"
                    CUDA_LOADED=true
                    break
                fi
            done
            
            if [ "$CUDA_LOADED" = false ]; then
                print_error "Failed to load any CUDA module. Available versions:"
                module avail 2>&1 | grep -i cuda || echo "No CUDA modules found"
                print_error "Please manually load a CUDA module first, e.g.:"
                print_error "  module load cuda-11.8.0-gcc-12.1.0"
                exit 1
            fi
        else
            print_error "CUDA not available and module system not found"
            print_error "Please ensure CUDA is installed and nvcc is in PATH"
            exit 1
        fi
    fi
    
    # Verify nvcc is now available
    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc still not found after attempting to load CUDA"
        exit 1
    fi
    
    # Get CUDA version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9\.]*\).*/\1/')
    print_status "Found CUDA version: $CUDA_VERSION"
    
    # Parse version numbers
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    # Check version requirements
    if (( CUDA_MAJOR < REQUIRED_CUDA_MAJOR || (CUDA_MAJOR == REQUIRED_CUDA_MAJOR && CUDA_MINOR < REQUIRED_CUDA_MINOR) )); then
        print_warning "CUDA $CUDA_VERSION is older than recommended $REQUIRED_CUDA_MAJOR.$REQUIRED_CUDA_MINOR"
        print_warning "Build may still work but is not tested"
    fi
    
    # Set CUDA_HOME if not set
    if [[ -z "$CUDA_HOME" ]]; then
        if [[ -n "$CUDA_ROOT" ]]; then
            export CUDA_HOME=$CUDA_ROOT
            print_status "Set CUDA_HOME=$CUDA_HOME (from CUDA_ROOT)"
        else
            # Try to detect CUDA home
            NVCC_PATH=$(which nvcc)
            CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
            export CUDA_HOME
            print_status "Set CUDA_HOME=$CUDA_HOME (detected from nvcc)"
        fi
    fi
    
    # Verify CUDA_HOME exists
    if [[ ! -d "$CUDA_HOME" ]]; then
        print_error "CUDA_HOME directory does not exist: $CUDA_HOME"
        exit 1
    fi
    
    # Update PATH and LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    print_status "CUDA environment configured"
}

# GPU Compute Capability Detection
detect_gpu_capabilities() {
    print_status "Detecting GPU compute capabilities..."
    
    if command -v nvidia-smi &> /dev/null; then
        # Get unique compute capabilities
        COMPUTE_CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -u | tr '\n' ';' | sed 's/;$//')
        
        if [[ -n "$COMPUTE_CAPS" ]]; then
            export TORCH_CUDA_ARCH_LIST="$COMPUTE_CAPS"
            print_status "Detected GPU compute capabilities: $COMPUTE_CAPS"
        else
            print_warning "Could not detect GPU compute capabilities"
            export TORCH_CUDA_ARCH_LIST="7.5"
            print_status "Using fallback compute capability: 7.5"
        fi
    else
        print_warning "nvidia-smi not available"
        export TORCH_CUDA_ARCH_LIST="7.5"
        print_status "Using fallback compute capability: 7.5"
    fi
}

# Python and PyTorch verification
check_python_environment() {
    print_status "Checking Python environment..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_status "Found Python version: $PYTHON_VERSION"
    
    # Check if PyTorch is available
    if ! python -c "import torch" 2>/dev/null; then
        print_error "PyTorch not found. Please install PyTorch with CUDA support"
        print_error "Visit: https://pytorch.org/get-started/locally/"
        exit 1
    fi
    
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    print_status "Found PyTorch version: $TORCH_VERSION"
    
    # Check CUDA support in PyTorch
    TORCH_CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [[ "$TORCH_CUDA_AVAILABLE" != "True" ]]; then
        print_error "PyTorch CUDA support not available"
        print_error "Please install PyTorch with CUDA support"
        exit 1
    fi
    
    TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    print_status "PyTorch built with CUDA: $TORCH_CUDA_VERSION"
}

# Build hashencoder extension
build_hashencoder() {
    print_status "Building hashencoder CUDA extension..."
    
    if [[ ! -d "$HASHENCODER_DIR" ]]; then
        print_error "Hashencoder directory not found: $HASHENCODER_DIR"
        exit 1
    fi
    
    # Change to hashencoder directory
    cd "$HASHENCODER_DIR"
    
    # Verify required files exist
    REQUIRED_FILES=("setup.py" "src/hashencoder.cu" "src/bindings.cpp" "src/hashencoder.h")
    for file in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Clean previous build
    if [[ -d "build" ]]; then
        print_status "Cleaning previous build..."
        rm -rf build
    fi
    
    # Build extension in-place
    print_status "Running: python setup.py build_ext --inplace"
    python setup.py build_ext --inplace || {
        print_error "Build failed"
        exit 1
    }
    
    # Return to original directory
    cd - > /dev/null
    
    print_status "Hashencoder build completed"
}

# Test imports
test_imports() {
    print_status "Testing imports..."
    
    # Test hashencoder import
    python -c "
import sys
sys.path.insert(0, '.')
from encoders.xyzt.hashencoder.hashgrid import HashEncoder
print('HashEncoder import: OK')
" || {
        print_error "Failed to import HashEncoder"
        exit 1
    }
    
    # Test Earth4D import
    python -c "
import sys
sys.path.insert(0, '.')
from encoders.xyzt.earth4d import Earth4D
print('Earth4D import: OK')
" || {
        print_error "Failed to import Earth4D"
        exit 1
    }
    
    # Test complete integration
    python -c "
import sys
sys.path.insert(0, '.')
import torch
from encoders.xyzt import Earth4D

# Create a simple test
encoder = Earth4D()
coords = torch.rand(2, 4)
try:
    spatial_feat, temporal_feat = encoder(coords)
    print(f'Earth4D test: OK (spatial: {spatial_feat.shape}, temporal: {temporal_feat.shape})')
except Exception as e:
    print(f'Earth4D test failed: {e}')
    sys.exit(1)
" || {
        print_error "Earth4D integration test failed"
        exit 1
    }
    
    print_status "All imports successful!"
}

# Main installation process
main() {
    check_environment
    setup_cuda
    detect_gpu_capabilities
    check_python_environment
    build_hashencoder
    test_imports
    
    echo ""
    print_status "Installation completed successfully!"
    echo ""
    echo "Usage example:"
    echo "  python -c \"import sys; sys.path.insert(0, '.'); from encoders.xyzt import Earth4D; print('Ready!')\""
    echo ""
    echo "Environment variables set:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
}

# Run main function
main "$@"