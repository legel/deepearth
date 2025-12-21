#!/bin/bash
# Earth4D Installation Script
# Handles dependencies and CUDA compilation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Earth4D Installation"
echo "=========================================="
echo

# Check Python version
PYTHON_CHECK=$(python3 -c "import sys; major, minor = sys.version_info[:2]; print(f'{major}.{minor}'); exit(0 if (major, minor) >= (3, 7) else 1)" 2>/dev/null)
PYTHON_EXIT_CODE=$?
if [[ $PYTHON_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}[✓]${NC} Python $PYTHON_CHECK detected"
else
    echo -e "${RED}[✗]${NC} Python 3.7+ required (found $PYTHON_CHECK)"
    exit 1
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}[✓]${NC} CUDA $CUDA_VERSION detected"
else
    echo -e "${YELLOW}[⚠]${NC} CUDA not found. GPU acceleration will not be available."
fi

# Check PyTorch
if python3 -c "import torch" &> /dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}[✓]${NC} PyTorch $TORCH_VERSION detected"
else
    echo -e "${RED}[✗]${NC} PyTorch not found. Please install: pip install torch"
    exit 1
fi

# Check for ninja
if ! command -v ninja &> /dev/null; then
    echo -e "${YELLOW}[⚠]${NC} Installing ninja build system..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ninja-build
    else
        echo "Please install ninja manually: https://ninja-build.org/"
        exit 1
    fi
else
    echo -e "${GREEN}[✓]${NC} Ninja build system found"
fi

# Ensure setuptools and wheel are up-to-date
echo "Checking build dependencies..."
pip install --upgrade setuptools wheel -q 2>/dev/null || true
echo -e "${GREEN}[✓]${NC} Build dependencies ready"

# Set library path
TORCH_LIB_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -d "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    echo -e "${GREEN}[✓]${NC} Library paths configured"
fi

# Always clean and rebuild CUDA extension
echo
echo "Cleaning previous builds..."
cd hashencoder
rm -rf build dist *.egg-info __pycache__
rm -f hashencoder_cuda*.so
# Also clean JIT cache to avoid stale modules
rm -rf ~/.cache/torch_extensions/py310_cu126/hashencoder_cuda 2>/dev/null || true
echo -e "${GREEN}[✓]${NC} Cleaned previous builds"

# Build CUDA extension
echo
echo "Building CUDA extension..."

# Detect CUDA architecture from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    if [ ! -z "$GPU_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="${GPU_ARCH:0:1}.${GPU_ARCH:1}"
        echo "  Detected GPU architecture: ${TORCH_CUDA_ARCH_LIST}"
    fi
fi

echo "  Compiling CUDA kernels (this takes 5-10 minutes)..."

TEMP_BUILD_LOG=$(mktemp)
python3 setup.py build_ext --inplace > "$TEMP_BUILD_LOG" 2>&1 &
BUILD_PID=$!

while kill -0 $BUILD_PID 2>/dev/null; do
    echo -n "."
    sleep 2
done
wait $BUILD_PID
BUILD_EXIT_CODE=$?
echo ""

# Show build output (filter noise but don't fail on empty)
cat "$TEMP_BUILD_LOG" | grep -vE "(bdist_wheel|version bounds)" || true
rm -f "$TEMP_BUILD_LOG"

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    # Verify the .so file was created
    if [ -f "hashencoder_cuda.so" ]; then
        echo -e "${GREEN}[✓]${NC} CUDA extension built successfully!"
        ls -lh hashencoder_cuda.so
    else
        echo -e "${YELLOW}[⚠]${NC} Build completed but .so file not found in expected location."
        echo "  Checking build directory..."
        if [ -f "build/hashencoder_cuda.so" ]; then
            echo "  Found in build/, copying to current directory..."
            cp build/hashencoder_cuda.so .
            echo -e "${GREEN}[✓]${NC} CUDA extension ready!"
            ls -lh hashencoder_cuda.so
        else
            echo -e "${RED}[✗]${NC} Could not find built .so file"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}[⚠]${NC} CUDA build failed."
    exit 1
fi

cd ..

# Test installation
echo
echo "Testing installation..."
python3 -c "
import os, warnings
warnings.filterwarnings('ignore')
os.chdir('$(pwd)')
from earth4d import Earth4D
import torch
encoder = Earth4D(verbose=False)
if torch.cuda.is_available():
    encoder = encoder.cuda()
    coords = torch.tensor([[40.7, -74.0, 100, 0.5]]).cuda()
else:
    coords = torch.tensor([[40.7, -74.0, 100, 0.5]])
features = encoder(coords)
print(f'Test passed: {features.shape}')
"

echo
echo -e "${GREEN}Installation Complete!${NC}"
