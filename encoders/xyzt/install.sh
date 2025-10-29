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

# Ensure setuptools and wheel are up-to-date (prevents bdist_wheel warnings)
echo "Checking build dependencies..."
pip install --upgrade setuptools wheel -q 2>/dev/null || true
echo -e "${GREEN}[✓]${NC} Build dependencies ready"

# Set library path
TORCH_LIB_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -d "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
    echo -e "${GREEN}[✓]${NC} Library paths configured"
fi

# Check if already compiled
if [ -f "hashencoder/hashencoder_cuda.so" ]; then
    FILE_SIZE=$(du -h hashencoder/hashencoder_cuda.so | cut -f1)
    echo -e "${GREEN}[✓]${NC} CUDA extension already compiled (${FILE_SIZE}B)"
    echo "  To rebuild, delete hashencoder/hashencoder_cuda.so"
else
    # Build CUDA extension
    echo
    echo "Building CUDA extension..."
    cd hashencoder

    # Clean previous builds
    rm -rf build dist *.egg-info __pycache__
    rm -f hashencoder_cuda*.so

    # Detect CUDA architecture from nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        if [ ! -z "$GPU_ARCH" ]; then
            export TORCH_CUDA_ARCH_LIST="${GPU_ARCH:0:1}.${GPU_ARCH:1}"
            echo "  Detected GPU architecture: ${TORCH_CUDA_ARCH_LIST}"
        fi
    fi

    # Run build with progress indication
    # Note: bdist_wheel and g++ version warnings are harmless and can be ignored
    echo "  Compiling CUDA kernels (this may take 30-60 seconds)..."

    # Run build in background and show progress
    TEMP_BUILD_LOG=$(mktemp)
    python3 setup.py build_ext --inplace > "$TEMP_BUILD_LOG" 2>&1 &
    BUILD_PID=$!

    # Show progress dots while building
    while kill -0 $BUILD_PID 2>/dev/null; do
        echo -n "."
        sleep 2
    done
    wait $BUILD_PID
    BUILD_EXIT_CODE=$?
    echo ""

    # Show build output, filtering out known harmless warnings
    cat "$TEMP_BUILD_LOG" | grep -vE "(ModuleNotFoundError while trying to load entry-point bdist_wheel|No module named 'setuptools\.command\.bdist_wheel'|warnings\.warn\(f'There are no.*version bounds|There are no .* version bounds defined for CUDA)"
    rm -f "$TEMP_BUILD_LOG"

    if [ $BUILD_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}[✓]${NC} CUDA extension built successfully!"
    else
        echo -e "${YELLOW}[⚠]${NC} CUDA build failed. Will compile on first use."
    fi

    cd ..
fi

# Test installation
echo
echo "Testing installation..."
TEST_OUTPUT=$(python3 -c "
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress deprecation warnings during test

os.chdir('$(pwd)')
try:
    from earth4d import Earth4D
    import torch
    encoder = Earth4D(verbose=False)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        coords = torch.tensor([[40.7, -74.0, 100, 0.5]]).cuda()
        device = 'CUDA'
    else:
        coords = torch.tensor([[40.7, -74.0, 100, 0.5]])
        device = 'CPU'

    features = encoder(coords)
    spatial_dim = encoder.encoder.spatial_dim
    temporal_dim = encoder.encoder.temporal_dim
    print(f'SUCCESS:{device}:{features.shape}:spatial={spatial_dim}:temporal={temporal_dim}')
except Exception as e:
    print(f'ERROR:{e}')
" 2>/dev/null)

if [[ $TEST_OUTPUT == SUCCESS:* ]]; then
    IFS=':' read -r status device shape spatial temporal <<< "$TEST_OUTPUT"
    echo -e "${GREEN}[✓]${NC} Earth4D test passed on $device"
    echo "    Features shape: $shape ($spatial, $temporal)"
else
    echo -e "${RED}[✗]${NC} Test failed: ${TEST_OUTPUT#ERROR:}"
    exit 1
fi

# Success message
echo
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo
echo "To use Earth4D:"
echo "  from earth4d import Earth4D"
echo "  encoder = Earth4D().cuda()  # if using GPU"
echo "  coords = torch.tensor([[lat, lon, elev, time]]).cuda()"
echo "  features = encoder(coords)"
echo
echo "Run example:"
echo "  python earth4d.py"
echo
echo "Documentation: README.md"
