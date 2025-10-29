#!/bin/bash
# Earth4D Environment Setup Script
# Source this file to set up the environment for Earth4D
# Usage: source setup_env.sh

# Get the directory where this script is located
EARTH4D_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add Earth4D to Python path
export PYTHONPATH="${EARTH4D_DIR}:${PYTHONPATH}"

# Set up PyTorch library path for CUDA extensions
TORCH_LIB_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB_PATH" ] && [ -d "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${LD_LIBRARY_PATH}"
fi

# Function to run Earth4D scripts from anywhere
earth4d_run() {
    if [ -z "$1" ]; then
        echo "Usage: earth4d_run <script.py> [arguments]"
        return 1
    fi

    cd "${EARTH4D_DIR}" && python3 "$@"
}

# Alias for quick testing
alias earth4d_test="cd ${EARTH4D_DIR} && python3 run_earth4d.py"

# Print status
echo "Earth4D Environment Configured:"
echo "  Directory: ${EARTH4D_DIR}"
echo "  PYTHONPATH: ${PYTHONPATH}"
if [ -n "$TORCH_LIB_PATH" ]; then
    echo "  PyTorch libs: ${TORCH_LIB_PATH}"
fi
echo ""
echo "You can now:"
echo "  - Import Earth4D from any directory: from earth4d import Earth4D"
echo "  - Run scripts: earth4d_run test_high_resolution.py --help"
echo "  - Quick test: earth4d_test"