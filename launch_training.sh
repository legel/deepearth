#!/bin/bash
# =============================================================================
# DeepEarth Production Training Launch Script
# =============================================================================
#
# This script launches full-scale training with optimal settings for:
# - Multi-GPU distributed training
# - Mixed precision (FP16/BF16)
# - Large batch sizes with gradient accumulation
# - Comprehensive logging and checkpointing
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Experiment name
EXPERIMENT_NAME="deepearth_flowering_production"

# Data paths
CONFIG_PATH="/opt/ecodash/deepearth/models/flowering/flowering_light.yaml"  # Use lightweight config for faster iteration
DATA_DIR="/opt/ecodash/deepearth/models/flowering/data"
OUTPUT_BASE="/opt/ecodash/deepearth/experiments"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=32  # Increased since model is much smaller
LEARNING_RATE=1e-3
GRADIENT_ACCUMULATION=2  # Reduced since batch is smaller

# System settings
NUM_WORKERS=4  # Use workers for faster data loading
MIXED_PRECISION=true
COMPILE_MODEL=false  # Disabled - Earth4D HashEncoder not compatible with torch.compile

# Checkpointing
SAVE_EVERY=5
RESUME=""  # Set to checkpoint path to resume

# =============================================================================
# Environment Setup
# =============================================================================

echo "════════════════════════════════════════════════════════════════════════════"
echo "                    DeepEarth Production Training"
echo "════════════════════════════════════════════════════════════════════════════"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Date: $(date)"
echo "════════════════════════════════════════════════════════════════════════════"

# Increase shared memory limit for DataLoader workers
echo "Checking shared memory..."
SHM_SIZE=$(df -h /dev/shm | tail -1 | awk '{print $2}')
echo "Current shared memory: ${SHM_SIZE}"

# Install tensorboard if not available
/usr/bin/python3 -c "import tensorboard" 2>/dev/null || {
    echo "Installing tensorboard..."
    /usr/bin/python3 -m pip install tensorboard --quiet
}

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/opt/ecodash/deepearth"

# Optimize CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# CPU optimization
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# =============================================================================
# GPU Detection
# =============================================================================

# Detect available GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
echo ""
echo "GPU Configuration:"
echo "  Available GPUs: ${N_GPUS}"

if [ ${N_GPUS} -eq 0 ]; then
    echo "ERROR: No GPUs detected!"
    exit 1
fi

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# =============================================================================
# Build Launch Command
# =============================================================================

# Base command - use /usr/bin/python3 which has HashEncoder installed
CMD="/usr/bin/python3"

# Add distributed training launcher for multi-GPU
if [ ${N_GPUS} -gt 1 ]; then
    CMD="/usr/bin/python3 -u -m torch.distributed.run --nproc_per_node=${N_GPUS} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
    echo "Using distributed training with ${N_GPUS} GPUs"
else
    echo "Using single GPU training"
    CMD="/usr/bin/python3 -u"  # Add -u for unbuffered output
fi

# Main training script - use verbose version for detailed diagnostics
CMD="${CMD} /opt/ecodash/deepearth/train_deepearth_verbose.py"

# Add arguments
CMD="${CMD} --config ${CONFIG_PATH}"
CMD="${CMD} --data_dir ${DATA_DIR}"
CMD="${CMD} --output_dir ${OUTPUT_BASE}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --learning_rate ${LEARNING_RATE}"
CMD="${CMD} --gradient_accumulation ${GRADIENT_ACCUMULATION}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --save_every ${SAVE_EVERY}"

# Add flags
if [ "${MIXED_PRECISION}" = true ]; then
    CMD="${CMD} --mixed_precision"
fi

if [ "${COMPILE_MODEL}" = true ]; then
    CMD="${CMD} --compile"
fi

if [ ! -z "${RESUME}" ]; then
    CMD="${CMD} --resume ${RESUME}"
fi

# =============================================================================
# Memory Monitoring
# =============================================================================

# Create output directory first
mkdir -p ${OUTPUT_BASE}

# Start GPU memory monitoring in background
monitor_gpu() {
    while true; do
        nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader >> "${OUTPUT_BASE}/gpu_monitor.log"
        sleep 30
    done
}

# Start monitoring
monitor_gpu &
MONITOR_PID=$!
echo "Started GPU monitoring (PID: ${MONITOR_PID})"

# =============================================================================
# Launch Training
# =============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Launching training with command:"
echo "${CMD}"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Create output directory
mkdir -p ${OUTPUT_BASE}

# Save launch configuration
echo "${CMD}" > "${OUTPUT_BASE}/launch_command.txt"

# Launch training with output logging
${CMD} 2>&1 | tee "${OUTPUT_BASE}/training.log"

# Training exit code
TRAIN_EXIT=$?

# =============================================================================
# Cleanup
# =============================================================================

# Stop GPU monitoring
kill ${MONITOR_PID} 2>/dev/null || true

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "✅ Training completed successfully!"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "Results saved to: ${OUTPUT_BASE}"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "❌ Training failed with exit code ${TRAIN_EXIT}"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "Check logs at: ${OUTPUT_BASE}/training.log"
fi

exit ${TRAIN_EXIT}