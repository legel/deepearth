#!/bin/bash
# train_flowering_model.sh - Train DeepEarth on Angiosperm Flowering Dataset

set -e  # Exit on any error

# Configuration
EXPERIMENT="angiosperm_flowering"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/${EXPERIMENT}_${TIMESTAMP}"

echo "════════════════════════════════════════════════════════════════════════════"
echo "DeepEarth Training: Angiosperm Flowering Intelligence"
echo "════════════════════════════════════════════════════════════════════════════"
echo "Experiment: ${EXPERIMENT}"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════════════════════"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/opt/ecodash/deepearth"
export OMP_NUM_THREADS=12

# Log file
LOG_FILE="${OUTPUT_DIR}/training.log"

# Run training
python -u /opt/ecodash/deepearth/models/flowering/flowering_model.py \
    --config /opt/ecodash/deepearth/models/flowering/flowering.yaml \
    --data_dir /opt/ecodash/deepearth/models/flowering/data \
    --output_dir ${OUTPUT_DIR} \
    --compile \
    --context_sampling \
    --print_architecture \
    --epochs 50 \
    --batch_size 128 \
    2>&1 | tee ${LOG_FILE}

echo "════════════════════════════════════════════════════════════════════════════"
echo "Training complete!"
echo "Results: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Best model: ${OUTPUT_DIR}/checkpoints/best.pt"
echo "════════════════════════════════════════════════════════════════════════════"
