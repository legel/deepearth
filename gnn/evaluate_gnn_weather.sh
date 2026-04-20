#!/bin/bash
#
# Evaluation script for GNN weather forecasting models.
#
# This script evaluates trained models on the test set and computes
# performance metrics across different autoregressive rollout steps.

set -e
set -u
set -o pipefail

# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint to evaluate
CHECKPOINT_PATH="${1:-}"

# Evaluation configuration
EVAL_SPLIT="test"
AR_STEPS_EVAL=10

# Multi-scale node configuration (must match training)
USE_MULTISCALE_NODE=true
SPATIAL_LEVELS=24
TEMPORAL_LEVELS=24
FEATURES_PER_LEVEL=2
COORDINATE_SYSTEM="geographic"

# Output
OUTPUT_DIR="experiments/neural-lam-comparison/results/evaluation"

# =============================================================================
# Validation
# =============================================================================

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  $0 saved_models/my_experiment/last.ckpt"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# =============================================================================
# Display Configuration
# =============================================================================

echo "============================================================"
echo "GNN Weather Model Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Evaluation split: $EVAL_SPLIT"
echo "Autoregressive steps: $AR_STEPS_EVAL"
echo ""
echo "Multi-Scale Node Embedder:"
echo "  Enabled: $USE_MULTISCALE_NODE"
echo "  Spatial levels: $SPATIAL_LEVELS"
echo "  Temporal levels: $TEMPORAL_LEVELS"
echo "  Coordinate system: $COORDINATE_SYSTEM"
echo "============================================================"

# =============================================================================
# Evaluation
# =============================================================================

echo ""
echo "Starting evaluation..."
echo "Started at: $(date)"
echo ""

mkdir -p "$OUTPUT_DIR"

python -m neural_lam.train_model \
    --eval "$EVAL_SPLIT" \
    --load "$CHECKPOINT_PATH" \
    --ar_steps_eval "$AR_STEPS_EVAL" \
    $([ "$USE_MULTISCALE_NODE" = true ] && echo "--use_earth4d") \
    $([ "$USE_MULTISCALE_NODE" = true ] && echo "--earth4d_spatial_levels $SPATIAL_LEVELS") \
    $([ "$USE_MULTISCALE_NODE" = true ] && echo "--earth4d_temporal_levels $TEMPORAL_LEVELS") \
    $([ "$USE_MULTISCALE_NODE" = true ] && echo "--earth4d_features_per_level $FEATURES_PER_LEVEL") \
    $([ "$USE_MULTISCALE_NODE" = true ] && echo "--earth4d_coordinate_system $COORDINATE_SYSTEM")

# =============================================================================
# Completion
# =============================================================================

echo ""
echo "============================================================"
echo "Evaluation completed successfully"
echo "Finished at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
