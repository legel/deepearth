#!/bin/bash
#
# Training script for GNN weather forecasting with multi-scale spatiotemporal nodes.
#
# This script trains a graph neural network for weather prediction using the
# MultiScaleSpatioTemporalNode embedder with Earth4D encoding.
#
# Requirements:
#   - CUDA 11.8+
#   - PyTorch 2.7+
#   - Earth4D encoder installed

set -e
set -u
set -o pipefail

# =============================================================================
# Configuration
# =============================================================================

# Data configuration
CONFIG_PATH="experiments/neural-lam-comparison/shared/data/config.yaml"
GRAPH_NAME="multiscale"
MODEL="graph_lam"
OUTPUT_DIR="experiments/neural-lam-comparison/results"

# GNN architecture
HIDDEN_DIM=64
PROCESSOR_LAYERS=4
MESH_AGGR="sum"

# Multi-scale spatiotemporal node embedder configuration
USE_MULTISCALE_NODE=true
SPATIAL_LEVELS=24
TEMPORAL_LEVELS=24
FEATURES_PER_LEVEL=2
COORDINATE_SYSTEM="geographic"
ADAPTIVE_RANGE=true
RESOLUTION_MODE="balanced"

# Training configuration
EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=1e-3
AR_STEPS_TRAIN=3
AR_STEPS_EVAL=10
SEED=42

# Hardware
DEVICES=1

# =============================================================================
# Display Configuration
# =============================================================================

echo "============================================================"
echo "GNN Weather Forecasting with Multi-Scale Spatiotemporal Nodes"
echo "============================================================"
echo "Configuration:"
echo "  Data config: $CONFIG_PATH"
echo "  Graph: $GRAPH_NAME"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "GNN Architecture:"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Processor layers: $PROCESSOR_LAYERS"
echo "  Mesh aggregation: $MESH_AGGR"
echo ""
echo "Multi-Scale Node Embedder:"
echo "  Enabled: $USE_MULTISCALE_NODE"
echo "  Spatial levels: $SPATIAL_LEVELS"
echo "  Temporal levels: $TEMPORAL_LEVELS"
echo "  Features per level: $FEATURES_PER_LEVEL"
echo "  Coordinate system: $COORDINATE_SYSTEM"
echo "  Adaptive range: $ADAPTIVE_RANGE"
echo ""
echo "Training:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo "============================================================"

# =============================================================================
# Verification
# =============================================================================

echo ""
echo "Verifying dependencies..."

python -c "
import sys
try:
    from encoders.xyzt.earth4d import Earth4D
    import torch

    encoder = Earth4D(
        spatial_levels=$SPATIAL_LEVELS,
        temporal_levels=$TEMPORAL_LEVELS,
        features_per_level=$FEATURES_PER_LEVEL,
        verbose=False
    )

    total_params = sum(p.numel() for p in encoder.parameters())

    print('✓ Earth4D encoder available')
    print(f'✓ Total parameters: {total_params:,} ({total_params/1e6:.1f}M)')
    print(f'✓ Output dimension: {encoder.output_dim}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')

    if torch.cuda.is_available():
        print(f'✓ CUDA device: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠ Warning: CUDA not available, training will be slow')

except ImportError as e:
    print(f'✗ Error: {e}')
    print('Please ensure encoders.xyzt is in your Python path')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Dependency check failed. Exiting."
    exit 1
fi

# =============================================================================
# Training
# =============================================================================

echo ""
echo "Starting training..."
echo "Started at: $(date)"
echo ""

mkdir -p "$OUTPUT_DIR"

python -m neural_lam.train_model \
    --config_path "$CONFIG_PATH" \
    --model "$MODEL" \
    --graph "$GRAPH_NAME" \
    --hidden_dim "$HIDDEN_DIM" \
    --processor_layers "$PROCESSOR_LAYERS" \
    --mesh_aggr "$MESH_AGGR" \
    --lr "$LEARNING_RATE" \
    --val_interval 1 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --ar_steps_train "$AR_STEPS_TRAIN" \
    --ar_steps_eval "$AR_STEPS_EVAL" \
    --num_workers 0 \
    --seed "$SEED" \
    --use_earth4d \
    --earth4d_spatial_levels "$SPATIAL_LEVELS" \
    --earth4d_temporal_levels "$TEMPORAL_LEVELS" \
    --earth4d_features_per_level "$FEATURES_PER_LEVEL" \
    --earth4d_coordinate_system "$COORDINATE_SYSTEM" \
    $([ "$ADAPTIVE_RANGE" = true ] && echo "--earth4d_adaptive_range") \
    --earth4d_resolution_mode "$RESOLUTION_MODE"

# =============================================================================
# Completion
# =============================================================================

echo ""
echo "============================================================"
echo "Training completed successfully"
echo "Finished at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
