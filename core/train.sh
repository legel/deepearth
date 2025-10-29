#!/bin/bash
# run_deepearth.sh - Complete DeepEarth training and evaluation pipeline

set -e  # Exit on error

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME="deepearth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"

# ════════════════════════════════════════════════════════════════════════════
# Environment Setup
# ════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════"
echo "DeepEarth Training Pipeline"
echo "════════════════════════════════════════════════════════════════════════════"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════════════════════"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p data

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Logging
LOG_FILE="${OUTPUT_DIR}/training.log"
exec > >(tee -a ${LOG_FILE})
exec 2>&1

# ════════════════════════════════════════════════════════════════════════════
# Generate Sample Data (if needed)
# ════════════════════════════════════════════════════════════════════════════

if [ ! -f "data/earth_observations.csv" ]; then
    echo ""
    echo "Generating sample data..."
    python scripts/generate_sample_data.py
fi

# ════════════════════════════════════════════════════════════════════════════
# Create Configuration
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "Creating configuration..."

cat > ${OUTPUT_DIR}/config.yaml << 'EOF'
# DeepEarth Configuration
# ════════════════════════════════════════════════════════════════════════════

# Architecture
universal_dim: 1024
spacetime_dim: 512
dataset_embed_dim: 2
modality_embed_dim: 2
encoder_embed_dim: 2
mask_embed_dim: 4
context_position_dim: 2
modality_position_dim: 2

# Modality definitions
modalities:
  visual:
    name: visual
    encoder_name: BioCLIP
    position_dim: 2
    position_shape: [16, 16]
    max_tokens: 256
  spectral:
    name: spectral
    encoder_name: SpectralNet
    position_dim: 1
    position_shape: [64]
    max_tokens: 64
  weather:
    name: weather
    encoder_name: WeatherNet
    position_dim: 0
    position_shape: []
    max_tokens: 1

# Training
batch_size: 32
learning_rate: 0.0001
weight_decay: 0.01
num_epochs: 100
gradient_clip: 1.0
mixed_precision: true
compile_model: true

# Masking strategy
mask_spacetime_prob: 0.15
mask_data_prob: 0.15
mask_dataset_prob: 0.05
mask_modality_prob: 0.05
mask_encoder_prob: 0.05

# Perceiver architecture
num_latents: 256
latent_dim: 512
num_blocks: 8
num_cross_attention_heads: 8
num_self_attention_heads: 8
dropout: 0.1

# Sampling strategy
sampling_strategy:
  clusters_per_context: 4
  samples_per_cluster: 8
  time_of_day_weight: 0.1
  time_of_year_weight: 0.1
  time_of_history_weight: 0.1
  spatial_weight: 0.2
  modality_weight: 0.3
  universal_weight: 0.2
  sampling_type: contiguous

# UMAP configuration
umap_dim: 1
umap_max_samples: 1000000
umap_n_neighbors: 15
umap_min_dist: 0.1

# Hardware
device: cuda
num_workers: 4
pin_memory: true

# Output
output_dir: ${OUTPUT_DIR}
EOF

echo "Configuration created"

# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "Starting training..."
echo "────────────────────────────────────────────────────────────────────────────"

python -u deepearth/train.py \
    --config ${OUTPUT_DIR}/config.yaml \
    --input_csv data/earth_observations.csv \
    --output_dir ${OUTPUT_DIR} \
    --compile \
    --context_sampling \
    --print_architecture \
    --epochs 10  # Reduced for testing

# ════════════════════════════════════════════════════════════════════════════
# Inference Testing
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "Testing inference..."
echo "────────────────────────────────────────────────────────────────────────────"

python -u deepearth/inference.py \
    --checkpoint ${OUTPUT_DIR}/checkpoints/best.pt \
    --query data/test_queries.csv \
    --mask data/test_masks.csv \
    --output ${OUTPUT_DIR}/predictions.json \
    --batch_size 16

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Pipeline Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo "Results directory: ${OUTPUT_DIR}"
echo "Training log: ${LOG_FILE}"
echo "Best model: ${OUTPUT_DIR}/checkpoints/best.pt"
echo "Predictions: ${OUTPUT_DIR}/predictions.json"
echo "════════════════════════════════════════════════════════════════════════════"
