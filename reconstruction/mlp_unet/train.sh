#!/bin/bash

# Quick script to train with 10 species

echo "🌿 Training MLP U-Net with 10 Central Florida Plant Species"
echo "=========================================================="

# Activate virtual environment
source ../../venv_mlp_unet/bin/activate

# Step 1: Download images for 10 species
echo ""
echo "📥 Step 1: Downloading images for 10 species (200 per species)..."
python download_images.py --species 10 --images-per-species 200

# Step 2: Train with improvements
echo ""
echo "🚀 Step 2: Training with improved settings..."
echo "   - 10 species"
echo "   - Contrastive learning"
echo "   - Curriculum learning (progressive masking)"
echo "   - 50 epochs with early stopping"

python train.py \
    --species 10 \
    --epochs 50 \
    --batch-size 16 \
    --lr 5e-5 \
    --eval-every 5 \
    --use-contrastive \
    --warmup-epochs 5 \
    --images-per-species 200

# Step 3: Generate visualizations
echo ""
echo "📊 Step 3: Generating visualizations..."
python inference_fixed.py

echo ""
echo "✅ Training complete!"
echo "   Checkpoints: ./checkpoints/"
echo "   Visualizations: ./visualizations/"
echo "   View logs: tensorboard --logdir ./logs"
