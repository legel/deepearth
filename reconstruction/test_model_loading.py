#!/usr/bin/env python3
"""Test that models can be loaded"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("Testing model loading...")

try:
    from transformers import ViTModel, ViTImageProcessor
    print("✓ Imports successful")
    
    # Try loading ViT
    print("\nLoading ViT model (this may take a moment)...")
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    print(f"✓ Successfully loaded ViT model")
    print(f"  Model config: {model.config.hidden_size} hidden size")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
