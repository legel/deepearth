#!/bin/bash
# fix_transformers.sh - Fix transformers compatibility issues

echo "Fixing transformers/huggingface_hub compatibility..."
echo "==================================================="

# Show current versions
echo "Current versions:"
pip list | grep -E "transformers|huggingface-hub|accelerate" || echo "Packages not found"

echo ""
echo "Uninstalling conflicting packages..."
pip uninstall -y transformers huggingface-hub accelerate tokenizers

echo ""
echo "Installing compatible versions..."
# These versions are known to work together
pip install "huggingface-hub==0.20.3"
pip install "transformers==4.36.2"
pip install "accelerate==0.25.0"
pip install "tokenizers==0.15.0"

echo ""
echo "Verifying installation..."
python -c "
try:
    import transformers
    import huggingface_hub
    import accelerate
    print(f'✓ Transformers: {transformers.__version__}')
    print(f'✓ HuggingFace Hub: {huggingface_hub.__version__}')
    print(f'✓ Accelerate: {accelerate.__version__}')
    
    # Test specific imports
    from transformers import ViTModel, AutoModel
    print('✓ ViT and AutoModel imports successful')
    
    from huggingface_hub import snapshot_download
    print('✓ HuggingFace Hub imports successful')
    
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "Creating test script for model loading..."
cat > test_model_loading.py << 'EOF'
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
EOF

chmod +x test_model_loading.py

echo ""
echo "Testing model loading..."
python test_model_loading.py

echo ""
echo "==================================================="
echo "Fix complete!"
echo ""
echo "If you still have issues, try:"
echo "1. Clear the transformers cache:"
echo "   rm -rf ~/.cache/huggingface/transformers/*"
echo ""
echo "2. Or install specific working versions:"
echo "   pip install transformers==4.30.2 huggingface-hub==0.16.4 accelerate==0.20.3"
