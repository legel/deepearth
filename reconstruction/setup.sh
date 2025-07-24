#!/bin/bash

# setup.sh - Setup script for multimodal plant training

echo "Setting up multimodal plant classification training environment..."

# Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
umap-learn>=0.5.4
tqdm>=4.66.0
datasets>=2.14.0
huggingface-hub>=0.19.0
EOF

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p visualizations

# Create a simple training script
cat > train.py << 'EOF'
#!/usr/bin/env python3
"""
Simple training launcher script
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function from the multimodal script
from multimodal_plant_training import main

if __name__ == "__main__":
    print("Starting multimodal plant training...")
    print("-" * 60)
    main()
EOF

# Make train.py executable
chmod +x train.py

# Create a test script to verify setup
cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that all dependencies are installed correctly
"""
import sys

def check_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed")
        return False

print("Checking dependencies...")
print("-" * 40)

all_good = True
all_good &= check_import("torch")
all_good &= check_import("torchvision")
all_good &= check_import("transformers")
all_good &= check_import("PIL", "Pillow")
all_good &= check_import("numpy")
all_good &= check_import("matplotlib")
all_good &= check_import("sklearn", "scikit-learn")
all_good &= check_import("umap", "umap-learn")
all_good &= check_import("tqdm")
all_good &= check_import("datasets")

print("-" * 40)

if all_good:
    print("✓ All dependencies installed successfully!")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("! CUDA not available - will use CPU")
    except:
        pass
else:
    print("✗ Some dependencies are missing. Please run: pip install -r requirements.txt")
    sys.exit(1)
EOF

# Make verify script executable
chmod +x verify_setup.py

# Run verification
echo ""
echo "Verifying installation..."
python verify_setup.py

echo ""
echo "Setup complete!"
echo ""
echo "Directory structure created:"
echo "  checkpoints/     - Model checkpoints will be saved here"
echo "  logs/           - Training logs will be saved here"
echo "  visualizations/ - UMAP plots will be saved here"
echo ""
echo "Next steps:"
echo "  1. Test data loading:  python test_data_loading.py"
echo "  2. Start training:     python train.py"
echo ""
echo "Or if you need to customize settings:"
echo "  - Edit the MultimodalConfig class in multimodal_plant_training.py"
echo ""
