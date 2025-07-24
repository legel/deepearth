#!/bin/bash
# fix_issues.sh - Fix the setup issues

echo "Fixing setup issues..."
echo "====================="

# 1. Fix NumPy version issue
echo "1. Fixing NumPy version conflict..."
pip uninstall -y numpy
pip install "numpy<2.0"

# 2. Create updated species mapping for your actual folder structure
echo ""
echo "2. Creating updated species mapping..."
cat > update_species_mapping.py << 'EOF'
import json
import os

# Your actual folder structure based on the output
species_mapping = {
    # Common name folders
    "american_beautyberry": "Callicarpa americana",
    "beach_sunflower": "Helianthus debilis",
    "black_eyed_susan": "Rudbeckia hirta",
    "blanket_flower": "Gaillardia pulchella",
    "coontie": "Zamia integrifolia",
    "leavenworth_tickseed": "Coreopsis leavenworthii",
    "saw_palmetto": "Serenoa repens",
    "spiderwort": "Tradescantia ohiensis",
    "spotted_beebalm": "Monarda punctata",
    "tropical_sage": "Salvia coccinea",
    
    # Scientific name folders (keep as is)
    "callicarpa_americana": "Callicarpa americana",
    "coreopsis_leavenworthii": "Coreopsis leavenworthii",
    "gaillardia_pulchella": "Gaillardia pulchella",
    "helianthus_debilis": "Helianthus debilis",
    "monarda_punctata": "Monarda punctata",
    "rudbeckia_hirta": "Rudbeckia hirta",
    "salvia_coccinea": "Salvia coccinea",
    "tradescantia_ohiensis": "Tradescantia ohiensis",
    "zamia_integrifolia": "Zamia integrifolia"
}

# Save to the data directory
data_root = "/home/ubuntu/a/deepearth/reconstruction/mlp_unet/data/plants"
mapping_file = os.path.join(data_root, "species_mapping.json")

with open(mapping_file, 'w') as f:
    json.dump(species_mapping, f, indent=2)

print(f"Updated species mapping saved to: {mapping_file}")

# Also create a list of folders to use (avoiding duplicates)
folders_to_use = [
    "american_beautyberry",
    "beach_sunflower", 
    "black_eyed_susan",
    "blanket_flower",
    "coontie",
    "leavenworth_tickseed",
    "saw_palmetto",
    "spiderwort",
    "spotted_beebalm",
    "tropical_sage"
]

with open("recommended_folders.txt", 'w') as f:
    f.write('\n'.join(folders_to_use))

print(f"Recommended folders saved to: recommended_folders.txt")
print(f"Total species to use: {len(folders_to_use)}")
EOF

python update_species_mapping.py

# 3. Create a simple test to verify PyTorch works
echo ""
echo "3. Testing PyTorch installation..."
cat > test_pytorch.py << 'EOF'
import sys
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
    # Test creating tensors
    x = torch.randn(2, 3)
    if torch.cuda.is_available():
        x = x.cuda()
        print("Successfully created CUDA tensor")
    else:
        print("Successfully created CPU tensor")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

python test_pytorch.py

# 4. Check the multimodal_plant_training.py file location
echo ""
echo "4. Checking file locations..."
echo "Current directory: $(pwd)"
echo ""
echo "Looking for multimodal_plant_training.py..."
find . -name "multimodal_plant_training.py" -type f 2>/dev/null | head -5

echo ""
echo "Files in current directory:"
ls -la *.py 2>/dev/null | head -10

echo ""
echo "====================="
echo "Setup fixes complete!"
echo ""
echo "Next steps:"
echo "1. Make sure multimodal_plant_training.py is in: $(pwd)"
echo "2. Fix the indentation error on line 465 of multimodal_plant_training.py"
echo "3. Run: python test_data_loading.py"
