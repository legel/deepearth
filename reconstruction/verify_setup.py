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
