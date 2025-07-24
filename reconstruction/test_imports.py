#!/usr/bin/env python3
"""Test all imports work correctly"""

print("Testing imports...")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✓ Matplotlib {plt.matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    from transformers import ViTModel, AutoModel
    print("✓ Transformers imported")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import umap
    print("✓ UMAP imported")
except ImportError as e:
    print(f"✗ UMAP import failed: {e}")

print("\nAll critical imports tested!")
