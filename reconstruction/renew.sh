#!/bin/bash
# fix_numpy.sh - Fix NumPy version conflicts

echo "Fixing NumPy version conflicts..."
echo "================================="

# First, check current versions
echo "Current package versions:"
pip show numpy | grep Version || echo "NumPy not found"
pip show matplotlib | grep Version || echo "Matplotlib not found"
pip show torch | grep Version || echo "PyTorch not found"

echo ""
echo "Uninstalling and reinstalling packages in correct order..."

# Uninstall conflicting packages
pip uninstall -y numpy matplotlib

# Install compatible versions
# NumPy 1.26.x works with both PyTorch and matplotlib
pip install "numpy>=1.23,<2.0"
pip install matplotlib

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import numpy as np
import matplotlib
import torch
print(f'NumPy version: {np.__version__}')
print(f'Matplotlib version: {matplotlib.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "Creating minimal test script..."
cat > test_imports.py << 'EOF'
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
EOF

chmod +x test_imports.py

echo ""
echo "Running import test..."
python test_imports.py

echo ""
echo "================================="
echo "Fix complete!"
echo ""
echo "If you still have issues, try creating a fresh virtual environment:"
echo "  python -m venv fresh_env"
echo "  source fresh_env/bin/activate"
echo "  pip install -r requirements.txt"
