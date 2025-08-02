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
