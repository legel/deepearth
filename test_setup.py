"""
Test DeepEarth setup on Lambda Labs
"""

import torch
import time
from core.inductive_simulator import create_inductive_simulator

print("Testing DeepEarth setup...")

# 1. GPU Check
print(f"\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 2. Creating Inductive Simulator
print(f"\n2. Creating Inductive Simulator...")
start = time.time()

# THE FIX: Load the language model in 8-bit and place the model directly on the GPU.
simulator = create_inductive_simulator(
    preset="standard",
    language_precision="int8",
    device="cuda"
)

print(f"   Created in {time.time() - start:.2f} seconds")
print(f"   Parameters: {sum(p.numel() for p in simulator.parameters()):,}")

# 3. Testing forward pass
print(f"\n3. Testing forward pass...")
batch_size = 4
num_tokens = 100
# Ensure input features are on the GPU
features = torch.randn(batch_size, num_tokens, 2048).cuda()

# Use autocast for mixed-precision inference
with torch.cuda.amp.autocast():
    start = time.time()
    # The simulator is already on the GPU
    output = simulator(features)
    torch.cuda.synchronize()
    print(f"   Forward pass time: {time.time() - start:.3f} seconds")
    print(f"   Output shape: {output['simulated_features'].shape}")

print(f"\n✓ Setup complete! Ready for training.")
