#!/usr/bin/env python3
"""
Properly visualize reconstruction using actual chip boundaries from metadata
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from earth4d import Earth4D
from train_naip_rgb import RGBReconstructionModel, NAIPDataset

def load_model(checkpoint_path, enable_learned_probing=False, probing_range=4):
    """Load trained model"""
    model = RGBReconstructionModel(
        enable_learned_probing=enable_learned_probing,
        probing_range=probing_range
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Load data
data_path = Path('data/houston_316_full_xyztrgb.pt')
print("Loading data...")
data_dict = torch.load(data_path, weights_only=False)
data = data_dict['data']
metadata = data_dict['metadata']

dataset = NAIPDataset(data_path, normalize_coords=True)

print(f"Total points: {len(data):,}")
print(f"Total chips: {len(metadata)}")
print(f"Points per chip: {metadata[0]['n_points']}")

# Find a chip in validation set
n_train = int(len(data) * 0.8)
n_chips_train = n_train // 65536

# Pick a chip from validation set (chip index > n_chips_train)
chip_idx = n_chips_train + 5  # 6th validation chip
start_idx = chip_idx * 65536
end_idx = start_idx + 65536

print(f"\nUsing chip {chip_idx} (validation set)")
print(f"  Chip metadata: {metadata[chip_idx]['chm_file']}")
print(f"  Date: {metadata[chip_idx]['date']}")
print(f"  Lat range: {metadata[chip_idx]['lat_range']}")
print(f"  Lon range: {metadata[chip_idx]['lon_range']}")

# Extract chip data
coords_chip = data[start_idx:end_idx, :4]
rgb_chip = data[start_idx:end_idx, 4:]

print(f"  Chip size: {len(coords_chip)} points")

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nLoading models on {device}...")

model_baseline = load_model('naip_training_baseline/best_model.pt', False).to(device)
model_learned = load_model('naip_training_learned/best_model.pt', True, 32).to(device)

# Normalize and reconstruct
coords_norm = (coords_chip - dataset.coord_mean) / dataset.coord_std
coords_norm = coords_norm.to(device)

print("Reconstructing...")
with torch.no_grad():
    rgb_baseline = model_baseline(coords_norm).cpu()
    rgb_learned = model_learned(coords_norm).cpu()

# Reshape to 256x256 grid (proper chip structure)
size = 256
rgb_true_grid = rgb_chip.reshape(size, size, 3).numpy()
rgb_baseline_grid = rgb_baseline.reshape(size, size, 3).numpy()
rgb_learned_grid = rgb_learned.reshape(size, size, 3).numpy()

# Extract elevation (z coordinate) and reshape
elevation_chip = coords_chip[:, 2]  # z is 3rd coordinate
elevation_grid = elevation_chip.reshape(size, size).numpy()

# Plot with 4 panels
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# LiDAR Elevation (CHM)
im0 = axes[0].imshow(elevation_grid, cmap='terrain', aspect='auto')
axes[0].set_title('LiDAR Elevation (CHM)', fontsize=14, fontweight='bold')
axes[0].axis('off')
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im0, cax=cax, label='Elevation (m)')

# Ground Truth RGB
axes[1].imshow(rgb_true_grid, aspect='auto')
axes[1].set_title('Ground Truth RGB', fontsize=14, fontweight='bold')
axes[1].axis('off')

# Baseline Reconstruction
axes[2].imshow(rgb_baseline_grid, aspect='auto')
axes[2].set_title('Baseline Reconstruction', fontsize=14, fontweight='bold')
axes[2].axis('off')

# Learned Probing Reconstruction
axes[3].imshow(rgb_learned_grid, aspect='auto')
axes[3].set_title('Learned Probing Reconstruction', fontsize=14, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
output_path = 'visuals/rgb_reconstruction.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved to {output_path}")
plt.close()

# Compute errors
mse_baseline = ((rgb_chip - rgb_baseline) ** 2).mean().item()
mse_learned = ((rgb_chip - rgb_learned) ** 2).mean().item()

print(f"\nChip MSE:")
print(f"  Baseline: {mse_baseline:.6f}")
print(f"  Learned: {mse_learned:.6f}")
print(f"  Improvement: {(mse_baseline-mse_learned)/mse_baseline*100:.2f}%")
