#!/usr/bin/env python3
"""
Visualize NAIP RGB reconstruction results

Creates two plots:
1. Training/validation loss curves comparing baseline vs learned probing
2. Reconstruction quality: LiDAR elevation, ground truth RGB, baseline prediction, learned prediction

Usage:
    python visualize_naip_results.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from earth4d import Earth4D


class RGBReconstructionModel(nn.Module):
    """Earth4D + MLP for RGB reconstruction"""

    def __init__(self, enable_learned_probing=False, probing_range=4):
        super().__init__()

        self.earth4d = Earth4D(
            verbose=False,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
        )

        earth4d_dim = self.earth4d.encoder.output_dim

        self.rgb_head = nn.Sequential(
            nn.Linear(earth4d_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, coords):
        features = self.earth4d(coords)
        rgb = self.rgb_head(features)
        return rgb


def plot_loss_curves(baseline_history_path, learned_history_path, output_path):
    """Plot training and validation loss curves"""

    # Load histories
    with open(baseline_history_path, 'r') as f:
        baseline_history = json.load(f)

    with open(learned_history_path, 'r') as f:
        learned_history = json.load(f)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = baseline_history['epochs']

    # Training loss
    ax1.plot(epochs, baseline_history['train_loss'], 'b-', label='Baseline', linewidth=2)
    ax1.plot(epochs, learned_history['train_loss'], 'r-', label='Learned Probing', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Validation loss
    ax2.plot(epochs, baseline_history['val_loss'], 'b-', label='Baseline', linewidth=2)
    ax2.plot(epochs, learned_history['val_loss'], 'r-', label='Learned Probing', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add improvement percentage text
    baseline_best = min(baseline_history['val_loss'])
    learned_best = min(learned_history['val_loss'])
    improvement = ((baseline_best - learned_best) / baseline_best) * 100

    fig.text(0.5, 0.02,
             f'Best Val Loss: Baseline={baseline_best:.6f}, Learned={learned_best:.6f} ({improvement:.2f}% improvement)',
             ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curves saved to {output_path}")
    plt.close()

    return baseline_best, learned_best, improvement


def plot_reconstruction_comparison(
    data_path,
    baseline_model_path,
    learned_model_path,
    output_path,
    chip_idx=0,
    device='cuda',
    use_train_chip=False
):
    """
    Plot reconstruction comparison for a single chip

    Shows: LiDAR elevation | Ground truth RGB | Baseline RGB | Learned RGB

    Args:
        use_train_chip: If True, use training chip instead of validation chip
    """
    print("\nLoading data and models...")

    # Load data
    data_dict = torch.load(data_path, weights_only=False)
    data_tensor = data_dict['data']
    chip_sizes = data_dict['chip_sizes']

    # Load baseline model checkpoint
    baseline_checkpoint = torch.load(baseline_model_path, weights_only=False)
    baseline_val_indices = baseline_checkpoint['val_indices']
    baseline_train_indices = baseline_checkpoint['train_indices']

    # Find which chip to visualize
    cumsum = np.cumsum([0] + chip_sizes)

    if use_train_chip:
        # Get training chip indices
        chip_indices = []
        for i in range(len(chip_sizes)):
            start = cumsum[i]
            end = cumsum[i + 1]
            if start in baseline_train_indices:
                chip_indices.append(i)
        chip_type = "training"
    else:
        # Get validation chip indices
        chip_indices = []
        for i in range(len(chip_sizes)):
            start = cumsum[i]
            end = cumsum[i + 1]
            if start in baseline_val_indices:
                chip_indices.append(i)
        chip_type = "validation"

    if chip_idx >= len(chip_indices):
        chip_idx = 0

    actual_chip_idx = chip_indices[chip_idx]
    print(f"  Visualizing {chip_type} chip {chip_idx} (global chip index {actual_chip_idx})")

    # Extract chip data
    start = cumsum[actual_chip_idx]
    end = cumsum[actual_chip_idx + 1]
    chip_data = data_tensor[start:end]  # (65536, 7)

    # Parse chip data
    coords = chip_data[:, :4]  # lat, lon, elev, time
    rgb_gt = chip_data[:, 4:].numpy()  # ground truth RGB
    elevation = coords[:, 2].numpy()

    # Normalize coordinates (same as training)
    coord_mean = data_tensor[:, :4].mean(dim=0)
    coord_std = data_tensor[:, :4].std(dim=0)
    coord_std[coord_std < 1e-6] = 1.0
    coords_normalized = (coords - coord_mean) / coord_std

    # Reshape to 256x256 (chip is 256x256 pixels)
    h, w = 256, 256
    elevation_img = elevation.reshape(h, w)
    rgb_gt_img = rgb_gt.reshape(h, w, 3)

    # Load and run baseline model
    print("  Running baseline model...")
    baseline_model = RGBReconstructionModel(
        enable_learned_probing=False,
        probing_range=4
    ).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_model.eval()

    with torch.no_grad():
        coords_gpu = coords_normalized.to(device)
        rgb_baseline = baseline_model(coords_gpu).cpu().numpy()

    rgb_baseline_img = rgb_baseline.reshape(h, w, 3)

    # Free baseline model memory before loading learned model
    del baseline_model
    del baseline_checkpoint
    torch.cuda.empty_cache()

    # Load and run learned model
    print("  Running learned probing model...")
    learned_checkpoint = torch.load(learned_model_path, weights_only=False)
    learned_config = learned_checkpoint['config']

    learned_model = RGBReconstructionModel(
        enable_learned_probing=learned_config['enable_learned_probing'],
        probing_range=learned_config['probing_range']
    ).to(device)
    learned_model.load_state_dict(learned_checkpoint['model_state_dict'])
    learned_model.eval()

    with torch.no_grad():
        rgb_learned = learned_model(coords_gpu).cpu().numpy()

    rgb_learned_img = rgb_learned.reshape(h, w, 3)

    # Calculate errors
    baseline_mse = np.mean((rgb_baseline - rgb_gt) ** 2)
    learned_mse = np.mean((rgb_learned - rgb_gt) ** 2)
    improvement = ((baseline_mse - learned_mse) / baseline_mse) * 100

    # Create visualization
    print("  Creating visualization...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # LiDAR elevation
    im0 = axes[0].imshow(elevation_img, cmap='terrain')
    axes[0].set_title('LiDAR Elevation (CHM)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Ground truth RGB
    axes[1].imshow(rgb_gt_img)
    axes[1].set_title('Ground Truth RGB (NAIP)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Baseline reconstruction
    axes[2].imshow(rgb_baseline_img)
    axes[2].set_title(f'Baseline Reconstruction\nMSE: {baseline_mse:.6f}',
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Learned probing reconstruction
    axes[3].imshow(rgb_learned_img)
    axes[3].set_title(f'Learned Probing Reconstruction\nMSE: {learned_mse:.6f} ({improvement:+.2f}%)',
                     fontsize=14, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Reconstruction comparison saved to {output_path}")
    plt.close()

    return baseline_mse, learned_mse, improvement


def main():
    print("="*70)
    print("NAIP RGB Reconstruction Visualization")
    print("="*70)

    # Paths
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data/asu/parsed_xyztrgb.pt'
    baseline_model = base_dir / 'runs/asu_baseline_v2/checkpoint_epoch25.pt'
    learned_model = base_dir / 'runs/asu_learned_v2/checkpoint_epoch25.pt'
    baseline_history = base_dir / 'runs/asu_baseline_v2/history.json'
    learned_history = base_dir / 'runs/asu_learned_v2/history.json'

    output_dir = base_dir / 'visuals'
    output_dir.mkdir(exist_ok=True)

    # Check required files exist
    for path in [data_path, baseline_model, learned_model]:
        if not path.exists():
            print(f"✗ Missing file: {path}")
            return

    # Only plot loss curves if history files exist
    if baseline_history.exists() and learned_history.exists():
        print("\n1. Creating loss curve comparison...")
        baseline_best, learned_best, loss_improvement = plot_loss_curves(
            baseline_history,
            learned_history,
            output_dir / 'naip_loss_curves.png'
        )
    else:
        print("\n1. Skipping loss curves (history.json files not found - training still in progress)")
        baseline_best, learned_best, loss_improvement = None, None, None

    # Optimized visualization - load models once, reuse for all chips
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n2. Loading data and models (optimized)...")
    # Load data once
    data_dict = torch.load(data_path, weights_only=False)
    data_tensor = data_dict['data']
    chip_sizes = data_dict['chip_sizes']

    # Load baseline model once
    print("  Loading baseline model...")
    baseline_checkpoint = torch.load(baseline_model, weights_only=False)
    baseline_model_obj = RGBReconstructionModel(
        enable_learned_probing=False,
        probing_range=4
    ).to(device)
    baseline_model_obj.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_model_obj.eval()

    # Get train/val indices
    baseline_val_indices = baseline_checkpoint['val_indices']
    baseline_train_indices = baseline_checkpoint['train_indices']

    # Compute normalization parameters once
    coord_mean = data_tensor[:, :4].mean(dim=0)
    coord_std = data_tensor[:, :4].std(dim=0)
    coord_std[coord_std < 1e-6] = 1.0

    # Run baseline inference on all chips
    print("  Running baseline inference on all chips...")
    train_chips = [0, 1]
    val_chips = [0, 1, 2]

    baseline_results = {}
    with torch.inference_mode():
        for chip_idx, use_train in [(i, True) for i in train_chips] + [(i, False) for i in val_chips]:
            chip_key = f"{'train' if use_train else 'val'}_{chip_idx}"

            # Get chip data
            cumsum = np.cumsum([0] + chip_sizes)
            if use_train:
                chip_indices = [i for i in range(len(chip_sizes)) if cumsum[i] in baseline_train_indices]
            else:
                chip_indices = [i for i in range(len(chip_sizes)) if cumsum[i] in baseline_val_indices]

            actual_chip_idx = chip_indices[chip_idx]
            start = cumsum[actual_chip_idx]
            end = cumsum[actual_chip_idx + 1]
            chip_data = data_tensor[start:end]

            coords = chip_data[:, :4]
            coords_normalized = (coords - coord_mean) / coord_std
            coords_gpu = coords_normalized.to(device)

            rgb_baseline = baseline_model_obj(coords_gpu).cpu()
            baseline_results[chip_key] = (rgb_baseline, chip_data, actual_chip_idx)

    # Free baseline model
    del baseline_model_obj
    del baseline_checkpoint
    torch.cuda.empty_cache()

    # Load learned model once
    print("  Loading learned model...")
    learned_checkpoint = torch.load(learned_model, weights_only=False)
    learned_config = learned_checkpoint['config']
    learned_model_obj = RGBReconstructionModel(
        enable_learned_probing=learned_config['enable_learned_probing'],
        probing_range=learned_config['probing_range']
    ).to(device)
    learned_model_obj.load_state_dict(learned_checkpoint['model_state_dict'])
    learned_model_obj.eval()

    # Run learned inference on all chips
    print("  Running learned model inference on all chips...")
    learned_results = {}
    with torch.inference_mode():
        for chip_key, (_, chip_data, _) in baseline_results.items():
            coords = chip_data[:, :4]
            coords_normalized = (coords - coord_mean) / coord_std
            coords_gpu = coords_normalized.to(device)

            rgb_learned = learned_model_obj(coords_gpu).cpu()
            learned_results[chip_key] = rgb_learned

    # Free learned model
    del learned_model_obj
    del learned_checkpoint
    torch.cuda.empty_cache()

    # Generate all visualizations
    print("\n3. Generating visualizations...")
    train_results = []
    for i in train_chips:
        chip_key = f"train_{i}"
        rgb_baseline, chip_data, actual_chip_idx = baseline_results[chip_key]
        rgb_learned = learned_results[chip_key]

        print(f"  Creating training chip {i} (global chip {actual_chip_idx})...")

        # Extract data
        coords = chip_data[:, :4]
        rgb_gt = chip_data[:, 4:].numpy()
        elevation = coords[:, 2].numpy()

        # Reshape
        h, w = 256, 256
        elevation_img = elevation.reshape(h, w)
        rgb_gt_img = rgb_gt.reshape(h, w, 3)
        rgb_baseline_img = rgb_baseline.numpy().reshape(h, w, 3)
        rgb_learned_img = rgb_learned.numpy().reshape(h, w, 3)

        # Calculate errors
        baseline_mse = np.mean((rgb_baseline.numpy() - rgb_gt) ** 2)
        learned_mse = np.mean((rgb_learned.numpy() - rgb_gt) ** 2)
        improvement = ((baseline_mse - learned_mse) / baseline_mse) * 100

        # Create plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        im0 = axes[0].imshow(elevation_img, cmap='terrain')
        axes[0].set_title('LiDAR Elevation (CHM)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].imshow(rgb_gt_img)
        axes[1].set_title('Ground Truth RGB (NAIP)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(rgb_baseline_img)
        axes[2].set_title(f'Baseline Reconstruction\nMSE: {baseline_mse:.6f}', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(rgb_learned_img)
        axes[3].set_title(f'Learned Probing Reconstruction\nMSE: {learned_mse:.6f} ({improvement:+.2f}%)', fontsize=14, fontweight='bold')
        axes[3].axis('off')

        plt.tight_layout()
        output_path = output_dir / f'naip_reconstruction_train_chip{i}_epoch25.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

        train_results.append((baseline_mse, learned_mse, improvement))

    # Validation chips
    val_results = []
    for i in val_chips:
        chip_key = f"val_{i}"
        rgb_baseline, chip_data, actual_chip_idx = baseline_results[chip_key]
        rgb_learned = learned_results[chip_key]

        print(f"  Creating validation chip {i} (global chip {actual_chip_idx})...")

        # Extract data
        coords = chip_data[:, :4]
        rgb_gt = chip_data[:, 4:].numpy()
        elevation = coords[:, 2].numpy()

        # Reshape
        h, w = 256, 256
        elevation_img = elevation.reshape(h, w)
        rgb_gt_img = rgb_gt.reshape(h, w, 3)
        rgb_baseline_img = rgb_baseline.numpy().reshape(h, w, 3)
        rgb_learned_img = rgb_learned.numpy().reshape(h, w, 3)

        # Calculate errors
        baseline_mse = np.mean((rgb_baseline.numpy() - rgb_gt) ** 2)
        learned_mse = np.mean((rgb_learned.numpy() - rgb_gt) ** 2)
        improvement = ((baseline_mse - learned_mse) / baseline_mse) * 100

        # Create plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        im0 = axes[0].imshow(elevation_img, cmap='terrain')
        axes[0].set_title('LiDAR Elevation (CHM)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].imshow(rgb_gt_img)
        axes[1].set_title('Ground Truth RGB (NAIP)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(rgb_baseline_img)
        axes[2].set_title(f'Baseline Reconstruction\nMSE: {baseline_mse:.6f}', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(rgb_learned_img)
        axes[3].set_title(f'Learned Probing Reconstruction\nMSE: {learned_mse:.6f} ({improvement:+.2f}%)', fontsize=14, fontweight='bold')
        axes[3].axis('off')

        plt.tight_layout()
        output_path = output_dir / f'naip_reconstruction_val_chip{i}_epoch25.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")

        val_results.append((baseline_mse, learned_mse, improvement))

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    if baseline_best is not None:
        print(f"Best Validation Loss:")
        print(f"  Baseline:  {baseline_best:.6f}")
        print(f"  Learned:   {learned_best:.6f}")
        print(f"  Improvement: {loss_improvement:.2f}%")

    print(f"\nReconstruction MSE (Epoch 25):")
    print(f"\nTraining Chips:")
    for i, (baseline_mse, learned_mse, improvement) in enumerate(train_results):
        print(f"  Chip {i}:")
        print(f"    Baseline:  {baseline_mse:.6f}")
        print(f"    Learned:   {learned_mse:.6f}")
        print(f"    Improvement: {improvement:.2f}%")

    print(f"\nValidation Chips:")
    for i, (baseline_mse, learned_mse, improvement) in enumerate(val_results):
        print(f"  Chip {i}:")
        print(f"    Baseline:  {baseline_mse:.6f}")
        print(f"    Learned:   {learned_mse:.6f}")
        print(f"    Improvement: {improvement:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
