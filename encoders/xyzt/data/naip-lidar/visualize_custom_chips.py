#!/usr/bin/env python3
"""
Visualize specific chips by global chip index

Usage:
    python visualize_custom_chips.py --chips 194 234 834 --epoch 25 --output visuals/natural_areas.png
    python visualize_custom_chips.py --chips 0 1 8 11 15 --epoch 25  # Default visualized chips
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import os
import argparse

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


def visualize_chips(
    chip_indices,
    data_path,
    baseline_model_path,
    learned_model_path,
    metadata_path,
    output_path,
    device='cuda'
):
    """
    Visualize specific chips by global index

    Args:
        chip_indices: List of global chip indices to visualize
        data_path: Path to parsed_xyztrgb.pt
        baseline_model_path: Path to baseline checkpoint
        learned_model_path: Path to learned checkpoint
        metadata_path: Path to chip_metadata.json
        output_path: Where to save the visualization
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print("Custom Chip Visualization")
    print("="*70)

    # Load metadata
    with open(metadata_path) as f:
        chip_metadata = json.load(f)

    # Load data
    print("\nLoading data...")
    data_dict = torch.load(data_path, weights_only=False)
    data_tensor = data_dict['data']
    chip_sizes = data_dict['chip_sizes']

    # Compute chip boundaries
    cumsum = np.cumsum([0] + chip_sizes)

    # Compute normalization parameters
    coord_mean = data_tensor[:, :4].mean(dim=0)
    coord_std = data_tensor[:, :4].std(dim=0)
    coord_std[coord_std < 1e-6] = 1.0

    # Load models
    print("Loading baseline model...")
    baseline_checkpoint = torch.load(baseline_model_path, weights_only=False)
    baseline_model = RGBReconstructionModel(
        enable_learned_probing=False,
        probing_range=4
    ).to(device)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_model.eval()

    print("Loading learned probing model...")
    learned_checkpoint = torch.load(learned_model_path, weights_only=False)
    learned_config = learned_checkpoint['config']
    learned_model = RGBReconstructionModel(
        enable_learned_probing=learned_config['enable_learned_probing'],
        probing_range=learned_config['probing_range']
    ).to(device)
    learned_model.load_state_dict(learned_checkpoint['model_state_dict'])
    learned_model.eval()

    # Process each chip
    results = []

    print(f"\nProcessing {len(chip_indices)} chips...")
    with torch.inference_mode():
        for chip_idx in chip_indices:
            print(f"  Chip {chip_idx}: {chip_metadata[chip_idx]['chip_id']}")

            # Extract chip data
            start = cumsum[chip_idx]
            end = cumsum[chip_idx + 1]
            chip_data = data_tensor[start:end]

            coords = chip_data[:, :4]
            coords_normalized = (coords - coord_mean) / coord_std
            coords_gpu = coords_normalized.to(device)

            rgb_gt = chip_data[:, 4:].cpu().numpy()
            elevation = coords[:, 2].cpu().numpy()

            # Run models
            rgb_baseline = baseline_model(coords_gpu).cpu().numpy()
            rgb_learned = learned_model(coords_gpu).cpu().numpy()

            # Calculate MSE
            baseline_mse = np.mean((rgb_baseline - rgb_gt) ** 2)
            learned_mse = np.mean((rgb_learned - rgb_gt) ** 2)
            improvement = ((baseline_mse - learned_mse) / baseline_mse) * 100

            results.append({
                'chip_idx': chip_idx,
                'metadata': chip_metadata[chip_idx],
                'elevation': elevation.reshape(256, 256),
                'rgb_gt': rgb_gt.reshape(256, 256, 3),
                'rgb_baseline': rgb_baseline.reshape(256, 256, 3),
                'rgb_learned': rgb_learned.reshape(256, 256, 3),
                'baseline_mse': baseline_mse,
                'learned_mse': learned_mse,
                'improvement': improvement
            })

    # Clean up GPU memory
    del baseline_model, learned_model
    torch.cuda.empty_cache()

    # Create visualization
    print("\nCreating visualization...")
    n_chips = len(results)
    fig, axes = plt.subplots(n_chips, 4, figsize=(20, 5*n_chips))

    # Handle single chip case
    if n_chips == 1:
        axes = axes.reshape(1, -1)

    land_cover_names = {
        11: "Open Water",
        21: "Developed Open Space",
        22: "Developed Low Intensity",
        23: "Developed Medium Intensity",
        24: "Developed High Intensity",
        31: "Barren Land",
        41: "Deciduous Forest",
        42: "Evergreen Forest",
        43: "Mixed Forest",
        52: "Shrub/Scrub",
        71: "Grassland",
        81: "Pasture/Hay",
        82: "Cultivated Crops",
        90: "Woody Wetlands",
        95: "Emergent Wetlands"
    }

    for i, result in enumerate(results):
        metadata = result['metadata']
        lc_name = land_cover_names.get(metadata['land_cover'], f"LC {metadata['land_cover']}")

        # LiDAR elevation
        im0 = axes[i, 0].imshow(result['elevation'], cmap='terrain')
        axes[i, 0].set_title(f"LiDAR Elevation\nChip {result['chip_idx']}: {lc_name}",
                            fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Ground truth
        axes[i, 1].imshow(result['rgb_gt'])
        axes[i, 1].set_title(f"Ground Truth RGB\n{metadata['naip_date']}",
                            fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        # Baseline
        axes[i, 2].imshow(result['rgb_baseline'])
        axes[i, 2].set_title(f"Baseline\nMSE: {result['baseline_mse']:.6f}",
                            fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')

        # Learned
        axes[i, 3].imshow(result['rgb_learned'])
        axes[i, 3].set_title(f"Learned Probing\nMSE: {result['learned_mse']:.6f} ({result['improvement']:+.1f}%)",
                            fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for result in results:
        metadata = result['metadata']
        lc_name = land_cover_names.get(metadata['land_cover'], f"LC {metadata['land_cover']}")
        print(f"\nChip {result['chip_idx']}: {metadata['chip_id']}")
        print(f"  Land Cover: {lc_name}")
        print(f"  Baseline MSE:  {result['baseline_mse']:.6f}")
        print(f"  Learned MSE:   {result['learned_mse']:.6f}")
        print(f"  Improvement:   {result['improvement']:+.2f}%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Visualize specific chips by global index")
    parser.add_argument('--chips', type=int, nargs='+', required=True,
                       help='Global chip indices to visualize')
    parser.add_argument('--epoch', type=int, default=25,
                       help='Which epoch checkpoint to use (default: 25)')
    parser.add_argument('--output', type=str, default='visuals/custom_chips.png',
                       help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent
    data_path = base_dir / 'data/asu/parsed_xyztrgb.pt'
    baseline_model = base_dir / f'runs/asu_baseline_v2/checkpoint_epoch{args.epoch}.pt'
    learned_model = base_dir / f'runs/asu_learned_v2/checkpoint_epoch{args.epoch}.pt'
    metadata_path = base_dir / 'data/asu/chip_metadata.json'

    output_path = base_dir / args.output
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Check files exist
    for path in [data_path, baseline_model, learned_model, metadata_path]:
        if not path.exists():
            print(f"✗ Missing file: {path}")
            return 1

    visualize_chips(
        chip_indices=args.chips,
        data_path=data_path,
        baseline_model_path=baseline_model,
        learned_model_path=learned_model,
        metadata_path=metadata_path,
        output_path=output_path,
        device=args.device
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
