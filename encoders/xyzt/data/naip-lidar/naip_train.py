#!/usr/bin/env python3
"""
Train Earth4D for NAIP RGB reconstruction with chip-based train/test splits

Ensures zero spatial leakage by splitting on chip boundaries, not random points.

Usage:
    python naip_train.py --data data/asu/parsed_xyztrgb.pt --output runs/asu_baseline
    python naip_train.py --data data/asu/parsed_xyztrgb.pt --learned-probing --output runs/asu_learned
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import argparse

# Add parent directory for imports (earth4d.py is in encoders/xyzt/)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from earth4d import Earth4D


class NAIPDataset(Dataset):
    """NAIP dataset with coordinate normalization"""

    def __init__(self, data_tensor, normalize_coords=True):
        """
        Args:
            data_tensor: (N, 7) tensor [lat, lon, elev, time, r, g, b]
            normalize_coords: Normalize inputs to zero mean, unit std
        """
        self.coords = data_tensor[:, :4]  # lat, lon, elev, time
        self.rgb = data_tensor[:, 4:]     # r, g, b

        if normalize_coords:
            self.coord_mean = self.coords.mean(dim=0)
            self.coord_std = self.coords.std(dim=0)
            self.coord_std[self.coord_std < 1e-6] = 1.0
        else:
            self.coord_mean = None
            self.coord_std = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coords = self.coords[idx]
        rgb = self.rgb[idx]

        if self.coord_mean is not None:
            coords = (coords - self.coord_mean) / self.coord_std

        return coords, rgb


class RGBReconstructionModel(nn.Module):
    """Earth4D + MLP for RGB reconstruction"""

    def __init__(self, enable_learned_probing=False, probing_range=4):
        super().__init__()

        self.earth4d = Earth4D(
            verbose=True,
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


def chip_based_split(chip_sizes, train_ratio=0.8):
    """
    Split data by chips (not points) to avoid spatial leakage

    Args:
        chip_sizes: List of point counts per chip
        train_ratio: Fraction of chips for training

    Returns:
        train_indices, val_indices (lists of point indices)
    """
    n_chips = len(chip_sizes)
    n_train_chips = int(n_chips * train_ratio)

    # Shuffle chips
    chip_indices = np.random.permutation(n_chips)
    train_chips = chip_indices[:n_train_chips]
    val_chips = chip_indices[n_train_chips:]

    # Convert chip indices to point indices
    cumsum = np.cumsum([0] + chip_sizes)

    train_indices = []
    val_indices = []

    for chip_idx in train_chips:
        start = cumsum[chip_idx]
        end = cumsum[chip_idx + 1]
        train_indices.extend(range(start, end))

    for chip_idx in val_chips:
        start = cumsum[chip_idx]
        end = cumsum[chip_idx + 1]
        val_indices.extend(range(start, end))

    return train_indices, val_indices


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for coords, rgb_target in dataloader:
        coords = coords.to(device)
        rgb_target = rgb_target.to(device)

        optimizer.zero_grad()
        rgb_pred = model(coords)
        loss = criterion(rgb_pred, rgb_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for coords, rgb_target in dataloader:
            coords = coords.to(device)
            rgb_target = rgb_target.to(device)

            rgb_pred = model(coords)
            loss = criterion(rgb_pred, rgb_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    data_path,
    output_dir,
    enable_learned_probing=False,
    probing_range=4,
    epochs=25,
    batch_size=8192,
    lr=1e-4,
    train_split=0.8,
    device='cuda'
):
    """Train RGB reconstruction model with chip-based splits"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("NAIP RGB Reconstruction Training")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Learned probing: {enable_learned_probing}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}\n")

    # Load data
    print("Loading dataset...")
    data_dict = torch.load(data_path, weights_only=False)
    data_tensor = data_dict['data']
    chip_sizes = data_dict.get('chip_sizes', None)
    n_chips = data_dict.get('n_chips', 0)

    print(f"  Total points: {len(data_tensor):,}")
    print(f"  Total chips: {n_chips}")

    # Create dataset
    dataset = NAIPDataset(data_tensor, normalize_coords=True)

    # Chip-based split to avoid spatial leakage
    if chip_sizes is not None:
        print(f"\n✓ Using chip-based train/test split (prevents spatial leakage)")
        train_indices, val_indices = chip_based_split(chip_sizes, train_split)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        n_train_chips = int(n_chips * train_split)
        n_val_chips = n_chips - n_train_chips
        print(f"  Train: {len(train_indices):,} points ({n_train_chips} chips)")
        print(f"  Val: {len(val_indices):,} points ({n_val_chips} chips)")
    else:
        print(f"\n⚠ No chip metadata - using random point split (may have spatial leakage)")
        n_train = int(len(dataset) * train_split)
        n_val = len(dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        print(f"  Train: {n_train:,} points")
        print(f"  Val: {n_val:,} points")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = RGBReconstructionModel(
        enable_learned_probing=enable_learned_probing,
        probing_range=probing_range
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("="*70)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch)

        # Save best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'enable_learned_probing': enable_learned_probing,
                    'probing_range': probing_range,
                    'batch_size': batch_size,
                    'lr': lr,
                },
                'train_indices': train_indices if chip_sizes else None,
                'val_indices': val_indices if chip_sizes else None,
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")

        # Save periodic checkpoints for reconstruction quality analysis
        # Save at epochs 10, 20, 30, 40, 50 (every 10 epochs)
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'enable_learned_probing': enable_learned_probing,
                    'probing_range': probing_range,
                    'batch_size': batch_size,
                    'lr': lr,
                },
                'train_indices': train_indices if chip_sizes else None,
                'val_indices': val_indices if chip_sizes else None,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: epoch {epoch}")

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Output: {output_dir}")
    print("="*70)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train NAIP RGB reconstruction')
    parser.add_argument('--data', required=True, help='Path to parsed_xyztrgb.pt')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--learned-probing', action='store_true', help='Enable learned probing')
    parser.add_argument('--probing-range', type=int, default=4, help='Probing range')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8192, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_dir=args.output,
        enable_learned_probing=args.learned_probing,
        probing_range=args.probing_range,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_split=args.train_split,
        device=args.device
    )


if __name__ == '__main__':
    main()
