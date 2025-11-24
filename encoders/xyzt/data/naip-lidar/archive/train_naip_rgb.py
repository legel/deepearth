#!/usr/bin/env python3
"""
Train Earth4D to reconstruct RGB from NAIP aerial imagery

Training objective: Input (x,y,z,t) → Output (R,G,B)
- x = latitude
- y = longitude
- z = elevation (canopy height)
- t = timestamp
- R,G,B = aerial imagery colors

This demonstrates Earth4D learning real geospatial patterns!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from earth4d import Earth4D


class NAIPDataset(Dataset):
    """Dataset for NAIP RGB reconstruction"""

    def __init__(self, data_path, normalize_coords=True):
        """
        Args:
            data_path: Path to .pt file with tensor shape (N, 7)
            normalize_coords: Normalize input coordinates to [-1, 1]
        """
        print(f"Loading dataset from {data_path}...")
        data_dict = torch.load(data_path, weights_only=False)
        self.data = data_dict['data']
        self.metadata = data_dict['metadata']
        self.normalize_coords = normalize_coords

        # Split into inputs (x,y,z,t) and targets (r,g,b)
        self.coords = self.data[:, :4]  # lat, lon, elev, time
        self.rgb = self.data[:, 4:]     # r, g, b

        print(f"  Loaded {len(self.data):,} points")
        print(f"  Coords shape: {self.coords.shape}")
        print(f"  RGB shape: {self.rgb.shape}")

        # Compute normalization statistics
        if normalize_coords:
            self.coord_mean = self.coords.mean(dim=0)
            self.coord_std = self.coords.std(dim=0)
            # Avoid division by zero
            self.coord_std[self.coord_std < 1e-6] = 1.0
            print(f"  Coord means: {self.coord_mean.numpy()}")
            print(f"  Coord stds: {self.coord_std.numpy()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords = self.coords[idx]
        rgb = self.rgb[idx]

        if self.normalize_coords:
            coords = (coords - self.coord_mean) / self.coord_std

        return coords, rgb


class RGBReconstructionModel(nn.Module):
    """Earth4D + MLP head for RGB reconstruction"""

    def __init__(self, enable_learned_probing=False, probing_range=4):
        super().__init__()

        self.earth4d = Earth4D(
            verbose=True,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
        )

        # Get Earth4D output dimension
        # Earth4D with 24 levels × 4 grids × 2 features = 192 features
        earth4d_dim = self.earth4d.encoder.output_dim

        # MLP head for RGB prediction
        self.rgb_head = nn.Sequential(
            nn.Linear(earth4d_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # R, G, B
            nn.Sigmoid()  # Output in [0, 1]
        )

        print(f"\nModel Architecture:")
        print(f"  Earth4D output dim: {earth4d_dim}")
        print(f"  Learned probing: {enable_learned_probing}")
        if enable_learned_probing:
            print(f"  Probing range: {probing_range}")

    def forward(self, coords):
        """
        Args:
            coords: (B, 4) tensor with [lat, lon, elev, time]

        Returns:
            rgb: (B, 3) tensor with [r, g, b] in [0, 1]
        """
        features = self.earth4d(coords)
        rgb = self.rgb_head(features)
        return rgb


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (coords, rgb_target) in enumerate(dataloader):
        coords = coords.to(device)
        rgb_target = rgb_target.to(device)

        # Forward pass
        optimizer.zero_grad()
        rgb_pred = model(coords)
        loss = criterion(rgb_pred, rgb_target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for coords, rgb_target in dataloader:
            coords = coords.to(device)
            rgb_target = rgb_target.to(device)

            rgb_pred = model(coords)
            loss = criterion(rgb_pred, rgb_target)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


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
    """
    Train RGB reconstruction model

    Args:
        data_path: Path to dataset .pt file
        output_dir: Directory to save outputs
        enable_learned_probing: Use learned probing in Earth4D
        probing_range: Number of probe slots
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        train_split: Fraction of data for training
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("NAIP RGB Reconstruction Training")
    print("="*70)
    print(f"Learned probing: {enable_learned_probing}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load dataset
    dataset = NAIPDataset(data_path, normalize_coords=True)

    # Train/val split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    print(f"\nDataset split:")
    print(f"  Train: {n_train:,} points")
    print(f"  Val: {n_val:,} points")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    model = RGBReconstructionModel(
        enable_learned_probing=enable_learned_probing,
        probing_range=probing_range
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("="*70)

    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch)

        # Save best model
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
                }
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Final train loss: {train_loss:.6f}")
    print(f"  Final val loss: {val_loss:.6f}")
    print(f"  Outputs saved to: {output_dir}")
    print("="*70)

    return history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NAIP RGB reconstruction')
    parser.add_argument('--data', required=True, help='Path to dataset .pt file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--learned-probing', action='store_true', help='Enable learned probing')
    parser.add_argument('--probing-range', type=int, default=4, help='Probing range')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8192, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        enable_learned_probing=args.learned_probing,
        probing_range=args.probing_range,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
