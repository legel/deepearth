#!/usr/bin/env python3
"""
Earth4D with Input Features - Training Script.

Streamflow prediction using Earth4D + meteorological inputs (precipitation, temperature, snow).
Implements Lance's multi-modal fusion architecture.

Usage:
    python -m benchmarks.caravan.train_inputs --epochs 50 --batch-size 4096
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local imports
from .data_inputs import CaravanDatasetInputs
from .model_inputs import Earth4DWithInputs
from .data import compute_streamflow_metrics


def train_epoch(model, dataset, optimizer, batch_size, device):
    """Train one epoch."""
    model.train()
    train_indices = torch.randperm(dataset.train_size, device=device)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(train_indices), batch_size):
        batch_indices = train_indices[i:i + batch_size]
        batch_data = dataset.get_batch_data(batch_indices, split='train')

        # Forward pass
        predictions = model(batch_data)
        targets = batch_data['streamflow']

        # Compute loss
        loss = nn.functional.mse_loss(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataset, split='test'):
    """Evaluate model on train or test set."""
    model.eval()

    if split == 'train':
        all_indices = torch.arange(dataset.train_size, device='cuda')
    else:
        all_indices = torch.arange(dataset.test_size, device='cuda')

    all_preds = []
    all_targets = []
    all_targets_raw = []

    batch_size = 8192  # Larger batch for evaluation
    with torch.no_grad():
        for i in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[i:i + batch_size]
            batch_data = dataset.get_batch_data(batch_indices, split=split)

            predictions = model(batch_data)
            targets = batch_data['streamflow']
            targets_raw = batch_data['streamflow_raw']

            all_preds.append(predictions)
            all_targets.append(targets)
            all_targets_raw.append(targets_raw)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_targets_raw = torch.cat(all_targets_raw)

    # Denormalize predictions for metrics (from log-normalized back to mm/day)
    preds_log = all_preds * dataset.streamflow_max_log
    preds_mm = (torch.exp(preds_log) - 1.0).clamp(min=0)

    # Compute metrics (compute_streamflow_metrics needs batch_data, but we just need NSE/MAE)
    # Simplified metrics computation
    nse = 1 - torch.sum((all_targets_raw - preds_mm) ** 2) / torch.sum((all_targets_raw - all_targets_raw.mean()) ** 2)
    mae = torch.mean(torch.abs(all_targets_raw - preds_mm))
    mse_loss = nn.functional.mse_loss(all_preds, all_targets)

    metrics = {
        'nse': nse.item(),
        'mae': mae.item(),
        'mse_loss': mse_loss.item(),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Earth4D with input features')
    parser.add_argument('--data-path', type=str,
                       default='benchmarks/caravan/data/caravan_alzhanov_147basins_inputs.csv',
                       help='Path to input features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='benchmarks/caravan/outputs/inputs_ecef',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--basin-dim', type=int, default=256, help='Basin embedding dimension')
    parser.add_argument('--feature-dim', type=int, default=32, help='Feature embedding dimension')
    parser.add_argument('--coordinate-system', type=str, default='ecef',
                       choices=['ecef', 'latlon'], help='Coordinate system')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--fused-adam', action='store_true', help='Use fused Adam optimizer')

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda'
    print("=" * 80)
    print("EARTH4D WITH INPUT FEATURES")
    print("=" * 80)
    print(f"Architecture: (x,y,z,t) + P + T + Snow → streamflow")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset: {args.data_path}")
    print("")

    # Load dataset
    dataset = CaravanDatasetInputs(
        csv_path=args.data_path,
        coordinate_system=args.coordinate_system,
        device=device,
    )

    # Create model
    model = Earth4DWithInputs(
        basin_dim=args.basin_dim,
        num_basins=dataset.num_basins,
        feature_embedding_dim=args.feature_dim,
        coordinate_system=args.coordinate_system,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Optimizer
    if args.fused_adam:
        try:
            from apex.optimizers import FusedAdam
            optimizer = FusedAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print("Using Fused Adam optimizer (10x faster)")
        except ImportError:
            print("apex not available, falling back to standard Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print("")

    best_test_nse = -float('inf')
    metrics_list = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, dataset, optimizer, args.batch_size, device)

        # Evaluate
        train_metrics = evaluate(model, dataset, split='train')
        test_metrics = evaluate(model, dataset, split='test')

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train:  NSE={train_metrics['nse']:.4f}  "
              f"MAE={train_metrics['mae']:.4f}  "
              f"Loss={train_metrics['mse_loss']:.6f}")
        print(f"  Test:   NSE={test_metrics['nse']:.4f}  "
              f"MAE={test_metrics['mae']:.4f}  "
              f"Loss={test_metrics['mse_loss']:.6f}")

        # Save best model
        if test_metrics['nse'] > best_test_nse:
            best_test_nse = test_metrics['nse']
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  → Saved best model (NSE={best_test_nse:.4f})")

        # Record metrics
        metrics_list.append({
            'epoch': epoch,
            'epoch_time': epoch_time,
            'train_nse': train_metrics['nse'],
            'train_mae': train_metrics['mae'],
            'train_loss': train_metrics['mse_loss'],
            'test_nse': test_metrics['nse'],
            'test_mae': test_metrics['mae'],
            'test_loss': test_metrics['mse_loss'],
        })

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')

    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = output_dir / f'metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest test NSE: {best_test_nse:.4f}")
    print("")


if __name__ == '__main__':
    main()
