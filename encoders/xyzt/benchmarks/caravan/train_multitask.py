#!/usr/bin/env python3
"""
Multi-task Caravan Benchmark Training Script.

Streamflow + Precipitation + Temperature prediction using Earth4D.

Implements Lance's loss balancing methodology:
1. Track individual loss terms (streamflow, precip, temp) separately
2. Run first epoch/checkpoint to observe raw loss magnitudes
3. Compute balancing coefficients so A*loss_a ≈ B*loss_b ≈ C*loss_c
4. Prioritize primary task (streamflow) by slightly increasing its coefficient

Usage:
    python -m benchmarks.caravan.train_multitask --epochs 50 --output-dir ./outputs
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Parent package imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Local imports
from .constants import DEFAULT_CARAVAN_CSV_PATH
from .data_multitask import CaravanDatasetMultitask
from .model_multitask import StreamflowMultitaskModel
from .data import compute_streamflow_metrics
from .utils import MetricsEMA


def create_multitask_loss_fn(loss_weights=None):
    """
    Create multi-task loss function.

    Args:
        loss_weights: Dict with keys 'streamflow', 'precipitation', 'temperature'
                     If None, uses equal weights (1.0, 1.0, 1.0)
    """
    if loss_weights is None:
        loss_weights = {'streamflow': 1.0, 'precipitation': 1.0, 'temperature': 1.0}

    criterion = nn.MSELoss()

    def loss_fn(predictions, targets, batch_data, model):
        """
        Compute multi-task loss.

        Args:
            predictions: Dict with 'streamflow', 'precipitation', 'temperature' tensors
            targets: Dict with same keys
            batch_data: Batch data dictionary
            model: Model instance

        Returns:
            Dict with individual loss terms and total loss
        """
        # Compute individual task losses
        loss_streamflow = criterion(predictions['streamflow'], targets['streamflow'])
        loss_precipitation = criterion(predictions['precipitation'], targets['precipitation'])
        loss_temperature = criterion(predictions['temperature'], targets['temperature'])

        # Weighted combination
        total_loss = (
            loss_weights['streamflow'] * loss_streamflow +
            loss_weights['precipitation'] * loss_precipitation +
            loss_weights['temperature'] * loss_temperature
        )

        return {
            '_total_loss_tensor': total_loss,
            'task_loss': total_loss,
            'loss_streamflow': loss_streamflow,
            'loss_precipitation': loss_precipitation,
            'loss_temperature': loss_temperature,
            'weighted_loss_streamflow': loss_weights['streamflow'] * loss_streamflow,
            'weighted_loss_precipitation': loss_weights['precipitation'] * loss_precipitation,
            'weighted_loss_temperature': loss_weights['temperature'] * loss_temperature,
        }

    return loss_fn


def train_epoch_multitask(
    model, dataset, indices, optimizer, loss_fn, batch_size=4096, device='cuda'
):
    """Train one epoch with multi-task loss."""
    model.train()

    # Shuffle indices
    perm = torch.randperm(len(indices), device=device)
    shuffled_indices = indices[perm]

    n_batches = (len(indices) + batch_size - 1) // batch_size

    epoch_losses = {
        'total': 0.0,
        'streamflow': 0.0,
        'precipitation': 0.0,
        'temperature': 0.0,
        'weighted_streamflow': 0.0,
        'weighted_precipitation': 0.0,
        'weighted_temperature': 0.0,
    }

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = shuffled_indices[start_idx:end_idx]

        # Get batch data
        batch_data = dataset.get_batch_data(batch_indices)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_data)

        # Compute loss
        loss_dict = loss_fn(predictions, batch_data['targets'], batch_data, model)
        total_loss = loss_dict['_total_loss_tensor']

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Accumulate losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['streamflow'] += loss_dict['loss_streamflow'].item()
        epoch_losses['precipitation'] += loss_dict['loss_precipitation'].item()
        epoch_losses['temperature'] += loss_dict['loss_temperature'].item()
        epoch_losses['weighted_streamflow'] += loss_dict['weighted_loss_streamflow'].item()
        epoch_losses['weighted_precipitation'] += loss_dict['weighted_loss_precipitation'].item()
        epoch_losses['weighted_temperature'] += loss_dict['weighted_loss_temperature'].item()

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= n_batches

    return epoch_losses


def evaluate_multitask(model, dataset, indices, batch_size=8192):
    """Evaluate multi-task model."""
    model.eval()

    all_preds_streamflow = []
    all_preds_precipitation = []
    all_preds_temperature = []
    all_targets_streamflow = []
    all_targets_precipitation = []
    all_targets_temperature = []
    all_streamflow_raw = []

    n_batches = (len(indices) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]

            batch_data = dataset.get_batch_data(batch_indices)
            predictions = model(batch_data)

            all_preds_streamflow.append(predictions['streamflow'])
            all_preds_precipitation.append(predictions['precipitation'])
            all_preds_temperature.append(predictions['temperature'])
            all_targets_streamflow.append(batch_data['targets']['streamflow'])
            all_targets_precipitation.append(batch_data['targets']['precipitation'])
            all_targets_temperature.append(batch_data['targets']['temperature'])
            all_streamflow_raw.append(batch_data['streamflow_raw'])

    # Concatenate all predictions
    preds_streamflow = torch.cat(all_preds_streamflow)
    preds_precipitation = torch.cat(all_preds_precipitation)
    preds_temperature = torch.cat(all_preds_temperature)
    targets_streamflow = torch.cat(all_targets_streamflow)
    targets_precipitation = torch.cat(all_targets_precipitation)
    targets_temperature = torch.cat(all_targets_temperature)
    streamflow_raw = torch.cat(all_streamflow_raw)

    # Compute streamflow metrics (primary task)
    # Note: compute_streamflow_metrics expects (predictions, targets, batch_data)
    # but we just need raw streamflow values for denormalization
    # Create minimal batch_data for compatibility
    batch_data_for_metrics = {'streamflow_raw': streamflow_raw}
    metrics = compute_streamflow_metrics(preds_streamflow, targets_streamflow, batch_data_for_metrics)

    # Compute auxiliary task MSE
    mse_precipitation = nn.MSELoss()(preds_precipitation, targets_precipitation).item()
    mse_temperature = nn.MSELoss()(preds_temperature, targets_temperature).item()

    metrics['precip_mse'] = mse_precipitation
    metrics['temp_mse'] = mse_temperature

    return metrics


def run_training_session(args) -> Dict[str, Any]:
    """Run complete multi-task training session."""

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}", flush=True)

    # Load dataset
    print(f"\n{'='*80}", flush=True)
    print("LOADING DATASET", flush=True)
    print(f"{'='*80}\n", flush=True)

    dataset = CaravanDatasetMultitask(
        data_path=args.data_path,
        device=device,
        use_temporal_split=True
    )

    # Create model
    print(f"\n{'='*80}", flush=True)
    print("CREATING MODEL", flush=True)
    print(f"{'='*80}\n", flush=True)

    model = StreamflowMultitaskModel(
        n_basins=dataset.n_basins,
        basin_dim=args.basin_dim,
        use_adaptive_range=args.adaptive_range,
        verbose=True,
        coordinate_system=args.coordinate_system,
        resolution_mode=args.resolution_mode,
        base_temporal_resolution=args.base_temporal_res,
        temporal_growth_factor=args.temporal_growth,
        latlon_growth_factor=args.latlon_growth,
        elev_growth_factor=args.elev_growth,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:", flush=True)
    print(f"  Total: {total_params:,}", flush=True)
    print(f"  Trainable: {trainable_params:,}", flush=True)

    # Create optimizer
    if args.fused_adam:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=True
        )
        print(f"Using Fused Adam optimizer (10x faster)", flush=True)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Initialize loss weights
    loss_weights = {
        'streamflow': args.weight_streamflow,
        'precipitation': args.weight_precipitation,
        'temperature': args.weight_temperature,
    }

    loss_fn = create_multitask_loss_fn(loss_weights)

    print(f"\nLoss weights:", flush=True)
    print(f"  Streamflow: {loss_weights['streamflow']}", flush=True)
    print(f"  Precipitation: {loss_weights['precipitation']}", flush=True)
    print(f"  Temperature: {loss_weights['temperature']}", flush=True)

    # Training loop
    print(f"\n{'='*80}", flush=True)
    print("TRAINING", flush=True)
    print(f"{'='*80}\n", flush=True)

    metrics_history = []
    best_test_nse = -float('inf')

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # Train
        train_losses = train_epoch_multitask(
            model, dataset, dataset.train_indices, optimizer, loss_fn,
            batch_size=args.batch_size, device=device
        )

        # Evaluate
        train_metrics = evaluate_multitask(model, dataset, dataset.train_indices)
        test_metrics = evaluate_multitask(model, dataset, dataset.test_indices)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)", flush=True)
        print(f"  Loss (raw):      Flow={train_losses['streamflow']:.6f}  "
              f"Precip={train_losses['precipitation']:.6f}  "
              f"Temp={train_losses['temperature']:.6f}", flush=True)
        print(f"  Loss (weighted): Flow={train_losses['weighted_streamflow']:.6f}  "
              f"Precip={train_losses['weighted_precipitation']:.6f}  "
              f"Temp={train_losses['weighted_temperature']:.6f}", flush=True)
        print(f"  Train:  NSE={train_metrics['nse']:.4f}  MAE={train_metrics['mae']:.4f}  "
              f"Precip_MSE={train_metrics['precip_mse']:.6f}  Temp_MSE={train_metrics['temp_mse']:.6f}", flush=True)
        print(f"  Test:   NSE={test_metrics['nse']:.4f}  MAE={test_metrics['mae']:.4f}  "
              f"Precip_MSE={test_metrics['precip_mse']:.6f}  Temp_MSE={test_metrics['temp_mse']:.6f}", flush=True)

        # Save metrics
        metrics_row = {
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            'loss_total': train_losses['total'],
            'loss_streamflow_raw': train_losses['streamflow'],
            'loss_precipitation_raw': train_losses['precipitation'],
            'loss_temperature_raw': train_losses['temperature'],
            'loss_streamflow_weighted': train_losses['weighted_streamflow'],
            'loss_precipitation_weighted': train_losses['weighted_precipitation'],
            'loss_temperature_weighted': train_losses['weighted_temperature'],
            'train_nse': train_metrics['nse'],
            'train_mae': train_metrics['mae'],
            'train_precip_mse': train_metrics['precip_mse'],
            'train_temp_mse': train_metrics['temp_mse'],
            'test_nse': test_metrics['nse'],
            'test_mae': test_metrics['mae'],
            'test_precip_mse': test_metrics['precip_mse'],
            'test_temp_mse': test_metrics['temp_mse'],
        }
        metrics_history.append(metrics_row)

        # Save checkpoint if best
        if test_metrics['nse'] > best_test_nse:
            best_test_nse = test_metrics['nse']
            checkpoint_path = args.output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_nse': best_test_nse,
                'loss_weights': loss_weights,
            }, checkpoint_path)
            print(f"  → Saved best model (NSE={best_test_nse:.4f})", flush=True)

    # Save final model
    final_model_path = args.output_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_weights': loss_weights,
    }, final_model_path)

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_history)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = args.output_dir / f'metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}", flush=True)

    # Print loss balancing analysis
    print(f"\n{'='*80}", flush=True)
    print("LOSS BALANCING ANALYSIS (for next experiment)", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Use first epoch losses as baseline
    first_epoch = metrics_df.iloc[0]
    loss_flow = first_epoch['loss_streamflow_raw']
    loss_precip = first_epoch['loss_precipitation_raw']
    loss_temp = first_epoch['loss_temperature_raw']

    print(f"Raw loss magnitudes (Epoch 1):", flush=True)
    print(f"  Streamflow:    {loss_flow:.6f}", flush=True)
    print(f"  Precipitation: {loss_precip:.6f}", flush=True)
    print(f"  Temperature:   {loss_temp:.6f}", flush=True)

    # Compute balancing coefficients (normalize to streamflow)
    coef_flow = 1.0
    coef_precip = loss_flow / loss_precip if loss_precip > 0 else 1.0
    coef_temp = loss_flow / loss_temp if loss_temp > 0 else 1.0

    print(f"\nSuggested balancing coefficients (A*a ≈ B*b ≈ C*c):", flush=True)
    print(f"  Streamflow:    {coef_flow:.4f}", flush=True)
    print(f"  Precipitation: {coef_precip:.4f}", flush=True)
    print(f"  Temperature:   {coef_temp:.4f}", flush=True)

    # Prioritize streamflow by 1.5x
    coef_flow_prioritized = coef_flow * 1.5

    print(f"\nPrioritized coefficients (streamflow × 1.5):", flush=True)
    print(f"  Streamflow:    {coef_flow_prioritized:.4f}", flush=True)
    print(f"  Precipitation: {coef_precip:.4f}", flush=True)
    print(f"  Temperature:   {coef_temp:.4f}", flush=True)

    print(f"\nNext experiment command:", flush=True)
    print(f"  --weight-streamflow {coef_flow_prioritized:.4f} \\", flush=True)
    print(f"  --weight-precipitation {coef_precip:.4f} \\", flush=True)
    print(f"  --weight-temperature {coef_temp:.4f}", flush=True)

    return {
        'best_test_nse': best_test_nse,
        'metrics_history': metrics_history,
        'suggested_weights': {
            'streamflow': coef_flow_prioritized,
            'precipitation': coef_precip,
            'temperature': coef_temp,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-task Caravan Benchmark Training')

    # Data
    parser.add_argument('--data-path', type=str,
                       default='benchmarks/caravan/data/caravan_alzhanov_147basins_multitask.csv',
                       help='Path to multi-task dataset CSV')

    # Model architecture
    parser.add_argument('--basin-dim', type=int, default=256,
                       help='Dimension of basin embeddings')
    parser.add_argument('--adaptive-range', action='store_true',
                       help='Use adaptive range for Earth4D')

    # Earth4D coordinate system
    parser.add_argument('--coordinate-system', type=str, default='ecef',
                       choices=['ecef', 'geographic'],
                       help='Coordinate system: ecef or geographic')
    parser.add_argument('--resolution-mode', type=str, default='balanced',
                       help='Resolution scaling mode')
    parser.add_argument('--base-temporal-res', type=float, default=32.0,
                       help='Base temporal resolution')
    parser.add_argument('--temporal-growth', type=float, default=2.0,
                       help='Temporal growth factor')
    parser.add_argument('--latlon-growth', type=float, default=None,
                       help='Lat/lon growth factor')
    parser.add_argument('--elev-growth', type=float, default=None,
                       help='Elevation growth factor')
    parser.add_argument('--lat-coverage', type=float, default=None,
                       help='Latitude coverage (geographic mode)')
    parser.add_argument('--lon-coverage', type=float, default=None,
                       help='Longitude coverage (geographic mode)')
    parser.add_argument('--elev-coverage', type=float, default=None,
                       help='Elevation coverage (geographic mode)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.00025,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                       help='Weight decay')
    parser.add_argument('--fused-adam', action='store_true',
                       help='Use fused Adam optimizer')

    # Multi-task loss weights
    parser.add_argument('--weight-streamflow', type=float, default=1.0,
                       help='Streamflow loss weight')
    parser.add_argument('--weight-precipitation', type=float, default=1.0,
                       help='Precipitation loss weight')
    parser.add_argument('--weight-temperature', type=float, default=1.0,
                       help='Temperature loss weight')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints and metrics')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')

    args = parser.parse_args()

    # Handle geographic coverage args
    if args.coordinate_system == 'geographic':
        if args.lat_coverage is not None:
            args.latlon_growth = args.lat_coverage
        if args.lon_coverage is not None:
            args.latlon_growth = args.lon_coverage  # Simplified for now
        if args.elev_coverage is not None:
            args.elev_growth = args.elev_coverage

    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}", flush=True)
    print("MULTI-TASK CARAVAN BENCHMARK", flush=True)
    print(f"{'='*80}\n", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Dataset: {args.data_path}", flush=True)
    print(f"Coordinate system: {args.coordinate_system}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Learning rate: {args.lr}", flush=True)
    print(f"Basin embedding dim: {args.basin_dim}", flush=True)

    # Run training
    results = run_training_session(args)

    print(f"\n{'='*80}", flush=True)
    print("TRAINING COMPLETE", flush=True)
    print(f"{'='*80}\n", flush=True)
    print(f"Best test NSE: {results['best_test_nse']:.4f}", flush=True)


if __name__ == '__main__':
    main()
