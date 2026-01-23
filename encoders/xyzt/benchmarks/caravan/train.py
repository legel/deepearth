#!/usr/bin/env python3
"""
Caravan Benchmark Training Script.

Streamflow prediction using Earth4D positional encoding with basin embeddings.

Usage:
    python -m benchmarks.caravan.train --epochs 500 --output-dir ./outputs
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
from ops import (
    train_epoch, train_epoch_precomputed, train_epoch_fused_adam, evaluate, compute_loss
)

# Local imports
from .constants import DEFAULT_CARAVAN_CSV_PATH
from .data import CaravanDataset, get_temporal_splits, compute_streamflow_metrics
from .model import StreamflowModel
from .utils import MetricsEMA, print_sample_predictions


def create_streamflow_loss_fn(model):
    """Create streamflow-specific loss function."""
    criterion = nn.MSELoss()

    def loss_fn(preds, targets, batch_data, mdl):
        if hasattr(mdl.earth4d, 'compute_loss'):
            return mdl.earth4d.compute_loss(
                preds, targets,
                enable_probe_entropy_loss=True,
                probe_entropy_weight=0.5
            )
        else:
            loss = criterion(preds, targets)
            return {'_total_loss_tensor': loss, 'task_loss': loss}

    return loss_fn


def create_probe_update_fn(model):
    """Create callback for learned probing updates."""
    if not hasattr(model.earth4d.xyz_encoder, 'update_probe_indices'):
        return None

    def post_batch_fn():
        model.earth4d.xyz_encoder.update_probe_indices()
        model.earth4d.xyt_encoder.update_probe_indices()
        model.earth4d.yzt_encoder.update_probe_indices()
        model.earth4d.xzt_encoder.update_probe_indices()

    return post_batch_fn


def evaluate_with_samples(model, dataset, indices):
    """Evaluate and return metrics plus sample predictions."""
    result = evaluate(model, dataset, indices, metrics_fn=compute_streamflow_metrics)

    metrics = result['metrics']
    preds = result['predictions']
    targets = result['targets']
    batch_data = result['batch_data']

    # Get sample predictions (denormalize to mm/day)
    from .constants import MAX_LOG_STREAMFLOW

    sample_idx = torch.randperm(len(preds))[:5]
    sample_preds_norm = preds[sample_idx]
    sample_targets_norm = targets[sample_idx]

    # Denormalize
    sample_preds_log = sample_preds_norm * MAX_LOG_STREAMFLOW
    sample_targets_log = sample_targets_norm * MAX_LOG_STREAMFLOW

    sample_preds_mm = (torch.exp(sample_preds_log) - 1.0).clamp(min=0).cpu().numpy()
    sample_targets_mm = (torch.exp(sample_targets_log) - 1.0).clamp(min=0).cpu().numpy()

    return metrics, sample_targets_mm.tolist(), sample_preds_mm.tolist()


def run_training_session(args) -> Dict[str, Any]:
    """Run complete training session."""

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda'
    print(f"Random seed: {args.seed}", flush=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("EARTH4D CARAVAN BENCHMARK", flush=True)
    print("Streamflow prediction from (x,y,z,t) + basin embeddings", flush=True)
    print("=" * 80, flush=True)

    # Load dataset
    dataset = CaravanDataset(args.data_path, device, use_temporal_split=True)
    splits = get_temporal_splits(dataset)

    # Model - with geographic coordinate system support
    coordinate_system = getattr(args, 'coordinate_system', 'ecef')
    resolution_mode = getattr(args, 'resolution_mode', 'balanced')
    base_temporal_res = getattr(args, 'base_temporal_res', 32.0)
    temporal_growth = getattr(args, 'temporal_growth', 2.0)
    latlon_growth = getattr(args, 'latlon_growth', None)
    elev_growth = getattr(args, 'elev_growth', None)

    model = StreamflowModel(
        dataset.n_basins,
        basin_dim=args.basin_dim,
        use_adaptive_range=getattr(args, 'use_adaptive_range', False),
        verbose=True,
        coordinate_system=coordinate_system,
        resolution_mode=resolution_mode,
        base_temporal_resolution=base_temporal_res,
        temporal_growth_factor=temporal_growth,
        latlon_growth_factor=latlon_growth,
        elev_growth_factor=elev_growth,
    ).to(device)

    # Fit geographic range if using geographic coordinates
    if coordinate_system == 'geographic':
        lat_cov = getattr(args, 'lat_coverage', 0.25)
        lon_cov = getattr(args, 'lon_coverage', 0.25)
        elev_cov = getattr(args, 'elev_coverage', 0.15)
        time_cov = getattr(args, 'time_coverage', 1.0)
        print(f"\nFitting geographic range (lat={lat_cov}, lon={lon_cov}, elev={elev_cov}, time={time_cov})...", flush=True)
        all_coords = dataset.coords.cpu()
        model.earth4d.fit_geo_range(all_coords, lat_cov, lon_cov, elev_cov, time_cov)

    # Precompute if enabled
    use_precomputed = not getattr(args, 'no_precomputed', False)
    if use_precomputed:
        print(f"\nPrecomputing hash indices for {len(dataset.coords):,} coordinates...", flush=True)
        precomp_stats = model.earth4d.precompute(dataset.coords)
        print(f"  Precomputation: {precomp_stats['total_mb']:.1f} MB", flush=True)

    # Count parameters
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters())
    basin_params = sum(p.numel() for p in model.basin_embeddings.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel architecture:", flush=True)
    print(f"  Earth4D parameters: {earth4d_params:,}", flush=True)
    print(f"  Basin embedding parameters: {basin_params:,} ({dataset.n_basins} basins × {args.basin_dim} dims)", flush=True)
    print(f"  MLP parameters: {mlp_params:,}", flush=True)
    print(f"  Total parameters: {total_params:,}", flush=True)

    # Check for fused Adam mode
    use_fused_adam = getattr(args, 'fused_adam', False)

    # Configure optimizer
    weight_decay = getattr(args, 'weight_decay', 0.001)

    if use_fused_adam:
        # Fused mode: MLP and basin embeddings with AdamW
        # Encoder embeddings updated via fused CUDA kernel
        mlp_params = list(model.mlp.parameters()) + list(model.basin_embeddings.parameters())
        index_lr_multiplier = 10.0
        optimizer_params = [
            {'params': mlp_params, 'lr': args.lr},
        ]
        # Add index_logits for learned probing (with higher LR)
        if hasattr(model.earth4d.xyz_encoder, 'index_logits'):
            optimizer_params.extend([
                {'params': model.earth4d.xyz_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
                {'params': model.earth4d.xyt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
                {'params': model.earth4d.yzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
                {'params': model.earth4d.xzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            ])
        try:
            optimizer = optim.AdamW(optimizer_params, weight_decay=weight_decay, fused=True)
        except TypeError:
            optimizer = optim.AdamW(optimizer_params, weight_decay=weight_decay)
        print(f"\n  Fused Adam mode: encoder embeddings updated via CUDA kernel", flush=True)
        print(f"  Weight decay: {weight_decay}", flush=True)
    else:
        # Standard mode: all parameters with AdamW
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        print(f"\n  Standard AdamW optimizer", flush=True)
        print(f"  Weight decay: {weight_decay}", flush=True)

    # Callbacks
    loss_fn = create_streamflow_loss_fn(model)
    post_batch_fn = create_probe_update_fn(model)

    # Tracking
    metrics_history = []
    metrics_ema = MetricsEMA(alpha=0.1)

    print("\n" + "=" * 80, flush=True)
    print("TRAINING", flush=True)
    print("-" * 80, flush=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        if use_fused_adam:
            # Fused Adam: 10x faster
            train_result = train_epoch_fused_adam(
                model, dataset, splits['train'], optimizer, args.batch_size,
                loss_fn=loss_fn, metrics_fn=compute_streamflow_metrics, post_batch_fn=post_batch_fn,
                lr=args.lr, weight_decay=weight_decay
            )
        elif use_precomputed:
            train_result = train_epoch_precomputed(
                model, dataset, splits['train'], optimizer, args.batch_size,
                loss_fn=loss_fn, metrics_fn=compute_streamflow_metrics, post_batch_fn=post_batch_fn
            )
        else:
            train_result = train_epoch(
                model, dataset, splits['train'], optimizer, args.batch_size,
                loss_fn=loss_fn, metrics_fn=compute_streamflow_metrics, post_batch_fn=post_batch_fn
            )

        # Evaluate
        test_metrics, test_gt, test_pred = evaluate_with_samples(model, dataset, splits['test'])

        dt = time.time() - t0

        # Track
        current_metrics = {
            'epoch': epoch,
            'time': dt,
            'train_mae': train_result['metrics']['mae'],
            'train_rmse': train_result['metrics']['rmse'],
            'train_r2': train_result['metrics']['r2'],
            'train_nse': train_result['metrics']['nse'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_nse': test_metrics['nse']
        }

        ema_metrics = metrics_ema.update(current_metrics)
        current_metrics.update(ema_metrics)
        metrics_history.append(current_metrics)

        # Print
        print(f"\nEPOCH {epoch:3d} ({dt:.1f}s)", flush=True)
        print(f"  TRAIN: MAE={current_metrics['train_mae']:.3f} mm/day, RMSE={current_metrics['train_rmse']:.3f}, R²={current_metrics['train_r2']:.3f}, NSE={current_metrics['train_nse']:.3f}", flush=True)
        print(f"  TEST:  MAE={current_metrics['test_mae']:.3f} mm/day, RMSE={current_metrics['test_rmse']:.3f}, R²={current_metrics['test_r2']:.3f}, NSE={current_metrics['test_nse']:.3f}", flush=True)

        # Sample predictions
        print_sample_predictions(test_gt, test_pred, prefix="TEST ")

        # LR decay
        for g in optimizer.param_groups:
            g['lr'] *= 0.99995
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

    print("=" * 80, flush=True)

    # Save model
    model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_basins': dataset.n_basins,
        'basin_dim': args.basin_dim,
        'basin_to_idx': dataset.basin_to_idx
    }, model_path)
    print(f"\nModel saved: {model_path}", flush=True)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = output_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved: {metrics_path}", flush=True)

    # Final summary
    final = metrics_history[-1]
    print(f"\nFINAL RESULTS (Epoch {args.epochs}):", flush=True)
    print(f"  Training: MAE={final['train_mae']:.3f} mm/day, RMSE={final['train_rmse']:.3f}, R²={final['train_r2']:.3f}, NSE={final['train_nse']:.3f}", flush=True)
    print(f"  Test:     MAE={final['test_mae']:.3f} mm/day, RMSE={final['test_rmse']:.3f}, R²={final['test_r2']:.3f}, NSE={final['test_nse']:.3f}", flush=True)

    return {
        'test_mae': final['test_mae'],
        'test_rmse': final['test_rmse'],
        'test_r2': final['test_r2'],
        'test_nse': final['test_nse']
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Earth4D Caravan Benchmark')

    # Data arguments
    parser.add_argument('--data-path', default=DEFAULT_CARAVAN_CSV_PATH,
                       help='Path to processed Caravan CSV')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0008,
                       help='Learning rate (matches LFMC default)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')

    # Model arguments
    parser.add_argument('--basin-dim', type=int, default=256,
                       help='Dimension of learnable basin embeddings')

    # Geographic coordinate system arguments (for SoTA performance)
    parser.add_argument('--coordinate-system', type=str, default='ecef',
                       choices=['geographic', 'ecef'],
                       help='Coordinate system: geographic (SoTA) or ecef (legacy)')
    parser.add_argument('--resolution-mode', type=str, default='balanced',
                       help='Resolution mode for hash grid')
    parser.add_argument('--base-temporal-res', type=float, default=32.0,
                       help='Base temporal resolution')
    parser.add_argument('--temporal-growth', type=float, default=2.0,
                       help='Temporal growth factor (1.2 optimal for SoTA)')
    parser.add_argument('--latlon-growth', type=float, default=None,
                       help='Growth factor for lat/lon dimensions')
    parser.add_argument('--elev-growth', type=float, default=None,
                       help='Growth factor for elevation dimension')
    parser.add_argument('--lat-coverage', type=float, default=0.25,
                       help='Latitude coverage for geographic mode (0.05 for SoTA)')
    parser.add_argument('--lon-coverage', type=float, default=0.25,
                       help='Longitude coverage for geographic mode (0.05 for SoTA)')
    parser.add_argument('--elev-coverage', type=float, default=0.15,
                       help='Elevation coverage for geographic mode (0.01 for SoTA)')
    parser.add_argument('--time-coverage', type=float, default=1.0,
                       help='Time coverage for geographic mode')

    # Enhancement flags
    parser.add_argument('--use-adaptive-range', action='store_true',
                       help='Enable adaptive range normalization (fit to data extent)')
    parser.add_argument('--no-precomputed', action='store_true',
                       help='Disable precomputed hash indices (enabled by default)')
    parser.add_argument('--fused-adam', action='store_true',
                       help='Enable fused Adam optimization (10x speedup, recommended)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                       help='Weight decay for regularization')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for model and metrics')

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    run_training_session(args)


if __name__ == "__main__":
    main()
