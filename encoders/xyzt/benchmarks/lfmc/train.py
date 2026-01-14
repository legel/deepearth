#!/usr/bin/env python3
"""
LFMC Benchmark Training Script.

Species-aware Live Fuel Moisture Content prediction using Earth4D positional encoding.

Usage:
    python -m benchmarks.lfmc.train --epochs 100 --output-dir ./outputs
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

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
    train_epoch, train_epoch_precomputed, train_epoch_batched_forward,
    train_epoch_fused_sgd, train_epoch_fused_adam, evaluate, compute_loss
)

# Local package imports
from .constants import MAX_LFMC_VALUE
from .data import FullyGPUDataset, get_ai2_splits, compute_lfmc_metrics
from .model import SpeciesAwareLFMCModel
from .utils import MetricsEMA, print_predictions_table, export_test_predictions_csv
from .visualization import (
    create_error_histogram,
    create_geospatial_visualization,
    create_temporal_visualization,
    create_combined_scientific_figure
)


def create_lfmc_loss_fn(model):
    """
    Create LFMC-specific loss function with optional entropy regularization.

    Returns a loss function callback compatible with generic training.
    Note: Returns tensors (not .item() floats) to support CUDA graph capture.
    """
    criterion = nn.MSELoss()

    def loss_fn(preds, targets, batch_data, mdl):
        # Use compute_loss with entropy regularization if available
        if hasattr(mdl.earth4d, 'compute_loss'):
            return mdl.earth4d.compute_loss(
                preds, targets,
                enable_probe_entropy_loss=True,
                probe_entropy_weight=0.5
            )
        else:
            loss = criterion(preds, targets)
            # Return tensor, not .item() - enables CUDA graph capture
            return {'_total_loss_tensor': loss, 'task_loss': loss}

    return loss_fn


def create_probe_update_fn(model):
    """
    Create post-batch callback for updating probe indices.

    Returns a callback for learned probing update, or None if not applicable.
    """
    if not hasattr(model.earth4d.xyz_encoder, 'update_probe_indices'):
        return None

    def post_batch_fn():
        model.earth4d.xyz_encoder.update_probe_indices()
        model.earth4d.xyt_encoder.update_probe_indices()
        model.earth4d.yzt_encoder.update_probe_indices()
        model.earth4d.xzt_encoder.update_probe_indices()

    return post_batch_fn


def evaluate_with_samples(model, dataset, indices):
    """
    Evaluate model and return metrics plus sample predictions for monitoring.

    Returns:
        Tuple of (overall_metrics, unique_metrics, degen_metrics,
                  sample_gt, sample_pred, sample_types)
    """
    result = evaluate(model, dataset, indices, metrics_fn=compute_lfmc_metrics)

    metrics = result['metrics']
    preds = result['predictions']
    targets = result['targets']
    batch_data = result['batch_data']

    # Extract sample predictions for monitoring
    is_degenerate = batch_data.get('is_degenerate', torch.zeros(len(preds), dtype=torch.bool))
    unique_mask = ~is_degenerate
    degen_mask = is_degenerate

    unique_idx = torch.where(unique_mask)[0]
    degen_idx = torch.where(degen_mask)[0]

    sample_preds = []
    sample_trues = []
    sample_types = []

    # Get up to 3 unique and 2 degenerate samples
    if len(unique_idx) > 0:
        n_unique_samples = min(3, len(unique_idx))
        for i in range(n_unique_samples):
            idx = unique_idx[i * len(unique_idx) // n_unique_samples]
            sample_preds.append(preds[idx].item())
            sample_trues.append(targets[idx].item())
            sample_types.append('U')

    if len(degen_idx) > 0:
        n_degen_samples = min(2, len(degen_idx))
        for i in range(n_degen_samples):
            idx = degen_idx[i * len(degen_idx) // n_degen_samples]
            sample_preds.append(preds[idx].item())
            sample_trues.append(targets[idx].item())
            sample_types.append('D')

    # Split metrics by overall/unique/degenerate
    overall = metrics.get('overall', metrics)
    unique = metrics.get('unique', {'mae': 0, 'r2': 0})
    degen = metrics.get('degenerate', {'mae': 0, 'r2': 0})

    return overall, unique, degen, sample_trues, sample_preds, sample_types


def run_training_session(args, run_name: str = "") -> Dict[str, Any]:
    """
    Run a complete LFMC training session.

    Args:
        args: Parsed command-line arguments
        run_name: Optional suffix for output directory

    Returns:
        Dictionary with final metrics and training statistics
    """
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda'
    print(f"Random seed: {args.seed}", flush=True)

    # Create output directory
    output_suffix = f"_{run_name}" if run_name else ""
    output_dir = Path(args.output_dir + output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"EARTH4D LFMC BENCHMARK", flush=True)
    print(f"Using learnable {args.species_dim}D species embeddings", flush=True)
    print("=" * 80, flush=True)

    # Load dataset with AI2 splits
    dataset = FullyGPUDataset(args.data_path, device, use_ai2_splits=True)
    splits = get_ai2_splits(dataset)

    # Model with learnable species embeddings
    use_adaptive_range = getattr(args, 'use_adaptive_range', False)
    coordinate_system = getattr(args, 'coordinate_system', 'ecef')
    resolution_mode = getattr(args, 'resolution_mode', 'balanced')
    base_temporal_res = getattr(args, 'base_temporal_res', 32)
    temporal_growth = getattr(args, 'temporal_growth', 2.0)
    latlon_growth = getattr(args, 'latlon_growth', None)
    elev_growth = getattr(args, 'elev_growth', None)

    model = SpeciesAwareLFMCModel(
        dataset.n_species,
        species_dim=args.species_dim,
        use_adaptive_range=use_adaptive_range,
        verbose=True,
        coordinate_system=coordinate_system,
        resolution_mode=resolution_mode,
        base_temporal_resolution=float(base_temporal_res),
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

    # Fit adaptive range if enabled (ECEF mode)
    if use_adaptive_range and coordinate_system == 'ecef':
        print("\nFitting adaptive range to training data...", flush=True)
        train_coords = dataset.coords[splits['train']].cpu()
        model.earth4d.fit_range(train_coords, buffer_fraction=0.25)
        print("Analyzing active levels...", flush=True)
        model.earth4d.analyze_levels(train_coords)

    # Check for precomputed mode
    use_precomputed = not getattr(args, 'no_precomputed', False)
    if use_precomputed:
        print(f"\nPrecomputing hash indices for {len(dataset.coords):,} coordinates...", flush=True)
        precomp_stats = model.earth4d.precompute(dataset.coords)
        print(f"  Precomputation complete: {precomp_stats['total_mb']:.1f} MB total", flush=True)
        print(f"  ({precomp_stats['bytes_per_coord']:.1f} bytes/coordinate)", flush=True)

    # Count parameters
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters())
    species_params = sum(p.numel() for p in model.species_embeddings.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:", flush=True)
    print(f"  Earth4D parameters: {earth4d_params:,}", flush=True)
    print(f"  Species embedding parameters: {species_params:,} (Learnable, {dataset.n_species} species × {args.species_dim} dims)", flush=True)
    print(f"  MLP parameters: {mlp_params:,}", flush=True)
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"  Trainable parameters: {trainable_params:,}", flush=True)

    # Check for fused optimization modes
    use_fused_sgd = getattr(args, 'fused_sgd', False)
    use_fused_adam = getattr(args, 'fused_adam', False)

    # Configure optimizer
    weight_decay = getattr(args, 'weight_decay', 0.001)
    if use_fused_sgd or use_fused_adam:
        # Fused mode: MLP, species embeddings, and index_logits with AdamW
        # Encoder embeddings are updated via fused CUDA kernel
        mlp_params = list(model.mlp.parameters()) + list(model.species_embeddings.parameters())
        # Also include index_logits for learned probing (with higher LR)
        index_lr_multiplier = 10.0
        optimizer_params = [
            {'params': mlp_params, 'lr': args.lr},
        ]
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
        mode_name = "Adam" if use_fused_adam else "SGD"
        print(f"\n  Fused {mode_name} mode: encoder embeddings updated via CUDA kernel", flush=True)
        print(f"  Weight decay: {weight_decay}", flush=True)
    elif hasattr(model.earth4d.xyz_encoder, 'index_logits'):
        index_lr_multiplier = 10.0
        optimizer_params = [
            {'params': model.earth4d.xyz_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.xyt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.yzt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.xzt_encoder.embeddings, 'lr': args.lr},
            {'params': model.species_embeddings.parameters(), 'lr': args.lr},
            {'params': model.mlp.parameters(), 'lr': args.lr},
            {'params': model.earth4d.xyz_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.xyt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.yzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.xzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
        ]
        # Use fused AdamW for faster optimizer steps (single CUDA kernel)
        try:
            optimizer = optim.AdamW(optimizer_params, weight_decay=weight_decay, fused=True)
            fused_msg = " (fused)"
        except TypeError:
            optimizer = optim.AdamW(optimizer_params, weight_decay=weight_decay)
            fused_msg = ""
        print(f"\n Using {index_lr_multiplier}x higher LR for index_logits: {args.lr * index_lr_multiplier:.6f}{fused_msg}", flush=True)
    else:
        # Use fused AdamW for faster optimizer steps (single CUDA kernel)
        try:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, fused=True)
        except TypeError:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    # Create callbacks
    loss_fn = create_lfmc_loss_fn(model)
    post_batch_fn = create_probe_update_fn(model)

    # Tracking
    metrics_history = []
    metrics_ema = MetricsEMA(alpha=0.1)
    epoch_times = []

    # Track current LR for fused modes (separate from optimizer which only handles MLP)
    current_encoder_lr = args.lr

    # Determine training mode
    use_batched_forward = getattr(args, 'batched_forward', False)
    use_amp = getattr(args, 'amp', False)

    # Setup GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("\n" + "=" * 80, flush=True)
    print("Training with species embeddings (UNIQUE=Single species, MULTI=Multi-species):", flush=True)
    mode_parts = []
    if use_fused_adam:
        mode_parts.append("fused-adam")
    elif use_fused_sgd:
        mode_parts.append("fused-sgd")
    if use_batched_forward:
        mode_parts.append("batched-forward")
    if use_amp:
        mode_parts.append("AMP")
    if use_precomputed and not use_batched_forward and not use_fused_sgd and not use_fused_adam:
        mode_parts.append("precomputed")
    mode_str = " + ".join(mode_parts) if mode_parts else "standard"
    print(f"  Mode: {mode_str}", flush=True)
    print("-" * 80, flush=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Select training function
        if use_fused_adam:
            train_result = train_epoch_fused_adam(
                model, dataset, splits['train'], optimizer, args.batch_size,
                lr=current_encoder_lr, weight_decay=weight_decay,
                loss_fn=loss_fn, metrics_fn=compute_lfmc_metrics, post_batch_fn=post_batch_fn
            )
        elif use_fused_sgd:
            train_result = train_epoch_fused_sgd(
                model, dataset, splits['train'], optimizer, args.batch_size,
                lr=args.lr, metrics_fn=compute_lfmc_metrics
            )
        elif use_batched_forward and use_precomputed:
            train_result = train_epoch_batched_forward(
                model, dataset, splits['train'], optimizer, args.batch_size,
                loss_fn=loss_fn, metrics_fn=compute_lfmc_metrics, post_batch_fn=post_batch_fn,
                use_amp=use_amp, scaler=scaler
            )
        elif use_precomputed:
            train_result = train_epoch_precomputed(
                model, dataset, splits['train'], optimizer, args.batch_size,
                loss_fn=loss_fn, metrics_fn=compute_lfmc_metrics, post_batch_fn=post_batch_fn
            )
        else:
            train_result = train_epoch(
                model, dataset, splits['train'], optimizer, args.batch_size,
                loss_fn=loss_fn, metrics_fn=compute_lfmc_metrics, post_batch_fn=post_batch_fn,
                use_block_shuffle=False, block_size=1024
            )

        # Extract train metrics
        trn_metrics = train_result['metrics']
        trn_overall = trn_metrics.get('overall', trn_metrics)
        trn_unique = trn_metrics.get('unique', {'mae': 0, 'r2': 0})
        trn_degen = trn_metrics.get('degenerate', {'mae': 0, 'r2': 0})

        # Evaluate test split
        test_overall, test_unique, test_degen, test_gt, test_pred, test_types = \
            evaluate_with_samples(model, dataset, splits['test'])

        dt = time.time() - t0
        epoch_times.append(dt)

        # Track metrics
        current_metrics = {
            'epoch': epoch,
            'time': dt,
            'train_rmse': trn_overall['rmse'],
            'train_mae': trn_overall['mae'],
            'train_r2': trn_overall['r2'],
            'train_unique_mae': trn_unique['mae'],
            'train_unique_r2': trn_unique['r2'],
            'train_degen_mae': trn_degen['mae'],
            'train_degen_r2': trn_degen['r2'],
            'test_rmse': test_overall['rmse'],
            'test_mae': test_overall['mae'],
            'test_r2': test_overall['r2'],
            'test_unique_mae': test_unique['mae'],
            'test_unique_r2': test_unique['r2'],
            'test_degen_mae': test_degen['mae'],
            'test_degen_r2': test_degen['r2']
        }

        ema_metrics = metrics_ema.update(current_metrics)
        current_metrics.update(ema_metrics)
        metrics_history.append(current_metrics)

        # Print formatted metrics
        print(f"\nEPOCH {epoch:3d} ({dt:.1f}s)", flush=True)
        print(f"  TRAIN ALL: [RMSE: {trn_overall['rmse']:5.1f}pp, MAE: {trn_overall['mae']:5.1f}pp, R²: {trn_overall['r2']:.3f}]", flush=True)
        print(f"        UNIQUE: MAE={trn_unique['mae']:5.1f}pp, R²={trn_unique['r2']:.3f}  |  MULTI: MAE={trn_degen['mae']:5.1f}pp, R²={trn_degen['r2']:.3f}", flush=True)

        print(f"\n  TEST: [RMSE: {test_overall['rmse']:5.1f}pp, MAE: {test_overall['mae']:5.1f}pp, R²: {test_overall['r2']:.3f}]", flush=True)
        print(f"        UNIQUE: MAE={test_unique['mae']:5.1f}pp, R²={test_unique['r2']:.3f}  |  MULTI: MAE={test_degen['mae']:5.1f}pp, R²={test_degen['r2']:.3f}", flush=True)

        # Show sample predictions
        print_predictions_table(test_gt, test_pred, test_types)

        # Learning rate decay
        for g in optimizer.param_groups:
            g['lr'] *= 0.99995
        # Also decay encoder LR for fused modes
        if use_fused_adam or use_fused_sgd:
            current_encoder_lr *= 0.99995
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

    print("=" * 80, flush=True)

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_species': dataset.n_species,
        'species_dim': args.species_dim,
        'species_to_idx': dataset.species_to_idx,
        'idx_to_species': dataset.idx_to_species
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}", flush=True)

    # Save metrics history
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = output_dir / f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}", flush=True)

    # Print final summary
    final = metrics_history[-1]
    print(f"\nFINAL RESULTS (Epoch {args.epochs}) - Absolute Errors (LFMC Percentage Points):", flush=True)
    print(f"  Baseline: Galileo (Johnson et al. 2025): RMSE=18.9pp, MAE=12.6pp, R²=0.72", flush=True)
    print(f"\n  Overall Performance:", flush=True)
    print(f"    Training:  RMSE={final['train_rmse']:.1f}pp, MAE={final['train_mae']:.1f}pp, R²={final['train_r2']:.3f}", flush=True)
    print(f"    Test:      RMSE={final['test_rmse']:.1f}pp, MAE={final['test_mae']:.1f}pp, R²={final['test_r2']:.3f}", flush=True)

    print(f"\n  Unique Species Locations Only:", flush=True)
    print(f"    Training:  MAE={final['train_unique_mae']:.1f}pp, R²={final['train_unique_r2']:.3f}", flush=True)
    print(f"    Test:      MAE={final['test_unique_mae']:.1f}pp, R²={final['test_unique_r2']:.3f}", flush=True)

    print(f"\n  Multi-Species Locations Only:", flush=True)
    print(f"    Training:  MAE={final['train_degen_mae']:.1f}pp, R²={final['train_degen_r2']:.3f}", flush=True)
    print(f"    Test:      MAE={final['test_degen_mae']:.1f}pp, R²={final['test_degen_r2']:.3f}", flush=True)

    print("\nTraining complete!", flush=True)

    # Generate visualizations
    print("\nGenerating visualizations...", flush=True)

    with torch.no_grad():
        model.eval()

        if len(splits['test']) > 0:
            batch_data = dataset.get_batch_data(splits['test'])
            test_preds = model(batch_data)

            test_preds_np = test_preds.cpu().numpy()
            test_gts_np = batch_data['targets'].cpu().numpy()
            test_indices = splits['test']
            train_count = len(splits['train'])

            # Package for visualization functions
            all_preds = {'test': test_preds_np}
            all_gts = {'test': test_gts_np}
            all_indices = {'test': test_indices}

            # Create temporal visualization
            create_temporal_visualization(
                dataset, all_preds, all_gts, all_indices, output_dir,
                epoch=args.epochs, total_epochs=args.epochs, train_samples=train_count
            )

            # Create geospatial visualization
            create_geospatial_visualization(
                dataset, test_preds_np, test_gts_np, test_indices, output_dir,
                epoch=args.epochs, train_samples=train_count
            )

            # Create combined scientific figure
            print("\nCreating combined scientific figure...", flush=True)
            create_combined_scientific_figure(
                dataset, test_preds_np, test_gts_np, test_indices,
                all_preds, all_gts, all_indices, output_dir,
                epoch=args.epochs, train_samples=train_count
            )

    print(f"Visualizations saved to {output_dir}", flush=True)

    # Export test predictions to CSV
    print("\nExporting test predictions to CSV...", flush=True)
    csv_path = export_test_predictions_csv(
        dataset, test_preds_np, test_gts_np, test_indices, output_dir
    )

    # Create error distribution histogram
    print("\nCreating error distribution histogram...", flush=True)
    create_error_histogram(csv_path, output_dir)

    # Return results
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    return {
        'test_rmse': final['test_rmse'],
        'test_mae': final['test_mae'],
        'test_r2': final['test_r2'],
        'train_rmse': final['train_rmse'],
        'train_mae': final['train_mae'],
        'train_r2': final['train_r2'],
        'avg_epoch_time_s': avg_epoch_time,
        'total_epochs': args.epochs,
        'use_adaptive_range': use_adaptive_range,
        'batch_size': args.batch_size,
    }


def main():
    """Main entry point for LFMC benchmark training."""
    parser = argparse.ArgumentParser(
        description='Earth4D LFMC Benchmark Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data-path', default=None,
                       help='Path to LFMC CSV. If not provided, downloads AI2 official CSV.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=2500,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0008,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')

    # Model arguments
    parser.add_argument('--species-dim', type=int, default=256,
                       help='Dimension of learnable species embeddings')

    # Enhancement flags
    parser.add_argument('--use-adaptive-range', action='store_true',
                       help='Enable adaptive range normalization (fit to data extent)')
    parser.add_argument('--no-precomputed', action='store_true',
                       help='Disable precomputed hash indices (enabled by default)')

    # Training acceleration flags
    parser.add_argument('--batched-forward', action='store_true',
                       help='Use batched forward pass (single kernel for all samples, per-batch gradients)')
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision (fp16) for faster training')
    parser.add_argument('--fused-sgd', action='store_true',
                       help='Use fused CUDA backward+SGD for encoder embeddings (22x faster, SGD instead of AdamW for embeddings)')
    parser.add_argument('--fused-adam', action='store_true', default=True,
                       help='Use sparse Adam CUDA kernel for encoder embeddings (default: enabled)')
    parser.add_argument('--no-fused-adam', action='store_false', dest='fused_adam',
                       help='Disable fused Adam, use standard training')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                       help='Weight decay for AdamW (default: 0.001)')

    # Geographic mode arguments
    parser.add_argument('--coordinate-system', type=str, default='ecef',
                       choices=['geographic', 'ecef'],
                       help='Coordinate system to use (default: ecef)')
    parser.add_argument('--resolution-mode', type=str, default='balanced',
                       help='Resolution mode for geographic coordinates')
    parser.add_argument('--lat-coverage', type=float, default=0.25,
                       help='Latitude coverage fraction (default: 0.25)')
    parser.add_argument('--lon-coverage', type=float, default=0.25,
                       help='Longitude coverage fraction (default: 0.25)')
    parser.add_argument('--elev-coverage', type=float, default=0.15,
                       help='Elevation coverage fraction (default: 0.15)')
    parser.add_argument('--time-coverage', type=float, default=1.0,
                       help='Time coverage fraction (default: 1.0)')
    parser.add_argument('--base-temporal-res', type=int, default=32,
                       help='Base temporal resolution (default: 32)')
    parser.add_argument('--temporal-growth', type=float, default=2.0,
                       help='Temporal growth factor per level (default: 2.0)')
    parser.add_argument('--latlon-growth', type=float, default=None,
                       help='Lat/lon growth factor (decoupled from elevation). Default: match encoder baseline')
    parser.add_argument('--elev-growth', type=float, default=None,
                       help='Elevation growth factor (decoupled from lat/lon). Default: match encoder baseline')
    parser.add_argument('--fast', action='store_true',
                       help='Skip visualizations and CSV export for faster runs')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for model and visualizations')

    args = parser.parse_args()

    # GPU settings
    torch.backends.cuda.matmul.allow_tf32 = True

    run_training_session(args)


if __name__ == "__main__":
    main()
