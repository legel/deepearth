#!/usr/bin/env python3
"""
Earth4D LFMC - Learned Probing Comparison
==========================================
Compares standard hash encoding vs learned hash probing on real LFMC dataset.

Runs identical training procedures with the only difference being:
- Baseline: Standard hash encoding (enable_learned_probing=False)
- Learned: Learned hash probing (enable_learned_probing=True)

Tracks timing for all key operations to measure overhead.
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Import the original script's functionality
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import everything from the original script
from earth4d_to_lfmc import *


def train_epoch_gpu_with_probing(model, dataset, indices, optimizer, batch_size=20000,
                                 enable_learned_probing=False, enable_entropy_loss=False,
                                 entropy_weight=0.01):
    """
    Ultra-fast training with learned probing support.

    This is a modified version of train_epoch_gpu that:
    1. Supports entropy regularization for learned probing
    2. Calls update_probe_indices() after each optimizer step when learned probing is enabled
    """
    model.train()
    n = len(indices)

    # Shuffle ON GPU
    perm = torch.randperm(n, device=indices.device)
    indices = indices[perm]

    # Accumulate for metrics
    all_preds = []
    all_targets = []
    all_degens = []

    # Process batches
    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        # Pure GPU ops
        coords = dataset.coords[batch_idx]
        targets = dataset.targets[batch_idx]
        species = dataset.species_idx[batch_idx]

        preds = model(coords, species)

        # Compute loss with optional entropy regularization
        loss_dict = model.earth4d.compute_loss(
            preds, targets,
            enable_probe_entropy_loss=enable_entropy_loss,
            probe_entropy_weight=entropy_weight
        )

        total_loss = loss_dict['_total_loss_tensor']

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # CRITICAL: Update probe indices after optimizer step (for learned probing)
        if enable_learned_probing:
            model.earth4d.encoder.xyz_encoder.update_probe_indices()
            model.earth4d.encoder.xyt_encoder.update_probe_indices()
            model.earth4d.encoder.yzt_encoder.update_probe_indices()
            model.earth4d.encoder.xzt_encoder.update_probe_indices()

        # Store for metrics
        all_preds.append(preds.detach())
        all_targets.append(targets)
        all_degens.append(dataset.is_degenerate[batch_idx])

    # Compute metrics on full epoch predictions
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_degens = torch.cat(all_degens)

    overall, unique, degen = compute_metrics_gpu(all_preds, all_targets, all_degens)

    return overall, unique, degen

# Override SpeciesAwareLFMCModel to accept learned probing parameters
class SpeciesAwareLFMCModelWithProbing(nn.Module):
    """LFMC model with learnable species embeddings and configurable hash probing."""

    def __init__(self, n_species, species_dim=32,
                 enable_learned_probing=False,
                 probing_range=4,
                 index_codebook_size=512):
        super().__init__()

        self.earth4d = Earth4D(
            verbose=True,
            enable_learned_probing=enable_learned_probing,
            probing_range=probing_range,
            index_codebook_size=index_codebook_size
        )

        earth4d_dim = self.earth4d.get_output_dim()

        # Learnable species embeddings
        self.species_embeddings = nn.Embedding(n_species, species_dim)
        nn.init.normal_(self.species_embeddings.weight, mean=0.0, std=0.1)

        if enable_learned_probing:
            print(f"  Using LEARNED HASH PROBING (N_p={probing_range}, N_c={index_codebook_size})", flush=True)
        else:
            print(f"  Using STANDARD HASH ENCODING (no learned probing)", flush=True)
        print(f"  Using learnable species embeddings: ({n_species}, {species_dim})", flush=True)

        # MLP that takes concatenated Earth4D features and species embedding
        input_dim = earth4d_dim + species_dim
        print(f"  MLP input dimension: {input_dim} (Earth4D: {earth4d_dim} + Species: {species_dim})", flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Constrain output to [0, 1] to match normalized targets
        )

        # Initialize Earth4D parameters
        with torch.no_grad():
            for p in self.earth4d.parameters():
                if p.dim() > 1:
                    nn.init.uniform_(p, -0.1, 0.1)

        self.n_species = n_species
        self.species_dim = species_dim
        self.enable_learned_probing = enable_learned_probing

    def forward(self, coords, species_idx):
        # Get Earth4D spatiotemporal features
        earth4d_features = self.earth4d(coords)

        # Get species embeddings
        species_features = self.species_embeddings(species_idx)

        # Concatenate features
        combined_features = torch.cat([earth4d_features, species_features], dim=-1)

        # Predict LFMC
        return self.mlp(combined_features).squeeze(-1)


def run_training_with_timing(args, run_name="", enable_learned_probing=False,
                             probing_range=4, index_codebook_size=512, index_lr_multiplier=10.0,
                             enable_entropy_loss=False, entropy_weight=0.01):
    """Run training session with detailed timing instrumentation."""

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

    # Timing dict
    timing = {
        'dataset_load': 0.0,
        'model_creation': 0.0,
        'epoch_times': [],
        'forward_pass_times': [],
        'backward_pass_times': [],
        'total_training': 0.0
    }

    # Create output directory with run name suffix
    output_suffix = f"_{run_name}" if run_name else ""
    output_dir = Path(args.output_dir + output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80, flush=True)
    print(f"EARTH4D LFMC BENCHMARK - {run_name.upper()}", flush=True)
    if enable_learned_probing:
        print(f"Mode: LEARNED HASH PROBING (N_p={probing_range}, N_c={index_codebook_size})", flush=True)
        if enable_entropy_loss:
            print(f"  ✓ Entropy regularization ENABLED (weight={entropy_weight})", flush=True)
        else:
            print(f"  ✗ Entropy regularization DISABLED", flush=True)
    else:
        print(f"Mode: STANDARD HASH ENCODING (Baseline)", flush=True)
    print(f"Using learnable {args.species_dim}D species embeddings", flush=True)
    print("="*80, flush=True)

    # Load dataset with AI2 splits (TIMED)
    t0 = time.time()
    dataset = FullyGPUDataset(args.data_path, device, use_ai2_splits=True)
    splits = get_ai2_splits(dataset)
    timing['dataset_load'] = time.time() - t0
    print(f"\n⏱  Dataset loaded in {timing['dataset_load']:.2f}s", flush=True)

    # Model creation (TIMED)
    t0 = time.time()
    model = SpeciesAwareLFMCModelWithProbing(
        dataset.n_species,
        species_dim=args.species_dim,
        enable_learned_probing=enable_learned_probing,
        probing_range=probing_range,
        index_codebook_size=index_codebook_size
    ).to(device)
    timing['model_creation'] = time.time() - t0
    print(f"\n⏱  Model created in {timing['model_creation']:.2f}s", flush=True)

    # Count parameters
    earth4d_params = sum(p.numel() for p in model.earth4d.parameters())
    species_params = sum(p.numel() for p in model.species_embeddings.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:", flush=True)
    print(f"  Earth4D parameters: {earth4d_params:,}", flush=True)
    print(f"  Species embedding parameters: {species_params:,}", flush=True)
    print(f"  MLP parameters: {mlp_params:,}", flush=True)
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"  Trainable parameters: {trainable_params:,}", flush=True)

    # Optimizer with different learning rates for different parameter groups
    if enable_learned_probing:
        # Setup parameter groups with higher LR for index_logits
        optimizer_params = [
            # Earth4D embedding parameters (base LR)
            {'params': model.earth4d.encoder.xyz_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.xyt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.yzt_encoder.embeddings, 'lr': args.lr},
            {'params': model.earth4d.encoder.xzt_encoder.embeddings, 'lr': args.lr},
            # Species embeddings (base LR)
            {'params': model.species_embeddings.parameters(), 'lr': args.lr},
            # MLP parameters (base LR)
            {'params': model.mlp.parameters(), 'lr': args.lr},
            # Index logits (higher LR - critical for learned probing!)
            {'params': model.earth4d.encoder.xyt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.encoder.yzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
            {'params': model.earth4d.encoder.xzt_encoder.index_logits, 'lr': args.lr * index_lr_multiplier},
        ]
        optimizer = optim.AdamW(optimizer_params, weight_decay=0.001)
        print(f"\n✓ Using {index_lr_multiplier:.1f}× higher learning rate for index_logits: {args.lr * index_lr_multiplier:.6f}", flush=True)
    else:
        # Standard optimizer for baseline
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)

    # Tracking metrics
    metrics_history = []
    metrics_ema = MetricsEMA(alpha=0.1)

    print("\n" + "="*80, flush=True)
    print("Training with species embeddings (UNIQUE=Single species, MULTI=Multi-species):", flush=True)
    print("-"*80, flush=True)

    # Training loop with timing
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train (with timing and learned probing support)
        trn_overall, trn_unique, trn_degen = train_epoch_gpu_with_probing(
            model, dataset, splits['train'], optimizer, args.batch_size,
            enable_learned_probing=enable_learned_probing,
            enable_entropy_loss=enable_entropy_loss,
            entropy_weight=entropy_weight
        )

        # Evaluate test split
        test_overall, test_unique, test_degen, test_gt, test_pred, test_types = evaluate_split(
            model, dataset, splits['test']
        )

        epoch_time = time.time() - epoch_start
        timing['epoch_times'].append(epoch_time)

        # Update EMAs
        current_metrics = {
            'epoch': epoch,
            'time': epoch_time,
            # Train overall metrics
            'train_rmse': trn_overall['rmse'],
            'train_mae': trn_overall['mae'],
            'train_r2': trn_overall['r2'],
            # Train unique metrics
            'train_unique_mae': trn_unique['mae'],
            'train_unique_r2': trn_unique['r2'],
            # Train degenerate metrics
            'train_degen_mae': trn_degen['mae'],
            'train_degen_r2': trn_degen['r2'],
            # Test metrics
            'test_rmse': test_overall['rmse'],
            'test_mae': test_overall['mae'],
            'test_r2': test_overall['r2'],
            'test_unique_mae': test_unique['mae'],
            'test_unique_r2': test_unique['r2'],
            'test_degen_mae': test_degen['mae'],
            'test_degen_r2': test_degen['r2']
        }

        # Update EMAs and store metrics
        ema_metrics = metrics_ema.update(current_metrics)
        current_metrics.update(ema_metrics)
        metrics_history.append(current_metrics)

        # Store sample predictions for EMA tracking
        if epoch == 1:
            metrics_ema.sample_predictions['test_gt'] = test_gt
        metrics_ema.sample_predictions[f'test_pred_{epoch}'] = test_pred

        # Print every 100 epochs or last epoch
        if epoch % 100 == 0 or epoch == args.epochs:
            print(f"\nEPOCH {epoch:4d} ({epoch_time:.1f}s)", flush=True)
            print(f"  TRAIN: RMSE={trn_overall['rmse']:5.1f}pp, MAE={trn_overall['mae']:5.1f}pp, R²={trn_overall['r2']:.3f}", flush=True)
            print(f"  TEST:  RMSE={test_overall['rmse']:5.1f}pp, MAE={test_overall['mae']:5.1f}pp, R²={test_overall['r2']:.3f}", flush=True)

        # LR decay
        for g in optimizer.param_groups:
            g['lr'] *= 0.9995

    timing['total_training'] = time.time() - training_start

    print("="*80, flush=True)
    print(f"\n⏱  Total training time: {timing['total_training']:.2f}s ({timing['total_training']/60:.1f} min)", flush=True)
    print(f"⏱  Average epoch time: {np.mean(timing['epoch_times']):.2f}s", flush=True)

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_species': dataset.n_species,
        'species_dim': args.species_dim,
        'species_to_idx': dataset.species_to_idx,
        'idx_to_species': dataset.idx_to_species,
        'enable_learned_probing': enable_learned_probing,
        'probing_range': probing_range,
        'index_codebook_size': index_codebook_size
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}", flush=True)

    # Save metrics history to CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = output_dir / f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}", flush=True)

    # Save timing info
    timing_path = output_dir / 'timing_info.json'
    with open(timing_path, 'w') as f:
        json.dump({
            'dataset_load_seconds': timing['dataset_load'],
            'model_creation_seconds': timing['model_creation'],
            'total_training_seconds': timing['total_training'],
            'total_training_minutes': timing['total_training'] / 60,
            'average_epoch_seconds': float(np.mean(timing['epoch_times'])),
            'min_epoch_seconds': float(np.min(timing['epoch_times'])),
            'max_epoch_seconds': float(np.max(timing['epoch_times'])),
            'total_epochs': len(timing['epoch_times']),
            'enable_learned_probing': enable_learned_probing,
            'probing_range': probing_range if enable_learned_probing else None,
            'index_codebook_size': index_codebook_size if enable_learned_probing else None
        }, f, indent=2)
    print(f"Timing info saved to: {timing_path}", flush=True)

    # Print final summary
    final = metrics_history[-1]
    print(f"\nFINAL RESULTS (Epoch {args.epochs}):", flush=True)
    print(f"  Test RMSE: {final['test_rmse']:.1f}pp", flush=True)
    print(f"  Test MAE:  {final['test_mae']:.1f}pp", flush=True)
    print(f"  Test R²:   {final['test_r2']:.3f}", flush=True)

    print("\nTraining complete!", flush=True)

    # Return final metrics and timing for comparison
    return {
        'final_metrics': final,
        'timing': timing,
        'total_params': total_params
    }


def run_comparison(args):
    """Run both baseline and learned probing, then compare."""

    print("\n" + "="*80)
    print("LEARNED HASH PROBING COMPARISON ON REAL LFMC DATASET")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Species embedding dim: {args.species_dim}")
    print(f"Random seed: {args.seed}")
    print()

    results = {}

    # ========================================================================
    # RUN 1: BASELINE (Standard Hash Encoding)
    # ========================================================================

    # Check if we should load baseline from file or run it
    if hasattr(args, 'baseline_results_path') and args.baseline_results_path:
        print("\n" + "="*80)
        print("LOADING BASELINE RESULTS FROM FILE")
        print("="*80)
        print(f"Path: {args.baseline_results_path}")

        with open(args.baseline_results_path, 'r') as f:
            baseline_comparison = json.load(f)
            baseline_loaded = baseline_comparison['baseline']

            # Reconstruct the results dict structure expected later
            results['baseline'] = {
                'final_metrics': {
                    'test_rmse': baseline_loaded['test_rmse'],
                    'test_mae': baseline_loaded['test_mae'],
                    'test_r2': baseline_loaded['test_r2'],
                },
                'timing': {
                    'dataset_load': baseline_loaded.get('dataset_load_seconds', 0.0),
                    'model_creation': baseline_loaded.get('model_creation_seconds', 0.0),
                    'total_training': baseline_loaded['total_training_seconds'],
                    'epoch_times': [baseline_loaded['average_epoch_seconds']] * args.epochs,  # Approximate
                },
                'total_params': baseline_loaded['total_params']
            }

        print(f"\nBaseline metrics:")
        print(f"  Test RMSE: {results['baseline']['final_metrics']['test_rmse']:.2f}pp")
        print(f"  Test MAE: {results['baseline']['final_metrics']['test_mae']:.2f}pp")
        print(f"  Test R²: {results['baseline']['final_metrics']['test_r2']:.4f}")

    elif not (hasattr(args, 'skip_baseline') and args.skip_baseline):
        print("\n" + "="*80)
        print("RUN 1: BASELINE - STANDARD HASH ENCODING")
        print("="*80)

        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.output_dir = args.output_dir + "_baseline"

        results['baseline'] = run_training_with_timing(
            baseline_args,
            run_name="baseline",
            enable_learned_probing=False
        )
    else:
        print("\n⏭  Skipping baseline (--skip-baseline flag set)")
        results['baseline'] = None

    # ========================================================================
    # RUN 2: LEARNED HASH PROBING
    # ========================================================================
    print("\n\n" + "="*80)
    print("RUN 2: LEARNED HASH PROBING")
    print("="*80)

    learned_args = argparse.Namespace(**vars(args))
    learned_args.output_dir = args.output_dir + "_learned"

    results['learned'] = run_training_with_timing(
        learned_args,
        run_name="learned",
        enable_learned_probing=True,
        probing_range=args.probing_range,
        index_codebook_size=args.index_codebook_size,
        index_lr_multiplier=args.index_lr_multiplier,
        enable_entropy_loss=args.enable_entropy_loss,
        entropy_weight=args.entropy_weight
    )

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)

    baseline_final = results['baseline']['final_metrics'] if results['baseline'] else None
    learned_final = results['learned']['final_metrics']

    baseline_timing = results['baseline']['timing'] if results['baseline'] else None
    learned_timing = results['learned']['timing']

    baseline_params = results['baseline']['total_params'] if results['baseline'] else None
    learned_params = results['learned']['total_params']

    print("\n" + "-"*80)
    print("ACCURACY COMPARISON (Test Set)")
    print("-"*80)

    if baseline_final is not None:
        print(f"{'Metric':<20} {'Baseline':<15} {'Learned':<15} {'Difference':<15}")
        print("-"*80)

        # RMSE
        rmse_diff = learned_final['test_rmse'] - baseline_final['test_rmse']
        rmse_pct = (rmse_diff / baseline_final['test_rmse']) * 100
        print(f"{'RMSE (pp)':<20} {baseline_final['test_rmse']:>14.2f} {learned_final['test_rmse']:>14.2f} {rmse_diff:>+14.2f} ({rmse_pct:+.1f}%)")

        # MAE
        mae_diff = learned_final['test_mae'] - baseline_final['test_mae']
        mae_pct = (mae_diff / baseline_final['test_mae']) * 100
        print(f"{'MAE (pp)':<20} {baseline_final['test_mae']:>14.2f} {learned_final['test_mae']:>14.2f} {mae_diff:>+14.2f} ({mae_pct:+.1f}%)")

        # R²
        r2_diff = learned_final['test_r2'] - baseline_final['test_r2']
        r2_pct = (r2_diff / baseline_final['test_r2']) * 100 if baseline_final['test_r2'] != 0 else 0
        print(f"{'R²':<20} {baseline_final['test_r2']:>14.4f} {learned_final['test_r2']:>14.4f} {r2_diff:>+14.4f} ({r2_pct:+.1f}%)")
    else:
        print(f"{'Metric':<20} {'Learned':<15}")
        print("-"*80)
        print(f"{'RMSE (pp)':<20} {learned_final['test_rmse']:>14.2f}")
        print(f"{'MAE (pp)':<20} {learned_final['test_mae']:>14.2f}")
        print(f"{'R²':<20} {learned_final['test_r2']:>14.4f}")
        rmse_diff = mae_diff = r2_diff = 0
        rmse_pct = mae_pct = r2_pct = 0

    if baseline_timing is not None:
        print("\n" + "-"*80)
        print("TIMING COMPARISON")
        print("-"*80)
        print(f"{'Operation':<30} {'Baseline (s)':<15} {'Learned (s)':<15} {'Overhead':<15}")
        print("-"*80)

        # Dataset load
        load_diff = learned_timing['dataset_load'] - baseline_timing['dataset_load']
        load_pct = (load_diff / baseline_timing['dataset_load']) * 100
        print(f"{'Dataset Load':<30} {baseline_timing['dataset_load']:>14.2f} {learned_timing['dataset_load']:>14.2f} {load_pct:>+14.1f}%")

        # Model creation
        create_diff = learned_timing['model_creation'] - baseline_timing['model_creation']
        create_pct = (create_diff / baseline_timing['model_creation']) * 100
        print(f"{'Model Creation':<30} {baseline_timing['model_creation']:>14.2f} {learned_timing['model_creation']:>14.2f} {create_pct:>+14.1f}%")

        # Average epoch time
        baseline_avg_epoch = np.mean(baseline_timing['epoch_times'])
        learned_avg_epoch = np.mean(learned_timing['epoch_times'])
        epoch_diff = learned_avg_epoch - baseline_avg_epoch
        epoch_pct = (epoch_diff / baseline_avg_epoch) * 100
        print(f"{'Average Epoch Time':<30} {baseline_avg_epoch:>14.2f} {learned_avg_epoch:>14.2f} {epoch_pct:>+14.1f}%")

        # Total training
        train_diff = learned_timing['total_training'] - baseline_timing['total_training']
        train_pct = (train_diff / baseline_timing['total_training']) * 100
        print(f"{'Total Training Time':<30} {baseline_timing['total_training']:>14.2f} {learned_timing['total_training']:>14.2f} {train_pct:>+14.1f}%")

        print("\n" + "-"*80)
        print("PARAMETER COMPARISON")
        print("-"*80)
        param_diff = learned_params - baseline_params
        param_pct = (param_diff / baseline_params) * 100
        print(f"{'Total Parameters':<30} {baseline_params:>14,} {learned_params:>14,} {param_pct:>+14.1f}%")
    else:
        print("\n(Baseline skipped - no timing/parameter comparison available)")
        # Initialize timing variables for JSON output
        train_pct = epoch_pct = 0

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nLearned Hash Probing trades off:")
    print("  + Memory: Can use smaller hash tables (4× reduction with N_p=4)")
    print("  + Quality: May improve or maintain accuracy via learned collision resolution")
    print("  - Speed: Adds training overhead from probe selection")

    if baseline_final is not None:
        if mae_pct < 0:
            print(f"\n✓ Learned probing IMPROVED accuracy by {abs(mae_pct):.1f}%")
        elif mae_pct > 0:
            print(f"\n⚠ Learned probing REDUCED accuracy by {mae_pct:.1f}%")
        else:
            print(f"\n= Learned probing had NO CHANGE in accuracy")

        if baseline_timing is not None:
            print(f"\n⏱ Training overhead: {train_pct:+.1f}%")
            print(f"   ({learned_timing['total_training']/60:.1f} min vs {baseline_timing['total_training']/60:.1f} min)")
    else:
        print(f"\nLearned probing results:")
        print(f"  Test RMSE: {learned_final['test_rmse']:.2f}pp")
        print(f"  Test MAE: {learned_final['test_mae']:.2f}pp")
        print(f"  Test R²: {learned_final['test_r2']:.4f}")

    # Save comparison summary
    comparison_path = Path(args.output_dir) / 'comparison_summary.json'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    with open(comparison_path, 'w') as f:
        # Save baseline results if available
        baseline_dict = None
        if baseline_final is not None and baseline_timing is not None and baseline_params is not None:
            baseline_dict = {
                'test_rmse': baseline_final['test_rmse'],
                'test_mae': baseline_final['test_mae'],
                'test_r2': baseline_final['test_r2'],
                'dataset_load_seconds': baseline_timing['dataset_load'],
                'model_creation_seconds': baseline_timing['model_creation'],
                'total_training_seconds': baseline_timing['total_training'],
                'average_epoch_seconds': float(np.mean(baseline_timing['epoch_times'])),
                'total_params': baseline_params
            }

        json.dump({
            'baseline': baseline_dict,
            'learned': {
                'test_rmse': learned_final['test_rmse'],
                'test_mae': learned_final['test_mae'],
                'test_r2': learned_final['test_r2'],
                'dataset_load_seconds': learned_timing['dataset_load'],
                'model_creation_seconds': learned_timing['model_creation'],
                'total_training_seconds': learned_timing['total_training'],
                'average_epoch_seconds': float(np.mean(learned_timing['epoch_times'])),
                'total_params': learned_params,
                'probing_range': args.probing_range,
                'index_codebook_size': args.index_codebook_size
            },
            'differences': {
                'test_rmse_pp': rmse_diff,
                'test_rmse_pct': rmse_pct,
                'test_mae_pp': mae_diff,
                'test_mae_pct': mae_pct,
                'test_r2_diff': r2_diff,
                'test_r2_pct': r2_pct,
                'training_time_overhead_pct': train_pct,
                'epoch_time_overhead_pct': epoch_pct
            },
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'species_dim': args.species_dim,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)

    print(f"\n✓ Comparison summary saved to: {comparison_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None,
                       help='Path to LFMC CSV. If not provided, downloads AI2 official CSV.')
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--batch-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.0125)
    parser.add_argument('--output-dir', type=str, default='./outputs_comparison')
    parser.add_argument('--species-dim', type=int, default=768,
                       help='Dimension of learnable species embeddings')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility (default: 0)')

    # Learned probing parameters
    parser.add_argument('--probing-range', type=int, default=4,
                       help='Probing range (N_p) for learned hash probing (default: 4)')
    parser.add_argument('--index-codebook-size', type=int, default=512,
                       help='Index codebook size (N_c) for learned hash probing (default: 512)')
    parser.add_argument('--index-lr-multiplier', type=float, default=10.0,
                       help='Learning rate multiplier for index_logits (default: 10.0)')
    parser.add_argument('--enable-entropy-loss', action='store_true',
                       help='Enable entropy regularization for learned hash probing')
    parser.add_argument('--entropy-weight', type=float, default=0.01,
                       help='Weight for entropy regularization loss (default: 0.01)')

    # Baseline control (for grid search efficiency)
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline training (only train learned variant)')
    parser.add_argument('--baseline-results-path', type=str, default=None,
                       help='Path to baseline comparison_summary.json to reuse (skips baseline training)')

    args = parser.parse_args()

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True

    # Run comparison
    run_comparison(args)


if __name__ == "__main__":
    main()
