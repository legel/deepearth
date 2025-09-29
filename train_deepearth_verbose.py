#!/usr/bin/python3
"""
DeepEarth Verbose Training Script
==================================

Enhanced training with detailed diagnostics for understanding:
- Data flow through each PerceiverProjector
- Gradient propagation
- Loss components
- Masking statistics
- Parameter updates
"""

import os
import sys
sys.path.append('/opt/ecodash/deepearth')

# Ensure unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from core.config import DeepEarthConfig
from core.perceiver import DeepEarthPerceiver
from models.flowering.preprocess_flowering_data import FloweringDatasetPreprocessor


class TrainingDiagnostics:
    """Comprehensive training diagnostics tracker."""

    def __init__(self, model, window_size=10):
        self.model = model
        self.window_size = window_size

        # Track loss components over time
        self.loss_history = defaultdict(lambda: deque(maxlen=window_size))

        # Track gradient statistics
        self.gradient_history = defaultdict(lambda: deque(maxlen=window_size))

        # Track encoder usage statistics
        self.encoder_stats = defaultdict(lambda: defaultdict(int))

        # Track masking statistics
        self.masking_stats = defaultdict(lambda: deque(maxlen=window_size))

        # Track parameter changes
        self.param_changes = {}
        self.prev_params = {}
        self._store_params()

    def _store_params(self):
        """Store current parameters for change detection."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.data.clone()

    def analyze_batch(self, batch, outputs):
        """Analyze a training batch."""
        stats = {}

        # Analyze data distribution
        if isinstance(batch['encoded_data'], list):
            # Count encoder usage
            encoder_counts = defaultdict(int)
            encoder_dims = defaultdict(list)

            for b_idx in range(len(batch['encoded_data'])):
                for s_idx in range(len(batch['encoded_data'][b_idx])):
                    enc_id = batch['dataset_modality_encoder'][b_idx, s_idx, 2].item()
                    encoder_counts[enc_id] += 1

                    if torch.is_tensor(batch['encoded_data'][b_idx][s_idx]):
                        encoder_dims[enc_id].append(batch['encoded_data'][b_idx][s_idx].shape[-1])

            stats['encoder_usage'] = dict(encoder_counts)
            stats['encoder_dims'] = {k: list(set(v)) for k, v in encoder_dims.items()}

        # Analyze masking
        if 'mask_stats' in outputs:
            stats['masking'] = outputs['mask_stats']

        # Analyze loss components
        if 'component_losses' in outputs:
            stats['losses'] = {k: v.item() for k, v in outputs['component_losses'].items()}
            for k, v in stats['losses'].items():
                self.loss_history[k].append(v)

        return stats

    def analyze_gradients(self):
        """Analyze gradient flow through the model."""
        grad_stats = {}

        # Check major components
        components = {
            'earth4d': 'earth4d_raw',
            'multimodal_fusion': 'multimodal_fusion',
            'perceiver': 'perceiver',
            'decoder': 'decoder'
        }

        for comp_name, attr_name in components.items():
            if hasattr(self.model, attr_name):
                component = getattr(self.model, attr_name)

                # Collect gradient statistics
                grad_norms = []
                grad_means = []
                zero_grads = 0
                total_params = 0

                for param in component.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        grad_mean = param.grad.data.mean().item()
                        grad_norms.append(grad_norm)
                        grad_means.append(grad_mean)

                        if grad_norm < 1e-8:
                            zero_grads += 1
                    else:
                        zero_grads += 1
                    total_params += 1

                if grad_norms:
                    grad_stats[comp_name] = {
                        'max_norm': max(grad_norms),
                        'mean_norm': np.mean(grad_norms),
                        'min_norm': min(grad_norms),
                        'mean_grad': np.mean(grad_means),
                        'zero_grad_ratio': zero_grads / total_params
                    }
                else:
                    grad_stats[comp_name] = {'error': 'No gradients found'}

                self.gradient_history[comp_name].append(grad_stats.get(comp_name, {}))

        return grad_stats

    def analyze_parameter_changes(self):
        """Analyze how parameters are changing."""
        param_changes = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.prev_params:
                change = (param.data - self.prev_params[name]).norm(2).item()
                param_changes[name] = change

        # Group by component
        component_changes = defaultdict(list)
        for name, change in param_changes.items():
            if 'earth4d' in name:
                component_changes['earth4d'].append(change)
            elif 'multimodal' in name:
                component_changes['multimodal'].append(change)
            elif 'perceiver' in name:
                component_changes['perceiver'].append(change)
            elif 'decoder' in name:
                component_changes['decoder'].append(change)
            else:
                component_changes['other'].append(change)

        # Compute statistics
        stats = {}
        for comp, changes in component_changes.items():
            if changes:
                stats[comp] = {
                    'max_change': max(changes),
                    'mean_change': np.mean(changes),
                    'num_params': len(changes)
                }

        # Store current params for next iteration
        self._store_params()

        return stats

    def get_summary(self):
        """Get a summary of recent training statistics."""
        summary = {}

        # Loss trends
        if self.loss_history:
            summary['loss_trends'] = {}
            for k, v in self.loss_history.items():
                if v:
                    recent = list(v)[-5:]  # Last 5 values
                    summary['loss_trends'][k] = {
                        'current': recent[-1],
                        'mean': np.mean(recent),
                        'trend': 'decreasing' if len(recent) > 1 and recent[-1] < recent[0] else 'increasing'
                    }

        # Gradient health
        if self.gradient_history:
            summary['gradient_health'] = {}
            for comp, history in self.gradient_history.items():
                if history and history[-1]:
                    latest = history[-1]
                    summary['gradient_health'][comp] = {
                        'norm': latest.get('mean_norm', 0),
                        'zero_ratio': latest.get('zero_grad_ratio', 1)
                    }

        return summary


def create_verbose_batch(indices_list, data, config, device):
    """Create batch with detailed tracking."""
    indices = torch.tensor([idx.item() if torch.is_tensor(idx) else idx for idx in indices_list])

    batch_size = len(indices) // config.context_window
    if batch_size == 0:
        batch_size = 1

    indices = indices[:batch_size * config.context_window]
    indices = indices.reshape(batch_size, config.context_window)

    batch_xyzt = []
    batch_dme = []
    batch_encoded = []

    # Track encoder usage in this batch
    encoder_counts = defaultdict(int)

    for b in range(batch_size):
        seq_xyzt = []
        seq_dme = []
        seq_encoded = []

        for s in range(config.context_window):
            idx = indices[b, s].item()
            sample_idx = idx % data['n_samples']

            seq_xyzt.append(data['xyzt'][sample_idx])
            seq_dme.append(data['dataset_modality_encoder'][idx])

            encoder_id = data['encoded_file_indices'][idx].item()
            encoder_counts[encoder_id] += 1

            row_idx = data['encoded_row_indices'][idx].item()

            if encoder_id == 1:  # Earth4D
                encoded_vec = data['xyzt'][sample_idx]
            else:
                encoded_vec = data['encoded_data'][encoder_id][row_idx]

            seq_encoded.append(encoded_vec)

        batch_xyzt.append(torch.stack(seq_xyzt))
        batch_dme.append(torch.stack(seq_dme))
        batch_encoded.append(seq_encoded)

    batch = {
        'xyzt': torch.stack(batch_xyzt),
        'dataset_modality_encoder': torch.stack(batch_dme).long(),
        'encoded_data': batch_encoded,
    }

    # Add targets
    target_indices = indices[:, 0] % data['n_samples']
    batch['target'] = data['target'][target_indices]

    # Add metadata
    batch['encoder_counts'] = dict(encoder_counts)

    return batch


def show_reconstruction_examples(batch, outputs, model):
    """Show detailed reconstruction examples for masked components."""
    import numpy as np

    # Get first sample from batch for detailed view
    sample_idx = 0
    seq_idx = 0

    # Debug: Show what we have
    print(f"  DEBUG - Output keys: {list(outputs.keys())}")

    # Get masks
    masks = outputs.get('masks', {})
    reconstructed = outputs.get('reconstructed')
    original = outputs.get('original')

    if reconstructed is None or original is None:
        print("  ⚠️ No reconstruction data available (model may not be returning reconstruction data)")
        print(f"  Available outputs: {list(outputs.keys())}")
        return

    # Debug: Show shapes and mask info
    print(f"  DEBUG - Reconstructed shape: {reconstructed.shape if reconstructed is not None else 'None'}")
    print(f"  DEBUG - Original shape: {original.shape if original is not None else 'None'}")
    print(f"  DEBUG - Mask types: {list(masks.keys())}")

    # Show mask statistics
    for mask_name, mask_tensor in masks.items():
        if torch.is_tensor(mask_tensor):
            print(f"  DEBUG - {mask_name} mask: shape={mask_tensor.shape}, any_masked={mask_tensor.any().item()}, count={mask_tensor.sum().item()}")

    # Show one example of each masked component type
    shown_types = set()

    # ALWAYS show spacetime reconstruction from first masked position
    if 'spacetime' in masks:
        spacetime_mask = masks['spacetime']
        # Find ANY masked position
        masked_positions = torch.where(spacetime_mask)

        if len(masked_positions[0]) > 0:
            # Use first masked position
            mask_sample_idx = masked_positions[0][0].item()
            mask_seq_idx = masked_positions[1][0].item()

            print(f"\n  📍 SPACETIME (X,Y,Z,T) RECONSTRUCTION (sample {mask_sample_idx}, seq {mask_seq_idx}):")

            # Get spacetime portion of the tokens
            start_idx = 0
            end_idx = model.spacetime_dim if hasattr(model, 'spacetime_dim') else 162

            # Just show the first 4 values (X,Y,Z,T)
            orig_vals = original[mask_sample_idx, mask_seq_idx, start_idx:start_idx+4].detach().cpu().numpy()
            recon_vals = reconstructed[mask_sample_idx, mask_seq_idx, start_idx:start_idx+4].detach().cpu().numpy()

            print(f"    Original:  [{orig_vals[0]:.4f}, {orig_vals[1]:.4f}, {orig_vals[2]:.4f}, {orig_vals[3]:.4f}]")
            print(f"    Predicted: [{recon_vals[0]:.4f}, {recon_vals[1]:.4f}, {recon_vals[2]:.4f}, {recon_vals[3]:.4f}]")

            delta = recon_vals - orig_vals
            print(f"    Δ (Delta): [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}, {delta[3]:.4f}]")

            mape = np.mean(np.abs(delta / (np.abs(orig_vals) + 1e-8))) * 100
            print(f"    MAPE: {mape:.2f}%")
        else:
            print("  DEBUG - No spacetime masks found!")

    # ALWAYS show data reconstruction from first masked position
    if 'data' in masks:
        data_mask = masks['data']
        # Find ANY masked position
        masked_positions = torch.where(data_mask)

        if len(masked_positions[0]) > 0:
            # Use first masked position
            mask_sample_idx = masked_positions[0][0].item()
            mask_seq_idx = masked_positions[1][0].item()

            # Determine which encoder this is from
            # dataset_modality_encoder is shape [batch, seq, 3] where 3 = [dataset_id, modality_id, encoder_id]
            dme = batch['dataset_modality_encoder'][mask_sample_idx, mask_seq_idx]
            encoder_id = dme[2].item()  # Get the encoder_id (third element)

            encoder_names = {0: 'AlphaEarth', 1: 'Earth4D', 2: 'BioCLIP', 3: 'PhenoVision'}
            encoder_name = encoder_names.get(encoder_id, f'Encoder_{encoder_id}')
            print(f"\n  🎨 {encoder_name.upper()} EMBEDDING RECONSTRUCTION (sample {mask_sample_idx}, seq {mask_seq_idx}):")

            # Get data portion
            start_idx = model.spacetime_dim if hasattr(model, 'spacetime_dim') else 162
            end_idx = start_idx + (model.data_dim if hasattr(model, 'data_dim') else 84)

            # Show first 8 channels for brevity
            num_channels = min(8, end_idx - start_idx)

            orig_vals = original[mask_sample_idx, mask_seq_idx, start_idx:start_idx+num_channels].detach().cpu().numpy()
            recon_vals = reconstructed[mask_sample_idx, mask_seq_idx, start_idx:start_idx+num_channels].detach().cpu().numpy()

            print(f"    Original (first {num_channels} dims):")
            print(f"      {np.array2string(orig_vals, precision=3, separator=', ', max_line_width=120)}")
            print(f"    Predicted:")
            print(f"      {np.array2string(recon_vals, precision=3, separator=', ', max_line_width=120)}")

            delta = recon_vals - orig_vals
            print(f"    Δ (Delta):")
            print(f"      {np.array2string(delta, precision=3, separator=', ', max_line_width=120)}")

            mape = np.mean(np.abs(delta / (np.abs(orig_vals) + 1e-8))) * 100
            rmse = np.sqrt(np.mean(delta**2))
            print(f"    MAPE: {mape:.2f}%, RMSE: {rmse:.4f}")

            # Special case for PhenoVision (1D flowering probability)
            if encoder_id == 3:
                print(f"    🌸 Flowering Probability:")
                print(f"      Original:  {orig_vals[0]:.3f}")
                print(f"      Predicted: {recon_vals[0]:.3f}")
                print(f"      Δ (Delta): {delta[0]:.3f}")
        else:
            print("  DEBUG - No data masks found!")

    sys.stdout.flush()


def train_epoch_verbose(model, data, train_indices, optimizer, config, diagnostics, device, scaler=None):
    """Train one epoch with detailed diagnostics."""
    model.train()

    # Create data loader
    train_loader = DataLoader(
        train_indices,
        batch_size=config.batch_size * config.context_window,
        shuffle=True,
        collate_fn=lambda x: create_verbose_batch(x, data, config, device),
        num_workers=0,
        drop_last=True
    )

    epoch_stats = defaultdict(list)

    print("\n" + "="*100)
    print("STARTING TRAINING EPOCH")
    print("="*100)

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        batch['xyzt'] = batch['xyzt'].to(device)
        batch['dataset_modality_encoder'] = batch['dataset_modality_encoder'].to(device)
        batch['target'] = batch['target'].to(device)

        # Move encoded data
        for b in range(len(batch['encoded_data'])):
            for s in range(len(batch['encoded_data'][b])):
                if torch.is_tensor(batch['encoded_data'][b][s]):
                    batch['encoded_data'][b][s] = batch['encoded_data'][b][s].to(device)

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/{len(train_loader)}")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Print batch composition
        print("\n📊 BATCH COMPOSITION:")
        print(f"  Shape: {batch['xyzt'].shape} (batch_size × context_window × dims)")
        print(f"  Encoder usage in batch:")
        encoder_names = {1: 'Earth4D', 2: 'AlphaEarth', 3: 'BioCLIP', 4: 'PhenoVision'}
        for enc_id, count in batch['encoder_counts'].items():
            name = encoder_names.get(enc_id, f'Encoder_{enc_id}')
            percentage = (count / (config.batch_size * config.context_window)) * 100
            print(f"    {name}: {count} tokens ({percentage:.1f}%)")

        # Forward pass with timing
        print("\n🔄 FORWARD PASS:")
        sys.stdout.flush()

        forward_start = time.time()
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(batch, return_latents=True)
                loss = outputs['loss']
        else:
            outputs = model(batch, return_latents=True)
            loss = outputs['loss']
        forward_time = time.time() - forward_start

        # Print loss components
        print("\n📉 LOSS BREAKDOWN:")
        print(f"  Total loss: {loss.item():.6f}")

        # Separate universal space and modality space losses
        print("\n  Universal Space Losses (256D token space):")
        if 'component_losses' in outputs:
            for comp, comp_loss in outputs['component_losses'].items():
                if 'modality_loss' not in comp:
                    print(f"    {comp}: {comp_loss.item():.6f}")

        print("\n  Modality Space Losses (original dimensions via PerceiverProjector decoders):")
        if 'component_losses' in outputs:
            modality_losses_found = False
            for comp, comp_loss in outputs['component_losses'].items():
                if 'modality_loss' in comp:
                    print(f"    {comp}: {comp_loss.item():.6f}")
                    modality_losses_found = True
            if not modality_losses_found:
                print("    ⚠️ No modality losses computed (projector decoders not being used)")

        sys.stdout.flush()  # Force output

        # Show reconstruction examples every batch
        if 'reconstructed' in outputs:
            print("\n🔍 RECONSTRUCTION EXAMPLES:")
            show_reconstruction_examples(batch, outputs, model)

        # Analyze batch
        batch_stats = diagnostics.analyze_batch(batch, outputs)

        # Backward pass with timing
        print("\n⬅️ BACKWARD PASS:")
        backward_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            scaler.scale(loss).backward()

            # Unscale for gradient analysis
            scaler.unscale_(optimizer)

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            print(f"  Gradient norm (before clipping): {grad_norm:.6f}")

            # Gradient analysis
            grad_stats = diagnostics.analyze_gradients()
            print("\n📐 GRADIENT FLOW:")
            for comp, stats in grad_stats.items():
                if 'error' not in stats:
                    print(f"  {comp}:")
                    print(f"    Mean norm: {stats['mean_norm']:.6f}")
                    print(f"    Max norm: {stats['max_norm']:.6f}")
                    print(f"    Zero gradient ratio: {stats['zero_grad_ratio']:.2%}")

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            backward_time = time.time() - backward_start

            # Display timing breakdown
            print("\n⏱️ TIMING BREAKDOWN:")
            print(f"  Forward pass:  {forward_time:.2f}s")
            print(f"  Backward pass: {backward_time:.2f}s")
            print(f"  Ratio (back/fwd): {backward_time/forward_time:.1f}x")
            sys.stdout.flush()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            print(f"  Gradient norm (before clipping): {grad_norm:.6f}")

            backward_time = time.time() - backward_start

            # Display timing breakdown
            print("\n⏱️ TIMING BREAKDOWN:")
            print(f"  Forward pass:  {forward_time:.2f}s")
            print(f"  Backward pass: {backward_time:.2f}s")
            print(f"  Ratio (back/fwd): {backward_time/forward_time:.1f}x")
            sys.stdout.flush()

            # Gradient analysis
            grad_stats = diagnostics.analyze_gradients()
            print("\n📐 GRADIENT FLOW:")
            for comp, stats in grad_stats.items():
                if 'error' not in stats:
                    print(f"  {comp}:")
                    print(f"    Mean norm: {stats['mean_norm']:.6f}")
                    print(f"    Max norm: {stats['max_norm']:.6f}")
                    print(f"    Zero gradient ratio: {stats['zero_grad_ratio']:.2%}")

            optimizer.step()

        # Parameter changes
        param_changes = diagnostics.analyze_parameter_changes()
        print("\n🔧 PARAMETER UPDATES:")
        for comp, stats in param_changes.items():
            print(f"  {comp}: mean_change={stats['mean_change']:.8f}, max_change={stats['max_change']:.8f}")

        # Store stats
        epoch_stats['loss'].append(loss.item())
        epoch_stats['grad_norm'].append(grad_norm.item())

        # Print summary every 10 batches
        if (batch_idx + 1) % 10 == 0:
            summary = diagnostics.get_summary()
            print("\n" + "="*80)
            print("SUMMARY (last 10 batches)")
            print("="*80)

            if 'loss_trends' in summary:
                print("\n📊 LOSS TRENDS:")
                for comp, trend in summary['loss_trends'].items():
                    print(f"  {comp}: {trend['current']:.6f} (trend: {trend['trend']})")

            if 'gradient_health' in summary:
                print("\n🏥 GRADIENT HEALTH:")
                for comp, health in summary['gradient_health'].items():
                    status = "✅" if health['zero_ratio'] < 0.5 else "⚠️"
                    print(f"  {status} {comp}: norm={health['norm']:.6f}, zero_ratio={health['zero_ratio']:.2%}")

        # Break after a few batches for testing
        # Remove the limit - process all batches
        # if batch_idx >= 5:  # Process only 6 batches for quick testing
        #     break

    return epoch_stats


def main():
    parser = argparse.ArgumentParser(description='Verbose DeepEarth Training')
    parser.add_argument('--config', type=str,
                       default='/opt/ecodash/deepearth/models/flowering/flowering.yaml')
    parser.add_argument('--data_dir', type=str,
                       default='/opt/ecodash/deepearth/models/flowering/data')
    parser.add_argument('--output_dir', type=str,
                       default='/opt/ecodash/deepearth/experiments')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load config
    config = DeepEarthConfig.from_yaml(args.config)
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.mixed_precision = args.mixed_precision
    config.device = device

    print("\n" + "="*100)
    print("DEEPEARTH VERBOSE TRAINING DIAGNOSTICS")
    print("="*100)
    print(f"Batch size: {config.batch_size}")
    print(f"Context window: {config.context_window}")
    print(f"Total tokens per batch: {config.batch_size * config.context_window}")

    # Load dataset
    print("\n📚 Loading dataset...")
    preprocessor = FloweringDatasetPreprocessor(config, args.data_dir)
    data = preprocessor.load_dataset()
    print(f"✅ Loaded {data['n_samples']:,} samples, {data['n_observations']:,} observations")

    # Setup encoder configs
    encoder_configs = {}
    for encoder_id, tensor in data['encoded_data'].items():
        encoder_name = [k for k, v in data['encoder_map'].items() if v == encoder_id][0]
        if encoder_id == 1:  # Earth4D
            encoder_configs[encoder_id] = {'name': encoder_name, 'input_dim': 4}
        else:
            encoder_configs[encoder_id] = {'name': encoder_name, 'input_dim': tensor.shape[-1]}

    print("\n🎼 Encoder configurations (Symphony of Experts):")
    for enc_id, cfg in encoder_configs.items():
        print(f"  [{enc_id}] {cfg['name']}: {cfg['input_dim']}D → PerceiverProjection → 500D")

    # Initialize model
    print("\n🏗️ Initializing model...")
    model = DeepEarthPerceiver(config, encoder_configs)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    earth4d_params = sum(p.numel() for p in model.earth4d_raw.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params/1e6:.1f}M")
    print(f"  Earth4D parameters: {earth4d_params/1e6:.1f}M ({100*earth4d_params/total_params:.1f}%)")

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None

    # Initialize diagnostics
    diagnostics = TrainingDiagnostics(model)

    # Create train split
    n_train = int(0.8 * data['n_observations'])
    train_indices = torch.arange(n_train)

    print(f"\n🏃 Starting training on {n_train:,} observations...")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {args.mixed_precision}")

    # Train
    epoch_stats = train_epoch_verbose(
        model, data, train_indices, optimizer, config,
        diagnostics, device, scaler
    )

    # Final summary
    print("\n" + "="*100)
    print("TRAINING COMPLETE")
    print("="*100)
    print(f"Average loss: {np.mean(epoch_stats['loss']):.6f}")
    print(f"Final loss: {epoch_stats['loss'][-1]:.6f}")
    print(f"Loss reduction: {(epoch_stats['loss'][0] - epoch_stats['loss'][-1]):.6f}")

    # Check if model is learning
    if epoch_stats['loss'][-1] < epoch_stats['loss'][0]:
        print("✅ Model is learning! Loss decreased during training.")
    else:
        print("⚠️ Model may not be learning effectively. Check gradients and learning rate.")


if __name__ == '__main__':
    main()