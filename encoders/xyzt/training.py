"""
Training utilities for Earth4D encoder.

Provides generic training infrastructure with Protocol classes for type safety
and callback patterns for task-specific customization.

Includes optimizations:
- Mixed precision (AMP) support for reduced memory bandwidth
- Batched forward passes to reduce kernel launch overhead
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Callable, Protocol, runtime_checkable, Tuple

from sorting import block_shuffle_indices


# =============================================================================
# Mixed Precision Support
# =============================================================================

# Check if AMP is available
_AMP_AVAILABLE = hasattr(torch.cuda.amp, 'autocast')


def get_amp_context(enabled: bool = True, dtype: torch.dtype = torch.float16):
    """Get autocast context manager for mixed precision."""
    if enabled and _AMP_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return torch.cuda.amp.autocast(enabled=False)


# =============================================================================
# Protocol Classes
# =============================================================================

@runtime_checkable
class TrainableModel(Protocol):
    """Protocol for models compatible with generic training functions."""

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using batch data dictionary."""
        ...

    def train(self, mode: bool = True) -> 'TrainableModel':
        """Set training mode."""
        ...

    def eval(self) -> 'TrainableModel':
        """Set evaluation mode."""
        ...

    def parameters(self):
        """Return model parameters."""
        ...


@runtime_checkable
class TrainableDataset(Protocol):
    """Protocol for datasets compatible with generic training functions."""

    coords: torch.Tensor  # (N, D) coordinate tensor
    targets: torch.Tensor  # (N,) or (N, K) target tensor
    device: str  # Device string ('cuda' or 'cpu')
    n: int  # Number of samples

    def get_batch_data(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get all data needed for a batch given indices."""
        ...


# =============================================================================
# Generic Metrics
# =============================================================================

def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute generic regression metrics (MSE, RMSE, MAE, R2).

    Does NOT apply any task-specific denormalization - works on raw values.

    Args:
        predictions: (N,) tensor of predictions
        targets: (N,) tensor of targets
        prefix: Optional prefix for metric names (e.g., "train_")

    Returns:
        Dictionary with mse, rmse, mae, r2, n_samples
    """
    if len(predictions) == 0 or len(targets) == 0:
        return {
            f'{prefix}mse': 0.0,
            f'{prefix}rmse': 0.0,
            f'{prefix}mae': 0.0,
            f'{prefix}r2': 0.0,
            f'{prefix}n_samples': 0
        }

    errors = predictions - targets
    mse = (errors ** 2).mean()
    rmse = torch.sqrt(mse)
    mae = torch.abs(errors).mean()

    ss_res = (errors ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0, device=predictions.device)

    return {
        f'{prefix}mse': mse.item(),
        f'{prefix}rmse': rmse.item(),
        f'{prefix}mae': mae.item(),
        f'{prefix}r2': r2.item(),
        f'{prefix}n_samples': len(predictions)
    }


# =============================================================================
# Generic Training Functions
# =============================================================================

def train_epoch(
    model: TrainableModel,
    dataset: TrainableDataset,
    indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 20000,
    criterion: Optional[nn.Module] = None,
    loss_fn: Optional[Callable] = None,
    metrics_fn: Optional[Callable] = None,
    post_batch_fn: Optional[Callable] = None,
    use_block_shuffle: bool = False,
    block_size: int = 1024
) -> Dict[str, Any]:
    """
    Generic training epoch with callback pattern.

    Args:
        model: Model implementing TrainableModel protocol
        dataset: Dataset implementing TrainableDataset protocol
        indices: Training sample indices
        optimizer: PyTorch optimizer
        batch_size: Batch size
        criterion: Loss function (default: MSELoss)
        loss_fn: Optional custom loss: (preds, targets, batch_data, model) -> loss_dict
        metrics_fn: Optional metrics: (preds, targets, batch_data) -> metrics_dict
        post_batch_fn: Called after each batch (e.g., for probe updates)
        use_block_shuffle: Use locality-preserving shuffle
        block_size: Block size for block shuffle

    Returns:
        Dictionary with 'metrics' and 'all_predictions', 'all_targets'
    """
    model.train()
    n = len(indices)

    if criterion is None:
        criterion = nn.MSELoss()

    # Shuffle indices
    if use_block_shuffle:
        perm = block_shuffle_indices(n, block_size=block_size, device=indices.device)
    else:
        perm = torch.randperm(n, device=indices.device)
    indices = indices[perm]

    # Accumulate predictions and targets
    all_preds = []
    all_targets = []
    all_batch_data = []

    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        # Get batch data using dataset protocol method
        batch_data = dataset.get_batch_data(batch_idx)
        targets = batch_data['targets']

        # Forward pass
        preds = model.forward(batch_data)

        # Compute loss
        if loss_fn is not None:
            loss_dict = loss_fn(preds, targets, batch_data, model)
            loss = loss_dict['_total_loss_tensor']
        else:
            loss = criterion(preds, targets)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Post-batch callback (e.g., update probe indices)
        if post_batch_fn is not None:
            post_batch_fn()

        # Store for epoch metrics
        all_preds.append(preds.detach())
        all_targets.append(targets)
        all_batch_data.append(batch_data)

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    if metrics_fn is not None:
        # Merge batch data for metrics computation
        merged_batch_data = {}
        for key in all_batch_data[0].keys():
            if key != 'targets':  # Already have targets
                merged_batch_data[key] = torch.cat([bd[key] for bd in all_batch_data])
        metrics = metrics_fn(all_preds, all_targets, merged_batch_data)
    else:
        metrics = compute_regression_metrics(all_preds, all_targets)

    return {
        'metrics': metrics,
        'all_predictions': all_preds,
        'all_targets': all_targets
    }


def train_epoch_precomputed(
    model: TrainableModel,
    dataset: TrainableDataset,
    indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 20000,
    criterion: Optional[nn.Module] = None,
    loss_fn: Optional[Callable] = None,
    metrics_fn: Optional[Callable] = None,
    post_batch_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Training epoch using precomputed hash indices.

    Requires model to support forward_precomputed() method and dataset
    to include 'indices' in batch_data for indexing precomputed buffers.

    Args:
        Same as train_epoch except no shuffle options (precomputed indices fixed)

    Returns:
        Dictionary with 'metrics' and 'all_predictions', 'all_targets'
    """
    model.train()
    n = len(indices)

    if criterion is None:
        criterion = nn.MSELoss()

    # Standard random shuffle
    perm = torch.randperm(n, device=indices.device)
    indices = indices[perm]

    # Accumulate predictions and targets
    all_preds = []
    all_targets = []
    all_batch_data = []

    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        # Get batch data - must include 'indices' for precomputed lookup
        batch_data = dataset.get_batch_data(batch_idx)
        batch_data['indices'] = batch_idx  # Add indices for precomputed forward
        targets = batch_data['targets']

        # Forward pass using precomputed indices
        if hasattr(model, 'forward_precomputed'):
            preds = model.forward_precomputed(batch_data)
        else:
            preds = model.forward(batch_data)

        # Compute loss
        if loss_fn is not None:
            loss_dict = loss_fn(preds, targets, batch_data, model)
            loss = loss_dict['_total_loss_tensor']
        else:
            loss = criterion(preds, targets)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Post-batch callback
        if post_batch_fn is not None:
            post_batch_fn()

        # Store for epoch metrics
        all_preds.append(preds.detach())
        all_targets.append(targets)
        all_batch_data.append(batch_data)

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    if metrics_fn is not None:
        merged_batch_data = {}
        for key in all_batch_data[0].keys():
            if key not in ('targets', 'indices'):
                merged_batch_data[key] = torch.cat([bd[key] for bd in all_batch_data])
        metrics = metrics_fn(all_preds, all_targets, merged_batch_data)
    else:
        metrics = compute_regression_metrics(all_preds, all_targets)

    return {
        'metrics': metrics,
        'all_predictions': all_preds,
        'all_targets': all_targets
    }


def train_epoch_batched_forward(
    model: 'TrainableModel',
    dataset: 'TrainableDataset',
    indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 1024,
    criterion: Optional[nn.Module] = None,
    loss_fn: Optional[Callable] = None,
    metrics_fn: Optional[Callable] = None,
    post_batch_fn: Optional[Callable] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, Any]:
    """
    Training epoch with batched forward pass - single kernel launch for all samples.

    Key optimization: Instead of N forward kernel launches (one per batch),
    we do ONE forward pass for ALL samples, then loop for loss/backward per batch.
    This preserves small-batch gradient dynamics while reducing kernel launch overhead.

    Args:
        model: Model implementing TrainableModel protocol with forward_precomputed
        dataset: Dataset implementing TrainableDataset protocol
        indices: Training sample indices
        optimizer: PyTorch optimizer
        batch_size: Batch size for gradient computation
        criterion: Loss function (default: MSELoss)
        loss_fn: Optional custom loss callback
        metrics_fn: Optional metrics callback
        post_batch_fn: Optional post-batch callback
        use_amp: Enable automatic mixed precision
        scaler: GradScaler for AMP (created if None and use_amp=True)

    Returns:
        Dictionary with 'metrics', 'all_predictions', 'all_targets'
    """
    model.train()
    n = len(indices)

    if criterion is None:
        criterion = nn.MSELoss()

    # Setup AMP scaler if needed
    if use_amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()

    # Shuffle indices for the epoch
    perm = torch.randperm(n, device=indices.device)
    shuffled_indices = indices[perm]

    # Get ALL data for the epoch at once
    all_batch_data = dataset.get_batch_data(shuffled_indices)
    all_batch_data['indices'] = shuffled_indices
    all_targets = all_batch_data['targets']

    # SINGLE forward pass for ALL samples (one kernel launch!)
    with get_amp_context(use_amp):
        if hasattr(model, 'forward_precomputed'):
            all_preds = model.forward_precomputed(all_batch_data)
        else:
            all_preds = model.forward(all_batch_data)

    # Now loop through batches for loss/backward (preserves small-batch gradients)
    n_batches = (n + batch_size - 1) // batch_size
    batch_preds_list = []
    batch_targets_list = []

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)

        # Slice the pre-computed predictions and targets
        batch_preds = all_preds[start:end]
        batch_targets = all_targets[start:end]

        # Build batch_data for loss function
        batch_data = {
            'targets': batch_targets,
            'is_degenerate': all_batch_data.get('is_degenerate',
                torch.zeros(end - start, dtype=torch.bool, device=batch_preds.device))[start:end]
        }

        # Compute loss
        with get_amp_context(use_amp):
            if loss_fn is not None:
                loss_dict = loss_fn(batch_preds, batch_targets, batch_data, model)
                loss = loss_dict['_total_loss_tensor']
            else:
                loss = criterion(batch_preds, batch_targets)

        # Backward pass with optional scaling
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward(retain_graph=(i < n_batches - 1))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(retain_graph=(i < n_batches - 1))
            optimizer.step()

        # Post-batch callback - only call on last batch to avoid modifying
        # tensors while graph is retained (probe indices update is in-place)
        if post_batch_fn is not None and i == n_batches - 1:
            post_batch_fn()

        batch_preds_list.append(batch_preds.detach())
        batch_targets_list.append(batch_targets)

    # Compute metrics
    final_preds = torch.cat(batch_preds_list)
    final_targets = torch.cat(batch_targets_list)

    if metrics_fn is not None:
        merged_batch_data = {}
        for key in all_batch_data.keys():
            if key not in ('targets', 'indices'):
                merged_batch_data[key] = all_batch_data[key]
        metrics = metrics_fn(final_preds, final_targets, merged_batch_data)
    else:
        metrics = compute_regression_metrics(final_preds, final_targets)

    return {
        'metrics': metrics,
        'all_predictions': final_preds,
        'all_targets': final_targets
    }


def train_epoch_fused_sgd(
    model,
    dataset: 'TrainableDataset',
    indices: torch.Tensor,
    optimizer_mlp: torch.optim.Optimizer,
    batch_size: int = 256,
    lr: float = 0.01,
    criterion: Optional[nn.Module] = None,
    metrics_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Training epoch with fused backward+SGD for encoder embeddings.

    Uses fused CUDA kernel for encoder embedding updates (SGD), while using
    the provided optimizer for MLP and other parameters. ~22x faster than
    standard training for the encoder portion.

    Args:
        model: Model with earth4d encoder (e.g., SpeciesAwareLFMCModel)
        dataset: Dataset implementing TrainableDataset protocol
        indices: Training sample indices
        optimizer_mlp: Optimizer for MLP and non-encoder parameters
        batch_size: Batch size
        lr: Learning rate for encoder SGD updates
        criterion: Loss function (default: MSELoss)
        metrics_fn: Optional metrics callback

    Returns:
        Dictionary with 'metrics', 'all_predictions', 'all_targets'
    """
    model.train()
    n = len(indices)

    if criterion is None:
        criterion = nn.MSELoss()

    # Get encoder reference
    encoder = model.earth4d

    # Shuffle indices
    perm = torch.randperm(n, device=indices.device)
    shuffled_indices = indices[perm]

    all_preds = []
    all_targets = []
    all_is_degenerate = []

    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = shuffled_indices[start:end]

        batch_data = dataset.get_batch_data(batch_idx)
        targets = batch_data['targets']

        # Forward encoder with no grad (we update manually)
        with torch.no_grad():
            enc_output = encoder.forward_precomputed(batch_idx)

        # Clone with grad for backprop through MLP
        enc_output_grad = enc_output.clone().requires_grad_(True)

        # Species embedding + MLP (these use standard autograd)
        species_emb = model.species_embeddings(batch_data['species_idx'])
        combined = torch.cat([enc_output_grad, species_emb], dim=-1)
        preds = model.mlp(combined).squeeze(-1)

        loss = criterion(preds, targets)

        # Backward for MLP and species embeddings
        optimizer_mlp.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_mlp.step()

        # Get gradient w.r.t encoder output
        grad_enc = enc_output_grad.grad

        # Fused backward + SGD for each encoder
        for enc_name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
            enc = getattr(encoder, enc_name)
            out_dim = enc.output_dim

            if enc_name == 'xyz_encoder':
                grad_slice = grad_enc[:, :out_dim]
            elif enc_name == 'xyt_encoder':
                start_idx = encoder.xyz_encoder.output_dim
                grad_slice = grad_enc[:, start_idx:start_idx + out_dim]
            elif enc_name == 'yzt_encoder':
                start_idx = encoder.xyz_encoder.output_dim + encoder.xyt_encoder.output_dim
                grad_slice = grad_enc[:, start_idx:start_idx + out_dim]
            else:  # xzt_encoder
                start_idx = encoder.xyz_encoder.output_dim + encoder.xyt_encoder.output_dim + encoder.yzt_encoder.output_dim
                grad_slice = grad_enc[:, start_idx:start_idx + out_dim]

            enc.backward_sgd_fused(grad_slice, batch_idx, lr)

        all_preds.append(preds.detach())
        all_targets.append(targets)
        if 'is_degenerate' in batch_data:
            all_is_degenerate.append(batch_data['is_degenerate'])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Build combined batch_data for metrics
    combined_batch_data = {}
    if all_is_degenerate:
        combined_batch_data['is_degenerate'] = torch.cat(all_is_degenerate)

    if metrics_fn is not None:
        metrics = metrics_fn(all_preds, all_targets, combined_batch_data)
    else:
        metrics = compute_regression_metrics(all_preds, all_targets)

    return {
        'metrics': metrics,
        'all_predictions': all_preds,
        'all_targets': all_targets
    }


def train_epoch_fused_adam(
    model,
    dataset: 'TrainableDataset',
    indices: torch.Tensor,
    optimizer_mlp: torch.optim.Optimizer,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 0.001,
    criterion: Optional[nn.Module] = None,
    loss_fn: Optional[Callable] = None,
    metrics_fn: Optional[Callable] = None,
    post_batch_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Training epoch with sparse Adam for encoder embeddings.

    Uses CUDA kernel that only updates embeddings touched by each batch,
    providing ~20x speedup while maintaining Adam optimization dynamics.

    IMPORTANT: This function is designed to produce IDENTICAL results to the
    standard training pipeline. It uses soft gradients for learned probing,
    supports entropy regularization via loss_fn, and calls post_batch_fn
    for probe index updates.

    Args:
        model: Model with earth4d encoder (e.g., SpeciesAwareLFMCModel)
        dataset: Dataset implementing TrainableDataset protocol
        indices: Training sample indices
        optimizer_mlp: Optimizer for MLP, species embeddings, AND index_logits
        batch_size: Batch size
        lr: Learning rate for encoder Adam updates
        weight_decay: AdamW weight decay (default: 0.001)
        criterion: Loss function (default: MSELoss) - used if loss_fn is None
        loss_fn: Custom loss callback: (preds, targets, batch_data, model) -> loss_dict
                 Should return dict with '_total_loss_tensor' key
        metrics_fn: Optional metrics callback
        post_batch_fn: Called after each batch (e.g., for probe index updates)

    Returns:
        Dictionary with 'metrics', 'all_predictions', 'all_targets'
    """
    model.train()
    n = len(indices)

    if criterion is None:
        criterion = nn.MSELoss()

    encoder = model.earth4d

    # Initialize or update sparse Adam with current lr and weight_decay
    for enc_name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
        enc = getattr(encoder, enc_name)
        if not enc._adam_initialized:
            enc.init_sparse_adam(lr=lr, weight_decay=weight_decay)
        else:
            # Update lr in case it changed (for LR decay)
            enc.set_adam_lr(lr)

    # Shuffle indices
    perm = torch.randperm(n, device=indices.device)
    shuffled_indices = indices[perm]

    all_preds = []
    all_targets = []
    all_is_degenerate = []

    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = shuffled_indices[start:end]

        batch_data = dataset.get_batch_data(batch_idx)
        targets = batch_data['targets']

        # Forward encoder with no grad (we update manually via sparse Adam)
        with torch.no_grad():
            enc_output = encoder.forward_precomputed(batch_idx)

        # Clone with grad for backprop through MLP
        enc_output_grad = enc_output.clone().requires_grad_(True)

        # Species embedding + MLP
        species_emb = model.species_embeddings(batch_data['species_idx'])
        combined = torch.cat([enc_output_grad, species_emb], dim=-1)
        preds = model.mlp(combined).squeeze(-1)

        # Compute loss - use loss_fn if provided (includes entropy regularization)
        if loss_fn is not None:
            loss_dict = loss_fn(preds, targets, batch_data, model)
            loss = loss_dict['_total_loss_tensor']
        else:
            loss = criterion(preds, targets)

        # Backward for MLP, species embeddings, and index_logits
        optimizer_mlp.zero_grad(set_to_none=True)
        loss.backward()

        # Get gradient w.r.t encoder output BEFORE optimizer step
        grad_enc = enc_output_grad.grad

        # Sparse Adam update for each encoder's embeddings
        # Also accumulate gradients for index_logits via soft backward
        for enc_name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
            enc = getattr(encoder, enc_name)
            out_dim = enc.output_dim

            if enc_name == 'xyz_encoder':
                grad_slice = grad_enc[:, :out_dim]
            elif enc_name == 'xyt_encoder':
                start_idx = encoder.xyz_encoder.output_dim
                grad_slice = grad_enc[:, start_idx:start_idx + out_dim]
            elif enc_name == 'yzt_encoder':
                start_idx = encoder.xyz_encoder.output_dim + encoder.xyt_encoder.output_dim
                grad_slice = grad_enc[:, start_idx:start_idx + out_dim]
            else:  # xzt_encoder
                start_idx = encoder.xyz_encoder.output_dim + encoder.xyt_encoder.output_dim + encoder.yzt_encoder.output_dim
                grad_slice = grad_enc[:, start_idx:start_idx + out_dim]

            # Accumulate gradients (uses soft gradients for learned probing)
            enc.accumulate_grad(grad_slice, batch_idx)

            # Transfer index_logits gradients to .grad for optimizer
            enc.transfer_index_logits_grad()

            # Apply sparse Adam step for embeddings
            enc.adam_step(batch_idx)

        # Now step the optimizer for MLP, species embeddings, and index_logits
        optimizer_mlp.step()

        # Post-batch callback (e.g., update probe indices)
        if post_batch_fn is not None:
            post_batch_fn()

        all_preds.append(preds.detach())
        all_targets.append(targets)
        if 'is_degenerate' in batch_data:
            all_is_degenerate.append(batch_data['is_degenerate'])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Build combined batch_data for metrics
    combined_batch_data = {}
    if all_is_degenerate:
        combined_batch_data['is_degenerate'] = torch.cat(all_is_degenerate)

    if metrics_fn is not None:
        metrics = metrics_fn(all_preds, all_targets, combined_batch_data)
    else:
        metrics = compute_regression_metrics(all_preds, all_targets)

    return {
        'metrics': metrics,
        'all_predictions': all_preds,
        'all_targets': all_targets
    }


@torch.no_grad()
def evaluate(
    model: TrainableModel,
    dataset: TrainableDataset,
    indices: torch.Tensor,
    metrics_fn: Optional[Callable] = None,
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generic evaluation function.

    Args:
        model: Model implementing TrainableModel protocol
        dataset: Dataset implementing TrainableDataset protocol
        indices: Evaluation sample indices
        metrics_fn: Optional metrics computation callback
        batch_size: Optional batching for large datasets (None = single batch)

    Returns:
        Dictionary with 'metrics', 'predictions', 'targets', and batch_data fields
    """
    model.eval()

    if batch_size is None or len(indices) <= batch_size:
        # Single batch evaluation
        batch_data = dataset.get_batch_data(indices)
        targets = batch_data['targets']
        preds = model.forward(batch_data)

        if metrics_fn is not None:
            metrics = metrics_fn(preds, targets, batch_data)
        else:
            metrics = compute_regression_metrics(preds, targets)

        return {
            'metrics': metrics,
            'predictions': preds,
            'targets': targets,
            'batch_data': batch_data
        }

    # Batched evaluation for large datasets
    all_preds = []
    all_targets = []
    all_batch_data = []

    n = len(indices)
    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]

        batch_data = dataset.get_batch_data(batch_idx)
        targets = batch_data['targets']
        preds = model.forward(batch_data)

        all_preds.append(preds)
        all_targets.append(targets)
        all_batch_data.append(batch_data)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Merge batch data
    merged_batch_data = {}
    for key in all_batch_data[0].keys():
        if key != 'targets':
            merged_batch_data[key] = torch.cat([bd[key] for bd in all_batch_data])

    if metrics_fn is not None:
        metrics = metrics_fn(all_preds, all_targets, merged_batch_data)
    else:
        metrics = compute_regression_metrics(all_preds, all_targets)

    return {
        'metrics': metrics,
        'predictions': all_preds,
        'targets': all_targets,
        'batch_data': merged_batch_data
    }


# =============================================================================
# Loss Computation
# =============================================================================

def compute_loss(predictions: torch.Tensor, targets: torch.Tensor, encoder,
                 criterion: Optional[Any] = None, enable_learned_probing: bool = False,
                 probe_entropy_weight: float = 0.5, enable_probe_entropy_loss: Optional[bool] = None,
                 enable_gradient_validation: bool = False) -> Dict:
    """Compute loss with optional entropy regularization for learned hash probing."""
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if enable_probe_entropy_loss is None:
        enable_probe_entropy_loss = enable_learned_probing

    task_loss = criterion(predictions, targets)
    total_loss = task_loss
    loss_dict = {'task_loss': task_loss, 'total_loss': task_loss}

    if enable_probe_entropy_loss and hasattr(encoder, 'xyt_encoder') and \
       getattr(encoder.xyt_encoder, 'enable_learned_probing', False):
        entropy = _compute_probe_entropy(encoder)
        total_loss = total_loss - probe_entropy_weight * entropy
        loss_dict['probe_entropy_loss'] = entropy
        loss_dict['total_loss'] = total_loss

    loss_dict['_total_loss_tensor'] = total_loss
    return loss_dict


def _compute_probe_entropy(encoder) -> torch.Tensor:
    """Compute entropy of probe distributions."""
    total_entropy = 0.0
    num_encoders = 0
    for name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
        enc = getattr(encoder, name, None)
        if enc is None or not getattr(enc, 'enable_learned_probing', False):
            continue
        if not hasattr(enc, 'index_logits') or enc.index_logits is None:
            continue
        probs = torch.softmax(enc.index_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        total_entropy = total_entropy + entropy
        num_encoders += 1
    if num_encoders == 0:
        return torch.tensor(0.0, device=encoder.xyz_encoder.embeddings.device)
    return total_entropy / num_encoders


# =============================================================================
# Reporting
# =============================================================================

def print_resolution_info(encoder, config: Dict[str, Any], adaptive_range: Optional[Any] = None):
    """Print detailed resolution information for Earth4D encoder."""
    results = _calculate_resolution_scales(encoder)

    print("\n" + "="*80)
    print("EARTH4D INITIALIZATION REPORT")
    print("="*80)

    print("\n┌─ ENHANCEMENT CONFIGURATION ─────────────────────────────────────────────┐")
    print(f"│  Adaptive Range:     {'ENABLED' if config.get('use_adaptive_range') else 'disabled':12}                                  │")
    lp_str = f"ENABLED (N_p={config.get('probing_range', 0)})" if config.get('enable_learned_probing') else 'disabled'
    print(f"│  Learned Probing:    {lp_str:24}              │")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "-"*80)
    print("RESOLUTION SCALE TABLE")
    print("-"*80)

    effective_multiplier = 1.0
    if config.get('use_adaptive_range') and adaptive_range is not None:
        coverage = adaptive_range.get_coordinate_coverage()
        avg_coverage = sum(coverage.values()) / 3
        if avg_coverage > 0:
            effective_multiplier = 1.0 / avg_coverage

    print("\nSPATIAL ENCODER (XYZ):")
    print(f"{'Level':<6} {'Grid Res':<12} {'Meters/Cell':<15} {'KM/Cell':<12}")
    print("-" * 70)
    for item in results['spatial']:
        meters = item['meters_per_cell']
        if meters >= 1000:
            meters_str = f"{meters/1000:.1f}km"
        elif meters >= 1:
            meters_str = f"{meters:.2f}m"
        else:
            meters_str = f"{meters:.3f}m"
        km_str = f"{item['km_per_cell']:.3f}" if item['km_per_cell'] < 1 else f"{item['km_per_cell']:.2f}"
        print(f"{item['level']:<6} {item['grid_resolution']:<12} {meters_str:<15} {km_str:<12}")

    print("\nSPATIOTEMPORAL ENCODERS (XYT, YZT, XZT):")
    print(f"{'Level':<6} {'Grid Res':<12} {'Seconds/Cell':<15} {'Days/Cell':<12}")
    print("-" * 70)
    for item in results['temporal']['xyt']:
        print(f"{item['level']:<6} {item['grid_resolution']:<12} {item['seconds_per_cell']:<15.1f} {item['days_per_cell']:<12.2f}")

    spatial_params = encoder.xyz_encoder.embeddings.numel()
    temporal_params = sum(getattr(encoder, f'{n}_encoder').embeddings.numel() for n in ['xyt', 'yzt', 'xzt'])
    total_params = spatial_params + temporal_params
    total_memory = total_params * 4 / (1024 * 1024)

    spatial_hash_entries = 2 ** encoder.spatial_log2_hashmap_size
    temporal_hash_entries = 2 ** encoder.temporal_log2_hashmap_size

    print(f"\nHASH TABLE CONFIGURATION:")
    print(f"  Spatial: 2^{encoder.spatial_log2_hashmap_size} = {spatial_hash_entries:,} entries")
    print(f"  Spatiotemporal: 2^{encoder.temporal_log2_hashmap_size} = {temporal_hash_entries:,} entries")
    print(f"  Total capacity: {spatial_hash_entries + temporal_hash_entries*3:,} entries")

    print(f"\nACTUAL PARAMETERS (MEMORY FOOTPRINT):")
    print(f"  Spatial encoders: {spatial_params:,} params = {spatial_params * 4 / (1024*1024):.2f} MB")
    print(f"  Spatiotemporal encoders: {temporal_params:,} params = {temporal_params * 4 / (1024*1024):.2f} MB")
    print(f"  Total: {total_params:,} params = {total_memory:.2f} MB")
    print(f"  During training (4x): ~{total_memory * 4:.2f} MB")


def _calculate_resolution_scales(encoder) -> Dict:
    """Calculate resolution scales for all encoders."""
    earth_radius = 6371000.0
    physical_range = 2 * earth_radius
    results = {'spatial': [], 'temporal': {'xyt': [], 'yzt': [], 'xzt': []}}

    spatial_encoder = encoder.xyz_encoder
    for level in range(spatial_encoder.num_levels):
        base_res = spatial_encoder.base_resolution[0].item()
        scale = spatial_encoder.per_level_scale[0].item()
        grid_resolution = np.ceil(base_res * (scale ** level))
        meters_per_cell = physical_range / grid_resolution
        results['spatial'].append({
            'level': level, 'grid_resolution': int(grid_resolution),
            'meters_per_cell': meters_per_cell, 'km_per_cell': meters_per_cell / 1000
        })

    seconds_per_year = 365.25 * 24 * 3600
    for name, enc in [('xyt', encoder.xyt_encoder), ('yzt', encoder.yzt_encoder), ('xzt', encoder.xzt_encoder)]:
        for level in range(enc.num_levels):
            base_res = enc.base_resolution[0].item()
            scale = enc.per_level_scale[0].item()
            grid_resolution = np.ceil(base_res * (scale ** level))
            seconds_per_cell = seconds_per_year / grid_resolution
            results['temporal'][name].append({
                'level': level, 'grid_resolution': int(grid_resolution),
                'seconds_per_cell': seconds_per_cell, 'days_per_cell': seconds_per_cell / 86400
            })
    return results
