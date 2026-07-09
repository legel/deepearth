"""
Earth4D Trainer - Train models with Earth4D spatiotemporal encoding.

Example:
    from trainer import Earth4DTrainer

    # Create trainer
    trainer = Earth4DTrainer(model, dataset, train_indices)

    # Training loop
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch()
        print(f"Epoch {epoch}: {metrics}")
        trainer.decay_lr()

    # Evaluate
    test_metrics = trainer.evaluate(test_indices)

For a complete example with custom loss functions and visualization,
see benchmarks/lfmc/train.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Callable, List

from ops import train_epoch_fused_adam, train_epoch_precomputed, evaluate


class Earth4DTrainer:
    """
    Train models that use Earth4D encoding.

    Args:
        model: Your model (must have a .earth4d attribute)
        dataset: Dataset with .coords and get_batch_data() method
        train_indices: Indices of training samples
        lr: Learning rate (default: 0.00025)
        weight_decay: Weight decay (default: 0.001)
        batch_size: Batch size (default: 256)
        lr_decay: Learning rate decay per epoch (default: 0.99995)
        use_fused_adam: Use fast training mode (default: True)
        verbose: Print progress (default: True)
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        train_indices: torch.Tensor,
        lr: float = 0.00025,
        weight_decay: float = 0.001,
        batch_size: int = 256,
        lr_decay: float = 0.99995,
        use_fused_adam: bool = True,
        index_lr_multiplier: float = 10.0,
        verbose: bool = True,
    ):
        self.model = model
        self.dataset = dataset
        self.train_indices = train_indices
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.use_fused_adam = use_fused_adam
        self.index_lr_multiplier = index_lr_multiplier
        self.verbose = verbose

        self._encoder_lr = lr
        self._setup()

    def _setup(self):
        """Prepare model for training."""
        earth4d = self.model.earth4d

        if self.verbose:
            print(f"Preparing trainer for {len(self.dataset.coords):,} samples...")

        # Cache coordinate encodings for faster training
        stats = earth4d.precompute(self.dataset.coords)

        if self.verbose:
            print(f"  Ready ({stats['total_mb']:.0f} MB)")

        if self.use_fused_adam:
            self._setup_fused_optimizer()
        else:
            self._setup_standard_optimizer()

    def _get_non_encoder_params(self) -> List[nn.Parameter]:
        """Get model parameters excluding encoder embeddings."""
        earth4d = self.model.earth4d
        encoder_param_ids = set()

        for enc_name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
            enc = getattr(earth4d, enc_name, None)
            if enc is not None:
                encoder_param_ids.add(id(enc.embeddings))

        return [p for p in self.model.parameters()
                if id(p) not in encoder_param_ids and p.requires_grad]

    def _get_index_logits(self) -> List[nn.Parameter]:
        """Get learned probing parameters if enabled."""
        earth4d = self.model.earth4d
        params = []
        for enc_name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
            enc = getattr(earth4d, enc_name, None)
            if enc is not None and hasattr(enc, 'index_logits') and enc.index_logits is not None:
                params.append(enc.index_logits)
        return params

    def _setup_fused_optimizer(self):
        """Setup optimizer for fast training mode."""
        non_encoder_params = self._get_non_encoder_params()
        index_logits = self._get_index_logits()

        index_logit_ids = {id(p) for p in index_logits}
        base_params = [p for p in non_encoder_params if id(p) not in index_logit_ids]

        param_groups = [{'params': base_params, 'lr': self.lr}]

        if index_logits:
            param_groups.append({
                'params': index_logits,
                'lr': self.lr * self.index_lr_multiplier
            })

        try:
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay, fused=True)
        except TypeError:
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)

    def _setup_standard_optimizer(self):
        """Setup standard optimizer."""
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr,
                weight_decay=self.weight_decay, fused=True
            )
        except TypeError:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr,
                weight_decay=self.weight_decay
            )

    def train_epoch(
        self,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
        post_batch_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run one training epoch.

        Args:
            loss_fn: Custom loss function (optional)
            metrics_fn: Custom metrics function (optional)
            post_batch_fn: Callback after each batch (optional)

        Returns:
            Dictionary with training metrics
        """
        if self.use_fused_adam:
            return train_epoch_fused_adam(
                self.model, self.dataset, self.train_indices,
                self.optimizer, self.batch_size,
                lr=self._encoder_lr, weight_decay=self.weight_decay,
                loss_fn=loss_fn, metrics_fn=metrics_fn, post_batch_fn=post_batch_fn
            )
        else:
            return train_epoch_precomputed(
                self.model, self.dataset, self.train_indices,
                self.optimizer, self.batch_size,
                loss_fn=loss_fn, metrics_fn=metrics_fn, post_batch_fn=post_batch_fn
            )

    def evaluate(
        self,
        indices: torch.Tensor,
        metrics_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on given sample indices."""
        return evaluate(self.model, self.dataset, indices, metrics_fn=metrics_fn)

    def decay_lr(self):
        """Apply learning rate decay."""
        for g in self.optimizer.param_groups:
            g['lr'] *= self.lr_decay
        if self.use_fused_adam:
            self._encoder_lr *= self.lr_decay

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'encoder_lr': self._encoder_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        self.optimizer.load_state_dict(state['optimizer'])
        self._encoder_lr = state['encoder_lr']
