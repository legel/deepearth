import enum
from math import ceil
from cachetools import cached
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Use torch.amp instead of deprecated torch.cuda.amp
try:
    from torch.amp import custom_bwd, custom_fwd
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, track_collisions=False, collision_indices=None, example_offset=0, max_tracked_examples=0, probe_indices=None, index_logits=None, N_f=0, N_p=1, N_c=0):
        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()
        per_level_scale = per_level_scale.contiguous()
        base_resolution = base_resolution.contiguous()

        if probe_indices is not None:
            probe_indices = probe_indices.contiguous()
        else:
            probe_indices = torch.empty(0, dtype=torch.int32, device=inputs.device)

        if index_logits is not None:
            index_logits = index_logits.contiguous().to(torch.float32)
        else:
            index_logits = torch.empty(0, dtype=torch.float32, device=inputs.device)

        B, D = inputs.shape
        L = offsets.shape[0] - 1
        C = embeddings.shape[1]
        per_level_scale = torch.log2(per_level_scale)

        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = torch.empty(1, device=inputs.device, dtype=embeddings.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c)

        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, probe_indices, index_logits)
        ctx.dims = [B, D, C, L]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.learned_probing_params = [N_f, N_p, N_c]

        return outputs
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, probe_indices, index_logits = ctx.saved_tensors
        B, D, C, L = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs
        N_f, N_p, N_c = ctx.learned_probing_params

        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_inputs, grad_embeddings, grad_index_logits = _hash_encode_second_backward.apply(grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, index_logits, N_f, N_p, N_c)

        if index_logits.requires_grad and index_logits.numel() > 0:
            grad_index_logits_return = grad_index_logits
        else:
            grad_index_logits_return = None

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None, None, grad_index_logits_return, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None, None, None, None, None, None, grad_index_logits_return, None, None, None


class _hash_encode_second_backward(Function):
    @staticmethod
    def forward(ctx, grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, index_logits, N_f, N_p, N_c):
        device = inputs.device
        grad_inputs_f32 = torch.zeros_like(inputs, device=device, dtype=embeddings.dtype)
        grad_embeddings = torch.zeros_like(embeddings, device=device)

        if index_logits is not None and index_logits.numel() > 0:
            grad_index_logits = torch.zeros_like(index_logits, device=device, dtype=torch.float32)
        else:
            grad_index_logits = torch.empty(0, device=device, dtype=torch.float32)

        ctx.save_for_backward(grad, inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, grad_inputs_f32, grad_embeddings, probe_indices, index_logits, grad_index_logits)
        ctx.dims = [B, D, C, L]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.learned_probing_params = [N_f, N_p, N_c]

        if index_logits.numel() > 0 and index_logits.dtype != torch.float32:
            index_logits_f32 = index_logits.to(torch.float32)
        else:
            index_logits_f32 = index_logits

        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs_f32, probe_indices, index_logits_f32, grad_index_logits, N_f, N_p, N_c)

        if inputs.dtype != embeddings.dtype:
            grad_inputs = grad_inputs_f32.to(inputs.dtype)
        else:
            grad_inputs = grad_inputs_f32

        return grad_inputs, grad_embeddings, grad_index_logits

    @staticmethod
    def backward(ctx, grad_grad_inputs, grad_grad_embeddings, grad_grad_index_logits):
        grad, inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, grad_inputs, grad_embeddings, probe_indices, index_logits, grad_index_logits = ctx.saved_tensors
        B, D, C, L = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        device = grad.device
        grad_grad = torch.zeros_like(grad, device=device)
        grad2_embeddings = torch.zeros_like(embeddings, device=device)

        _backend.hash_encode_second_backward(grad, inputs, embeddings, offsets,
                                             B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx,
                                             grad_grad_inputs,
                                             grad_grad, grad2_embeddings)

        return grad_grad, None, grad2_embeddings, None, None, None, None, None, None, None, None, None, None, None, None, None, None


hash_encode = _hash_encode.apply


# =============================================================================
# Precomputed Hash Encoding Autograd Function
# =============================================================================

class _hash_encode_precomputed(Function):
    """
    Hash encoding using precomputed indices and weights.

    For fixed coordinate sets, this avoids recomputing hash indices and
    interpolation weights every forward pass. Only embedding lookups change.

    With learned probing, uses soft gradients (matching standard backward) to enable
    gradient flow to index_logits.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, embeddings, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits, B, D, C, L, N_p, N_c):
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()

        if probe_indices is not None:
            probe_indices = probe_indices.contiguous()
        else:
            probe_indices = torch.empty(0, dtype=torch.int32, device=embeddings.device)

        if index_logits is not None:
            index_logits = index_logits.contiguous().to(torch.float32)
        else:
            index_logits = torch.empty(0, dtype=torch.float32, device=embeddings.device)

        outputs = torch.empty(L, B, C, device=embeddings.device, dtype=embeddings.dtype)

        _backend.hash_encode_forward_precomputed(
            embeddings, offsets,
            precomp_h1, precomp_h2, precomp_weights,
            probe_indices, outputs,
            B, D, C, L, N_p, N_c
        )

        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits)
        ctx.dims = [B, D, C, L]
        ctx.probing_params = [N_p, N_c]
        ctx.embeddings_shape = embeddings.shape

        return outputs

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits = ctx.saved_tensors
        B, D, C, L = ctx.dims
        N_p, N_c = ctx.probing_params

        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros(ctx.embeddings_shape, device=grad.device, dtype=grad.dtype)

        # Allocate grad_index_logits if we have index_logits (for soft gradient mode)
        if index_logits.numel() > 0:
            grad_index_logits = torch.zeros_like(index_logits, dtype=torch.float32)
        else:
            grad_index_logits = torch.empty(0, dtype=torch.float32, device=grad.device)

        _backend.hash_encode_backward_precomputed(
            grad, offsets,
            precomp_h1, precomp_h2, precomp_weights,
            probe_indices, index_logits, grad_index_logits, grad_embeddings,
            B, D, C, L, N_p, N_c
        )

        # Return gradients for: embeddings, offsets, h1, h2, weights, probe_indices, index_logits, B, D, C, L, N_p, N_c
        grad_index_logits_return = grad_index_logits if index_logits.numel() > 0 else None
        return grad_embeddings, None, None, None, None, None, grad_index_logits_return, None, None, None, None, None, None


hash_encode_precomputed = _hash_encode_precomputed.apply


# =============================================================================
# HashEncoder Module
# =============================================================================

class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, enable_learned_probing=False, probing_range=4, index_codebook_size=1024):
        super().__init__()

        if type(base_resolution) is int:
            base_resolution = np.array([base_resolution for _ in range(input_dim)], dtype=np.float64)
        else:
            assert len(base_resolution) == input_dim
            base_resolution = np.array(base_resolution, dtype=np.float64)

        if desired_resolution is not None:
            if type(desired_resolution) is int:
                desired_resolution = np.array([desired_resolution for _ in range(input_dim)], dtype=np.float64)
            else:
                assert len(desired_resolution) == input_dim
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        else:
            if type(per_level_scale) is int or type(per_level_scale) is float:
                per_level_scale = np.array([per_level_scale for _ in range(input_dim)], dtype=np.float64)
            else:
                assert len(per_level_scale) == input_dim
                per_level_scale = np.array(per_level_scale, dtype=np.float64)

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size

        self.enable_learned_probing = enable_learned_probing
        if enable_learned_probing:
            self.N_p = probing_range
            self.N_c = index_codebook_size
            self.max_params = (2 ** log2_hashmap_size) // probing_range
            self.N_f = self.max_params
        else:
            self.max_params = 2 ** log2_hashmap_size
            self.N_p = 1
            self.N_c = 1
            self.N_f = self.max_params

        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16!')

        offsets = []
        offset = 0
        for i in range(num_levels):
            resolution = np.ceil(base_resolution * per_level_scale ** i)
            if enable_learned_probing:
                params_in_level = self.max_params * self.N_p
            else:
                params_in_level = min(self.max_params, int(np.prod(resolution)))
            offsets.append(offset)
            offset += int(params_in_level)
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        self.register_buffer('per_level_scale', torch.tensor(per_level_scale, dtype=torch.float32))
        self.register_buffer('base_resolution', torch.tensor(base_resolution, dtype=torch.float32))

        self.n_params = offsets[-1] * level_dim
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        if enable_learned_probing:
            self.index_logits = nn.Parameter(torch.randn(num_levels, index_codebook_size, probing_range) * 0.01)
            self.register_buffer('probe_indices', torch.zeros(num_levels, index_codebook_size, dtype=torch.int32))
        else:
            self.index_logits = None
            self.probe_indices = None

        # Precomputation buffers (initialized when precompute() is called)
        self._precomputed = False
        self._precomp_h1 = None
        self._precomp_h2 = None
        self._precomp_weights = None
        self._precomp_pos_deriv = None
        self._precomp_B = 0

        # Sparse Adam state (initialized when init_sparse_adam() is called)
        self._adam_initialized = False
        self._adam_exp_avg = None      # First moment
        self._adam_exp_avg_sq = None   # Second moment
        self._adam_step = 0            # Step counter
        self._adam_grad_buffer = None  # Gradient buffer for sparse updates

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-1
        self.embeddings.data.uniform_(-std, std)

    def update_probe_indices(self):
        """Update discrete probe indices from logits (call after optimizer step)"""
        if self.enable_learned_probing and self.index_logits is not None:
            with torch.no_grad():
                self.probe_indices.copy_(torch.argmax(self.index_logits, dim=-1))

    def __repr__(self):
        precomp_str = f" precomputed={self._precomputed}" if self._precomputed else ""
        if self.enable_learned_probing:
            return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} learned_probing=True N_f={self.N_f} N_p={self.N_p} N_c={self.N_c} params={tuple(self.embeddings.shape)}{precomp_str}"
        else:
            return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} params={tuple(self.embeddings.shape)}{precomp_str}"

    def precompute(self, inputs, size=1.0):
        """
        Precompute hash indices and interpolation weights for a fixed set of coordinates.
        
        Call this once before training with all training coordinates.
        Subsequent forward_precomputed() calls will use cached values.
        
        Args:
            inputs: [..., input_dim] coordinates in [-size, size]
            size: coordinate range normalization
            
        Returns:
            dict with memory usage statistics
        """
        inputs = (inputs + size) / (2 * size)  # map to [0, 1]
        inputs = inputs.view(-1, self.input_dim).contiguous()
        
        B = inputs.shape[0]
        D = self.input_dim
        L = self.num_levels
        C = self.level_dim
        n_corners = 2 ** D
        
        # Allocate precomputation buffers
        self._precomp_h1 = torch.zeros(B, L, n_corners, dtype=torch.int32, device=inputs.device)
        self._precomp_h2 = torch.zeros(B, L, n_corners, dtype=torch.int32, device=inputs.device)
        self._precomp_weights = torch.zeros(B, L, n_corners, dtype=torch.float32, device=inputs.device)
        self._precomp_pos_deriv = torch.zeros(B, L, D, dtype=torch.float32, device=inputs.device)
        
        # Run CUDA precomputation kernel
        _backend.hash_encode_precompute(
            inputs, self.offsets,
            self._precomp_h1, self._precomp_h2, self._precomp_weights, self._precomp_pos_deriv,
            B, D, C, L,
            self.per_level_scale, self.base_resolution,
            self.N_f, self.N_p
        )
        
        self._precomputed = True
        self._precomp_B = B
        
        # Calculate memory usage
        h1_bytes = self._precomp_h1.numel() * 4
        h2_bytes = self._precomp_h2.numel() * 4
        weights_bytes = self._precomp_weights.numel() * 4
        pos_deriv_bytes = self._precomp_pos_deriv.numel() * 4
        total_bytes = h1_bytes + h2_bytes + weights_bytes + pos_deriv_bytes
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'bytes_per_coord': total_bytes / B,
            'num_coords': B,
            'num_levels': L,
            'num_corners': n_corners,
            'breakdown': {
                'h1': h1_bytes,
                'h2': h2_bytes,
                'weights': weights_bytes,
                'pos_deriv': pos_deriv_bytes
            }
        }

    def clear_precomputed(self):
        """Clear precomputed buffers to free memory."""
        self._precomputed = False
        self._precomp_h1 = None
        self._precomp_h2 = None
        self._precomp_weights = None
        self._precomp_pos_deriv = None
        self._precomp_B = 0

    def forward_precomputed(self, batch_indices=None):
        """
        Forward pass using precomputed indices and weights.

        Must call precompute() first with the full coordinate set.

        Args:
            batch_indices: Optional tensor of indices into the precomputed coordinates.
                          If None, uses all precomputed coordinates.

        Returns:
            [..., num_levels * level_dim] encoded features
        """
        if not self._precomputed:
            raise RuntimeError("Must call precompute() before forward_precomputed()")

        # Update probe indices if training with learned probing
        if self.enable_learned_probing and self.training:
            self.update_probe_indices()

        if batch_indices is not None:
            # Select subset of precomputed values
            h1 = self._precomp_h1[batch_indices]
            h2 = self._precomp_h2[batch_indices]
            weights = self._precomp_weights[batch_indices]
            B = batch_indices.shape[0]
        else:
            h1 = self._precomp_h1
            h2 = self._precomp_h2
            weights = self._precomp_weights
            B = self._precomp_B

        probe_indices = self.probe_indices if self.enable_learned_probing else None
        index_logits = self.index_logits if self.enable_learned_probing else None

        outputs = hash_encode_precomputed(
            self.embeddings, self.offsets,
            h1, h2, weights,
            probe_indices, index_logits,
            B, self.input_dim, self.level_dim, self.num_levels,
            self.N_p if self.enable_learned_probing else 1,
            self.N_c if self.enable_learned_probing else 0
        )

        return outputs

    def forward_precomputed_multibatch(self, sample_indices):
        """
        Forward pass for multiple mini-batches in a single kernel launch.

        Uses sample_indices to gather from precomputed buffers, enabling
        processing of multiple batches without separate kernel launches.

        Args:
            sample_indices: [N_total] tensor of indices into precomputed buffers

        Returns:
            [N_total, num_levels * level_dim] encoded features
        """
        if not self._precomputed:
            raise RuntimeError("Must call precompute() before forward_precomputed_multibatch()")

        sample_indices = sample_indices.contiguous().to(torch.int32)
        N_total = sample_indices.shape[0]

        if self.enable_learned_probing and self.training:
            self.update_probe_indices()

        probe_indices = self.probe_indices if self.enable_learned_probing else None
        if probe_indices is None:
            probe_indices = torch.empty(0, dtype=torch.int32, device=self.embeddings.device)

        outputs = torch.empty(self.num_levels, N_total, self.level_dim,
                              device=self.embeddings.device, dtype=self.embeddings.dtype)

        _backend.hash_encode_forward_precomputed_multibatch(
            self.embeddings, self.offsets,
            self._precomp_h1, self._precomp_h2, self._precomp_weights,
            probe_indices, sample_indices, outputs,
            N_total, self.input_dim, self.level_dim, self.num_levels,
            self.N_p if self.enable_learned_probing else 1,
            self.N_c if self.enable_learned_probing else 0
        )

        return outputs.permute(1, 0, 2).reshape(N_total, self.output_dim)

    def backward_sgd_fused(self, grad, batch_indices, lr):
        """
        Fused backward pass + SGD update for embeddings.

        Eliminates gradient buffer and optimizer step overhead by directly
        updating embeddings in the backward kernel. For small batch training
        where optimizer overhead dominates.

        Args:
            grad: [B, num_levels * level_dim] output gradients
            batch_indices: [B] indices into precomputed buffers
            lr: learning rate (will be negated internally for descent)

        Note: This bypasses autograd - use only for embeddings when doing
        pure SGD optimization.
        """
        if not self._precomputed:
            raise RuntimeError("Must call precompute() before backward_sgd_fused()")

        B = batch_indices.shape[0]
        grad = grad.view(B, self.num_levels, self.level_dim).permute(1, 0, 2).contiguous()

        h1 = self._precomp_h1[batch_indices].contiguous()
        h2 = self._precomp_h2[batch_indices].contiguous()
        weights = self._precomp_weights[batch_indices].contiguous()

        probe_indices = self.probe_indices if self.enable_learned_probing else None
        if probe_indices is None:
            probe_indices = torch.empty(0, dtype=torch.int32, device=self.embeddings.device)

        # Pass negative lr for gradient descent
        _backend.hash_encode_backward_sgd_fused(
            grad, self.offsets,
            h1, h2, weights,
            probe_indices, self.embeddings.data,
            B, self.input_dim, self.level_dim, self.num_levels,
            self.N_p if self.enable_learned_probing else 1,
            self.N_c if self.enable_learned_probing else 0,
            -lr  # Negative for descent: embedding += (-lr) * grad = embedding - lr * grad
        )

    def init_sparse_adam(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.001):
        """
        Initialize sparse Adam optimizer state for this encoder.

        Call once before training. The sparse Adam update only touches
        embeddings accessed in each batch, providing massive speedup for
        small batch training.

        Args:
            lr: Learning rate
            beta1: First moment decay (default: 0.9)
            beta2: Second moment decay (default: 0.999)
            eps: Numerical stability epsilon (default: 1e-8)
            weight_decay: AdamW weight decay (default: 0.001)
        """
        device = self.embeddings.device
        dtype = self.embeddings.dtype
        shape = self.embeddings.shape

        # Initialize moment buffers to zero (all float32 for CUDA atomicExch support)
        self._adam_exp_avg = torch.zeros(shape, dtype=torch.float32, device=device)
        self._adam_exp_avg_sq = torch.zeros(shape, dtype=torch.float32, device=device)
        self._adam_grad_buffer = torch.zeros(shape, dtype=torch.float32, device=device)
        self._adam_step = 0

        # Initialize index_logits gradient buffer if learned probing is enabled
        if self.enable_learned_probing and self.index_logits is not None:
            self._adam_grad_index_logits = torch.zeros_like(self.index_logits, dtype=torch.float32, device=device)
        else:
            self._adam_grad_index_logits = None

        # Store hyperparameters
        self._adam_lr = lr
        self._adam_beta1 = beta1
        self._adam_beta2 = beta2
        self._adam_eps = eps
        self._adam_weight_decay = weight_decay
        self._adam_initialized = True

    def set_adam_lr(self, lr: float):
        """Update the learning rate for sparse Adam optimizer."""
        if self._adam_initialized:
            self._adam_lr = lr

    def accumulate_grad(self, grad, batch_indices):
        """
        Accumulate gradients into the sparse Adam gradient buffer.

        Call this after computing gradients for a batch. The gradients are
        accumulated (for hash collisions) and will be applied by adam_step().

        With learned probing, also accumulates gradients for index_logits using
        soft gradient distribution (matching standard backward).

        Args:
            grad: [B, output_dim] gradients w.r.t. encoder output
            batch_indices: [B] indices into precomputed buffers
        """
        if not self._adam_initialized:
            raise RuntimeError("Must call init_sparse_adam() first")
        if not self._precomputed:
            raise RuntimeError("Must call precompute() first")

        B = batch_indices.shape[0]
        # Cast to float32 since grad_buffer is float32 for atomicExch support
        grad = grad.float().view(B, self.num_levels, self.level_dim).permute(1, 0, 2).contiguous()

        h1 = self._precomp_h1[batch_indices].contiguous()
        h2 = self._precomp_h2[batch_indices].contiguous()
        weights = self._precomp_weights[batch_indices].contiguous()

        probe_indices = self.probe_indices if self.enable_learned_probing else None
        if probe_indices is None:
            probe_indices = torch.empty(0, dtype=torch.int32, device=self.embeddings.device)

        # For learned probing, pass index_logits to enable soft gradient distribution
        index_logits = self.index_logits if self.enable_learned_probing else None
        if index_logits is None:
            index_logits = torch.empty(0, dtype=torch.float32, device=self.embeddings.device)
        else:
            index_logits = index_logits.contiguous().to(torch.float32)

        grad_index_logits = self._adam_grad_index_logits if self._adam_grad_index_logits is not None else torch.empty(0, dtype=torch.float32, device=self.embeddings.device)

        # Use backward with soft gradients to accumulate both embedding and index_logits gradients
        _backend.hash_encode_backward_precomputed(
            grad, self.offsets,
            h1, h2, weights,
            probe_indices, index_logits, grad_index_logits, self._adam_grad_buffer,
            B, self.input_dim, self.level_dim, self.num_levels,
            self.N_p if self.enable_learned_probing else 1,
            self.N_c if self.enable_learned_probing else 0
        )

    def adam_step(self, batch_indices):
        """
        Apply sparse Adam update to embeddings touched by this batch.

        Uses CUDA kernel that only updates embeddings referenced by batch_indices,
        providing ~20x speedup over full optimizer step for small batches.

        Args:
            batch_indices: [B] indices into precomputed buffers (same as accumulate_grad)
        """
        if not self._adam_initialized:
            raise RuntimeError("Must call init_sparse_adam() first")

        self._adam_step += 1
        B = batch_indices.shape[0]

        h1 = self._precomp_h1[batch_indices].contiguous()
        h2 = self._precomp_h2[batch_indices].contiguous()

        probe_indices = self.probe_indices if self.enable_learned_probing else None
        if probe_indices is None:
            probe_indices = torch.empty(0, dtype=torch.int32, device=self.embeddings.device)

        _backend.hash_encode_adam_sparse_update(
            self.offsets,
            h1, h2,
            probe_indices,
            self.embeddings.data,
            self._adam_grad_buffer,
            self._adam_exp_avg,
            self._adam_exp_avg_sq,
            B, self.input_dim, self.level_dim, self.num_levels,
            self.N_p if self.enable_learned_probing else 1,
            self.N_c if self.enable_learned_probing else 0,
            self._adam_lr, self._adam_beta1, self._adam_beta2, self._adam_eps,
            self._adam_weight_decay,
            self._adam_step
        )

    def transfer_index_logits_grad(self):
        """
        Transfer accumulated index_logits gradients to the parameter's .grad attribute.

        Call this after accumulate_grad() and before optimizer.step() for index_logits.
        The gradients are accumulated via the CUDA backward kernel into a separate buffer,
        and this method transfers them to index_logits.grad for the regular optimizer.

        Also zeros the accumulation buffer for the next batch.
        """
        if self._adam_grad_index_logits is not None and self.index_logits is not None:
            # Transfer to .grad (add if already exists, else set)
            if self.index_logits.grad is None:
                self.index_logits.grad = self._adam_grad_index_logits.clone()
            else:
                self.index_logits.grad.add_(self._adam_grad_index_logits)
            # Zero the buffer for next accumulation
            self._adam_grad_index_logits.zero_()

    def forward(self, inputs, size=1, collision_tracking=None):
        """Standard forward pass (recomputes hash indices each time)."""
        if self.enable_learned_probing and self.training:
            self.update_probe_indices()

        inputs = (inputs + size) / (2 * size)

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        if collision_tracking is not None:
            track_collisions = True
            collision_indices = collision_tracking['collision_indices']
            example_offset = collision_tracking['example_offset']
            max_tracked_examples = collision_tracking['max_tracked_examples']
        else:
            track_collisions = False
            collision_indices = torch.empty(1, dtype=torch.int32, device=inputs.device)
            example_offset = 0
            max_tracked_examples = 0

        if self.enable_learned_probing:
            outputs = hash_encode(
                inputs, self.embeddings, self.offsets, self.per_level_scale,
                self.base_resolution, inputs.requires_grad,
                track_collisions, collision_indices,
                example_offset, max_tracked_examples,
                self.probe_indices, self.index_logits, self.N_f, self.N_p, self.N_c
            )
        else:
            outputs = hash_encode(
                inputs, self.embeddings, self.offsets, self.per_level_scale,
                self.base_resolution, inputs.requires_grad,
                track_collisions, collision_indices,
                example_offset, max_tracked_examples,
                None, None, 0, 1, 0
            )
        outputs = outputs.view(prefix_shape + [self.output_dim])

        return outputs
