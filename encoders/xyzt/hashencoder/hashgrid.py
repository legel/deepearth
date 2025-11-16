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
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # probe_indices: [L, N_c], int32 (optional, for learned probing)
        # index_logits: [L, N_c, N_p], float (optional, for learned probing backward)
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()
        per_level_scale = per_level_scale.contiguous()
        base_resolution = base_resolution.contiguous()

        # Handle probe_indices and index_logits (may be None if learned probing disabled)
        if probe_indices is not None:
            probe_indices = probe_indices.contiguous()
        else:
            # Create empty tensor if not provided (backend will handle nullptr)
            probe_indices = torch.empty(0, dtype=torch.int32, device=inputs.device)

        if index_logits is not None:
            index_logits = index_logits.contiguous()
        else:
            # Create empty tensor if not provided (backend will handle nullptr)
            index_logits = torch.empty(0, dtype=torch.float32, device=inputs.device)

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        # S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        per_level_scale = torch.log2(per_level_scale)
        # H = base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        # Use embeddings dtype (float32) for outputs, even if inputs are float64
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = torch.empty(1, device=inputs.device, dtype=embeddings.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, probe_indices, index_logits)
        ctx.dims = [B, D, C, L]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.learned_probing_params = [N_f, N_p, N_c]  # Save for backward

        return outputs
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):

        inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, probe_indices, index_logits = ctx.saved_tensors
        B, D, C, L = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs
        N_f, N_p, N_c = ctx.learned_probing_params

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_inputs, grad_embeddings, grad_index_logits = _hash_encode_second_backward.apply(grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, index_logits, N_f, N_p, N_c)

        # Return gradients for all 15 input parameters:
        # inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs,
        # track_collisions, collision_indices, example_offset, max_tracked_examples,
        # probe_indices, index_logits, N_f, N_p, N_c

        # Only return grad_index_logits if index_logits requires grad and is not empty
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
        grad_inputs = torch.zeros_like(inputs, device=device)
        grad_embeddings = torch.zeros_like(embeddings, device=device)

        # Create grad_index_logits tensor if index_logits is provided
        if index_logits is not None and index_logits.numel() > 0:
            grad_index_logits = torch.zeros_like(index_logits, device=device, dtype=torch.float32)
        else:
            grad_index_logits = torch.empty(0, device=device, dtype=torch.float32)

        ctx.save_for_backward(grad, inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, grad_inputs, grad_embeddings, probe_indices, index_logits, grad_index_logits)
        ctx.dims = [B, D, C, L]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.learned_probing_params = [N_f, N_p, N_c]

        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c)

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

        # Return gradients for all 17 parameters
        return grad_grad, None, grad2_embeddings, None, None, None, None, None, None, None, None, None, None, None, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, enable_learned_probing=False, probing_range=4, index_codebook_size=1024):
        super().__init__()

        if type(base_resolution) is int:
            base_resolution = np.array([base_resolution for _ in range(input_dim)], dtype=np.float64)
        else:
            assert len(base_resolution) == input_dim
            base_resolution = np.array(base_resolution, dtype=np.float64)

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            if type(desired_resolution) is int:
                desired_resolution = np.array([desired_resolution for _ in range(input_dim)], dtype=np.float64)
            else:
                assert len(desired_resolution) == input_dim
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        else:
            # Handle both scalar and array per_level_scale
            if type(per_level_scale) is int or type(per_level_scale) is float:
                per_level_scale = np.array([per_level_scale for _ in range(input_dim)], dtype=np.float64)
            else:
                assert len(per_level_scale) == input_dim
                per_level_scale = np.array(per_level_scale, dtype=np.float64)

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size

        # Learned hash probing parameters
        self.enable_learned_probing = enable_learned_probing
        if enable_learned_probing:
            self.N_p = probing_range
            self.N_c = index_codebook_size
            # Adjust max params for learned probing: N_f = hashmap_size / N_p
            self.max_params = (2 ** log2_hashmap_size) // probing_range
            self.N_f = self.max_params  # Store for CUDA kernels
        else:
            self.max_params = 2 ** log2_hashmap_size
            self.N_p = 1  # Default to 1 (no probing)
            self.N_c = 1
            self.N_f = self.max_params

        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        offsets = []
        offset = 0
        for i in range(num_levels):
            resolution = np.ceil(base_resolution * per_level_scale ** i)
            if enable_learned_probing:
                # With learned probing: ALWAYS allocate N_f × N_p features per level
                # Cannot use smaller sizes because index = N_p * h1 + probe requires full range
                params_in_level = self.max_params * self.N_p
            else:
                # Original: just hashmap_size features per level
                params_in_level = min(self.max_params, int(np.prod(resolution)))
            offsets.append(offset)
            offset += int(params_in_level)
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        self.register_buffer('per_level_scale', torch.tensor(per_level_scale, dtype=torch.float32))
        self.register_buffer('base_resolution', torch.tensor(base_resolution, dtype=torch.float32))

        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        # Learned probing index codebooks (only if enabled)
        if enable_learned_probing:
            # Training codebook: (L, N_c, N_p) logits for softmax
            self.index_logits = nn.Parameter(torch.randn(num_levels, index_codebook_size, probing_range) * 0.01)

            # Inference codebook: (L, N_c) discrete probe indices
            self.register_buffer('probe_indices', torch.zeros(num_levels, index_codebook_size, dtype=torch.int32))
        else:
            self.index_logits = None
            self.probe_indices = None

        self.reset_parameters()

    def reset_parameters(self):
        # With large hash tables and high collision ratios, we need stronger initialization
        # to ensure gradients flow properly
        std = 1e-1  # Increased to 0.1 for better gradient flow with large hash tables
        self.embeddings.data.uniform_(-std, std)

    def update_probe_indices(self):
        """Update discrete probe indices from logits (call after optimizer step)"""
        if self.enable_learned_probing and self.index_logits is not None:
            with torch.no_grad():
                # D_c = argmax(D̃_c, dim=-1)
                self.probe_indices.copy_(torch.argmax(self.index_logits, dim=-1))

    def __repr__(self):
        if self.enable_learned_probing:
            return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} learned_probing=True N_f={self.N_f} N_p={self.N_p} N_c={self.N_c} params={tuple(self.embeddings.shape)}"
        else:
            return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} params={tuple(self.embeddings.shape)}"

    def forward(self, inputs, size=1, collision_tracking=None):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # collision_tracking: Optional dict with collision tracking data for this grid
        # return: [..., num_levels * level_dim]

        # Update probe indices before forward pass (if learned probing enabled)
        if self.enable_learned_probing and self.training:
            self.update_probe_indices()

        inputs = (inputs + size) / (2 * size) # map to [0, 1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        # Extract collision tracking parameters if provided
        if collision_tracking is not None:
            track_collisions = True
            collision_indices = collision_tracking['collision_indices']
            example_offset = collision_tracking['example_offset']
            max_tracked_examples = collision_tracking['max_tracked_examples']
        else:
            track_collisions = False
            # Create dummy tensor for when collision tracking is disabled
            collision_indices = torch.empty(1, dtype=torch.int32, device=inputs.device)
            example_offset = 0
            max_tracked_examples = 0

        # Call hash_encode with collision tracking and learned probing parameters
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
                None, None, 0, 1, 0  # No learned probing
            )
        outputs = outputs.view(prefix_shape + [self.output_dim])

        return outputs