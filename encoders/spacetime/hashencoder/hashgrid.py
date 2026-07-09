import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from torch.amp import custom_bwd, custom_fwd
except ImportError:                                          # older PyTorch
    from torch.cuda.amp import custom_bwd, custom_fwd

from .backend import _backend


class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type='cuda')
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, track_collisions=False, collision_indices=None, example_offset=0, max_tracked_examples=0, probe_indices=None, index_logits=None, N_f=0, N_p=1, N_c=0):
        inputs = inputs.contiguous(); embeddings = embeddings.contiguous(); offsets = offsets.contiguous()
        per_level_scale = per_level_scale.contiguous(); base_resolution = base_resolution.contiguous()
        probe_indices = probe_indices.contiguous() if probe_indices is not None else torch.empty(0, dtype=torch.int32, device=inputs.device)
        index_logits = index_logits.contiguous().to(torch.float32) if index_logits is not None else torch.empty(0, dtype=torch.float32, device=inputs.device)

        B, D = inputs.shape; L = offsets.shape[0] - 1; C = embeddings.shape[1]
        per_level_scale = per_level_scale.reshape(-1).contiguous()          # [L, D] log2 resolutions -> [L*D] for the kernel
        calc_grad_inputs = bool(calc_grad_inputs or ctx.needs_input_grad[3])   # per-level dy_dx needed to learn resolutions

        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
        dy_dx = (torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype) if calc_grad_inputs
                 else torch.empty(1, device=inputs.device, dtype=embeddings.dtype))

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c)
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, probe_indices, index_logits)
        ctx.dims = [B, D, C, L]; ctx.calc_grad_inputs = calc_grad_inputs; ctx.learned_probing_params = [N_f, N_p, N_c]
        return outputs

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, probe_indices, index_logits = ctx.saved_tensors
        B, D, C, L = ctx.dims; calc_grad_inputs = ctx.calc_grad_inputs; N_f, N_p, N_c = ctx.learned_probing_params
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_inputs, grad_embeddings, grad_index_logits = _hash_encode_second_backward.apply(grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, index_logits, N_f, N_p, N_c)
        grad_index_logits_return = grad_index_logits if (index_logits.requires_grad and index_logits.numel() > 0) else None

        # Gradient w.r.t. the per-level log2 resolutions (learnable frequencies), from the per-level dy_dx the kernel
        # returns: for scale_l[d] = exp2(pls[l,d])*base[d]-1, d(out)/d(pls) = ln2*(scale+1)/scale *
        # (dy_dx/scale contracted with grad_out) * input. Computed in Python -> no backward-kernel change.
        grad_pls = None
        if ctx.needs_input_grad[3] and calc_grad_inputs:
            gb = grad.permute(1, 0, 2); dd = dy_dx.view(B, L, D, C)                               # [B, L, C], [B, L, D, C]
            contrib = (torch.einsum('blc,bldc->bld', gb.float(), dd.float()) * inputs.float().unsqueeze(1)).sum(0)  # [L,D]
            scale = torch.exp2(per_level_scale.view(L, D).float()) * base_resolution.view(1, D).float() - 1.0
            grad_pls = (0.6931471805599453 * (scale + 1.0) / scale.clamp_min(1e-6) * contrib).to(per_level_scale.dtype)  # [L,D]

        grad_in = grad_inputs if calc_grad_inputs else None
        return grad_in, grad_embeddings, None, grad_pls, None, None, None, None, None, None, None, grad_index_logits_return, None, None, None


class _hash_encode_second_backward(Function):
    @staticmethod
    def forward(ctx, grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, index_logits, N_f, N_p, N_c):
        device = inputs.device
        grad_inputs_f32 = torch.zeros_like(inputs, device=device, dtype=embeddings.dtype)
        grad_embeddings = torch.zeros_like(embeddings, device=device)
        grad_index_logits = (torch.zeros_like(index_logits, device=device, dtype=torch.float32) if index_logits.numel() > 0
                             else torch.empty(0, device=device, dtype=torch.float32))

        ctx.save_for_backward(grad, inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, grad_inputs_f32, grad_embeddings, probe_indices, index_logits, grad_index_logits)
        ctx.dims = [B, D, C, L]; ctx.calc_grad_inputs = calc_grad_inputs; ctx.learned_probing_params = [N_f, N_p, N_c]

        index_logits_f32 = index_logits.to(torch.float32) if (index_logits.numel() > 0 and index_logits.dtype != torch.float32) else index_logits
        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs_f32, probe_indices, index_logits_f32, grad_index_logits, N_f, N_p, N_c)

        grad_inputs = grad_inputs_f32.to(inputs.dtype) if inputs.dtype != embeddings.dtype else grad_inputs_f32
        return grad_inputs, grad_embeddings, grad_index_logits

    @staticmethod
    def backward(ctx, grad_grad_inputs, grad_grad_embeddings, grad_grad_index_logits):
        grad, inputs, embeddings, offsets, per_level_scale, base_resolution, dy_dx, grad_inputs, grad_embeddings, probe_indices, index_logits, grad_index_logits = ctx.saved_tensors
        B, D, C, L = ctx.dims; calc_grad_inputs = ctx.calc_grad_inputs
        grad_grad = torch.zeros_like(grad, device=grad.device)
        grad2_embeddings = torch.zeros_like(embeddings, device=grad.device)
        _backend.hash_encode_second_backward(grad, inputs, embeddings, offsets, B, D, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_grad_inputs, grad_grad, grad2_embeddings)
        return grad_grad, None, grad2_embeddings, None, None, None, None, None, None, None, None, None, None, None, None, None, None


hash_encode = _hash_encode.apply


# =============================================================================
# torch.library custom op for the hash encode. Registering the encode as one opaque operator lets torch.compile treat
# it as a single node instead of graph-breaking at the C++ extension call, unlocking full-model compilation and CUDA
# graphs. First-order only (training never takes a gradient of a gradient) and numerically identical to _hash_encode.
# Returns (outputs, dy_dx); dy_dx is the input-derivative buffer the backward kernel needs, kept so autograd can save it.
# =============================================================================

@torch.library.custom_op("earth4d::hash_encode", mutates_args=())
def _hash_encode_cuda(inputs: torch.Tensor, embeddings: torch.Tensor, offsets: torch.Tensor,
                      per_level_scale: torch.Tensor, base_resolution: torch.Tensor,
                      probe_indices: torch.Tensor, index_logits: torch.Tensor,
                      N_f: int, N_p: int, N_c: int, calc_grad_inputs: bool) -> "tuple[torch.Tensor, torch.Tensor]":
    inputs = inputs.contiguous(); embeddings = embeddings.contiguous(); offsets = offsets.contiguous()
    B, D = inputs.shape; L = offsets.shape[0] - 1; C = embeddings.shape[1]
    pls = per_level_scale.reshape(-1).contiguous(); base = base_resolution.contiguous()   # [L,D] log2 res -> [L*D]
    outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
    dy_dx = (torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype) if calc_grad_inputs
             else torch.empty(1, device=inputs.device, dtype=embeddings.dtype))
    empty_ci = torch.empty(1, dtype=torch.int32, device=inputs.device)
    _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, pls, base,
                                 calc_grad_inputs, dy_dx, False, empty_ci, 0, 0, probe_indices, N_f, N_p, N_c)
    return outputs.permute(1, 0, 2).reshape(B, L * C), dy_dx


@_hash_encode_cuda.register_fake
def _(inputs, embeddings, offsets, per_level_scale, base_resolution, probe_indices, index_logits,
      N_f, N_p, N_c, calc_grad_inputs):
    B, D = inputs.shape; L = offsets.shape[0] - 1; C = embeddings.shape[1]
    return (inputs.new_empty(B, L * C),
            inputs.new_empty(B, L * D * C) if calc_grad_inputs else inputs.new_empty(1))


@torch.library.custom_op("earth4d::hash_encode_backward", mutates_args=())
def _hash_encode_bwd_cuda(grad: torch.Tensor, inp: torch.Tensor, emb: torch.Tensor, off: torch.Tensor,
                          pls_log2: torch.Tensor, base: torch.Tensor, dy_dx: torch.Tensor,
                          probe: torch.Tensor, idx_f32: torch.Tensor,
                          N_f: int, N_p: int, N_c: int, calc: bool,
                          B: int, D: int, C: int, L: int) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
    grad_emb = torch.zeros_like(emb); grad_inp = torch.zeros_like(inp, dtype=emb.dtype)
    grad_idx = torch.zeros_like(idx_f32) if idx_f32.numel() > 0 else torch.empty(0, dtype=torch.float32, device=grad.device)
    _backend.hash_encode_backward(grad, inp, emb, off, grad_emb, B, D, C, L, pls_log2, base, calc, dy_dx,
                                  grad_inp, probe, idx_f32, grad_idx, N_f, N_p, N_c)
    return grad_inp, grad_emb, grad_idx


@_hash_encode_bwd_cuda.register_fake
def _(grad, inp, emb, off, pls_log2, base, dy_dx, probe, idx_f32, N_f, N_p, N_c, calc, B, D, C, L):
    return (torch.empty_like(inp, dtype=emb.dtype), torch.empty_like(emb),
            torch.empty_like(idx_f32) if idx_f32.numel() > 0 else idx_f32.new_empty(0))


def _hash_encode_cuda_setup(ctx, inputs, output):
    inp, emb, off, pls, base, probe, idx_logits, N_f, N_p, N_c, calc = inputs
    _, dy_dx = output
    ctx.save_for_backward(inp, emb, off, pls.reshape(-1).contiguous(), base, dy_dx, probe, idx_logits)  # log2 [L*D]
    ctx.dims = (inp.shape[0], inp.shape[1], emb.shape[1], off.shape[0] - 1); ctx.probing = (N_f, N_p, N_c); ctx.calc = calc


def _hash_encode_cuda_backward(ctx, grad_out, grad_dy_dx):
    inp, emb, off, pls, base, dy_dx, probe, idx_logits = ctx.saved_tensors
    B, D, C, L = ctx.dims; N_f, N_p, N_c = ctx.probing; calc = ctx.calc
    idx_f32 = idx_logits.to(torch.float32) if idx_logits.numel() > 0 else idx_logits
    grad_inp, grad_emb, grad_idx = _hash_encode_bwd_cuda(grad_out, inp, emb, off, pls, base, dy_dx, probe, idx_f32, N_f, N_p, N_c, calc, B, D, C, L)
    # per-level log2 resolution gradient (same formula as the eager Function) -> learnable frequencies work under compile
    grad_pls = None
    if calc and dy_dx.numel() > 1:
        go = grad_out.reshape(B, L, C); dd = dy_dx.reshape(B, L, D, C)
        contrib = (torch.einsum('blc,bldc->bld', go.float(), dd.float()) * inp.float().unsqueeze(1)).sum(0)  # [L,D]
        scale = torch.exp2(pls.reshape(L, D).float()) * base.reshape(1, D).float() - 1.0
        grad_pls = (0.6931471805599453 * (scale + 1.0) / scale.clamp_min(1e-6) * contrib).to(emb.dtype)  # [L, D]
    return ((grad_inp.to(inp.dtype) if calc else None), grad_emb, None, grad_pls, None, None,
            (grad_idx if idx_logits.numel() > 0 else None), None, None, None, None)


_hash_encode_cuda.register_autograd(_hash_encode_cuda_backward, setup_context=_hash_encode_cuda_setup)


class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, enable_learned_probing=False, probing_range=4, index_codebook_size=1024):
        super().__init__()
        if type(base_resolution) is int:
            base_resolution = np.array([base_resolution] * input_dim, dtype=np.float64)
        else:
            assert len(base_resolution) == input_dim; base_resolution = np.array(base_resolution, dtype=np.float64)

        if desired_resolution is not None:
            if type(desired_resolution) is int:
                desired_resolution = np.array([desired_resolution] * input_dim, dtype=np.float64)
            else:
                assert len(desired_resolution) == input_dim
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        elif type(per_level_scale) in (int, float):
            per_level_scale = np.array([per_level_scale] * input_dim, dtype=np.float64)
        else:
            assert len(per_level_scale) == input_dim; per_level_scale = np.array(per_level_scale, dtype=np.float64)

        self.input_dim = input_dim; self.num_levels = num_levels; self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size; self.enable_learned_probing = enable_learned_probing
        if enable_learned_probing:
            self.N_p = probing_range; self.N_c = index_codebook_size
            self.max_params = (2 ** log2_hashmap_size) // probing_range; self.N_f = self.max_params
        else:
            self.max_params = 2 ** log2_hashmap_size; self.N_p = 1; self.N_c = 1; self.N_f = self.max_params
        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16!')

        offsets = []; offset = 0
        for i in range(num_levels):
            resolution = np.ceil(base_resolution * per_level_scale ** i)
            params_in_level = self.max_params * self.N_p if enable_learned_probing else min(self.max_params, int(np.prod(resolution)))
            offsets.append(offset); offset += int(params_in_level)
        offsets.append(offset)
        self.register_buffer('offsets', torch.from_numpy(np.array(offsets, dtype=np.int32)))
        # Per-level, per-axis log2 resolution: resolution_l[d] = exp2(per_level_scale[l,d]) * base[d]. Initialized to
        # the geometric ladder l * log2(growth_d); learnable because the kernel reads a per-level value directly (its
        # gradient is computed in Python from the per-level dy_dx).
        _lg = np.log2(np.asarray(per_level_scale, dtype=np.float64))                  # [D] log2 growth factor
        _pls = np.stack([i * _lg for i in range(num_levels)], axis=0)                 # [L, D] geometric log2 resolution
        self.per_level_scale = nn.Parameter(torch.tensor(_pls, dtype=torch.float32))   # learnable per-level frequencies
        self.register_buffer('base_resolution', torch.tensor(base_resolution, dtype=torch.float32))

        self.n_params = offsets[-1] * level_dim
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))
        if enable_learned_probing:
            self.index_logits = nn.Parameter(torch.randn(num_levels, index_codebook_size, probing_range) * 0.01)
            self.register_buffer('probe_indices', torch.zeros(num_levels, index_codebook_size, dtype=torch.int32))
        else:
            self.index_logits = None; self.probe_indices = None
        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings.data.uniform_(-1e-1, 1e-1)

    def update_probe_indices(self):
        """Update discrete probe indices from logits (call after optimizer step)."""
        if self.enable_learned_probing and self.index_logits is not None:
            with torch.no_grad():
                self.probe_indices.copy_(torch.argmax(self.index_logits, dim=-1))

    def __repr__(self):
        if self.enable_learned_probing:
            return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} learned_probing=True N_f={self.N_f} N_p={self.N_p} N_c={self.N_c} params={tuple(self.embeddings.shape)}"
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} params={tuple(self.embeddings.shape)}"

    def clamp_per_level_scale(self):
        """Keep every level's resolution >= 2 (scale = exp2(pls)*base - 1 >= 1 > 0). Below scale=0 the kernel reads out
        of bounds and the per-level resolution gradient sign-flips and explodes. Call after each optimizer step when
        per_level_scale is a learnable Parameter."""
        if isinstance(self.per_level_scale, torch.nn.Parameter):
            with torch.no_grad():
                floor = (1.0 - torch.log2(self.base_resolution)).view(1, -1)     # per-dim: exp2(floor)*base = 2
                self.per_level_scale.data.copy_(torch.maximum(self.per_level_scale.data, floor))

    # class-level switch: route the standard forward through the torch.library custom op so torch.compile can capture
    # the whole model. Numerically identical; enable once validated.
    use_custom_op = False

    def forward(self, inputs, size=1, collision_tracking=None):
        """Standard forward pass (recomputes hash indices each time)."""
        if self.enable_learned_probing and self.training:
            self.update_probe_indices()
        inputs = (inputs + size) / (2 * size)
        prefix_shape = list(inputs.shape[:-1]); inputs = inputs.view(-1, self.input_dim)

        if HashEncoder.use_custom_op and collision_tracking is None:
            if self.enable_learned_probing:
                probe, idx = self.probe_indices, self.index_logits
            else:
                probe = torch.empty(0, dtype=torch.int32, device=inputs.device)
                idx = torch.empty(0, dtype=torch.float32, device=inputs.device)
            outputs, _ = _hash_encode_cuda(inputs, self.embeddings, self.offsets, self.per_level_scale,
                                           self.base_resolution, probe, idx, self.N_f, self.N_p, self.N_c,
                                           inputs.requires_grad or self.per_level_scale.requires_grad)
            return outputs.view(prefix_shape + [self.output_dim])

        if collision_tracking is not None:
            track_collisions = True
            collision_indices = collision_tracking['collision_indices']
            example_offset = collision_tracking['example_offset']
            max_tracked_examples = collision_tracking['max_tracked_examples']
        else:
            track_collisions = False
            collision_indices = torch.empty(1, dtype=torch.int32, device=inputs.device)
            example_offset = 0; max_tracked_examples = 0

        if self.enable_learned_probing:
            outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution,
                                  inputs.requires_grad, track_collisions, collision_indices, example_offset,
                                  max_tracked_examples, self.probe_indices, self.index_logits, self.N_f, self.N_p, self.N_c)
        else:
            outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution,
                                  inputs.requires_grad, track_collisions, collision_indices, example_offset,
                                  max_tracked_examples, None, None, 0, 1, 0)
        return outputs.view(prefix_shape + [self.output_dim])
