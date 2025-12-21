#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// =============================================================================
// CORE HASH ENCODING
// =============================================================================

void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, at::Tensor dy_dx, const bool track_collisions, at::Tensor collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const at::Tensor probe_indices, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c);
void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs, const at::Tensor probe_indices, const at::Tensor index_logits, at::Tensor grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c);
void hash_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, const at::Tensor grad_grad_inputs, at::Tensor grad_grad, at::Tensor grad2_embeddings);

// =============================================================================
// PRECOMPUTED HASH ENCODING (Fixed coordinate optimization)
// =============================================================================
// Pre-computes invariant values (h1, h2, weights) once for fixed coordinate sets.
// During training, only embedding values change - everything else is cached.

void hash_encode_precompute(
    const at::Tensor inputs,        // [B, D] normalized coords in [0,1]
    const at::Tensor offsets,       // [L+1] level offsets
    at::Tensor precomp_h1,          // [B, L, 2^D] output: h1 values or direct indices
    at::Tensor precomp_h2,          // [B, L, 2^D] output: h2 values for learned probing
    at::Tensor precomp_weights,     // [B, L, 2^D] output: interpolation weights
    at::Tensor precomp_pos_deriv,   // [B, L, D] output: position derivatives for backward
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const at::Tensor per_level_scale,
    const at::Tensor base_resolution,
    const uint32_t N_f,             // learned probing: hashmap_size / N_p
    const uint32_t N_p              // learned probing: probe range
);

void hash_encode_forward_precomputed(
    const at::Tensor embeddings,    // [total_embeddings, C]
    const at::Tensor offsets,       // [L+1]
    const at::Tensor precomp_h1,    // [B, L, 2^D]
    const at::Tensor precomp_h2,    // [B, L, 2^D]
    const at::Tensor precomp_weights,// [B, L, 2^D]
    const at::Tensor probe_indices, // [L, N_c] or empty
    at::Tensor outputs,             // [L, B, C] output
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p,             // probe range
    const uint32_t N_c              // codebook size
);

void hash_encode_backward_precomputed(
    const at::Tensor grad,          // [L, B, C]
    const at::Tensor offsets,       // [L+1]
    const at::Tensor precomp_h1,    // [B, L, 2^D]
    const at::Tensor precomp_h2,    // [B, L, 2^D]
    const at::Tensor precomp_weights,// [B, L, 2^D]
    const at::Tensor probe_indices, // [L, N_c] or empty (for hard mode fallback)
    const at::Tensor index_logits,  // [L, N_c, N_p] for soft gradients or empty
    at::Tensor grad_index_logits,   // [L, N_c, N_p] output or empty
    at::Tensor grad_embeddings,     // [total_embeddings, C] output
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p,
    const uint32_t N_c
);

// =============================================================================
// OPTIMIZED KERNELS FOR SMALL BATCH TRAINING
// =============================================================================

// Fused backward + SGD update - eliminates gradient buffer and optimizer kernel
void hash_encode_backward_sgd_fused(
    const at::Tensor grad,          // [L, B, C]
    const at::Tensor offsets,       // [L+1]
    const at::Tensor precomp_h1,    // [B, L, 2^D]
    const at::Tensor precomp_h2,    // [B, L, 2^D]
    const at::Tensor precomp_weights,// [B, L, 2^D]
    const at::Tensor probe_indices, // [L, N_c] or empty
    at::Tensor embeddings,          // [total_embeddings, C] MODIFIED IN PLACE
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p,
    const uint32_t N_c,
    const float lr                  // Learning rate (negative for descent)
);

// Multi-batch forward - process all samples in single kernel launch
void hash_encode_forward_precomputed_multibatch(
    const at::Tensor embeddings,    // [total_embeddings, C]
    const at::Tensor offsets,       // [L+1]
    const at::Tensor precomp_h1,    // [N_all, L, 2^D] precomputed for ALL samples
    const at::Tensor precomp_h2,    // [N_all, L, 2^D]
    const at::Tensor precomp_weights,// [N_all, L, 2^D]
    const at::Tensor probe_indices, // [L, N_c] or empty
    const at::Tensor sample_indices,// [N_total] indices into precomputed buffers
    at::Tensor outputs,             // [L, N_total, C] output
    const uint32_t N_total, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p,
    const uint32_t N_c
);

// Sparse Adam update - only updates embeddings touched by this batch
void hash_encode_adam_sparse_update(
    const at::Tensor offsets,       // [L+1]
    const at::Tensor precomp_h1,    // [B, L, 2^D]
    const at::Tensor precomp_h2,    // [B, L, 2^D]
    const at::Tensor probe_indices, // [L, N_c] or empty
    at::Tensor embeddings,          // [total_embeddings, C] MODIFIED IN PLACE
    at::Tensor grad_embeddings,     // [total_embeddings, C] MODIFIED (zeroed after)
    at::Tensor exp_avg,             // [total_embeddings, C] Adam first moment
    at::Tensor exp_avg_sq,          // [total_embeddings, C] Adam second moment
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr, const float beta1, const float beta2, const float eps,
    const float weight_decay,       // AdamW weight decay
    const uint32_t step             // Current optimization step (1-indexed)
);

#endif
