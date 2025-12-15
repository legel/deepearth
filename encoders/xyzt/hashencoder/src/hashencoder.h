#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, L * C], float
// H: base resolution
// probe_indices: [L, N_c], int32 (optional, for learned probing)
void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, at::Tensor dy_dx, const bool track_collisions, at::Tensor collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const at::Tensor probe_indices, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c);
void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs, const at::Tensor probe_indices, const at::Tensor index_logits, at::Tensor grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c);
void hash_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, const at::Tensor grad_grad_inputs, at::Tensor grad_grad, at::Tensor grad2_embeddings);

// =============================================================================
// WARP-LEVEL YOHO (Automatic Intra-Warp Deduplication)
// =============================================================================
// Threads in the same warp that share a grid cell only compute once.
// This is automatic and requires no Python-side preprocessing.
// Provides speedup proportional to spatial coherence within each warp.
// Now supports learned hash probing via probe_indices parameter.
void hash_encode_forward_warp_yoho(
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets,
    at::Tensor outputs,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const at::Tensor per_level_scale,
    const at::Tensor base_resolution,
    const bool calc_grad_inputs,
    at::Tensor dy_dx,
    const at::Tensor probe_indices,  // Shape: (L, N_c) or empty tensor if disabled
    const uint32_t N_f,
    const uint32_t N_p,
    const uint32_t N_c,
    at::Tensor dedup_stats  // Shape: (L,) uint32 for tracking unique cells, or empty to disable
);

void hash_encode_backward_warp_yoho(
    const at::Tensor grad,
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets,
    at::Tensor grad_embeddings,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const at::Tensor per_level_scale,
    const at::Tensor base_resolution,
    const bool calc_grad_inputs,
    const at::Tensor dy_dx,
    at::Tensor grad_inputs,
    const at::Tensor probe_indices,  // Shape: (L, N_c) or empty tensor if disabled
    const at::Tensor index_logits,   // Shape: (L, N_c, N_p) for learned probing softmax
    at::Tensor grad_index_logits,    // Shape: (L, N_c, N_p) output gradients
    const uint32_t N_f,
    const uint32_t N_p,
    const uint32_t N_c
);

#endif