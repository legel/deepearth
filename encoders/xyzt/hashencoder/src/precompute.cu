/**
 * precompute.cu - Pre-computation kernels for hash encoding
 *
 * Pre-computes invariant values (h1, h2, weights) for fixed coordinate sets.
 * At training time, only embedding values change - everything else is cached.
 *
 * Memory layout per coordinate, per level, per corner (8 corners for D=3):
 *   - h1: uint32 (base hash for learned probing, or direct index)
 *   - h2: uint32 (codebook index for learned probing)
 *   - weight: float32 (interpolation weight)
 *   - needs_hash: bool (packed into flags, whether level needs hashing)
 */

#include "utils.cuh"

// =============================================================================
// PRECOMPUTATION KERNEL
// =============================================================================

template <typename input_t, uint32_t D, uint32_t C>
__global__ void kernel_precompute(
    const input_t * __restrict__ inputs,           // [B, D] normalized coords in [0,1]
    const int * __restrict__ offsets,              // [L+1] level offsets
    uint32_t * __restrict__ precomp_h1,            // [B, L, 2^D] h1 values or direct indices
    uint32_t * __restrict__ precomp_h2,            // [B, L, 2^D] h2 values (for learned probing)
    float * __restrict__ precomp_weights,          // [B, L, 2^D] interpolation weights
    float * __restrict__ precomp_pos_deriv,        // [B, L, D] position derivatives for backward
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,    // log2 of scale per dimension
    const float * __restrict__ base_resolution,
    const uint32_t N_f,                            // learned probing: hashmap_size / N_p
    const uint32_t N_p                             // learned probing: probe range
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t n_corners = 1 << D;

    // Locate input/output pointers
    inputs += b * D;
    precomp_h1 += b * L * n_corners + level * n_corners;
    precomp_h2 += b * L * n_corners + level * n_corners;
    precomp_weights += b * L * n_corners + level * n_corners;
    precomp_pos_deriv += b * L * D + level * D;

    // Check bounds
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) flag_oob = true;
    }

    if (flag_oob) {
        // Out of bounds: set weights to 0
        #pragma unroll
        for (uint32_t idx = 0; idx < n_corners; idx++) {
            precomp_h1[idx] = 0;
            precomp_h2[idx] = 0;
            precomp_weights[idx] = 0.0f;
        }
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            precomp_pos_deriv[d] = 0.0f;
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

    // Compute scale and resolution for this level
    double scale[D];
    uint32_t resolution[D];
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2(level * (double)per_level_scale[d]) * (double)base_resolution[d] - 1.0;
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    // Compute position with high precision
    double pos_hp[D];
    float pos[D];
    float pos_derivative[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos_hp[d] = (double)inputs[d] * scale[d];
        pos_grid[d] = (uint32_t)floor(pos_hp[d]);
        pos[d] = (float)(pos_hp[d] - (double)pos_grid[d]);
        pos_derivative[d] = smoothstep_derivative(pos[d]);
        pos[d] = smoothstep(pos[d]);
    }

    // Store position derivatives for backward pass
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        precomp_pos_deriv[d] = pos_derivative[d];
    }

    // Check if this level needs hashing
    uint64_t stride = 1;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        stride *= resolution[d];
    }
    bool needs_hash = (stride > hashmap_size);

    // Compute h1, h2, and weights for each corner
    #pragma unroll
    for (uint32_t idx = 0; idx < n_corners; idx++) {
        float w = 1.0f;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1.0f - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // Store interpolation weight
        precomp_weights[idx] = w;

        // Compute index or hash
        if (needs_hash) {
            // Compute h1 and h2 for learned probing
            uint64_t h1 = fast_hash<D>(pos_grid_local) % N_f;
            uint64_t h2 = fast_hash2<D>(pos_grid_local);
            precomp_h1[idx] = (uint32_t)h1;
            precomp_h2[idx] = (uint32_t)h2;  // Will be modulo N_c at runtime
        } else {
            // Direct indexing - store the direct index
            uint64_t direct_idx = 0;
            uint64_t s = 1;
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                direct_idx += (uint64_t)pos_grid_local[d] * s;
                s *= resolution[d];
            }
            precomp_h1[idx] = (uint32_t)(direct_idx % hashmap_size);
            precomp_h2[idx] = 0xFFFFFFFF;  // Sentinel: no hashing needed
        }
    }
}

// =============================================================================
// PRECOMPUTED FORWARD KERNEL
// =============================================================================

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_precomputed(
    const scalar_t * __restrict__ grid,            // [total_embeddings, C]
    const int * __restrict__ offsets,              // [L+1]
    const uint32_t * __restrict__ precomp_h1,      // [B, L, 2^D]
    const uint32_t * __restrict__ precomp_h2,      // [B, L, 2^D]
    const float * __restrict__ precomp_weights,    // [B, L, 2^D]
    const int * __restrict__ probe_indices,        // [L, N_c] learned probe indices
    scalar_t * __restrict__ outputs,               // [L, B, C]
    const uint32_t B, const uint32_t L,
    const uint32_t N_p,                            // probe range
    const uint32_t N_c                             // codebook size
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t n_corners = 1 << D;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

    // Locate pointers
    const scalar_t* level_grid = grid + (uint32_t)offsets[level] * C;
    const uint32_t* h1_ptr = precomp_h1 + b * L * n_corners + level * n_corners;
    const uint32_t* h2_ptr = precomp_h2 + b * L * n_corners + level * n_corners;
    const float* w_ptr = precomp_weights + b * L * n_corners + level * n_corners;
    scalar_t* out_ptr = outputs + level * B * C + b * C;

    scalar_t results[C] = {0};

    #pragma unroll
    for (uint32_t idx = 0; idx < n_corners; idx++) {
        float w = w_ptr[idx];
        if (w == 0.0f) continue;  // Skip zero-weight corners

        uint32_t h1 = h1_ptr[idx];
        uint32_t h2 = h2_ptr[idx];

        uint32_t index;
        if (h2 == 0xFFFFFFFF) {
            // Direct indexing (no hash needed)
            index = h1 * C;
        } else if (probe_indices != nullptr && N_c > 0) {
            // Learned probing
            uint32_t h2_mod = h2 % N_c;
            int probe = probe_indices[level * N_c + h2_mod];
            if (probe < 0 || (uint32_t)probe >= N_p) probe = 0;
            uint64_t final_idx = (uint64_t)N_p * h1 + probe;
            index = (uint32_t)((final_idx % hashmap_size) * C);
        } else {
            // Standard hashing (no learned probing)
            index = (uint32_t)((h1 % hashmap_size) * C);
        }

        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += (scalar_t)(w * (float)level_grid[index + ch]);
        }
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        out_ptr[ch] = results[ch];
    }
}

// =============================================================================
// PRECOMPUTED BACKWARD KERNEL (with soft gradient support for learned probing)
// =============================================================================
// When index_logits is provided, uses SOFT selection (softmax) for gradients
// to match the standard backward behavior. This enables gradient flow to
// index_logits and distributes embedding gradients across all probe positions.

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward_precomputed(
    const scalar_t * __restrict__ grad,            // [L, B, C]
    const int * __restrict__ offsets,              // [L+1]
    const uint32_t * __restrict__ precomp_h1,      // [B, L, 2^D]
    const uint32_t * __restrict__ precomp_h2,      // [B, L, 2^D]
    const float * __restrict__ precomp_weights,    // [B, L, 2^D]
    const int * __restrict__ probe_indices,        // [L, N_c] (used only for hard mode)
    const float * __restrict__ index_logits,       // [L, N_c, N_p] for soft gradients (nullptr = hard mode)
    float * __restrict__ grad_index_logits,        // [L, N_c, N_p] output gradients (nullptr = don't compute)
    scalar_t * __restrict__ grad_grid,             // [total_embeddings, C]
    const uint32_t B, const uint32_t L,
    const uint32_t N_p,
    const uint32_t N_c
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;
    const uint32_t n_corners = 1 << D;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

    // Locate pointers
    scalar_t* level_grad_grid = grad_grid + offsets[level] * C;
    const uint32_t* h1_ptr = precomp_h1 + b * L * n_corners + level * n_corners;
    const uint32_t* h2_ptr = precomp_h2 + b * L * n_corners + level * n_corners;
    const float* w_ptr = precomp_weights + b * L * n_corners + level * n_corners;
    const scalar_t* grad_ptr = grad + level * B * C + b * C + ch;

    scalar_t grad_cur[N_C];
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad_ptr[c];
    }

    #pragma unroll
    for (uint32_t idx = 0; idx < n_corners; idx++) {
        float w = w_ptr[idx];
        if (w == 0.0f) continue;

        uint32_t h1 = h1_ptr[idx];
        uint32_t h2 = h2_ptr[idx];

        // Direct indexing (no hash needed)
        if (h2 == 0xFFFFFFFF) {
            uint32_t index = h1 * C + ch;
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&level_grad_grid[index + c], (scalar_t)(w * grad_cur[c]));
            }
        }
        // Learned probing with SOFT gradients (matches standard backward)
        else if (index_logits != nullptr && N_c > 0 && N_p > 1) {
            uint32_t h2_mod = h2 % N_c;

            // Get logits for this codebook entry
            const float* logits = &index_logits[level * N_c * N_p + h2_mod * N_p];

            // Compute softmax weights (numerically stable)
            float weights[16];  // N_p <= 16 assumed
            float max_logit = logits[0];
            #pragma unroll
            for (uint32_t p = 1; p < N_p; ++p) {
                max_logit = max(max_logit, logits[p]);
            }

            float sum_exp = 0.0f;
            #pragma unroll
            for (uint32_t p = 0; p < N_p; ++p) {
                weights[p] = expf(logits[p] - max_logit);
                sum_exp += weights[p];
            }

            #pragma unroll
            for (uint32_t p = 0; p < N_p; ++p) {
                weights[p] /= sum_exp;
            }

            // Accumulate gradients to ALL probe positions with softmax weights
            #pragma unroll
            for (uint32_t p = 0; p < N_p; ++p) {
                uint32_t probe_index = (uint32_t)(((uint64_t)N_p * h1 + p) % hashmap_size) * C + ch;
                float combined_weight = w * weights[p];

                #pragma unroll
                for (uint32_t c = 0; c < N_C; c++) {
                    atomicAdd(&level_grad_grid[probe_index + c], (scalar_t)(combined_weight * grad_cur[c]));
                }
            }

            // Compute gradient for index_logits if requested
            if (grad_index_logits != nullptr) {
                // grad_weights[p] = sum over channels of (grad * w)
                float grad_weights[16] = {0};
                #pragma unroll
                for (uint32_t p = 0; p < N_p; ++p) {
                    #pragma unroll
                    for (uint32_t c = 0; c < N_C; c++) {
                        grad_weights[p] += (float)grad_cur[c] * w;
                    }
                }

                // Softmax gradient: d/d_logit[i] = softmax[i] * (grad[i] - sum_j(softmax[j] * grad[j]))
                float dot_product = 0.0f;
                #pragma unroll
                for (uint32_t p = 0; p < N_p; ++p) {
                    dot_product += weights[p] * grad_weights[p];
                }

                #pragma unroll
                for (uint32_t p = 0; p < N_p; ++p) {
                    float grad_logit = weights[p] * (grad_weights[p] - dot_product);
                    uint32_t logit_idx = level * N_c * N_p + h2_mod * N_p + p;
                    atomicAdd(&grad_index_logits[logit_idx], grad_logit);
                }
            }
        }
        // Hard probing or no learned probing
        else {
            uint32_t index;
            if (probe_indices != nullptr && N_c > 0) {
                uint32_t h2_mod = h2 % N_c;
                int probe = probe_indices[level * N_c + h2_mod];
                if (probe < 0 || (uint32_t)probe >= N_p) probe = 0;
                uint64_t final_idx = (uint64_t)N_p * h1 + probe;
                index = (uint32_t)((final_idx % hashmap_size) * C + ch);
            } else {
                index = (uint32_t)((h1 % hashmap_size) * C + ch);
            }

            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&level_grad_grid[index + c], (scalar_t)(w * grad_cur[c]));
            }
        }
    }
}

// =============================================================================
// SPARSE ADAM UPDATE KERNEL
// =============================================================================
// Updates only embeddings touched by the current batch, using precomputed
// indices to know which embeddings to update. Much faster than updating all
// embeddings when batch_size << total_embeddings.
//
// Uses atomicCAS on gradients for deduplication: first thread to process
// a non-zero gradient performs the Adam update and zeros the gradient.

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_adam_sparse_update(
    const int * __restrict__ offsets,              // [L+1]
    const uint32_t * __restrict__ precomp_h1,      // [B, L, 2^D]
    const uint32_t * __restrict__ precomp_h2,      // [B, L, 2^D]
    const int * __restrict__ probe_indices,        // [L, N_c]
    scalar_t * __restrict__ embeddings,            // [total_embeddings, C]
    float * __restrict__ grad_embeddings,          // [total_embeddings, C] - FLOAT buffer, zeroed after use
    float * __restrict__ exp_avg,                  // [total_embeddings, C] first moment
    float * __restrict__ exp_avg_sq,               // [total_embeddings, C] second moment
    const uint32_t B, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr,
    const float beta1, const float beta2, const float eps,
    const float weight_decay,                      // AdamW weight decay
    const float bias_correction1,                  // 1 - beta1^step
    const float bias_correction2                   // 1 - beta2^step
) {
    // Each thread handles one (batch, level, corner) tuple
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n_corners = 1 << D;
    const uint32_t total_tuples = B * L * n_corners;

    if (tid >= total_tuples) return;

    // Decode (b, level, corner) from linear index
    const uint32_t b = tid / (L * n_corners);
    const uint32_t level = (tid / n_corners) % L;
    const uint32_t corner = tid % n_corners;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const uint32_t level_offset = offsets[level];

    // Get precomputed hash values
    const uint32_t h1 = precomp_h1[b * L * n_corners + level * n_corners + corner];
    const uint32_t h2 = precomp_h2[b * L * n_corners + level * n_corners + corner];

    // Compute embedding index
    uint32_t emb_idx;
    if (h2 == 0xFFFFFFFF) {
        // Direct indexing
        emb_idx = level_offset + h1;
    } else if (probe_indices != nullptr && N_c > 0) {
        // Learned probing
        uint32_t h2_mod = h2 % N_c;
        int probe = probe_indices[level * N_c + h2_mod];
        if (probe < 0 || (uint32_t)probe >= N_p) probe = 0;
        uint64_t final_idx = (uint64_t)N_p * h1 + probe;
        emb_idx = level_offset + (uint32_t)(final_idx % hashmap_size);
    } else {
        // Standard hashing
        emb_idx = level_offset + (uint32_t)(h1 % hashmap_size);
    }

    // Process each channel
    #pragma unroll
    for (uint32_t c = 0; c < C; c++) {
        const uint32_t idx = emb_idx * C + c;

        // Atomically read and zero the gradient (deduplication) - float supports atomicExch
        float g = atomicExch(&grad_embeddings[idx], 0.0f);

        if (g == 0.0f) continue;  // Already processed by another thread

        // Load current Adam state
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        // Adam update
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        // Store updated state
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        // Bias-corrected estimates
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;

        // Update embedding (AdamW: decoupled weight decay)
        float param = (float)embeddings[idx];
        float update = lr * m_hat / (sqrtf(v_hat) + eps);
        param = param * (1.0f - lr * weight_decay) - update;
        embeddings[idx] = (scalar_t)param;
    }
}


// =============================================================================
// FUSED BACKWARD + SGD UPDATE KERNEL
// =============================================================================
// Directly updates embeddings without storing gradients - eliminates gradient
// buffer and reduces memory bandwidth. For small batch training where optimizer
// overhead dominates.

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_backward_sgd_fused(
    const scalar_t * __restrict__ grad,            // [L, B, C] output gradients
    const int * __restrict__ offsets,              // [L+1]
    const uint32_t * __restrict__ precomp_h1,      // [B, L, 2^D]
    const uint32_t * __restrict__ precomp_h2,      // [B, L, 2^D]
    const float * __restrict__ precomp_weights,    // [B, L, 2^D]
    const int * __restrict__ probe_indices,        // [L, N_c]
    scalar_t * __restrict__ embeddings,            // [total_embeddings, C] - MODIFIED IN PLACE
    const uint32_t B, const uint32_t L,
    const uint32_t N_p,
    const uint32_t N_c,
    const float lr                                 // Learning rate (negative for descent)
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;
    const uint32_t n_corners = 1 << D;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

    // Locate pointers
    scalar_t* level_embeddings = embeddings + offsets[level] * C;
    const uint32_t* h1_ptr = precomp_h1 + b * L * n_corners + level * n_corners;
    const uint32_t* h2_ptr = precomp_h2 + b * L * n_corners + level * n_corners;
    const float* w_ptr = precomp_weights + b * L * n_corners + level * n_corners;
    const scalar_t* grad_ptr = grad + level * B * C + b * C + ch;

    scalar_t grad_cur[N_C];
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad_ptr[c];
    }

    #pragma unroll
    for (uint32_t idx = 0; idx < n_corners; idx++) {
        float w = w_ptr[idx];
        if (w == 0.0f) continue;

        uint32_t h1 = h1_ptr[idx];
        uint32_t h2 = h2_ptr[idx];

        uint32_t index;
        if (h2 == 0xFFFFFFFF) {
            index = h1 * C + ch;
        } else if (probe_indices != nullptr && N_c > 0) {
            uint32_t h2_mod = h2 % N_c;
            int probe = probe_indices[level * N_c + h2_mod];
            if (probe < 0 || (uint32_t)probe >= N_p) probe = 0;
            uint64_t final_idx = (uint64_t)N_p * h1 + probe;
            index = (uint32_t)((final_idx % hashmap_size) * C + ch);
        } else {
            index = (uint32_t)((h1 % hashmap_size) * C + ch);
        }

        // Fused SGD update: embedding -= lr * grad
        // lr is passed as negative, so we add
        #pragma unroll
        for (uint32_t c = 0; c < N_C; c++) {
            atomicAdd(&level_embeddings[index + c], (scalar_t)(lr * w * grad_cur[c]));
        }
    }
}


// =============================================================================
// MULTI-BATCH FORWARD KERNEL
// =============================================================================
// Process multiple mini-batches in a SINGLE kernel launch.
// Each thread processes one sample, batch boundaries are implicit.
// Reduces kernel launch overhead for small batch training.

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_precomputed_multibatch(
    const scalar_t * __restrict__ grid,            // [total_embeddings, C]
    const int * __restrict__ offsets,              // [L+1]
    const uint32_t * __restrict__ precomp_h1,      // [N_total, L, 2^D]
    const uint32_t * __restrict__ precomp_h2,      // [N_total, L, 2^D]
    const float * __restrict__ precomp_weights,    // [N_total, L, 2^D]
    const int * __restrict__ probe_indices,        // [L, N_c]
    const int * __restrict__ sample_indices,       // [N_total] indices into precomputed buffers
    scalar_t * __restrict__ outputs,               // [L, N_total, C]
    const uint32_t N_total,                        // Total samples across all batches
    const uint32_t L,
    const uint32_t N_p,
    const uint32_t N_c
) {
    const uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N_total) return;

    const uint32_t level = blockIdx.y;
    const uint32_t n_corners = 1 << D;
    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

    // Get the actual index into precomputed buffers
    const uint32_t precomp_idx = sample_indices[sample];

    // Locate pointers
    const scalar_t* level_grid = grid + (uint32_t)offsets[level] * C;
    const uint32_t* h1_ptr = precomp_h1 + precomp_idx * L * n_corners + level * n_corners;
    const uint32_t* h2_ptr = precomp_h2 + precomp_idx * L * n_corners + level * n_corners;
    const float* w_ptr = precomp_weights + precomp_idx * L * n_corners + level * n_corners;
    scalar_t* out_ptr = outputs + level * N_total * C + sample * C;

    scalar_t results[C] = {0};

    #pragma unroll
    for (uint32_t idx = 0; idx < n_corners; idx++) {
        float w = w_ptr[idx];
        if (w == 0.0f) continue;

        uint32_t h1 = h1_ptr[idx];
        uint32_t h2 = h2_ptr[idx];

        uint32_t index;
        if (h2 == 0xFFFFFFFF) {
            index = h1 * C;
        } else if (probe_indices != nullptr && N_c > 0) {
            uint32_t h2_mod = h2 % N_c;
            int probe = probe_indices[level * N_c + h2_mod];
            if (probe < 0 || (uint32_t)probe >= N_p) probe = 0;
            uint64_t final_idx = (uint64_t)N_p * h1 + probe;
            index = (uint32_t)((final_idx % hashmap_size) * C);
        } else {
            index = (uint32_t)((h1 % hashmap_size) * C);
        }

        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += (scalar_t)(w * (float)level_grid[index + ch]);
        }
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        out_ptr[ch] = results[ch];
    }
}


// =============================================================================
// WRAPPER FUNCTIONS
// =============================================================================

template <typename scalar_t, uint32_t D>
void kernel_backward_sgd_fused_wrapper(
    const scalar_t *grad, const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2, const float *precomp_weights,
    const int *probe_indices, scalar_t *embeddings,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c, const float lr
) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C);
    const dim3 blocks = { div_round_up(B * C / N_C, N_THREAD), L, 1 };

    switch (C) {
        case 1: kernel_backward_sgd_fused<scalar_t, D, 1, 1><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, embeddings, B, L, N_p, N_c, lr); break;
        case 2: kernel_backward_sgd_fused<scalar_t, D, 2, 2><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, embeddings, B, L, N_p, N_c, lr); break;
        case 4: kernel_backward_sgd_fused<scalar_t, D, 4, 2><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, embeddings, B, L, N_p, N_c, lr); break;
        case 8: kernel_backward_sgd_fused<scalar_t, D, 8, 2><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, embeddings, B, L, N_p, N_c, lr); break;
        default: throw std::runtime_error{"Fused backward: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t, uint32_t D>
void kernel_adam_sparse_update_wrapper(
    const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2,
    const int *probe_indices,
    scalar_t *embeddings, float *grad_embeddings,  // grad_embeddings is always float
    float *exp_avg, float *exp_avg_sq,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr, const float beta1, const float beta2, const float eps,
    const float weight_decay,
    const float bias_correction1, const float bias_correction2
) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t n_corners = 1 << D;
    const uint32_t total_tuples = B * L * n_corners;
    const dim3 blocks = { div_round_up(total_tuples, N_THREAD), 1, 1 };

    switch (C) {
        case 1: kernel_adam_sparse_update<scalar_t, D, 1><<<blocks, N_THREAD>>>(offsets, precomp_h1, precomp_h2, probe_indices, embeddings, grad_embeddings, exp_avg, exp_avg_sq, B, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2); break;
        case 2: kernel_adam_sparse_update<scalar_t, D, 2><<<blocks, N_THREAD>>>(offsets, precomp_h1, precomp_h2, probe_indices, embeddings, grad_embeddings, exp_avg, exp_avg_sq, B, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2); break;
        case 4: kernel_adam_sparse_update<scalar_t, D, 4><<<blocks, N_THREAD>>>(offsets, precomp_h1, precomp_h2, probe_indices, embeddings, grad_embeddings, exp_avg, exp_avg_sq, B, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2); break;
        case 8: kernel_adam_sparse_update<scalar_t, D, 8><<<blocks, N_THREAD>>>(offsets, precomp_h1, precomp_h2, probe_indices, embeddings, grad_embeddings, exp_avg, exp_avg_sq, B, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2); break;
        default: throw std::runtime_error{"Adam sparse update: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grid_precomputed_multibatch_wrapper(
    const scalar_t *grid, const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2, const float *precomp_weights,
    const int *probe_indices, const int *sample_indices, scalar_t *outputs,
    const uint32_t N_total, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks = { div_round_up(N_total, N_THREAD), L, 1 };

    switch (C) {
        case 1: kernel_grid_precomputed_multibatch<scalar_t, D, 1><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, sample_indices, outputs, N_total, L, N_p, N_c); break;
        case 2: kernel_grid_precomputed_multibatch<scalar_t, D, 2><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, sample_indices, outputs, N_total, L, N_p, N_c); break;
        case 4: kernel_grid_precomputed_multibatch<scalar_t, D, 4><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, sample_indices, outputs, N_total, L, N_p, N_c); break;
        case 8: kernel_grid_precomputed_multibatch<scalar_t, D, 8><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, sample_indices, outputs, N_total, L, N_p, N_c); break;
        default: throw std::runtime_error{"Multibatch forward: C must be 1, 2, 4, or 8."};
    }
}

template <typename input_t, uint32_t D>
void kernel_precompute_wrapper(
    const input_t *inputs, const int *offsets,
    uint32_t *precomp_h1, uint32_t *precomp_h2,
    float *precomp_weights, float *precomp_pos_deriv,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const float *per_level_scale, const float *base_resolution,
    const uint32_t N_f, const uint32_t N_p
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks = { div_round_up(B, N_THREAD), L, 1 };

    switch (C) {
        case 1: kernel_precompute<input_t, D, 1><<<blocks, N_THREAD>>>(inputs, offsets, precomp_h1, precomp_h2, precomp_weights, precomp_pos_deriv, B, L, per_level_scale, base_resolution, N_f, N_p); break;
        case 2: kernel_precompute<input_t, D, 2><<<blocks, N_THREAD>>>(inputs, offsets, precomp_h1, precomp_h2, precomp_weights, precomp_pos_deriv, B, L, per_level_scale, base_resolution, N_f, N_p); break;
        case 4: kernel_precompute<input_t, D, 4><<<blocks, N_THREAD>>>(inputs, offsets, precomp_h1, precomp_h2, precomp_weights, precomp_pos_deriv, B, L, per_level_scale, base_resolution, N_f, N_p); break;
        case 8: kernel_precompute<input_t, D, 8><<<blocks, N_THREAD>>>(inputs, offsets, precomp_h1, precomp_h2, precomp_weights, precomp_pos_deriv, B, L, per_level_scale, base_resolution, N_f, N_p); break;
        default: throw std::runtime_error{"Precompute: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grid_precomputed_wrapper(
    const scalar_t *grid, const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2, const float *precomp_weights,
    const int *probe_indices, scalar_t *outputs,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks = { div_round_up(B, N_THREAD), L, 1 };

    switch (C) {
        case 1: kernel_grid_precomputed<scalar_t, D, 1><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, outputs, B, L, N_p, N_c); break;
        case 2: kernel_grid_precomputed<scalar_t, D, 2><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, outputs, B, L, N_p, N_c); break;
        case 4: kernel_grid_precomputed<scalar_t, D, 4><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, outputs, B, L, N_p, N_c); break;
        case 8: kernel_grid_precomputed<scalar_t, D, 8><<<blocks, N_THREAD>>>(grid, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, outputs, B, L, N_p, N_c); break;
        default: throw std::runtime_error{"Precomputed forward: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_precomputed_wrapper(
    const scalar_t *grad, const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2, const float *precomp_weights,
    const int *probe_indices, const float *index_logits, float *grad_index_logits,
    scalar_t *grad_grid,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c
) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C);
    const dim3 blocks = { div_round_up(B * C / N_C, N_THREAD), L, 1 };

    switch (C) {
        case 1: kernel_grid_backward_precomputed<scalar_t, D, 1, 1><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits, grad_index_logits, grad_grid, B, L, N_p, N_c); break;
        case 2: kernel_grid_backward_precomputed<scalar_t, D, 2, 2><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits, grad_index_logits, grad_grid, B, L, N_p, N_c); break;
        case 4: kernel_grid_backward_precomputed<scalar_t, D, 4, 2><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits, grad_index_logits, grad_grid, B, L, N_p, N_c); break;
        case 8: kernel_grid_backward_precomputed<scalar_t, D, 8, 2><<<blocks, N_THREAD>>>(grad, offsets, precomp_h1, precomp_h2, precomp_weights, probe_indices, index_logits, grad_index_logits, grad_grid, B, L, N_p, N_c); break;
        default: throw std::runtime_error{"Precomputed backward: C must be 1, 2, 4, or 8."};
    }
}

// =============================================================================
// ENTRY POINTS
// =============================================================================

void hash_encode_precompute(
    const at::Tensor inputs,
    const at::Tensor offsets,
    at::Tensor precomp_h1,
    at::Tensor precomp_h2,
    at::Tensor precomp_weights,
    at::Tensor precomp_pos_deriv,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const at::Tensor per_level_scale,
    const at::Tensor base_resolution,
    const uint32_t N_f, const uint32_t N_p
) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(precomp_weights);
    CHECK_CUDA(precomp_pos_deriv);
    CHECK_CUDA(per_level_scale);
    CHECK_CUDA(base_resolution);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(precomp_h1);
    CHECK_CONTIGUOUS(precomp_h2);
    CHECK_CONTIGUOUS(precomp_weights);
    CHECK_CONTIGUOUS(precomp_pos_deriv);

    auto per_level_scale_log2 = torch::log2(per_level_scale);

    // Cast int32 pointers to uint32_t (same bit representation, safe reinterpret)
    uint32_t* h1_ptr = reinterpret_cast<uint32_t*>(precomp_h1.data_ptr<int>());
    uint32_t* h2_ptr = reinterpret_cast<uint32_t*>(precomp_h2.data_ptr<int>());

    if (inputs.scalar_type() == at::ScalarType::Double) {
        switch (D) {
            case 2: kernel_precompute_wrapper<double, 2>(inputs.data_ptr<double>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            case 3: kernel_precompute_wrapper<double, 3>(inputs.data_ptr<double>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            default: throw std::runtime_error{"Precompute: D must be 2 or 3."};
        }
    } else {
        switch (D) {
            case 2: kernel_precompute_wrapper<float, 2>(inputs.data_ptr<float>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            case 3: kernel_precompute_wrapper<float, 3>(inputs.data_ptr<float>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            default: throw std::runtime_error{"Precompute: D must be 2 or 3."};
        }
    }
}

void hash_encode_forward_precomputed(
    const at::Tensor embeddings,
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor precomp_weights,
    const at::Tensor probe_indices,
    at::Tensor outputs,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c
) {
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(precomp_weights);
    CHECK_CUDA(outputs);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    // Cast int32 pointers to uint32_t
    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "hash_encode_forward_precomputed", ([&] {
        switch (D) {
            case 2: kernel_grid_precomputed_wrapper<scalar_t, 2>(embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, outputs.data_ptr<scalar_t>(), B, C, L, N_p, N_c); break;
            case 3: kernel_grid_precomputed_wrapper<scalar_t, 3>(embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, outputs.data_ptr<scalar_t>(), B, C, L, N_p, N_c); break;
            default: throw std::runtime_error{"Precomputed forward: D must be 2 or 3."};
        }
    }));
}

void hash_encode_backward_precomputed(
    const at::Tensor grad,
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor precomp_weights,
    const at::Tensor probe_indices,
    const at::Tensor index_logits,
    at::Tensor grad_index_logits,
    at::Tensor grad_embeddings,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c
) {
    CHECK_CUDA(grad);
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(precomp_weights);
    CHECK_CUDA(grad_embeddings);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    float *index_logits_ptr = nullptr;
    if (index_logits.defined() && index_logits.numel() > 0) {
        CHECK_CUDA(index_logits);
        index_logits_ptr = index_logits.data_ptr<float>();
    }

    float *grad_index_logits_ptr = nullptr;
    if (grad_index_logits.defined() && grad_index_logits.numel() > 0) {
        CHECK_CUDA(grad_index_logits);
        grad_index_logits_ptr = grad_index_logits.data_ptr<float>();
    }

    // Cast int32 pointers to uint32_t
    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "hash_encode_backward_precomputed", ([&] {
        switch (D) {
            case 2: kernel_grid_backward_precomputed_wrapper<scalar_t, 2>(grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, index_logits_ptr, grad_index_logits_ptr, grad_embeddings.data_ptr<scalar_t>(), B, C, L, N_p, N_c); break;
            case 3: kernel_grid_backward_precomputed_wrapper<scalar_t, 3>(grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, index_logits_ptr, grad_index_logits_ptr, grad_embeddings.data_ptr<scalar_t>(), B, C, L, N_p, N_c); break;
            default: throw std::runtime_error{"Precomputed backward: D must be 2 or 3."};
        }
    }));
}


// =============================================================================
// NEW OPTIMIZED ENTRY POINTS
// =============================================================================

void hash_encode_backward_sgd_fused(
    const at::Tensor grad,
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor precomp_weights,
    const at::Tensor probe_indices,
    at::Tensor embeddings,  // Modified in place!
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr
) {
    CHECK_CUDA(grad);
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(precomp_weights);
    CHECK_CUDA(embeddings);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "hash_encode_backward_sgd_fused", ([&] {
        switch (D) {
            case 2: kernel_backward_sgd_fused_wrapper<scalar_t, 2>(grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, embeddings.data_ptr<scalar_t>(), B, C, L, N_p, N_c, lr); break;
            case 3: kernel_backward_sgd_fused_wrapper<scalar_t, 3>(grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, embeddings.data_ptr<scalar_t>(), B, C, L, N_p, N_c, lr); break;
            default: throw std::runtime_error{"Fused backward: D must be 2 or 3."};
        }
    }));
}

void hash_encode_forward_precomputed_multibatch(
    const at::Tensor embeddings,
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor precomp_weights,
    const at::Tensor probe_indices,
    const at::Tensor sample_indices,  // [N_total] indices into precomputed buffers
    at::Tensor outputs,
    const uint32_t N_total, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c
) {
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(precomp_weights);
    CHECK_CUDA(sample_indices);
    CHECK_CUDA(outputs);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "hash_encode_forward_precomputed_multibatch", ([&] {
        switch (D) {
            case 2: kernel_grid_precomputed_multibatch_wrapper<scalar_t, 2>(embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, sample_indices.data_ptr<int>(), outputs.data_ptr<scalar_t>(), N_total, C, L, N_p, N_c); break;
            case 3: kernel_grid_precomputed_multibatch_wrapper<scalar_t, 3>(embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), probe_indices_ptr, sample_indices.data_ptr<int>(), outputs.data_ptr<scalar_t>(), N_total, C, L, N_p, N_c); break;
            default: throw std::runtime_error{"Multibatch forward: D must be 2 or 3."};
        }
    }));
}

void hash_encode_adam_sparse_update(
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor probe_indices,
    at::Tensor embeddings,           // Modified in place
    at::Tensor grad_embeddings,      // Modified in place (zeroed after use)
    at::Tensor exp_avg,              // Adam first moment, modified in place
    at::Tensor exp_avg_sq,           // Adam second moment, modified in place
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr, const float beta1, const float beta2, const float eps,
    const float weight_decay,        // AdamW weight decay
    const uint32_t step              // Current optimization step (1-indexed)
) {
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(exp_avg);
    CHECK_CUDA(exp_avg_sq);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());

    // Compute bias corrections
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    // grad_embeddings must be float type for atomicExch support
    TORCH_CHECK(grad_embeddings.scalar_type() == at::kFloat,
                "grad_embeddings must be float type for Adam sparse update");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "hash_encode_adam_sparse_update", ([&] {
        switch (D) {
            case 2: kernel_adam_sparse_update_wrapper<scalar_t, 2>(offsets.data_ptr<int>(), h1_ptr, h2_ptr, probe_indices_ptr, embeddings.data_ptr<scalar_t>(), grad_embeddings.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), B, C, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2); break;
            case 3: kernel_adam_sparse_update_wrapper<scalar_t, 3>(offsets.data_ptr<int>(), h1_ptr, h2_ptr, probe_indices_ptr, embeddings.data_ptr<scalar_t>(), grad_embeddings.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), B, C, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2); break;
            default: throw std::runtime_error{"Adam sparse update: D must be 2 or 3."};
        }
    }));
}
