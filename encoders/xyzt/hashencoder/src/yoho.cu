/**
 * yoho.cu - Warp-level YOHO (You Only Hash Once) hash encoding kernels
 *
 * Contains:
 * - kernel_grid_warp_yoho: Forward pass with intra-warp deduplication
 * - kernel_grid_backward_warp_yoho: Backward pass with warp-level reduction
 * - Entry points: hash_encode_forward_warp_yoho, hash_encode_backward_warp_yoho
 *
 * Warp-level YOHO exploits spatial coherence: threads in the same warp that
 * share a grid cell only compute once, then broadcast via shuffle.
 * This is automatic and requires no Python-side preprocessing.
 */

#include "utils.cuh"


// =============================================================================
// WARP-LEVEL YOHO FORWARD KERNEL
// =============================================================================

template <typename input_t, typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_warp_yoho(
    const input_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets,
    scalar_t * __restrict__ outputs,
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution,
    const bool calc_grad_inputs,
    scalar_t * __restrict__ dy_dx,
    // Learned probing parameters (optional, nullptr means disabled)
    const int * __restrict__ probe_indices = nullptr,  // Shape: (L, N_c) or nullptr
    const uint32_t N_f = 0,
    const uint32_t N_p = 1,
    const uint32_t N_c = 0,
    // Deduplication tracking (optional, nullptr means disabled)
    uint32_t * __restrict__ dedup_stats = nullptr  // Shape: (L,) - counts unique cells per level
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t level = blockIdx.y;
    const int lane_id = threadIdx.x & 31;  // Lane within warp (0-31)

    if (b >= B) return;

    // locate
    const scalar_t* level_grid = grid + (uint32_t)offsets[level] * C;
    const input_t* my_inputs = inputs + b * D;
    scalar_t* my_outputs = outputs + level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (my_inputs[d] < 0 || my_inputs[d] > 1) {
            flag_oob = true;
        }
    }

    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            my_outputs[ch] = 0;
        }
        if (calc_grad_inputs) {
            scalar_t* my_dy_dx = dy_dx + b * D * L * C + level * D * C;
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    my_dy_dx[d * C + ch] = 0;
                }
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    double scale[D];
    uint32_t resolution[D];
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2(level * (double)per_level_scale[d]) * (double)base_resolution[d] - 1.0;
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    // calculate coordinate using high precision
    double pos_hp[D];
    float pos[D];
    float pos_derivative[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos_hp[d] = (double)my_inputs[d] * scale[d];
        pos_grid[d] = (uint32_t)floor(pos_hp[d]);
        pos[d] = (float)(pos_hp[d] - (double)pos_grid[d]);
        pos_derivative[d] = smoothstep_derivative(pos[d]);
        pos[d] = smoothstep(pos[d]);
    }

    // WARP-LEVEL YOHO: Share embedding reads, but each thread computes own interpolation
    uint64_t cell_key = pack_cell_key<D>(pos_grid);
    unsigned int match_mask;
    int leader_lane;
    find_warp_cell_leader(cell_key, match_mask, leader_lane);

    bool is_leader = (lane_id == leader_lane);

    // Track deduplication: count unique cells (leaders) per level
    if (dedup_stats != nullptr && is_leader) {
        atomicAdd(&dedup_stats[level], 1);
    }

    // Results array
    scalar_t results[C] = {0};

    // For each of the 2^D corners
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        // Each thread computes its own interpolation weight
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // YOHO: Only leader reads from global memory, then broadcasts
        float corner_embeddings[C];

        if (is_leader) {
            uint32_t index;
            if (probe_indices != nullptr) {
                index = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
            } else {
                index = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
            }
            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                corner_embeddings[ch] = (float)level_grid[index + ch];
            }
        }

        // Broadcast corner embeddings from leader to all threads with same cell
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            corner_embeddings[ch] = __shfl_sync(match_mask, corner_embeddings[ch], leader_lane);
        }

        // Each thread computes its own weighted contribution
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += (scalar_t)(w * corner_embeddings[ch]);
        }
    }

    // All threads write to their output location
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        my_outputs[ch] = results[ch];
    }

    // Handle dy_dx (gradient computation) with YOHO
    if (calc_grad_inputs) {
        scalar_t* my_dy_dx = dy_dx + b * D * L * C + level * D * C;

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {
            float results_grad[C] = {0};

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale[gd];
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;
                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }

                // YOHO: Leader reads embedding difference, broadcasts
                float embed_diff[C];
                if (is_leader) {
                    pos_grid_local[gd] = pos_grid[gd];
                    uint32_t index_left;
                    if (probe_indices != nullptr) {
                        index_left = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
                    } else {
                        index_left = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                    }
                    pos_grid_local[gd] = pos_grid[gd] + 1;
                    uint32_t index_right;
                    if (probe_indices != nullptr) {
                        index_right = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
                    } else {
                        index_right = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                    }

                    #pragma unroll
                    for (uint32_t ch = 0; ch < C; ch++) {
                        embed_diff[ch] = (float)level_grid[index_right + ch] - (float)level_grid[index_left + ch];
                    }
                }

                // Broadcast embedding differences
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    embed_diff[ch] = __shfl_sync(match_mask, embed_diff[ch], leader_lane);
                }

                // Each thread computes its own gradient contribution
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * embed_diff[ch] * pos_derivative[gd];
                }
            }

            // Write gradients
            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                my_dy_dx[gd * C + ch] = (scalar_t)results_grad[ch];
            }
        }
    }
}


// =============================================================================
// WARP-LEVEL YOHO BACKWARD KERNEL
// =============================================================================

template <typename input_t, typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_backward_warp_yoho(
    const scalar_t * __restrict__ grad,
    const input_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets,
    scalar_t * __restrict__ grad_grid,
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution,
    // Learned probing parameters (optional, nullptr means disabled)
    const int * __restrict__ probe_indices = nullptr,  // Shape: (L, N_c) or nullptr
    const float * __restrict__ index_logits = nullptr, // Shape: (L, N_c, N_p) for softmax
    float * __restrict__ grad_index_logits = nullptr,  // Shape: (L, N_c, N_p) output gradients
    const uint32_t N_f = 0,
    const uint32_t N_p = 1,
    const uint32_t N_c = 0
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t level = blockIdx.y;
    const int lane_id = threadIdx.x & 31;

    if (b >= B) return;

    // locate
    scalar_t* level_grad_grid = grad_grid + offsets[level] * C;
    const input_t* my_inputs = inputs + b * D;
    const scalar_t* my_grad = grad + level * B * C + b * C;

    // check input range
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (my_inputs[d] < 0 || my_inputs[d] > 1) {
            return;
        }
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    double scale[D];
    uint32_t resolution[D];
    uint64_t stride = 1;
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2(level * (double)per_level_scale[d]) * (double)base_resolution[d] - 1.0;
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
        stride *= resolution[d];
    }

    // calculate coordinate
    double pos_hp[D];
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos_hp[d] = (double)my_inputs[d] * scale[d];
        pos_grid[d] = (uint32_t)floor(pos_hp[d]);
        pos[d] = (float)(pos_hp[d] - (double)pos_grid[d]);
        pos[d] = smoothstep(pos[d]);
    }

    // Fetch gradient to register
    scalar_t grad_cur[C];
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        grad_cur[ch] = my_grad[ch];
    }

    // YOHO: Find threads with same cell
    uint64_t cell_key = pack_cell_key<D>(pos_grid);
    unsigned int match_mask;
    int leader_lane;
    find_warp_cell_leader(cell_key, match_mask, leader_lane);

    bool is_leader = (lane_id == leader_lane);

    // For each corner, compute weighted gradient and reduce across threads with same cell
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        // Compute weighted gradients for this corner
        float weighted_grad[C];
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            weighted_grad[ch] = w * (float)grad_cur[ch];
        }

        // Warp reduction: sum gradients from all threads with same cell
        // IMPORTANT: Use __activemask() to handle warps where some threads have exited early
        // (e.g., b >= B or out-of-bounds inputs). Using 0xFFFFFFFF is undefined behavior
        // when not all 32 threads are active.
        unsigned int active_mask = __activemask();
        float summed_grad[C] = {0};
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            #pragma unroll
            for (int src_lane = 0; src_lane < 32; src_lane++) {
                // Only shuffle from active threads to avoid undefined behavior
                float val = __shfl_sync(active_mask, weighted_grad[ch], src_lane);
                // Only leader accumulates, and only from matching threads
                if (is_leader && (match_mask & (1u << src_lane))) {
                    summed_grad[ch] += val;
                }
            }
        }

        // Only leader does atomicAdd with summed gradient
        if (is_leader) {
            // Check if we need learned probing (stride > hashmap_size means collision)
            if (probe_indices != nullptr && index_logits != nullptr && stride > hashmap_size) {
                // Use learned hash probing with softmax weighting
                uint64_t h1 = fast_hash<D>(pos_grid_local) % N_f;
                uint64_t h2 = fast_hash2<D>(pos_grid_local) % N_c;

                // Get pointer to logits for this (level, h2)
                const float* logits = &index_logits[level * N_c * N_p + h2 * N_p];

                // Compute softmax weights
                float weights[16];  // Max N_p we support
                float max_logit = logits[0];
                for (uint32_t p = 1; p < N_p; ++p) {
                    max_logit = max(max_logit, logits[p]);
                }

                float sum_exp = 0.0f;
                for (uint32_t p = 0; p < N_p; ++p) {
                    weights[p] = expf(logits[p] - max_logit);
                    sum_exp += weights[p];
                }

                for (uint32_t p = 0; p < N_p; ++p) {
                    weights[p] /= sum_exp;
                }

                // Distribute gradients to all N_p probes weighted by softmax
                for (uint32_t p = 0; p < N_p; ++p) {
                    uint32_t probe_index = ((N_p * h1 + p) % hashmap_size) * C;
                    #pragma unroll
                    for (uint32_t ch = 0; ch < C; ch++) {
                        atomicAdd(&level_grad_grid[probe_index + ch], (scalar_t)(weights[p] * summed_grad[ch]));
                    }
                }

                // Compute gradients for index_logits (softmax backward)
                if (grad_index_logits != nullptr) {
                    // Compute total gradient magnitude for this corner
                    float total_grad = 0.0f;
                    #pragma unroll
                    for (uint32_t ch = 0; ch < C; ch++) {
                        total_grad += summed_grad[ch];
                    }

                    // Compute dot product for softmax backward
                    float dot_product = 0.0f;
                    for (uint32_t p = 0; p < N_p; ++p) {
                        dot_product += weights[p] * total_grad;
                    }

                    // Apply softmax backward: grad_logit[p] = weight[p] * (grad_weight[p] - dot_product)
                    for (uint32_t p = 0; p < N_p; ++p) {
                        float grad_logit = weights[p] * (total_grad - dot_product);
                        uint32_t logit_idx = level * N_c * N_p + h2 * N_p + p;
                        atomicAdd(&grad_index_logits[logit_idx], grad_logit);
                    }
                }
            } else {
                // Standard indexing (no learned probing or grid fits in hashmap)
                uint32_t index;
                if (probe_indices != nullptr && stride > hashmap_size) {
                    // Learned probing but no index_logits - use argmax probe
                    index = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
                } else {
                    // Standard indexing
                    index = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                }

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    atomicAdd(&level_grad_grid[index + ch], (scalar_t)summed_grad[ch]);
                }
            }
        }
    }
}


// =============================================================================
// WRAPPER FUNCTIONS
// =============================================================================

// Forward declaration for kernel_input_backward (defined in hashencoder.cu)
// We use a local implementation here to avoid cross-TU template issues
template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward_warp(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,
    scalar_t * __restrict__ grad_inputs,
    uint32_t B, uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= B * D) return;

    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    dy_dx += b * L * D * C;

    scalar_t result = 0;

    #pragma unroll
    for (int l = 0; l < L; l++) {
        #pragma unroll
        for (int ch = 0; ch < C; ch++) {
            result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }

    grad_inputs[t] = result;
}

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_warp_yoho_wrapper(const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, const int *probe_indices, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c, uint32_t *dedup_stats) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid_warp_yoho<input_t, scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, N_f, N_p, N_c, dedup_stats); break;
        case 2: kernel_grid_warp_yoho<input_t, scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, N_f, N_p, N_c, dedup_stats); break;
        case 4: kernel_grid_warp_yoho<input_t, scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, N_f, N_p, N_c, dedup_stats); break;
        case 8: kernel_grid_warp_yoho<input_t, scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, N_f, N_p, N_c, dedup_stats); break;
        default: throw std::runtime_error{"GridEncoding YOHO: C must be 1, 2, 4, or 8."};
    }
}

template <typename input_t, typename scalar_t>
void hash_encode_forward_warp_yoho_cuda(const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, const int *probe_indices, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c, uint32_t *dedup_stats) {
    switch (D) {
        case 2: kernel_grid_warp_yoho_wrapper<input_t, scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, N_f, N_p, N_c, dedup_stats); break;
        case 3: kernel_grid_warp_yoho_wrapper<input_t, scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, probe_indices, N_f, N_p, N_c, dedup_stats); break;
        default: throw std::runtime_error{"GridEncoding YOHO: D must be 2 or 3."};
    }
}

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_backward_warp_yoho_wrapper(const scalar_t *grad, const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, const scalar_t *dy_dx, scalar_t *grad_inputs, const int *probe_indices, const float *index_logits, float *grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid_backward_warp_yoho<input_t, scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        case 2: kernel_grid_backward_warp_yoho<input_t, scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        case 4: kernel_grid_backward_warp_yoho<input_t, scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        case 8: kernel_grid_backward_warp_yoho<input_t, scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        default: throw std::runtime_error{"GridEncoding YOHO backward: C must be 1, 2, 4, or 8."};
    }
    // Handle grad_inputs
    if (calc_grad_inputs) {
        static constexpr uint32_t N_THREAD_INPUT = 256;
        switch (C) {
            case 1: kernel_input_backward_warp<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD_INPUT), N_THREAD_INPUT>>>(grad, dy_dx, grad_inputs, B, L); break;
            case 2: kernel_input_backward_warp<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD_INPUT), N_THREAD_INPUT>>>(grad, dy_dx, grad_inputs, B, L); break;
            case 4: kernel_input_backward_warp<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD_INPUT), N_THREAD_INPUT>>>(grad, dy_dx, grad_inputs, B, L); break;
            case 8: kernel_input_backward_warp<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD_INPUT), N_THREAD_INPUT>>>(grad, dy_dx, grad_inputs, B, L); break;
            default: break;
        }
    }
}

template <typename input_t, typename scalar_t>
void hash_encode_backward_warp_yoho_cuda(const scalar_t *grad, const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, const scalar_t *dy_dx, scalar_t *grad_inputs, const int *probe_indices, const float *index_logits, float *grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c) {
    switch (D) {
        case 2: kernel_grid_backward_warp_yoho_wrapper<input_t, scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        case 3: kernel_grid_backward_warp_yoho_wrapper<input_t, scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        default: throw std::runtime_error{"GridEncoding YOHO backward: D must be 2 or 3."};
    }
}


// =============================================================================
// ENTRY POINTS
// =============================================================================

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
    const at::Tensor probe_indices,
    const uint32_t N_f,
    const uint32_t N_p,
    const uint32_t N_c,
    at::Tensor dedup_stats  // Shape: (L,) uint32 or empty tensor to disable tracking
) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(per_level_scale);
    CHECK_CUDA(base_resolution);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(per_level_scale);
    CHECK_CONTIGUOUS(base_resolution);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(per_level_scale);
    CHECK_IS_FLOATING(base_resolution);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(dy_dx);

    // Get probe_indices pointer (nullptr if empty tensor)
    const int* probe_ptr = (probe_indices.numel() > 0) ? probe_indices.data_ptr<int>() : nullptr;

    // Get dedup_stats pointer (nullptr if empty tensor)
    uint32_t* dedup_ptr = (dedup_stats.numel() > 0) ? dedup_stats.data_ptr<uint32_t>() : nullptr;

    if (inputs.scalar_type() == at::ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        embeddings.scalar_type(), "hash_encode_forward_warp_yoho", ([&] {
            hash_encode_forward_warp_yoho_cuda<double, scalar_t>(
                inputs.data_ptr<double>(),
                embeddings.data_ptr<scalar_t>(),
                offsets.data_ptr<int>(),
                outputs.data_ptr<scalar_t>(),
                B, D, C, L,
                per_level_scale.data_ptr<float>(),
                base_resolution.data_ptr<float>(),
                calc_grad_inputs,
                dy_dx.data_ptr<scalar_t>(),
                probe_ptr, N_f, N_p, N_c,
                dedup_ptr
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        inputs.scalar_type(), "hash_encode_forward_warp_yoho", ([&] {
            hash_encode_forward_warp_yoho_cuda<scalar_t, scalar_t>(
                inputs.data_ptr<scalar_t>(),
                embeddings.data_ptr<scalar_t>(),
                offsets.data_ptr<int>(),
                outputs.data_ptr<scalar_t>(),
                B, D, C, L,
                per_level_scale.data_ptr<float>(),
                base_resolution.data_ptr<float>(),
                calc_grad_inputs,
                dy_dx.data_ptr<scalar_t>(),
                probe_ptr, N_f, N_p, N_c,
                dedup_ptr
            );
        }));
    }
}

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
    const at::Tensor probe_indices,
    const at::Tensor index_logits,
    at::Tensor grad_index_logits,
    const uint32_t N_f,
    const uint32_t N_p,
    const uint32_t N_c
) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);
    CHECK_CUDA(per_level_scale);
    CHECK_CUDA(base_resolution);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);
    CHECK_CONTIGUOUS(per_level_scale);
    CHECK_CONTIGUOUS(base_resolution);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(per_level_scale);
    CHECK_IS_FLOATING(base_resolution);
    CHECK_IS_FLOATING(grad_embeddings);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_inputs);

    // Get probe_indices pointer (nullptr if empty tensor)
    const int* probe_ptr = (probe_indices.numel() > 0) ? probe_indices.data_ptr<int>() : nullptr;

    // Get index_logits pointer (nullptr if empty tensor)
    const float* logits_ptr = (index_logits.numel() > 0) ? index_logits.data_ptr<float>() : nullptr;

    // Get grad_index_logits pointer (nullptr if empty tensor)
    float* grad_logits_ptr = (grad_index_logits.numel() > 0) ? grad_index_logits.data_ptr<float>() : nullptr;

    if (inputs.scalar_type() == at::ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "hash_encode_backward_warp_yoho", ([&] {
            hash_encode_backward_warp_yoho_cuda<double, scalar_t>(
                grad.data_ptr<scalar_t>(),
                inputs.data_ptr<double>(),
                embeddings.data_ptr<scalar_t>(),
                offsets.data_ptr<int>(),
                grad_embeddings.data_ptr<scalar_t>(),
                B, D, C, L,
                per_level_scale.data_ptr<float>(),
                base_resolution.data_ptr<float>(),
                calc_grad_inputs,
                dy_dx.data_ptr<scalar_t>(),
                grad_inputs.data_ptr<scalar_t>(),
                probe_ptr, logits_ptr, grad_logits_ptr, N_f, N_p, N_c
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "hash_encode_backward_warp_yoho", ([&] {
            hash_encode_backward_warp_yoho_cuda<scalar_t, scalar_t>(
                grad.data_ptr<scalar_t>(),
                inputs.data_ptr<scalar_t>(),
                embeddings.data_ptr<scalar_t>(),
                offsets.data_ptr<int>(),
                grad_embeddings.data_ptr<scalar_t>(),
                B, D, C, L,
                per_level_scale.data_ptr<float>(),
                base_resolution.data_ptr<float>(),
                calc_grad_inputs,
                dy_dx.data_ptr<scalar_t>(),
                grad_inputs.data_ptr<scalar_t>(),
                probe_ptr, logits_ptr, grad_logits_ptr, N_f, N_p, N_c
            );
        }));
    }
}
