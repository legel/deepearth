/**
 * hashencoder.cu - Core hash encoding forward/backward kernels
 *
 * Contains:
 * - kernel_grid: Main forward kernel
 * - kernel_grid_backward: Backward pass for embeddings
 * - kernel_input_backward: Backward pass for inputs
 * - kernel_grid_second_backward_*: Second-order backward pass
 * - Entry points: hash_encode_forward, hash_encode_backward, hash_encode_second_backward
 */

#include "utils.cuh"

// =============================================================================
// MAIN FORWARD KERNEL
// =============================================================================

template <typename input_t, typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(
    const input_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets,
    scalar_t * __restrict__ outputs,
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution,
    const bool calc_grad_inputs,
    scalar_t * __restrict__ dy_dx,
    const bool track_collisions,
    int * __restrict__ collision_indices,
    const uint32_t example_offset,
    const uint32_t max_tracked_examples,
    // Learned probing parameters (optional, nullptr means disabled)
    const int * __restrict__ probe_indices = nullptr,  // Shape: (L, N_c) or nullptr
    const uint32_t N_f = 0,
    const uint32_t N_p = 1,
    const uint32_t N_c = 0
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B) return;

    const uint32_t level = blockIdx.y;

    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0;
        }
        if (calc_grad_inputs) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0;
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
    double pos_hp[D];  // High precision position
    float pos[D];      // Convert to float for interpolation
    float pos_derivative[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        // Use double precision for critical calculation
        pos_hp[d] = (double)inputs[d] * scale[d];
        pos_grid[d] = (uint32_t)floor(pos_hp[d]);
        // Convert to float for interpolation (after extracting integer part)
        pos[d] = (float)(pos_hp[d] - (double)pos_grid[d]);
        pos_derivative[d] = smoothstep_derivative(pos[d]);
        pos[d] = smoothstep(pos[d]);
    }

    // Collision tracking: capture hash table index
    if (track_collisions && collision_indices != nullptr) {
        const uint32_t global_example_idx = example_offset + b;
        if (global_example_idx < max_tracked_examples) {
            uint64_t stride = 1;
            uint64_t index = 0;

            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                index += (uint64_t)pos_grid[d] * stride;
                stride *= resolution[d];
            }

            if (stride > hashmap_size) {
                if (probe_indices != nullptr && N_c > 0 && N_f > 0 && N_p > 0) {
                    uint64_t h1 = fast_hash<D>(pos_grid) % N_f;
                    uint64_t h2 = fast_hash2<D>(pos_grid) % N_c;
                    uint64_t probe_idx = (uint64_t)level * N_c + h2;
                    int probe_raw = probe_indices[probe_idx];
                    uint32_t probe = (uint32_t)probe_raw;
                    if (probe >= N_p) probe = 0;
                    index = (uint64_t)N_p * h1 + probe;
                } else {
                    index = fast_hash<D>(pos_grid);
                }
            }

            uint32_t hash_table_index;
            if (stride > hashmap_size && probe_indices == nullptr) {
                hash_table_index = (uint32_t)(index % hashmap_size);
            } else {
                hash_table_index = (uint32_t)index;
            }
            collision_indices[global_example_idx * L + level] = (int)hash_table_index;
        }
    }

    // interpolate
    scalar_t results[C] = {0};

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

        uint32_t index;
        if (probe_indices != nullptr) {
            index = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
        } else {
            index = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
        }

        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }
    }

    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch];
    }

    // prepare dy_dx for calc_grad_inputs
    if (calc_grad_inputs) {
        dy_dx += b * D * L * C + level * D * C;

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {
            scalar_t results_grad[C] = {0};

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

                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left;
                uint32_t index_right;
                if (probe_indices != nullptr) {
                    index_left = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
                    pos_grid_local[gd] = pos_grid[gd] + 1;
                    index_right = get_grid_index_learned<D, C>(0, hashmap_size, N_f, N_p, N_c, level, resolution, pos_grid_local, probe_indices);
                } else {
                    index_left = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                    pos_grid_local[gd] = pos_grid[gd] + 1;
                    index_right = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                }

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_derivative[gd];
                }
            }

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                dy_dx[gd * C + ch] = results_grad[ch];
            }
        }
    }
}


// =============================================================================
// BACKWARD KERNEL (EMBEDDING GRADIENTS)
// =============================================================================

template <typename input_t, typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,
    const input_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets,
    scalar_t * __restrict__ grad_grid,
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution,
    const int * __restrict__ probe_indices = nullptr,
    const float * __restrict__ index_logits = nullptr,
    float * __restrict__ grad_index_logits = nullptr,
    const uint32_t N_f = 0,
    const uint32_t N_p = 1,
    const uint32_t N_c = 0
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    double scale[D];
    uint32_t resolution[D];
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2(level * (double)per_level_scale[d]) * (double)base_resolution[d] - 1.0;
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    // check input range
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return;
        }
    }

    // calculate coordinate with high precision
    double pos_hp[D];
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos_hp[d] = (double)inputs[d] * scale[d];
        pos_grid[d] = (uint32_t)floor(pos_hp[d]);
        pos[d] = (float)(pos_hp[d] - (double)pos_grid[d]);
        pos[d] = smoothstep(pos[d]);
    }

    scalar_t grad_cur[N_C] = {0};
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
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

        if (probe_indices != nullptr && index_logits != nullptr) {
            uint64_t stride = 1;
            uint64_t index_direct = 0;

            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                index_direct += (uint64_t)pos_grid_local[d] * stride;
                stride *= resolution[d];
            }

            if (stride > hashmap_size) {
                uint64_t h1 = fast_hash<D>(pos_grid_local) % N_f;
                uint64_t h2 = fast_hash2<D>(pos_grid_local) % N_c;

                const float* logits = &index_logits[level * N_c * N_p + h2 * N_p];

                float weights[16];
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

                #pragma unroll
                for (uint32_t p = 0; p < N_p; ++p) {
                    uint32_t probe_index = ((N_p * h1 + p) % hashmap_size) * C + ch;
                    float weight = w * weights[p];

                    if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
                        #pragma unroll
                        for (uint32_t c = 0; c < N_C; c += 2) {
                            __half2 v = {(__half)(weight * grad_cur[c]), (__half)(weight * grad_cur[c + 1])};
                            atomicAdd((__half2*)&grad_grid[probe_index + c], v);
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t c = 0; c < N_C; c++) {
                            atomicAdd(&grad_grid[probe_index + c], weight * grad_cur[c]);
                        }
                    }
                }

                if (grad_index_logits != nullptr) {
                    float grad_weights[16] = {0};

                    #pragma unroll
                    for (uint32_t p = 0; p < N_p; ++p) {
                        #pragma unroll
                        for (uint32_t c = 0; c < N_C; c++) {
                            grad_weights[p] += grad_cur[c] * w;
                        }
                    }

                    float dot_product = 0.0f;
                    #pragma unroll
                    for (uint32_t p = 0; p < N_p; ++p) {
                        dot_product += weights[p] * grad_weights[p];
                    }

                    #pragma unroll
                    for (uint32_t p = 0; p < N_p; ++p) {
                        float grad_logit = weights[p] * (grad_weights[p] - dot_product);
                        uint32_t logit_idx = level * N_c * N_p + h2 * N_p + p;
                        atomicAdd(&grad_index_logits[logit_idx], grad_logit);
                    }
                }
            } else {
                uint32_t index = (uint32_t)((index_direct % hashmap_size) * C + ch);

                if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
                    #pragma unroll
                    for (uint32_t c = 0; c < N_C; c += 2) {
                        __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                        atomicAdd((__half2*)&grad_grid[index + c], v);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t c = 0; c < N_C; c++) {
                        atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
                    }
                }
            }
        } else {
            uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);

            if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
                #pragma unroll
                for (uint32_t c = 0; c < N_C; c += 2) {
                    __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                    atomicAdd((__half2*)&grad_grid[index + c], v);
                }
            } else {
                #pragma unroll
                for (uint32_t c = 0; c < N_C; c++) {
                    atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
                }
            }
        }
    }
}


// =============================================================================
// INPUT BACKWARD KERNEL
// =============================================================================

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(
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

    # pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }

    grad_inputs[t] = result;
}


// =============================================================================
// SECOND BACKWARD KERNELS
// =============================================================================

template <typename input_t, typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_second_backward_grad(
    const scalar_t * __restrict__ grad,
    const input_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets,
    const scalar_t * __restrict__ grad_grad_inputs,
    const scalar_t * __restrict__ dy_dx,
    scalar_t * __restrict__ grad_grad,
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    grad_grad += level * B * C + b * C + ch;
    grad += level * B * C + b * C + ch;
    grad_grad_inputs += b * D;
    dy_dx += b * L * D * C + level * D * C + ch;

    scalar_t result[N_C] = {0};

    # pragma unroll
    for (int d = 0; d < D; d++) {
        # pragma unroll
        for (int c = 0; c < N_C; c++) {
            result[c] += grad_grad_inputs[d] * dy_dx[d * C + c];
        }
    }

    for (int c = 0; c < N_C; c++) {
        grad_grad[c] = result[c];
    }
}


template <typename input_t, typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_second_backward_embedding(
    const scalar_t * __restrict__ grad,
    const input_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets,
    const scalar_t * __restrict__ grad_grad_inputs,
    const scalar_t * __restrict__ dy_dx,
    scalar_t * __restrict__ grad2_embeddings,
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    grad2_embeddings += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch;
    grad_grad_inputs += b * D;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    double scale[D];
    uint32_t resolution[D];
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2(level * (double)per_level_scale[d]) * (double)base_resolution[d] - 1.0;
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return;
        }
    }

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

    scalar_t grad_cur[N_C] = {0};
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    scalar_t grad2_input_cur[D] = {0};
    for (uint32_t d = 0; d < D; d++) {
        grad2_input_cur[d] = grad_grad_inputs[d];
    }

    scalar_t grad_embeddings_cur[N_C * (1 << D)] = {0};

    #pragma unroll
    for (uint32_t gd = 0; gd < D; gd++) {
        scalar_t results_grad[C] = {0};

        #pragma unroll
        for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
            float w = scale[gd];
            uint32_t pos_grid_local[D];

            #pragma unroll
            for (uint32_t nd = 0; nd < D - 1; nd++) {
                const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                if ((idx & (1 << nd)) == 0) {
                    w *= 1 - pos[d];
                    pos_grid_local[d] = 0;
                } else {
                    w *= pos[d];
                    pos_grid_local[d] = 1;
                }
            }

            pos_grid_local[gd] = 0;
            uint32_t index_left = get_grid_index<D>(pos_grid_local);
            pos_grid_local[gd] = 1;
            uint32_t index_right = get_grid_index<D>(pos_grid_local);

            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                grad_embeddings_cur[index_right * N_C + c] += w * grad_cur[c] * grad2_input_cur[gd] * pos_derivative[gd];
                grad_embeddings_cur[index_left * N_C + c] -= w * grad_cur[c] * grad2_input_cur[gd] * pos_derivative[gd];
            }
        }
    }

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        uint32_t pos_grid_local[D];
        uint32_t cache_index = 0;
        uint32_t stride = 1;

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                pos_grid_local[d] = pos_grid[d];
            } else {
                pos_grid_local[d] = pos_grid[d] + 1;
                cache_index += stride;
            }
            stride *= 2;
        }

        uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);

        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                __half2 v = {(__half)(1.0 * grad_embeddings_cur[cache_index * N_C + c]),
                             (__half)(1.0 * grad_embeddings_cur[cache_index * N_C + c + 1])};
                atomicAdd((__half2*)&grad2_embeddings[index + c], v);
            }
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad2_embeddings[index + c], grad_embeddings_cur[cache_index * N_C + c]);
            }
        }
    }
}


// =============================================================================
// WRAPPER FUNCTIONS
// =============================================================================

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, const bool track_collisions, int *collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const int *probe_indices = nullptr, const uint32_t N_f = 0, const uint32_t N_p = 1, const uint32_t N_c = 0) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid<input_t, scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c); break;
        case 2: kernel_grid<input_t, scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c); break;
        case 4: kernel_grid<input_t, scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c); break;
        case 8: kernel_grid<input_t, scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename input_t, typename scalar_t>
void hash_encode_forward_cuda(const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, const bool track_collisions, int *collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const int *probe_indices = nullptr, const uint32_t N_f = 0, const uint32_t N_p = 1, const uint32_t N_c = 0) {
    switch (D) {
        case 2: kernel_grid_wrapper<input_t, scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c); break;
        case 3: kernel_grid_wrapper<input_t, scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c); break;
        default: throw std::runtime_error{"GridEncoding: D must be 2 or 3."};
    }
}

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(const scalar_t *grad, const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const float* per_level_scale, const float* base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs, const int *probe_indices, const float *index_logits, float *grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C);
    const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 1:
            kernel_grid_backward<input_t, scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 2:
            kernel_grid_backward<input_t, scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4:
            kernel_grid_backward<input_t, scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 8:
            kernel_grid_backward<input_t, scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, per_level_scale, base_resolution, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename input_t, typename scalar_t>
void hash_encode_backward_cuda(const scalar_t *grad, const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float* per_level_scale, const float* base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs, const int *probe_indices, const float *index_logits, float *grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c) {
    switch (D) {
        case 2: kernel_grid_backward_wrapper<input_t, scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        case 3: kernel_grid_backward_wrapper<input_t, scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_inputs, probe_indices, index_logits, grad_index_logits, N_f, N_p, N_c); break;
        default: throw std::runtime_error{"GridEncoding: D must be 2 or 3."};
    }
}

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_second_backward_wrapper(const scalar_t *grad, const input_t *inputs, const scalar_t *embeddings, const int *offsets,
    const uint32_t B, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs,
    const scalar_t *dy_dx, const scalar_t *grad_grad_inputs, scalar_t *grad_grad, scalar_t *grad2_embeddings) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C = std::min(2u, C);
    const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 2:
            kernel_grid_second_backward_grad<input_t, scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, per_level_scale, base_resolution);
            kernel_grid_second_backward_embedding<input_t, scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, per_level_scale, base_resolution);
            break;
        case 4:
            kernel_grid_second_backward_grad<input_t, scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, per_level_scale, base_resolution);
            kernel_grid_second_backward_embedding<input_t, scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, per_level_scale, base_resolution);
            break;
        case 8:
            kernel_grid_second_backward_grad<input_t, scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, per_level_scale, base_resolution);
            kernel_grid_second_backward_embedding<input_t, scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, per_level_scale, base_resolution);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 2, 4, or 8."};
    }
}

template <typename input_t, typename scalar_t>
void hash_encode_second_backward_cuda(const scalar_t *grad, const input_t *inputs, const scalar_t *embeddings,
    const int *offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution,
    const bool calc_grad_inputs, const scalar_t *dy_dx, const scalar_t *grad_grad_inputs, scalar_t *grad_grad, scalar_t *grad2_embeddings) {
    switch (D) {
        case 2: kernel_grid_second_backward_wrapper<input_t, scalar_t, 2>(grad, inputs, embeddings, offsets, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_grad_inputs, grad_grad, grad2_embeddings); break;
        case 3: kernel_grid_second_backward_wrapper<input_t, scalar_t, 3>(grad, inputs, embeddings, offsets, B, C, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, grad_grad_inputs, grad_grad, grad2_embeddings); break;
        default: throw std::runtime_error{"GridEncoding: D must be 2 or 3."};
    }
}


// =============================================================================
// ENTRY POINTS
// =============================================================================

void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, at::Tensor dy_dx, const bool track_collisions, at::Tensor collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const at::Tensor probe_indices, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c) {
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

    int *collision_indices_ptr = track_collisions ? collision_indices.data_ptr<int>() : nullptr;

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        CHECK_CONTIGUOUS(probe_indices);
        CHECK_IS_INT(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    if (inputs.scalar_type() == at::ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        embeddings.scalar_type(), "hash_encode_forward", ([&] {
            hash_encode_forward_cuda<double, scalar_t>(inputs.data_ptr<double>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), track_collisions, collision_indices_ptr, example_offset, max_tracked_examples, probe_indices_ptr, N_f, N_p, N_c);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        inputs.scalar_type(), "hash_encode_forward", ([&] {
            hash_encode_forward_cuda<scalar_t, scalar_t>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), track_collisions, collision_indices_ptr, example_offset, max_tracked_examples, probe_indices_ptr, N_f, N_p, N_c);
        }));
    }
}


void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs, const at::Tensor probe_indices, const at::Tensor index_logits, at::Tensor grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c) {
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

    int *probe_indices_ptr = nullptr;
    float *index_logits_ptr = nullptr;
    float *grad_index_logits_ptr = nullptr;

    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        CHECK_CONTIGUOUS(probe_indices);
        CHECK_IS_INT(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    if (index_logits.defined() && index_logits.numel() > 0) {
        CHECK_CUDA(index_logits);
        CHECK_CONTIGUOUS(index_logits);
        CHECK_IS_FLOATING(index_logits);
        index_logits_ptr = index_logits.data_ptr<float>();
    }

    if (grad_index_logits.defined() && grad_index_logits.numel() > 0) {
        CHECK_CUDA(grad_index_logits);
        CHECK_CONTIGUOUS(grad_index_logits);
        CHECK_IS_FLOATING(grad_index_logits);
        grad_index_logits_ptr = grad_index_logits.data_ptr<float>();
    }

    if (inputs.scalar_type() == at::ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "hash_encode_backward", ([&] {
            hash_encode_backward_cuda<double, scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<double>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>(), probe_indices_ptr, index_logits_ptr, grad_index_logits_ptr, N_f, N_p, N_c);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "hash_encode_backward", ([&] {
            hash_encode_backward_cuda<scalar_t, scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>(), probe_indices_ptr, index_logits_ptr, grad_index_logits_ptr, N_f, N_p, N_c);
        }));
    }
}


void hash_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs,
    const at::Tensor dy_dx, const at::Tensor grad_grad_inputs, at::Tensor grad_grad, at::Tensor grad2_embeddings) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_grad_inputs);
    CHECK_CUDA(grad_grad);
    CHECK_CUDA(grad2_embeddings);
    CHECK_CUDA(per_level_scale);
    CHECK_CUDA(base_resolution);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_grad_inputs);
    CHECK_CONTIGUOUS(grad_grad);
    CHECK_CONTIGUOUS(grad2_embeddings);
    CHECK_CONTIGUOUS(per_level_scale);
    CHECK_CONTIGUOUS(base_resolution);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_grad_inputs);
    CHECK_IS_FLOATING(grad_grad);
    CHECK_IS_FLOATING(grad2_embeddings);
    CHECK_IS_FLOATING(per_level_scale);
    CHECK_IS_FLOATING(base_resolution);

    if (inputs.scalar_type() == at::ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "hash_encode_second_backward", ([&] {
            hash_encode_second_backward_cuda<double, scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<double>(), embeddings.data_ptr<scalar_t>(),
            offsets.data_ptr<int>(), B, D, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_grad_inputs.data_ptr<scalar_t>(),
            grad_grad.data_ptr<scalar_t>(), grad2_embeddings.data_ptr<scalar_t>());
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "hash_encode_second_backward", ([&] {
            hash_encode_second_backward_cuda<scalar_t, scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(),
            offsets.data_ptr<int>(), B, D, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_grad_inputs.data_ptr<scalar_t>(),
            grad_grad.data_ptr<scalar_t>(), grad2_embeddings.data_ptr<scalar_t>());
        }));
    }
}
