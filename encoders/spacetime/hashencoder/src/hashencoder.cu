// hashencoder.cu - hash-encoding kernels: standard fwd/bwd + 2nd-order, precomputed recompute-fresh fwd/bwd, sparse geoNudge Adam. Shares resolve_index/compute_hashes (utils.cuh).

#include "utils.cuh"

// MAIN FORWARD KERNEL

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
        scale[d] = exp2((double)per_level_scale[level * D + d]) * (double)base_resolution[d] - 1.0;
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


// BACKWARD KERNEL (EMBEDDING GRADIENTS)

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
        scale[d] = exp2((double)per_level_scale[level * D + d]) * (double)base_resolution[d] - 1.0;
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


// INPUT BACKWARD KERNEL

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


// SECOND BACKWARD KERNELS

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
        scale[d] = exp2((double)per_level_scale[level * D + d]) * (double)base_resolution[d] - 1.0;
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


// WRAPPER FUNCTIONS

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const input_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs, scalar_t *dy_dx, const bool track_collisions, int *collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const int *probe_indices = nullptr, const uint32_t N_f = 0, const uint32_t N_p = 1, const uint32_t N_c = 0) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    #define _L(Cv) kernel_grid<input_t, scalar_t, D, Cv><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, per_level_scale, base_resolution, calc_grad_inputs, dy_dx, track_collisions, collision_indices, example_offset, max_tracked_examples, probe_indices, N_f, N_p, N_c)
    switch (C) { case 1: _L(1); break; case 2: _L(2); break; case 4: _L(4); break; case 8: _L(8); break;
                 default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."}; }
    #undef _L
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


// ENTRY POINTS

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


// PRECOMPUTED HASH ENCODING (merged from precompute.cu); Fixed-coordinate optimization: cache the discrete cell (h1/h2 + pos_grid); recompute all continuous terms; fresh from the current per_level_scale each forward. Shares resolve_index / compute_hashes (utils.cuh) with; the standard path so the two forwards cannot drift.

// PRECOMPUTATION KERNEL

template <typename input_t, uint32_t D, uint32_t C>
__global__ void kernel_precompute(
    const input_t * __restrict__ inputs,           // [B, D] normalized coords in [0,1]
    const int * __restrict__ offsets,              // [L+1] level offsets
    uint32_t * __restrict__ precomp_h1,            // [B, L, 2^D] h1 values or direct indices
    uint32_t * __restrict__ precomp_h2,            // [B, L, 2^D] h2 values (for learned probing)
    float * __restrict__ precomp_weights,          // [B, L, 2^D] interpolation weights
    float * __restrict__ precomp_pos_deriv,        // [B, L, D] position derivatives for backward
    int * __restrict__ precomp_pos_grid,           // [B, L, D] integer base cell (the cached discrete structure)
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
    precomp_pos_grid += b * L * D + level * D;

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
            precomp_pos_grid[d] = 0;
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];

    // scale = exp2(per_level_scale[level][d])*base[d] (learnable per-level log2 resolution, matching the forward)
    double scale[D];
    uint32_t resolution[D];
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2((double)per_level_scale[level * D + d]) * (double)base_resolution[d] - 1.0;
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

    // cache pos_deriv + integer base cell for the precomputed forward
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        precomp_pos_deriv[d] = pos_derivative[d];
        precomp_pos_grid[d] = (int)pos_grid[d];
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

// PRECOMPUTED FORWARD KERNEL

// Non-compromising precomputed forward: recompute ALL continuous terms (frac/pos/pos_deriv/weights/dy_dx) fresh from the
// CURRENT per_level_scale, and recompute the discrete cell (floor + hashes) fresh so a coordinate that crosses a cell
// boundary under resolution drift still resolves to the correct row -- bit-identical to kernel_grid at ANY scale. The
// cached cell (precomp_pos_grid + precomp_h1/h2) is used as a fast path: when the freshly-floored base cell still equals
// the cached one, the cached hashes are reused verbatim (identical bytes, no fast_hash); otherwise the hashes are
// recomputed inline. The h1/h2 actually used are emitted to h1_out/h2_out so the (unchanged) backward kernel scatters to
// exactly the rows this forward gathered.
template <typename input_t, typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_precomputed(
    const input_t * __restrict__ inputs,           // [B, D] normalized coords in [0,1]
    const scalar_t * __restrict__ grid,            // [total_embeddings, C]
    const int * __restrict__ offsets,              // [L+1]
    const uint32_t * __restrict__ precomp_h1,      // [B, L, 2^D] cached hashes
    const uint32_t * __restrict__ precomp_h2,      // [B, L, 2^D]
    const int * __restrict__ precomp_pos_grid,     // [B, L, D] cached integer base cell
    const int * __restrict__ probe_indices,        // [L, N_c] learned probe indices
    scalar_t * __restrict__ outputs,               // [L, B, C]
    uint32_t * __restrict__ h1_out,                // [B, L, 2^D] hashes actually used (fed to backward)
    uint32_t * __restrict__ h2_out,                // [B, L, 2^D]
    float * __restrict__ weights_out,              // [B, L, 2^D] freshly recomputed weights (fed to backward)
    scalar_t * __restrict__ dy_dx,                 // [B, L*D*C] per-level input derivative (for per_level_scale grad)
    const uint32_t B, const uint32_t L,
    const float * __restrict__ per_level_scale,
    const float * __restrict__ base_resolution,
    const bool calc_grad_inputs,
    const uint32_t N_f,                            // learned probing: hashmap_size / N_p
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
    const uint32_t* sh1 = precomp_h1 + b * L * n_corners + level * n_corners;   // cached hashes
    const uint32_t* sh2 = precomp_h2 + b * L * n_corners + level * n_corners;
    const int* spg = precomp_pos_grid + b * L * D + level * D;                  // cached base cell
    const input_t* in_ptr = inputs + b * D;
    scalar_t* out_ptr = outputs + level * B * C + b * C;
    uint32_t* oh1 = h1_out + b * L * n_corners + level * n_corners;
    uint32_t* oh2 = h2_out + b * L * n_corners + level * n_corners;
    float* w_out = weights_out + b * L * n_corners + level * n_corners;

    // OOB handling (mirrors kernel_grid): zero outputs / weights / dy_dx and return
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (in_ptr[d] < 0 || in_ptr[d] > 1) flag_oob = true;
    }
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) out_ptr[ch] = 0;
        #pragma unroll
        for (uint32_t idx = 0; idx < n_corners; idx++) { w_out[idx] = 0.0f; oh1[idx] = 0; oh2[idx] = 0; }
        if (calc_grad_inputs) {
            scalar_t* dd = dy_dx + b * D * L * C + level * D * C;
            #pragma unroll
            for (uint32_t d = 0; d < D; d++)
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) dd[d * C + ch] = 0;
        }
        return;
    }

    // Recompute scale, resolution, fresh base cell and smoothstep weights/derivative from the CURRENT scale
    double scale[D];
    uint32_t resolution[D];
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        scale[d] = exp2((double)per_level_scale[level * D + d]) * (double)base_resolution[d] - 1.0;
        resolution[d] = (uint32_t)ceil(scale[d]) + 1;
    }

    float pos[D];
    float pos_derivative[D];
    uint32_t pos_grid[D];
    bool cell_match = true;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        double pos_hp = (double)in_ptr[d] * scale[d];
        pos_grid[d] = (uint32_t)floor(pos_hp);
        if (pos_grid[d] != (uint32_t)spg[d]) cell_match = false;   // cached cell still valid?
        float frac = (float)(pos_hp - (double)pos_grid[d]);
        pos_derivative[d] = smoothstep_derivative(frac);
        pos[d] = smoothstep(frac);
    }

    // Interpolate: corner weight w = Π_d (pos or 1-pos); reuse cached hashes when the cell matches, else recompute
    scalar_t results[C] = {0};
    #pragma unroll
    for (uint32_t idx = 0; idx < n_corners; idx++) {
        float w = 1.0f;
        uint32_t pos_grid_local[D];
        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) { w *= 1.0f - pos[d]; pos_grid_local[d] = pos_grid[d]; }
            else                       { w *= pos[d];        pos_grid_local[d] = pos_grid[d] + 1; }
        }
        w_out[idx] = w;

        uint32_t h1, h2;
        if (cell_match) { h1 = sh1[idx]; h2 = sh2[idx]; }
        else            compute_hashes<D>(pos_grid_local, resolution, hashmap_size, N_f, h1, h2);
        oh1[idx] = h1; oh2[idx] = h2;

        uint32_t index = resolve_index(h1, h2, probe_indices, level, N_p, N_c, hashmap_size, 0, C);
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) results[ch] += w * level_grid[index + ch];
    }
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) out_ptr[ch] = results[ch];

    // dy_dx[b,l,gd,ch] = Σ_{other-dim sub-corners} scale[gd]·Πw·(emb[right]-emb[left])·pos_deriv[gd]
    if (calc_grad_inputs) {
        scalar_t* dd = dy_dx + b * D * L * C + level * D * C;
        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {
            scalar_t results_grad[C] = {0};
            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale[gd];
                uint32_t corner = 0;                                 // bit d set -> pos_grid[d]+1 (gd bit stays 0 = left)
                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;
                    if ((idx & (1 << nd)) == 0) w *= 1 - pos[d];
                    else                      { w *= pos[d]; corner |= (1u << d); }
                }
                uint32_t cr = corner | (1u << gd);
                uint32_t index_left  = resolve_index(oh1[corner], oh2[corner], probe_indices, level, N_p, N_c, hashmap_size, 0, C);
                uint32_t index_right = resolve_index(oh1[cr], oh2[cr], probe_indices, level, N_p, N_c, hashmap_size, 0, C);
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++)
                    results_grad[ch] += w * (level_grid[index_right + ch] - level_grid[index_left + ch]) * pos_derivative[gd];
            }
            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) dd[gd * C + ch] = results_grad[ch];
        }
    }
}

// PRECOMPUTED BACKWARD KERNEL: with index_logits, uses soft (softmax) selection matching the standard backward, so gradients flow to index_logits and spread over all probe positions

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

// SPARSE ADAM UPDATE KERNEL
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
    int * __restrict__ last_step,                  // [total_embeddings] per-row global step of last touch (0 = never)
    const uint32_t B, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr,
    const float beta1, const float beta2, const float eps,
    const float weight_decay,                      // AdamW weight decay
    const uint32_t step,                           // current global step t (1-indexed)
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

    // Per-row claim: many (b,level,corner) tuples collide onto one row; exactly one thread updates it this step. The
    // winner reads the fully-accumulated grad (accumulate_grad ran to completion first) and applies both the missed-step
    // replay and this step's Adam update to ALL channels; the last_step buffer is shared per row so it must be claimed
    // once, not per channel.
    const int t = (int)step;
    int s = atomicExch(&last_step[emb_idx], t);
    if (s >= t) return;                            // another thread already claimed this row this step
    const int k = t - s;                           // steps since this row was last touched (>= 1)

    // Closed-form geometric replay of the momentum-driven moves dense AdamW made on the k-1 untouched steps: with the
    // (frozen) moments the row carried, dense kept nudging it by a geometrically-decaying step r^i, r = b1/sqrt(b2).
    const float r = beta1 / sqrtf(beta2);
    const int n = k - 1;
    const float geo = (n > 0) ? r * (1.0f - powf(r, (float)n)) / (1.0f - r) : 0.0f;
    const float b1k = powf(beta1, (float)k);       // lazy moment decay over the k-step gap
    const float b2k = powf(beta2, (float)k);
    const float wdk = powf(1.0f - lr * weight_decay, (float)k);   // lazy weight decay over the gap

    #pragma unroll
    for (uint32_t c = 0; c < C; c++) {
        const uint32_t idx = emb_idx * C + c;

        float g = grad_embeddings[idx];
        grad_embeddings[idx] = 0.0f;               // clear for reuse next step

        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];
        float param = (float)embeddings[idx];

        // (1) geometric nudge from the pre-grad bias-corrected moments (first touch: m=v=0 -> 0, and geo=0 anyway)
        float mh0 = m / bias_correction1;
        float vh0 = v / bias_correction2;
        param -= lr * (mh0 / (sqrtf(vh0) + eps)) * geo;

        // (2) lazy-decayed Adam step with this touch's accumulated grad
        m = b1k * m + (1.0f - beta1) * g;
        v = b2k * v + (1.0f - beta2) * g * g;
        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        param -= lr * m_hat / (sqrtf(v_hat) + eps);

        // (3) lazy weight decay accumulated over the gap
        param *= wdk;

        embeddings[idx] = (scalar_t)param;
    }
}


// WRAPPER FUNCTIONS


template <typename scalar_t, uint32_t D>
void kernel_adam_sparse_update_wrapper(
    const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2,
    const int *probe_indices,
    scalar_t *embeddings, float *grad_embeddings,  // grad_embeddings is always float
    float *exp_avg, float *exp_avg_sq, int *last_step,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const uint32_t N_p, const uint32_t N_c,
    const float lr, const float beta1, const float beta2, const float eps,
    const float weight_decay,
    const uint32_t step, const float bias_correction1, const float bias_correction2
) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t n_corners = 1 << D;
    const uint32_t total_tuples = B * L * n_corners;
    const dim3 blocks = { div_round_up(total_tuples, N_THREAD), 1, 1 };

    #define _L(Cv) kernel_adam_sparse_update<scalar_t, D, Cv><<<blocks, N_THREAD>>>(offsets, precomp_h1, precomp_h2, probe_indices, embeddings, grad_embeddings, exp_avg, exp_avg_sq, last_step, B, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, step, bias_correction1, bias_correction2)
    switch (C) { case 1: _L(1); break; case 2: _L(2); break; case 4: _L(4); break; case 8: _L(8); break;
                 default: throw std::runtime_error{"Adam sparse update: C must be 1, 2, 4, or 8."}; }
    #undef _L
}


template <typename input_t, uint32_t D>
void kernel_precompute_wrapper(
    const input_t *inputs, const int *offsets,
    uint32_t *precomp_h1, uint32_t *precomp_h2,
    float *precomp_weights, float *precomp_pos_deriv, int *precomp_pos_grid,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const float *per_level_scale, const float *base_resolution,
    const uint32_t N_f, const uint32_t N_p
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks = { div_round_up(B, N_THREAD), L, 1 };

    #define _L(Cv) kernel_precompute<input_t, D, Cv><<<blocks, N_THREAD>>>(inputs, offsets, precomp_h1, precomp_h2, precomp_weights, precomp_pos_deriv, precomp_pos_grid, B, L, per_level_scale, base_resolution, N_f, N_p)
    switch (C) { case 1: _L(1); break; case 2: _L(2); break; case 4: _L(4); break; case 8: _L(8); break;
                 default: throw std::runtime_error{"Precompute: C must be 1, 2, 4, or 8."}; }
    #undef _L
}

template <typename input_t, typename scalar_t, uint32_t D>
void kernel_grid_precomputed_wrapper(
    const input_t *inputs, const scalar_t *grid, const int *offsets,
    const uint32_t *precomp_h1, const uint32_t *precomp_h2, const int *precomp_pos_grid,
    const int *probe_indices, scalar_t *outputs, uint32_t *h1_out, uint32_t *h2_out, float *weights_out, scalar_t *dy_dx,
    const uint32_t B, const uint32_t C, const uint32_t L,
    const float *per_level_scale, const float *base_resolution, const bool calc_grad_inputs,
    const uint32_t N_f, const uint32_t N_p, const uint32_t N_c
) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks = { div_round_up(B, N_THREAD), L, 1 };

    #define _L(Cv) kernel_grid_precomputed<input_t, scalar_t, D, Cv><<<blocks, N_THREAD>>>(inputs, grid, offsets, precomp_h1, precomp_h2, precomp_pos_grid, probe_indices, outputs, h1_out, h2_out, weights_out, dy_dx, B, L, per_level_scale, base_resolution, calc_grad_inputs, N_f, N_p, N_c)
    switch (C) { case 1: _L(1); break; case 2: _L(2); break; case 4: _L(4); break; case 8: _L(8); break;
                 default: throw std::runtime_error{"Precomputed forward: C must be 1, 2, 4, or 8."}; }
    #undef _L
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

// ENTRY POINTS

void hash_encode_precompute(
    const at::Tensor inputs,
    const at::Tensor offsets,
    at::Tensor precomp_h1,
    at::Tensor precomp_h2,
    at::Tensor precomp_weights,
    at::Tensor precomp_pos_deriv,
    at::Tensor precomp_pos_grid,
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
    CHECK_CUDA(precomp_pos_grid);
    CHECK_CUDA(per_level_scale);
    CHECK_CUDA(base_resolution);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(precomp_h1);
    CHECK_CONTIGUOUS(precomp_h2);
    CHECK_CONTIGUOUS(precomp_weights);
    CHECK_CONTIGUOUS(precomp_pos_deriv);
    CHECK_CONTIGUOUS(precomp_pos_grid);

    auto per_level_scale_log2 = per_level_scale;   // already [L,D] log2 resolution; pass through (do NOT log2 again)

    // int32 -> uint32_t (same bits)
    uint32_t* h1_ptr = reinterpret_cast<uint32_t*>(precomp_h1.data_ptr<int>());
    uint32_t* h2_ptr = reinterpret_cast<uint32_t*>(precomp_h2.data_ptr<int>());
    int* pos_grid_ptr = precomp_pos_grid.data_ptr<int>();

    if (inputs.scalar_type() == at::ScalarType::Double) {
        switch (D) {
            case 2: kernel_precompute_wrapper<double, 2>(inputs.data_ptr<double>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), pos_grid_ptr, B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            case 3: kernel_precompute_wrapper<double, 3>(inputs.data_ptr<double>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), pos_grid_ptr, B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            default: throw std::runtime_error{"Precompute: D must be 2 or 3."};
        }
    } else {
        switch (D) {
            case 2: kernel_precompute_wrapper<float, 2>(inputs.data_ptr<float>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), pos_grid_ptr, B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            case 3: kernel_precompute_wrapper<float, 3>(inputs.data_ptr<float>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, precomp_weights.data_ptr<float>(), precomp_pos_deriv.data_ptr<float>(), pos_grid_ptr, B, C, L, per_level_scale_log2.data_ptr<float>(), base_resolution.data_ptr<float>(), N_f, N_p); break;
            default: throw std::runtime_error{"Precompute: D must be 2 or 3."};
        }
    }
}

void hash_encode_forward_precomputed(
    const at::Tensor inputs,          // [B, D] normalized coords in [0,1]
    const at::Tensor embeddings,
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor precomp_pos_grid,// [B, L, D] cached integer base cell
    const at::Tensor per_level_scale,
    const at::Tensor base_resolution,
    const at::Tensor probe_indices,
    at::Tensor outputs,
    at::Tensor h1_out,                // [B, L, 2^D] hashes actually used (for backward)
    at::Tensor h2_out,                // [B, L, 2^D]
    at::Tensor weights_out,           // [B, L, 2^D] freshly recomputed weights (for backward)
    at::Tensor dy_dx,                 // [B, L*D*C] or dummy
    const bool calc_grad_inputs,
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const uint32_t N_f, const uint32_t N_p, const uint32_t N_c
) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(precomp_h1);
    CHECK_CUDA(precomp_h2);
    CHECK_CUDA(precomp_pos_grid);
    CHECK_CUDA(per_level_scale);
    CHECK_CUDA(base_resolution);
    CHECK_CUDA(outputs);
    CHECK_CUDA(h1_out);
    CHECK_CUDA(h2_out);
    CHECK_CUDA(weights_out);
    CHECK_CUDA(dy_dx);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    // Cast int32 pointers to uint32_t
    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());
    const int* pos_grid_ptr = precomp_pos_grid.data_ptr<int>();
    uint32_t* h1_out_ptr = reinterpret_cast<uint32_t*>(h1_out.data_ptr<int>());
    uint32_t* h2_out_ptr = reinterpret_cast<uint32_t*>(h2_out.data_ptr<int>());

    if (inputs.scalar_type() == at::ScalarType::Double) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "hash_encode_forward_precomputed", ([&] {
            switch (D) {
                case 2: kernel_grid_precomputed_wrapper<double, scalar_t, 2>(inputs.data_ptr<double>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, pos_grid_ptr, probe_indices_ptr, outputs.data_ptr<scalar_t>(), h1_out_ptr, h2_out_ptr, weights_out.data_ptr<float>(), dy_dx.data_ptr<scalar_t>(), B, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, N_f, N_p, N_c); break;
                case 3: kernel_grid_precomputed_wrapper<double, scalar_t, 3>(inputs.data_ptr<double>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, pos_grid_ptr, probe_indices_ptr, outputs.data_ptr<scalar_t>(), h1_out_ptr, h2_out_ptr, weights_out.data_ptr<float>(), dy_dx.data_ptr<scalar_t>(), B, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, N_f, N_p, N_c); break;
                default: throw std::runtime_error{"Precomputed forward: D must be 2 or 3."};
            }
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "hash_encode_forward_precomputed", ([&] {
            switch (D) {
                case 2: kernel_grid_precomputed_wrapper<scalar_t, scalar_t, 2>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, pos_grid_ptr, probe_indices_ptr, outputs.data_ptr<scalar_t>(), h1_out_ptr, h2_out_ptr, weights_out.data_ptr<float>(), dy_dx.data_ptr<scalar_t>(), B, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, N_f, N_p, N_c); break;
                case 3: kernel_grid_precomputed_wrapper<scalar_t, scalar_t, 3>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), h1_ptr, h2_ptr, pos_grid_ptr, probe_indices_ptr, outputs.data_ptr<scalar_t>(), h1_out_ptr, h2_out_ptr, weights_out.data_ptr<float>(), dy_dx.data_ptr<scalar_t>(), B, C, L, per_level_scale.data_ptr<float>(), base_resolution.data_ptr<float>(), calc_grad_inputs, N_f, N_p, N_c); break;
                default: throw std::runtime_error{"Precomputed forward: D must be 2 or 3."};
            }
        }));
    }
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


void hash_encode_adam_sparse_update(
    const at::Tensor offsets,
    const at::Tensor precomp_h1,
    const at::Tensor precomp_h2,
    const at::Tensor probe_indices,
    at::Tensor embeddings,           // Modified in place
    at::Tensor grad_embeddings,      // Modified in place (zeroed after use)
    at::Tensor exp_avg,              // Adam first moment, modified in place
    at::Tensor exp_avg_sq,           // Adam second moment, modified in place
    at::Tensor last_step,            // [total_embeddings] per-row last-touch step, modified in place
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
    CHECK_CUDA(last_step);

    int *probe_indices_ptr = nullptr;
    if (probe_indices.defined() && probe_indices.numel() > 0) {
        CHECK_CUDA(probe_indices);
        probe_indices_ptr = probe_indices.data_ptr<int>();
    }

    const uint32_t* h1_ptr = reinterpret_cast<const uint32_t*>(precomp_h1.data_ptr<int>());
    const uint32_t* h2_ptr = reinterpret_cast<const uint32_t*>(precomp_h2.data_ptr<int>());
    int* last_step_ptr = last_step.data_ptr<int>();

    // Compute bias corrections
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    // grad_embeddings must be float type for atomicExch support
    TORCH_CHECK(grad_embeddings.scalar_type() == at::kFloat,
                "grad_embeddings must be float type for Adam sparse update");
    TORCH_CHECK(last_step.scalar_type() == at::kInt, "last_step must be int32");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "hash_encode_adam_sparse_update", ([&] {
        switch (D) {
            case 2: kernel_adam_sparse_update_wrapper<scalar_t, 2>(offsets.data_ptr<int>(), h1_ptr, h2_ptr, probe_indices_ptr, embeddings.data_ptr<scalar_t>(), grad_embeddings.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), last_step_ptr, B, C, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, step, bias_correction1, bias_correction2); break;
            case 3: kernel_adam_sparse_update_wrapper<scalar_t, 3>(offsets.data_ptr<int>(), h1_ptr, h2_ptr, probe_indices_ptr, embeddings.data_ptr<scalar_t>(), grad_embeddings.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), last_step_ptr, B, C, L, N_p, N_c, lr, beta1, beta2, eps, weight_decay, step, bias_correction1, bias_correction2); break;
            default: throw std::runtime_error{"Adam sparse update: D must be 2 or 3."};
        }
    }));
}
