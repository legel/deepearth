/**
 * utils.cuh - Shared helper functions and utilities
 *
 * This header contains all device functions needed across multiple CUDA files.
 * Splitting into a header allows parallel compilation of kernel files.
 */

#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <stdint.h>
#include <cstdio>

// =============================================================================
// MACROS
// =============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

// =============================================================================
// ATOMIC ADD FOR HALF PRECISION
// =============================================================================

// half atomicAdd (CUDA>=10, ARCH>=70; slow vs float/__half2)
static inline __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
    return atomicAdd(reinterpret_cast<__half*>(address), val);
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

// =============================================================================
// HASH FUNCTIONS
// =============================================================================

template <uint32_t D>
__device__ uint64_t fast_hash(const uint32_t pos_grid[D]) {
    static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

    // 64-bit primes to avoid overflow issues with large grid coordinates
    constexpr uint64_t primes[7] = {
        1ULL, 2654435761ULL, 805459861ULL, 3674653429ULL,
        2097192037ULL, 1434869437ULL, 2165219737ULL
    };

    uint64_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        result ^= (uint64_t)pos_grid[i] * primes[i];
    }

    return result;
}

template <uint32_t D>
__device__ uint64_t fast_hash2(const uint32_t pos_grid[D]) {
    static_assert(D <= 7, "fast_hash2 can only hash up to 7 dimensions.");

    // Secondary hash function with different primes for decorrelation
    constexpr uint64_t primes[7] = {
        1ULL,          // Memory coherence
        3141592653ULL, // π × 10^9
        2718281829ULL, // e × 10^9
        1618033989ULL, // φ × 10^9
        2236067977ULL, // √5 × 10^9
        1732050807ULL, // √3 × 10^9
        4142135623ULL  // √17 × 10^9
    };

    uint64_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        result ^= (uint64_t)pos_grid[i] * primes[i];
    }

    return result;
}

// =============================================================================
// GRID INDEX FUNCTIONS
// =============================================================================

template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(
    const uint32_t ch,
    const uint32_t hashmap_size,
    const uint32_t* resolution,
    const uint32_t pos_grid[D]
) {
    uint64_t stride = 1;
    uint64_t index = 0;

    #pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += (uint64_t)pos_grid[d] * stride;
        stride *= resolution[d];
    }

    if (stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

    return (uint32_t)((index % hashmap_size) * C + ch);
}

template <uint32_t D>
__device__ uint64_t get_grid_index(const uint32_t pos_grid[D]) {
    uint64_t stride = 1;
    uint64_t index = 0;

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        index += (uint64_t)pos_grid[d] * stride;
        stride *= 2;
    }
    return index;
}

// Shared row resolver: maps a stored/computed (h1,h2) pair to the embedding row (idx*C+ch). h2==0xFFFFFFFF is the
// direct-index sentinel (h1 already reduced mod hashmap_size); otherwise probe with the learned codebook. Used by
// BOTH get_grid_index_learned (standard path) and the precomputed forward so the two paths cannot drift.
__device__ inline uint32_t resolve_index(
    const uint32_t h1, const uint32_t h2, const int* probe_indices,
    const uint32_t level, const uint32_t N_p, const uint32_t N_c,
    const uint32_t hashmap_size, const uint32_t ch, const uint32_t C
) {
    if (h2 == 0xFFFFFFFFu) return h1 * C + ch;                       // direct index (fits in hashmap)
    if (probe_indices != nullptr && N_c > 0) {                      // learned probing
        uint32_t h2_mod = h2 % N_c;
        uint32_t probe = (uint32_t)probe_indices[(uint64_t)level * N_c + h2_mod];
        uint64_t index = (uint64_t)N_p * h1 + probe;
        return (uint32_t)((index % hashmap_size) * C + ch);
    }
    return (uint32_t)(((uint64_t)h1 % hashmap_size) * C + ch);      // plain hashing (no probing)
}

// Compute the stored discrete hashes (h1,h2) for a cell corner: direct index + sentinel when the grid fits, otherwise
// the twin fast-hashes. This is the SAME convention the precompute buffers use, so the standard and precomputed paths
// share one hashing rule. h2 is raw (resolve_index applies % N_c).
template <uint32_t D>
__device__ inline void compute_hashes(
    const uint32_t pos_grid[D], const uint32_t* resolution, const uint32_t hashmap_size,
    const uint32_t N_f, uint32_t& h1, uint32_t& h2
) {
    uint64_t stride = 1;
    uint64_t index = 0;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        index += (uint64_t)pos_grid[d] * stride;
        stride *= resolution[d];
    }
    if (stride > hashmap_size) {
        h1 = (uint32_t)(fast_hash<D>(pos_grid) % N_f);
        h2 = (uint32_t)fast_hash2<D>(pos_grid);      // raw; resolve_index applies % N_c
    } else {
        h1 = (uint32_t)(index % hashmap_size);
        h2 = 0xFFFFFFFFu;                            // direct sentinel
    }
}

// Learned hash probing version of get_grid_index. Computes (h1,h2) then defers to resolve_index so both the standard
// and precomputed forward share one row-resolution rule.
template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index_learned(
    const uint32_t ch,
    const uint32_t hashmap_size,
    const uint32_t N_f,
    const uint32_t N_p,
    const uint32_t N_c,
    const uint32_t level,
    const uint32_t* resolution,
    const uint32_t pos_grid[D],
    const int* probe_indices  // Shape: (L, N_c)
) {
    uint32_t h1, h2;
    compute_hashes<D>(pos_grid, resolution, hashmap_size, N_f, h1, h2);
    return resolve_index(h1, h2, probe_indices, level, N_p, N_c, hashmap_size, ch, C);
}

// =============================================================================
// INTERPOLATION FUNCTIONS
// =============================================================================

__device__ inline float smoothstep(float val) {
    return val * val * (3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
    return 6 * val * (1.0f - val);
}

__device__ inline float identity_fun(float val) {
    return val;
}

__device__ inline float identity_derivative(float val) {
    return 1;
}

// =============================================================================
// LEARNED PROBING HELPERS
// =============================================================================

// Softmax computation for learned probing
template <uint32_t N_p>
__device__ inline void compute_softmax(const float* logits, float* weights) {
    // Find max for numerical stability
    float max_logit = logits[0];
    #pragma unroll
    for (uint32_t i = 1; i < N_p; ++i) {
        max_logit = max(max_logit, logits[i]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < N_p; ++i) {
        weights[i] = expf(logits[i] - max_logit);
        sum_exp += weights[i];
    }

    // Normalize
    #pragma unroll
    for (uint32_t i = 0; i < N_p; ++i) {
        weights[i] /= sum_exp;
    }
}

#endif // UTILS_CUH
