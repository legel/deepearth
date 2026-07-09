#include <torch/extension.h>

#include "hashencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Core hash encoding
    m.def("hash_encode_forward", &hash_encode_forward, "hash encode forward (CUDA)");
    m.def("hash_encode_backward", &hash_encode_backward, "hash encode backward (CUDA)");
    m.def("hash_encode_second_backward", &hash_encode_second_backward, "hash encode second backward (CUDA)");

    // Precomputed hash encoding (fixed coordinate optimization)
    m.def("hash_encode_precompute", &hash_encode_precompute, "Precompute hash indices and weights (CUDA)");
    m.def("hash_encode_forward_precomputed", &hash_encode_forward_precomputed, "Forward with precomputed indices (CUDA)");
    m.def("hash_encode_backward_precomputed", &hash_encode_backward_precomputed, "Backward with precomputed indices (CUDA)");

    // Optimized kernels for small batch training
    m.def("hash_encode_backward_sgd_fused", &hash_encode_backward_sgd_fused, "Fused backward + SGD update (CUDA)");
    m.def("hash_encode_forward_precomputed_multibatch", &hash_encode_forward_precomputed_multibatch, "Forward for multiple batches in single kernel (CUDA)");
    m.def("hash_encode_adam_sparse_update", &hash_encode_adam_sparse_update, "Sparse Adam update for touched embeddings only (CUDA)");
}
