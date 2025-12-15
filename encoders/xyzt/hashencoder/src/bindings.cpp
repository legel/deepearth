#include <torch/extension.h>

#include "hashencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Core hash encoding
    m.def("hash_encode_forward", &hash_encode_forward, "hash encode forward (CUDA)");
    m.def("hash_encode_backward", &hash_encode_backward, "hash encode backward (CUDA)");
    m.def("hash_encode_second_backward", &hash_encode_second_backward, "hash encode second backward (CUDA)");

    // Warp-level YOHO (automatic intra-warp deduplication)
    m.def("hash_encode_forward_warp_yoho", &hash_encode_forward_warp_yoho, "Warp-level YOHO forward (CUDA)");
    m.def("hash_encode_backward_warp_yoho", &hash_encode_backward_warp_yoho, "Warp-level YOHO backward (CUDA)");
}