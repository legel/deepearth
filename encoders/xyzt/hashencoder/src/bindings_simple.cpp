#include <torch/types.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

// Forward declarations
void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, 
                         at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, 
                         const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, 
                         const bool calc_grad_inputs, at::Tensor dy_dx);

void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, 
                          const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, 
                          const uint32_t D, const uint32_t C, const uint32_t L, 
                          const at::Tensor per_level_scale, const at::Tensor base_resolution, 
                          const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs);

void hash_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, 
                                 const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, 
                                 const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, 
                                 const bool calc_grad_inputs, const at::Tensor dy_dx, 
                                 const at::Tensor grad_grad_inputs, at::Tensor grad_grad, at::Tensor grad2_embeddings);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_encode_forward", &hash_encode_forward, "hash encode forward (CUDA)");
    m.def("hash_encode_backward", &hash_encode_backward, "hash encode backward (CUDA)");
    m.def("hash_encode_second_backward", &hash_encode_second_backward, "hash encode second backward (CUDA)");
}