#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, at::Tensor dy_dx, const bool track_collisions, at::Tensor collision_indices, const uint32_t example_offset, const uint32_t max_tracked_examples, const at::Tensor probe_indices, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c);
void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs, const at::Tensor probe_indices, const at::Tensor index_logits, at::Tensor grad_index_logits, const uint32_t N_f, const uint32_t N_p, const uint32_t N_c);
void hash_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const at::Tensor per_level_scale, const at::Tensor base_resolution, const bool calc_grad_inputs, const at::Tensor dy_dx, const at::Tensor grad_grad_inputs, at::Tensor grad_grad, at::Tensor grad2_embeddings);

#endif
