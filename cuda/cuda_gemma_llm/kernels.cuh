// Launch wrappers for the custom kernels; matrix products use cuBLAS (model.cu).
//
// Every kernel that depends on the sequence position reads the number of
// cached tokens from device memory (`length`). One decode step therefore has
// fixed launch parameters and can be captured once as a CUDA graph.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Deterministic uniform [-bound, bound] initialization (random-weight benchmarks).
void launch_fill_uniform(float* data, size_t count, float bound, uint64_t seed, cudaStream_t stream);
void launch_fill_uniform(__nv_bfloat16* data, size_t count, float bound, uint64_t seed, cudaStream_t stream);
void launch_fill_constant(float* data, size_t count, float value, cudaStream_t stream);

// out[row] = float(table[tokens[length + row]]): VLM.embed_tokens before projection.
void launch_embed_tokens(const float* table, const int* tokens, const int* length,
                         float* out, int rows, int dim, cudaStream_t stream);
void launch_embed_tokens(const __nv_bfloat16* table, const int* tokens, const int* length,
                         float* out, int rows, int dim, cudaStream_t stream);

// x += bias (the previous linear's deferred bias, if any); out = LayerNorm(x).
void launch_add_bias_layer_norm(float* x, const float* bias, const float* gamma, const float* beta,
                                float* out, int rows, int dim, float eps, cudaStream_t stream);

// On qkvg rows [q | k | v | gate]: add bias, rotate q/k with RoPE, write k/v
// into the cache at position length + row, and replace gate with sigmoid(gate).
void launch_qkv_rope_cache(float* qkvg, const float* bias, const float* rope_cos,
                           const float* rope_sin, float* k_cache, float* v_cache,
                           const int* length, int rows, int q_head, int kv_head, int head_dim,
                           int max_context, cudaStream_t stream);

// Causal grouped-query attention over the cache, multiplied by the per-head gate.
// out has shape [rows, q_head * head_dim].
void launch_attention(const float* qkvg, const float* k_cache, const float* v_cache,
                      const int* length, float* out, int rows, int q_head, int kv_head,
                      int head_dim, int max_context, cudaStream_t stream);

// out = gelu_tanh(gate + bias_gate) * (up + bias_up) for gate_up rows [gate | up].
void launch_gelu_mul(const float* gate_up, const float* bias, float* out, int rows, int ffn_dim,
                     cudaStream_t stream);

// Adds the classifier bias to logits in place, then greedily selects the first
// maximum: tokens[length + rows] = argmax, and length += rows.
int argmax_parts(int vocab);
void launch_argmax(float* logits, const float* bias, int vocab, float* part_values,
                   int* part_indices, int* tokens, int* length, int rows, cudaStream_t stream);
