// Custom kernels for the Gemma text decoder. See kernels.cuh for contracts.
#include "common.cuh"
#include "kernels.cuh"

#include <algorithm>
#include <climits>
#include <cmath>

namespace {

constexpr int kBlock = 256;  // Multiple of 32 for the warp reductions.

void check_launch() { CHECK_CUDA(cudaGetLastError()); }

unsigned grid_for(size_t count, int per_block = kBlock) {
    return static_cast<unsigned>(std::min<size_t>((count + per_block - 1) / per_block, 1u << 20));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform(size_t index, float bound, uint64_t seed) {
    // 24 random bits -> [0, 1) -> [-bound, bound).
    float unit = (splitmix64(seed ^ splitmix64(index)) >> 40) * (1.0f / 16777216.0f);
    return (2.0f * unit - 1.0f) * bound;
}

__global__ void fill_uniform_kernel(float* data, size_t count, float bound, uint64_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] = uniform(i, bound, seed);
}

__global__ void fill_uniform_bf16_kernel(__nv_bfloat16* data, size_t count, float bound, uint64_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] = __float2bfloat16(uniform(i, bound, seed));
}

__global__ void fill_constant_kernel(float* data, size_t count, float value) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] = value;
}

__device__ __forceinline__ float to_float(float value) { return value; }
__device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
__global__ void embed_tokens_kernel(const T* table, const int* tokens, const int* length,
                                    float* out, int dim) {
    const int row = blockIdx.x;
    const T* source = table + size_t(tokens[*length + row]) * dim;
    float* target = out + size_t(row) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        target[i] = to_float(source[i]);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Sum across the block; every thread receives the result. shared holds >= 32 floats.
__device__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    value = warp_sum(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < int(blockDim.x / 32) ? shared[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    value = shared[0];
    __syncthreads();  // Let every thread read before shared is reused.
    return value;
}

__global__ void add_bias_layer_norm_kernel(float* x, const float* bias, const float* gamma,
                                           const float* beta, float* out, int dim, float eps) {
    __shared__ float shared[32];
    float* row = x + size_t(blockIdx.x) * dim;
    float* target = out + size_t(blockIdx.x) * dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float value = row[i];
        if (bias) {
            value += bias[i];
            row[i] = value;  // The residual stream keeps the biased value.
        }
        sum += value;
    }
    const float mean = block_sum(sum, shared) / dim;
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = row[i] - mean;
        square_sum += diff * diff;
    }
    const float inv_std = rsqrtf(block_sum(square_sum, shared) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        target[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
}

__global__ void qkv_rope_cache_kernel(float* qkvg, const float* bias, const float* rope_cos,
                                      const float* rope_sin, float* k_cache, float* v_cache,
                                      const int* length, int q_head, int kv_head, int head_dim,
                                      int max_context) {
    const int row = blockIdx.x;
    const int position = *length + row;
    const int half = head_dim / 2;
    const int qkv_dim = (q_head + 2 * kv_head) * head_dim;
    float* values = qkvg + size_t(row) * (qkv_dim + q_head);
    const float* cos_row = rope_cos + size_t(position) * half;
    const float* sin_row = rope_sin + size_t(position) * half;

    // ROPEEmbedding rotates adjacent pairs (2i, 2i+1) by angle position * theta^(-2i/head_dim).
    for (int pair = threadIdx.x; pair < (q_head + kv_head) * half; pair += blockDim.x) {
        const int head = pair / half, i = pair % half;
        const int index = head * head_dim + 2 * i;
        const float a = values[index] + bias[index];
        const float b = values[index + 1] + bias[index + 1];
        const float rotated_a = a * cos_row[i] - b * sin_row[i];
        const float rotated_b = b * cos_row[i] + a * sin_row[i];
        if (head < q_head) {
            values[index] = rotated_a;
            values[index + 1] = rotated_b;
        } else {
            float* key = k_cache + (size_t(head - q_head) * max_context + position) * head_dim + 2 * i;
            key[0] = rotated_a;
            key[1] = rotated_b;
        }
    }
    const int v_offset = (q_head + kv_head) * head_dim;
    for (int i = threadIdx.x; i < kv_head * head_dim; i += blockDim.x) {
        const int kv = i / head_dim, d = i % head_dim;
        v_cache[(size_t(kv) * max_context + position) * head_dim + d] =
            values[v_offset + i] + bias[v_offset + i];
    }
    for (int head = threadIdx.x; head < q_head; head += blockDim.x) {
        const float gate = values[qkv_dim + head] + bias[qkv_dim + head];
        values[qkv_dim + head] = 1.0f / (1.0f + expf(-gate));
    }
}

// One block per (query head, new token). Groups of kLanes threads share one
// cached position: each lane holds HEAD_DIM / kLanes dimensions of q, k, v and
// the output accumulator, so loads are contiguous and the dot product needs
// only log2(kLanes) shuffles. Each group keeps an online softmax (running max,
// normalizer, weighted values); groups are merged through shared memory.
template <int HEAD_DIM>
__global__ void attention_kernel(const float* qkvg, const float* k_cache, const float* v_cache,
                                 const int* length, float* out, int q_head, int kv_head,
                                 int max_context, float scale) {
    constexpr int kLanes = 8;
    constexpr int kDims = HEAD_DIM / kLanes;
    constexpr int kGroupsPerWarp = 32 / kLanes;
    static_assert(kDims % 4 == 0, "float4 loads need four dimensions per lane");
    extern __shared__ float shared[];

    const int head = blockIdx.x, row = blockIdx.y;
    const int groups = blockDim.x / kLanes;
    const int warp = threadIdx.x / 32;
    const int sub = (threadIdx.x % 32) / kLanes;
    const int lane = threadIdx.x % kLanes;
    const int group = warp * kGroupsPerWarp + sub;
    const int qkv_dim = (q_head + 2 * kv_head) * HEAD_DIM;
    const float* token = qkvg + size_t(row) * (qkv_dim + q_head);
    const int kv = head / (q_head / kv_head);  // torch.repeat_interleave over kv heads
    const float* keys = k_cache + size_t(kv) * max_context * HEAD_DIM + lane * kDims;
    const float* values = v_cache + size_t(kv) * max_context * HEAD_DIM + lane * kDims;
    const int last = *length + row;  // Causal: attend to positions [0, last].

    float q[kDims], acc[kDims];
#pragma unroll
    for (int d = 0; d < kDims; ++d) {
        q[d] = token[head * HEAD_DIM + lane * kDims + d];
        acc[d] = 0.0f;
    }
    float running_max = -INFINITY, normalizer = 0.0f;

    // The loop bound depends only on the warp, so full-mask shuffles are safe.
    for (int base = warp * kGroupsPerWarp; base <= last; base += groups) {
        const int position = base + sub;
        const bool valid = position <= last;
        float dot = 0.0f;
        if (valid) {
            const float4* key = reinterpret_cast<const float4*>(keys + size_t(position) * HEAD_DIM);
#pragma unroll
            for (int d = 0; d < kDims / 4; ++d) {
                float4 k = key[d];
                dot += q[4 * d] * k.x + q[4 * d + 1] * k.y + q[4 * d + 2] * k.z + q[4 * d + 3] * k.w;
            }
        }
#pragma unroll
        for (int offset = kLanes / 2; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        if (valid) {
            const float score = dot * scale;
            const float new_max = fmaxf(running_max, score);
            const float correction = expf(running_max - new_max);
            const float weight = expf(score - new_max);
            normalizer = normalizer * correction + weight;
            const float4* value = reinterpret_cast<const float4*>(values + size_t(position) * HEAD_DIM);
#pragma unroll
            for (int d = 0; d < kDims / 4; ++d) {
                float4 v = value[d];
                acc[4 * d] = acc[4 * d] * correction + weight * v.x;
                acc[4 * d + 1] = acc[4 * d + 1] * correction + weight * v.y;
                acc[4 * d + 2] = acc[4 * d + 2] * correction + weight * v.z;
                acc[4 * d + 3] = acc[4 * d + 3] * correction + weight * v.w;
            }
            running_max = new_max;
        }
    }

    float* shared_acc = shared;                          // [groups, HEAD_DIM]
    float* shared_max = shared + groups * HEAD_DIM;      // [groups]
    float* shared_normalizer = shared_max + groups;      // [groups]
#pragma unroll
    for (int d = 0; d < kDims; ++d)
        shared_acc[group * HEAD_DIM + lane * kDims + d] = acc[d];
    if (lane == 0) {
        shared_max[group] = running_max;
        shared_normalizer[group] = normalizer;
    }
    __syncthreads();

    // Position 0 is always visible, so the merged maximum is finite; groups
    // that saw no positions contribute exp(-inf) = 0.
    const float gate = token[qkv_dim + head];
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float maximum = -INFINITY;
        for (int g = 0; g < groups; ++g) maximum = fmaxf(maximum, shared_max[g]);
        float total = 0.0f, value = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const float weight = expf(shared_max[g] - maximum);
            total += shared_normalizer[g] * weight;
            value += shared_acc[g * HEAD_DIM + d] * weight;
        }
        out[size_t(row) * q_head * HEAD_DIM + head * HEAD_DIM + d] = value / total * gate;
    }
}

__global__ void gelu_mul_kernel(const float* gate_up, const float* bias, float* out, int rows,
                                int ffn_dim) {
    const size_t count = size_t(rows) * ffn_dim;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const size_t row = i / ffn_dim;
        const int column = int(i % ffn_dim);
        const float* source = gate_up + row * 2 * ffn_dim;
        const float gate = source[column] + bias[column];
        const float up = source[ffn_dim + column] + bias[ffn_dim + column];
        // F.gelu(approximate="tanh"); gate_and_up.chunk(2) puts gate first.
        const float inner = 0.7978845608028654f * (gate + 0.044715f * gate * gate * gate);
        out[i] = 0.5f * gate * (1.0f + tanhf(inner)) * up;
    }
}

// Prefer the larger value, then the lower index, matching torch.argmax.
__device__ __forceinline__ void keep_best(float& value, int& index, float other_value, int other_index) {
    if (other_value > value || (other_value == value && other_index < index)) {
        value = other_value;
        index = other_index;
    }
}

__device__ void block_argmax(float& value, int& index, float* shared_values, int* shared_indices) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1)
        keep_best(value, index, __shfl_xor_sync(0xffffffffu, value, offset),
                  __shfl_xor_sync(0xffffffffu, index, offset));
    if (lane == 0) {
        shared_values[warp] = value;
        shared_indices[warp] = index;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < int(blockDim.x / 32) ? shared_values[lane] : -INFINITY;
        index = lane < int(blockDim.x / 32) ? shared_indices[lane] : INT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            keep_best(value, index, __shfl_xor_sync(0xffffffffu, value, offset),
                      __shfl_xor_sync(0xffffffffu, index, offset));
    }
}

__global__ void argmax_partial_kernel(float* logits, const float* bias, int vocab,
                                      float* part_values, int* part_indices) {
    __shared__ float shared_values[32];
    __shared__ int shared_indices[32];
    float best = -INFINITY;
    int best_index = INT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
        const float value = logits[i] + bias[i];
        logits[i] = value;
        if (value > best) {  // Indices increase per thread, so ties keep the first.
            best = value;
            best_index = i;
        }
    }
    block_argmax(best, best_index, shared_values, shared_indices);
    if (threadIdx.x == 0) {
        part_values[blockIdx.x] = best;
        part_indices[blockIdx.x] = best_index;
    }
}

__global__ void argmax_final_kernel(const float* part_values, const int* part_indices, int parts,
                                    int* tokens, int* length, int rows) {
    __shared__ float shared_values[32];
    __shared__ int shared_indices[32];
    float best = -INFINITY;
    int best_index = INT_MAX;
    for (int i = threadIdx.x; i < parts; i += blockDim.x)
        keep_best(best, best_index, part_values[i], part_indices[i]);
    block_argmax(best, best_index, shared_values, shared_indices);
    if (threadIdx.x == 0) {
        const int processed = *length + rows;
        tokens[processed] = best_index;
        *length = processed;
    }
}

}  // namespace

void launch_fill_uniform(float* data, size_t count, float bound, uint64_t seed, cudaStream_t stream) {
    fill_uniform_kernel<<<grid_for(count), kBlock, 0, stream>>>(data, count, bound, seed);
    check_launch();
}

void launch_fill_uniform(__nv_bfloat16* data, size_t count, float bound, uint64_t seed,
                         cudaStream_t stream) {
    fill_uniform_bf16_kernel<<<grid_for(count), kBlock, 0, stream>>>(data, count, bound, seed);
    check_launch();
}

void launch_fill_constant(float* data, size_t count, float value, cudaStream_t stream) {
    fill_constant_kernel<<<grid_for(count), kBlock, 0, stream>>>(data, count, value);
    check_launch();
}

void launch_embed_tokens(const float* table, const int* tokens, const int* length, float* out,
                         int rows, int dim, cudaStream_t stream) {
    embed_tokens_kernel<<<rows, kBlock, 0, stream>>>(table, tokens, length, out, dim);
    check_launch();
}

void launch_embed_tokens(const __nv_bfloat16* table, const int* tokens, const int* length,
                         float* out, int rows, int dim, cudaStream_t stream) {
    embed_tokens_kernel<<<rows, kBlock, 0, stream>>>(table, tokens, length, out, dim);
    check_launch();
}

void launch_add_bias_layer_norm(float* x, const float* bias, const float* gamma, const float* beta,
                                float* out, int rows, int dim, float eps, cudaStream_t stream) {
    add_bias_layer_norm_kernel<<<rows, kBlock, 0, stream>>>(x, bias, gamma, beta, out, dim, eps);
    check_launch();
}

void launch_qkv_rope_cache(float* qkvg, const float* bias, const float* rope_cos,
                           const float* rope_sin, float* k_cache, float* v_cache,
                           const int* length, int rows, int q_head, int kv_head, int head_dim,
                           int max_context, cudaStream_t stream) {
    qkv_rope_cache_kernel<<<rows, kBlock, 0, stream>>>(qkvg, bias, rope_cos, rope_sin, k_cache,
                                                       v_cache, length, q_head, kv_head,
                                                       head_dim, max_context);
    check_launch();
}

void launch_attention(const float* qkvg, const float* k_cache, const float* v_cache,
                      const int* length, float* out, int rows, int q_head, int kv_head,
                      int head_dim, int max_context, cudaStream_t stream) {
    const dim3 grid(q_head, rows);
    const int groups = kBlock / 8;
    const size_t shared_bytes = size_t(groups) * (head_dim + 2) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    switch (head_dim) {
        case 32: attention_kernel<32><<<grid, kBlock, shared_bytes, stream>>>(
            qkvg, k_cache, v_cache, length, out, q_head, kv_head, max_context, scale); break;
        case 64: attention_kernel<64><<<grid, kBlock, shared_bytes, stream>>>(
            qkvg, k_cache, v_cache, length, out, q_head, kv_head, max_context, scale); break;
        case 128: attention_kernel<128><<<grid, kBlock, shared_bytes, stream>>>(
            qkvg, k_cache, v_cache, length, out, q_head, kv_head, max_context, scale); break;
        case 256: attention_kernel<256><<<grid, kBlock, shared_bytes, stream>>>(
            qkvg, k_cache, v_cache, length, out, q_head, kv_head, max_context, scale); break;
        default: throw std::runtime_error("Unsupported head_dim");
    }
    check_launch();
}

void launch_gelu_mul(const float* gate_up, const float* bias, float* out, int rows, int ffn_dim,
                     cudaStream_t stream) {
    gelu_mul_kernel<<<grid_for(size_t(rows) * ffn_dim), kBlock, 0, stream>>>(gate_up, bias, out,
                                                                           rows, ffn_dim);
    check_launch();
}

int argmax_parts(int vocab) {
    // About four logits per thread in the first pass; at most 1024 partial results.
    return std::max(1, std::min(1024, (vocab + 4 * kBlock - 1) / (4 * kBlock)));
}

void launch_argmax(float* logits, const float* bias, int vocab, float* part_values,
                   int* part_indices, int* tokens, int* length, int rows, cudaStream_t stream) {
    const int parts = argmax_parts(vocab);
    argmax_partial_kernel<<<parts, kBlock, 0, stream>>>(logits, bias, vocab, part_values, part_indices);
    check_launch();
    argmax_final_kernel<<<1, 1024, 0, stream>>>(part_values, part_indices, parts, tokens, length, rows);
    check_launch();
}
