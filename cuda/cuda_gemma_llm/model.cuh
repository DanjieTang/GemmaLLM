// Float32 CUDA implementation of VLM's text-only greedy generation path.
#pragma once

#include "common.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <string>
#include <vector>

// Parameters of one LLMLayer. The attention gate is fused into the QKV
// projection: rows are qkv.weight followed by gate.weight, so a single GEMM
// produces [q | k | v | gate] for every token.
struct LayerWeights {
    DeviceBuffer<float> norm1_weight, norm1_bias;
    DeviceBuffer<float> qkvg_weight, qkvg_bias;
    DeviceBuffer<float> o_weight, o_bias;
    DeviceBuffer<float> norm2_weight, norm2_bias;
    DeviceBuffer<float> gate_up_weight, gate_up_bias;
    DeviceBuffer<float> down_weight, down_bias;
};

class GemmaLLM {
public:
    // max_prompt_length sizes the activation buffers used by the prefill step.
    GemmaLLM(const ModelConfig& config, int max_prompt_length);
    ~GemmaLLM();
    GemmaLLM(const GemmaLLM&) = delete;
    GemmaLLM& operator=(const GemmaLLM&) = delete;

    // PyTorch-style initialization: Linear U(+-1/sqrt(fan_in)), LayerNorm (1, 0),
    // and unit-variance embeddings. Values do not affect generation speed.
    void init_random(uint64_t seed);
    // Load <directory>/<state_dict name>.bin files written by export_weights.py.
    void load_weights(const std::string& directory);

    // Greedy decoding until the sequence (prompt included) holds total_tokens
    // tokens; returns the whole sequence. Optionally copies the logits that
    // follow the prompt. With use_graph, each decode step replays a CUDA graph.
    std::vector<int> generate(const std::vector<int>& prompt, int total_tokens, bool use_graph,
                              std::vector<float>* first_logits = nullptr);

    const ModelConfig& config() const { return config_; }
    size_t parameter_count() const;
    size_t embedding_bytes() const;

private:
    struct Parameter {
        std::string name;  // PyTorch state_dict key
        float* data;
        size_t count;
        enum Kind { Linear, Ones, Zeros } kind;
        int fan_in;
    };
    std::vector<Parameter> parameters() const;

    void forward(int rows);  // Enqueue one step for `rows` new tokens on stream_.
    void linear(const float* x, const float* weight, float* y, int rows, int in, int out, float beta);
    void capture_decode_graph();

    ModelConfig config_;
    int max_rows_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    cudaGraphExec_t decode_graph_ = nullptr;

    // Weights
    DeviceBuffer<__nv_bfloat16> embeddings_bf16_;
    DeviceBuffer<float> embeddings_f32_;
    DeviceBuffer<float> projection_weight_, projection_bias_;
    std::vector<LayerWeights> layers_;
    DeviceBuffer<float> output_norm_weight_, output_norm_bias_;
    DeviceBuffer<float> classifier_weight_, classifier_bias_;
    DeviceBuffer<float> rope_cos_, rope_sin_;  // [max_context, head_dim / 2]

    // Generation state: KV cache [layer, kv_head, max_context, head_dim], the
    // token sequence, and the number of tokens already in the cache.
    DeviceBuffer<float> k_cache_, v_cache_;
    DeviceBuffer<int> tokens_, length_;

    // Activations for up to max_rows_ tokens.
    DeviceBuffer<float> embedded_, x_, normed_, qkvg_, attention_, gate_up_, ffn_, logits_;
    DeviceBuffer<float> argmax_values_;
    DeviceBuffer<int> argmax_indices_;
    DeviceBuffer<char> blas_workspace_;
};
