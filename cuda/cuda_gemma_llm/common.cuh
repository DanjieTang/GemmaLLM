// Shared error checks, device buffers, and the model configuration.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#define CHECK_CUDA(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) \
        throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error)); \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error(std::string(#call) + ": " + cublasGetStatusString(status)); \
} while (0)

// Owns one GPU allocation; movable so layers can live in a std::vector.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count) CHECK_CUDA(cudaMalloc(&data_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(count_, other.count_);
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return data_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

enum class EmbeddingDtype { Float32, BFloat16 };

// Architecture of VLM's text path (embedding -> optional projection -> LLM).
// Key names follow sweep_config.yaml; vocab_size and text_dim come from the
// Gemma embedding table, which the YAML references by path only.
struct ModelConfig {
    int num_layer = 4;
    int vocab_size = 262144;   // gemma-4-31B-it embedding rows
    int text_dim = 5376;       // gemma-4-31B-it embedding width
    int projection_dim = 512;  // 0 (YAML null): the decoder uses text_dim
    int expansion_factor = 18;
    int head_dim = 64;
    int q_head = 8;            // 0 (YAML null): hidden_dim / head_dim, as in LLM
    int kv_head = 4;           // 0 (YAML null): hidden_dim / head_dim, as in LLM
    float theta = 10000.0f;
    int max_context_length = 256;
    float norm_eps = 1e-5f;    // nn.LayerNorm default
    EmbeddingDtype embedding_dtype = EmbeddingDtype::BFloat16;

    int hidden_dim() const { return projection_dim ? projection_dim : text_dim; }
    int ffn_dim() const { return hidden_dim() * expansion_factor; }
    bool has_projection() const { return text_dim != hidden_dim(); }
    int q_dim() const { return q_head * head_dim; }
    int qkv_dim() const { return (q_head + 2 * kv_head) * head_dim; }
    // Fused attention input projection: [q | k | v | gate] per token.
    int qkvg_dim() const { return qkv_dim() + q_head; }

    // Resolve null head counts and reject shapes the kernels do not support.
    void finalize();
    std::string summary() const;
};

// Apply every recognized top-level `key: value` in a flat YAML file (such as
// sweep_config.yaml or an exported config.yaml); unrelated keys are ignored.
void apply_yaml(ModelConfig& config, const std::string& path);
// Apply one key/value pair; returns false for keys that are not architecture settings.
bool apply_setting(ModelConfig& config, const std::string& key, const std::string& value);
