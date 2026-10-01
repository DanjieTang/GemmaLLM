// GemmaLLM: weights, KV cache, and the forward step. Mirrors, in order,
// VLM.embed_tokens -> LLM.forward -> greedy argmax from generate.py.
#include "model.cuh"

#include "kernels.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace {

constexpr size_t kBlasWorkspaceBytes = 32u << 20;

// Copy a raw little-endian file of exactly `bytes` bytes to device memory in chunks.
void load_file(const std::string& path, void* device, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open " + path);
    if (static_cast<size_t>(file.tellg()) != bytes)
        throw std::runtime_error(path + ": expected " + std::to_string(bytes) + " bytes, found " +
                                 std::to_string(static_cast<size_t>(file.tellg())));
    file.seekg(0);
    std::vector<char> chunk(std::min<size_t>(bytes, 64u << 20));
    for (size_t offset = 0; offset < bytes; offset += chunk.size()) {
        const size_t size = std::min(chunk.size(), bytes - offset);
        if (!file.read(chunk.data(), static_cast<std::streamsize>(size)))
            throw std::runtime_error("Cannot read " + path);
        CHECK_CUDA(cudaMemcpy(static_cast<char*>(device) + offset, chunk.data(), size,
                              cudaMemcpyHostToDevice));
    }
}

}  // namespace

GemmaLLM::GemmaLLM(const ModelConfig& config, int max_prompt_length)
    : config_(config), max_rows_(std::max(1, max_prompt_length)) {
    config_.finalize();
    const ModelConfig& c = config_;
    const int hidden = c.hidden_dim(), ffn = c.ffn_dim();
    if (max_rows_ > c.max_context_length)
        throw std::runtime_error("Prompt length exceeds max_context_length");
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasCreate(&blas_));
    CHECK_CUBLAS(cublasSetStream(blas_, stream_));
    // A fixed workspace keeps cuBLAS from allocating while a graph is captured.
    blas_workspace_ = DeviceBuffer<char>(kBlasWorkspaceBytes);
    CHECK_CUBLAS(cublasSetWorkspace(blas_, blas_workspace_.data(), blas_workspace_.bytes()));

    const size_t table = size_t(c.vocab_size) * c.text_dim;
    if (c.embedding_dtype == EmbeddingDtype::BFloat16) embeddings_bf16_ = DeviceBuffer<__nv_bfloat16>(table);
    else embeddings_f32_ = DeviceBuffer<float>(table);
    if (c.has_projection()) {
        projection_weight_ = DeviceBuffer<float>(size_t(hidden) * c.text_dim);
        projection_bias_ = DeviceBuffer<float>(hidden);
    }
    layers_.resize(c.num_layer);
    for (LayerWeights& layer : layers_) {
        layer.norm1_weight = DeviceBuffer<float>(hidden);
        layer.norm1_bias = DeviceBuffer<float>(hidden);
        layer.qkvg_weight = DeviceBuffer<float>(size_t(c.qkvg_dim()) * hidden);
        layer.qkvg_bias = DeviceBuffer<float>(c.qkvg_dim());
        layer.o_weight = DeviceBuffer<float>(size_t(hidden) * c.q_dim());
        layer.o_bias = DeviceBuffer<float>(hidden);
        layer.norm2_weight = DeviceBuffer<float>(hidden);
        layer.norm2_bias = DeviceBuffer<float>(hidden);
        layer.gate_up_weight = DeviceBuffer<float>(size_t(2) * ffn * hidden);
        layer.gate_up_bias = DeviceBuffer<float>(size_t(2) * ffn);
        layer.down_weight = DeviceBuffer<float>(size_t(hidden) * ffn);
        layer.down_bias = DeviceBuffer<float>(hidden);
    }
    output_norm_weight_ = DeviceBuffer<float>(hidden);
    output_norm_bias_ = DeviceBuffer<float>(hidden);
    classifier_weight_ = DeviceBuffer<float>(size_t(c.vocab_size) * hidden);
    classifier_bias_ = DeviceBuffer<float>(c.vocab_size);

    // Same float32 arithmetic as ROPEEmbedding.create_embedding.
    const int half = c.head_dim / 2;
    std::vector<float> cos_table(size_t(c.max_context_length) * half);
    std::vector<float> sin_table(cos_table.size());
    for (int i = 0; i < half; ++i) {
        const float frequency = std::pow(c.theta, -static_cast<float>(2 * i) / c.head_dim);
        for (int position = 0; position < c.max_context_length; ++position) {
            const float angle = frequency * static_cast<float>(position);
            cos_table[size_t(position) * half + i] = std::cos(angle);
            sin_table[size_t(position) * half + i] = std::sin(angle);
        }
    }
    rope_cos_ = DeviceBuffer<float>(cos_table.size());
    rope_sin_ = DeviceBuffer<float>(sin_table.size());
    CHECK_CUDA(cudaMemcpy(rope_cos_.data(), cos_table.data(), rope_cos_.bytes(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rope_sin_.data(), sin_table.data(), rope_sin_.bytes(), cudaMemcpyHostToDevice));

    const size_t cache = size_t(c.num_layer) * c.kv_head * c.max_context_length * c.head_dim;
    k_cache_ = DeviceBuffer<float>(cache);
    v_cache_ = DeviceBuffer<float>(cache);
    tokens_ = DeviceBuffer<int>(c.max_context_length + 1);
    length_ = DeviceBuffer<int>(1);
    CHECK_CUDA(cudaMemset(tokens_.data(), 0, tokens_.bytes()));
    CHECK_CUDA(cudaMemset(length_.data(), 0, length_.bytes()));

    if (c.has_projection()) embedded_ = DeviceBuffer<float>(size_t(max_rows_) * c.text_dim);
    x_ = DeviceBuffer<float>(size_t(max_rows_) * hidden);
    normed_ = DeviceBuffer<float>(size_t(max_rows_) * hidden);
    qkvg_ = DeviceBuffer<float>(size_t(max_rows_) * c.qkvg_dim());
    attention_ = DeviceBuffer<float>(size_t(max_rows_) * c.q_dim());
    gate_up_ = DeviceBuffer<float>(size_t(max_rows_) * 2 * ffn);
    ffn_ = DeviceBuffer<float>(size_t(max_rows_) * ffn);
    logits_ = DeviceBuffer<float>(c.vocab_size);
    argmax_values_ = DeviceBuffer<float>(argmax_parts(c.vocab_size));
    argmax_indices_ = DeviceBuffer<int>(argmax_parts(c.vocab_size));
}

GemmaLLM::~GemmaLLM() {
    if (decode_graph_) cudaGraphExecDestroy(decode_graph_);
    if (blas_) cublasDestroy(blas_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<GemmaLLM::Parameter> GemmaLLM::parameters() const {
    const ModelConfig& c = config_;
    const int hidden = c.hidden_dim(), ffn = c.ffn_dim();
    std::vector<Parameter> list;
    auto linear = [&](const std::string& name, const DeviceBuffer<float>& weight,
                      const DeviceBuffer<float>& bias, int in, int out) {
        list.push_back({name + ".weight", weight.data(), size_t(out) * in, Parameter::Linear, in});
        list.push_back({name + ".bias", bias.data(), size_t(out), Parameter::Linear, in});
    };
    auto norm = [&](const std::string& name, const DeviceBuffer<float>& weight,
                    const DeviceBuffer<float>& bias) {
        list.push_back({name + ".weight", weight.data(), size_t(hidden), Parameter::Ones, 0});
        list.push_back({name + ".bias", bias.data(), size_t(hidden), Parameter::Zeros, 0});
    };
    if (c.has_projection())
        linear("text_token_projection", projection_weight_, projection_bias_, c.text_dim, hidden);
    for (int i = 0; i < c.num_layer; ++i) {
        const LayerWeights& layer = layers_[i];
        const std::string prefix = "llm.transformer." + std::to_string(i) + ".";
        norm(prefix + "norm1", layer.norm1_weight, layer.norm1_bias);
        // Two PyTorch Linears share the fused buffers: qkv rows first, then gate rows.
        const size_t qkv_weights = size_t(c.qkv_dim()) * hidden;
        list.push_back({prefix + "mqa.qkv.weight", layer.qkvg_weight.data(), qkv_weights, Parameter::Linear, hidden});
        list.push_back({prefix + "mqa.qkv.bias", layer.qkvg_bias.data(), size_t(c.qkv_dim()), Parameter::Linear, hidden});
        list.push_back({prefix + "mqa.gate.weight", layer.qkvg_weight.data() + qkv_weights,
                        size_t(c.q_head) * hidden, Parameter::Linear, hidden});
        list.push_back({prefix + "mqa.gate.bias", layer.qkvg_bias.data() + c.qkv_dim(),
                        size_t(c.q_head), Parameter::Linear, hidden});
        linear(prefix + "mqa.o", layer.o_weight, layer.o_bias, c.q_dim(), hidden);
        norm(prefix + "norm2", layer.norm2_weight, layer.norm2_bias);
        linear(prefix + "ffn.gate_and_up", layer.gate_up_weight, layer.gate_up_bias, hidden, 2 * ffn);
        linear(prefix + "ffn.down", layer.down_weight, layer.down_bias, ffn, hidden);
    }
    norm("llm.output_norm", output_norm_weight_, output_norm_bias_);
    linear("llm.classifier", classifier_weight_, classifier_bias_, hidden, c.vocab_size);
    return list;
}

size_t GemmaLLM::parameter_count() const {
    size_t total = size_t(config_.vocab_size) * config_.text_dim;
    for (const Parameter& parameter : parameters()) total += parameter.count;
    return total;
}

size_t GemmaLLM::embedding_bytes() const {
    return embeddings_bf16_.bytes() + embeddings_f32_.bytes();
}

void GemmaLLM::init_random(uint64_t seed) {
    if (embeddings_bf16_.data())
        launch_fill_uniform(embeddings_bf16_.data(), embeddings_bf16_.count(), std::sqrt(3.0f), seed, stream_);
    else
        launch_fill_uniform(embeddings_f32_.data(), embeddings_f32_.count(), std::sqrt(3.0f), seed, stream_);
    uint64_t index = 0;
    for (const Parameter& parameter : parameters()) {
        ++index;
        if (parameter.kind == Parameter::Linear)
            launch_fill_uniform(parameter.data, parameter.count,
                                1.0f / std::sqrt(static_cast<float>(parameter.fan_in)),
                                seed + 0x1000 * index, stream_);
        else
            launch_fill_constant(parameter.data, parameter.count,
                                 parameter.kind == Parameter::Ones ? 1.0f : 0.0f, stream_);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

void GemmaLLM::load_weights(const std::string& directory) {
    if (embeddings_bf16_.data())
        load_file(directory + "/embeddings.bin", embeddings_bf16_.data(), embeddings_bf16_.bytes());
    else
        load_file(directory + "/embeddings.bin", embeddings_f32_.data(), embeddings_f32_.bytes());
    for (const Parameter& parameter : parameters())
        load_file(directory + "/" + parameter.name + ".bin", parameter.data, parameter.count * sizeof(float));
}

// Row-major y[rows, out] = x[rows, in] @ weight[out, in]^T + beta * y, i.e.
// nn.Linear without its bias. cuBLAS is column-major, so the row-major weight
// is read as its transpose without copying; a single token uses GEMV.
void GemmaLLM::linear(const float* x, const float* weight, float* y, int rows, int in, int out, float beta) {
    const float alpha = 1.0f;
    if (rows == 1)
        CHECK_CUBLAS(cublasSgemv(blas_, CUBLAS_OP_T, in, out, &alpha, weight, in, x, 1, &beta, y, 1));
    else
        CHECK_CUBLAS(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &alpha, weight, in,
                                 x, in, &beta, y, out));
}

void GemmaLLM::forward(int rows) {
    const ModelConfig& c = config_;
    const int hidden = c.hidden_dim(), ffn = c.ffn_dim();
    float* x = x_.data();

    // tensor = VLM.embed_tokens(token_ids). A Linear's bias is deferred and
    // added by the next add_bias_layer_norm, which also updates the residual.
    float* embedding_target = c.has_projection() ? embedded_.data() : x;
    if (embeddings_bf16_.data())
        launch_embed_tokens(embeddings_bf16_.data(), tokens_.data(), length_.data(), embedding_target,
                            rows, c.text_dim, stream_);
    else
        launch_embed_tokens(embeddings_f32_.data(), tokens_.data(), length_.data(), embedding_target,
                            rows, c.text_dim, stream_);
    if (c.has_projection()) linear(embedded_.data(), projection_weight_.data(), x, rows, c.text_dim, hidden, 0.0f);
    const float* pending_bias = c.has_projection() ? projection_bias_.data() : nullptr;

    const size_t layer_cache = size_t(c.kv_head) * c.max_context_length * c.head_dim;
    for (int i = 0; i < c.num_layer; ++i) {
        const LayerWeights& layer = layers_[i];
        float* k_cache = k_cache_.data() + i * layer_cache;
        float* v_cache = v_cache_.data() + i * layer_cache;
        // tensor = tensor + mqa(norm1(tensor))
        launch_add_bias_layer_norm(x, pending_bias, layer.norm1_weight.data(), layer.norm1_bias.data(),
                                   normed_.data(), rows, hidden, c.norm_eps, stream_);
        linear(normed_.data(), layer.qkvg_weight.data(), qkvg_.data(), rows, hidden, c.qkvg_dim(), 0.0f);
        launch_qkv_rope_cache(qkvg_.data(), layer.qkvg_bias.data(), rope_cos_.data(), rope_sin_.data(),
                              k_cache, v_cache, length_.data(), rows, c.q_head, c.kv_head, c.head_dim,
                              c.max_context_length, stream_);
        launch_attention(qkvg_.data(), k_cache, v_cache, length_.data(), attention_.data(), rows,
                         c.q_head, c.kv_head, c.head_dim, c.max_context_length, stream_);
        linear(attention_.data(), layer.o_weight.data(), x, rows, c.q_dim(), hidden, 1.0f);
        // tensor = tensor + ffn(norm2(tensor))
        launch_add_bias_layer_norm(x, layer.o_bias.data(), layer.norm2_weight.data(), layer.norm2_bias.data(),
                                   normed_.data(), rows, hidden, c.norm_eps, stream_);
        linear(normed_.data(), layer.gate_up_weight.data(), gate_up_.data(), rows, hidden, 2 * ffn, 0.0f);
        launch_gelu_mul(gate_up_.data(), layer.gate_up_bias.data(), ffn_.data(), rows, ffn, stream_);
        linear(ffn_.data(), layer.down_weight.data(), x, rows, ffn, hidden, 1.0f);
        pending_bias = layer.down_bias.data();
    }

    // Only the newest token's logits choose the next token.
    float* last = x + size_t(rows - 1) * hidden;
    launch_add_bias_layer_norm(last, pending_bias, output_norm_weight_.data(), output_norm_bias_.data(),
                               normed_.data(), 1, hidden, c.norm_eps, stream_);
    linear(normed_.data(), classifier_weight_.data(), logits_.data(), 1, hidden, c.vocab_size, 0.0f);
    launch_argmax(logits_.data(), classifier_bias_.data(), c.vocab_size, argmax_values_.data(),
                  argmax_indices_.data(), tokens_.data(), length_.data(), rows, stream_);
}

void GemmaLLM::capture_decode_graph() {
    if (decode_graph_) return;
    // Run one eager step first so cuBLAS picks its kernels outside capture.
    // generate() resets the cache length and tokens, so this leaves no state.
    CHECK_CUDA(cudaMemsetAsync(length_.data(), 0, sizeof(int), stream_));
    forward(1);
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    cudaGraph_t graph = nullptr;
    CHECK_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    try {
        forward(1);
    } catch (...) {
        cudaStreamEndCapture(stream_, &graph);
        if (graph) cudaGraphDestroy(graph);
        throw;
    }
    CHECK_CUDA(cudaStreamEndCapture(stream_, &graph));
    const cudaError_t status = cudaGraphInstantiate(&decode_graph_, graph, 0);
    cudaGraphDestroy(graph);
    CHECK_CUDA(status);
}

std::vector<int> GemmaLLM::generate(const std::vector<int>& prompt, int total_tokens, bool use_graph,
                                    std::vector<float>* first_logits) {
    const int prompt_length = static_cast<int>(prompt.size());
    if (prompt_length < 1 || prompt_length > max_rows_)
        throw std::runtime_error("Prompt must hold between 1 and max_prompt_length tokens");
    if (total_tokens <= prompt_length || total_tokens > config_.max_context_length)
        throw std::runtime_error("total_tokens must exceed the prompt and fit in max_context_length");
    for (int token : prompt)
        if (token < 0 || token >= config_.vocab_size)
            throw std::runtime_error("Prompt token outside the vocabulary");
    if (use_graph) capture_decode_graph();

    CHECK_CUDA(cudaMemsetAsync(length_.data(), 0, sizeof(int), stream_));
    CHECK_CUDA(cudaMemcpyAsync(tokens_.data(), prompt.data(), prompt.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));
    forward(prompt_length);  // Prefill the prompt; the next token lands in tokens_.
    if (first_logits) {
        first_logits->resize(config_.vocab_size);
        CHECK_CUDA(cudaMemcpyAsync(first_logits->data(), logits_.data(), logits_.bytes(),
                                   cudaMemcpyDeviceToHost, stream_));
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
    // Each step reads its input token and position on the device, so the host
    // only enqueues work and never waits for a token until the end.
    for (int length = prompt_length + 1; length < total_tokens; ++length) {
        if (use_graph) CHECK_CUDA(cudaGraphLaunch(decode_graph_, stream_));
        else forward(1);
    }
    std::vector<int> tokens(total_tokens);
    CHECK_CUDA(cudaMemcpyAsync(tokens.data(), tokens_.data(), tokens.size() * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    return tokens;
}
