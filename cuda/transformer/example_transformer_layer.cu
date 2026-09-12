// Matches PreNormTransformer.forward in example_transformer_layer.py.
// Build: nvcc -O2 -std=c++17 example_transformer_layer.cu -lcublas -o example_transformer_layer
// Run:   ./example_transformer_layer tensors
// Without a PyTorch output reference: ./example_transformer_layer tensors --no-verify
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

// Reuse one handle across GEMMs and forward calls, on the default CUDA stream.
struct CublasHandle {
    cublasHandle_t handle = nullptr;
    CublasHandle() {
        CHECK_CUBLAS(cublasCreate(&handle));
        try {
            // Keep full float32 precision to match the exported PyTorch reference.
            CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        } catch (...) {
            cublasDestroy(handle);
            throw;
        }
    }
    ~CublasHandle() { cublasDestroy(handle); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};

struct Config {
    int batch, sequence, hidden;
    float eps1, eps2;
};

Config load_config(const std::string& directory) {
    std::ifstream file(directory + "/config.txt");
    std::string version, extra;
    Config c{};
    if (!(file >> version >> c.batch >> c.sequence >> c.hidden >> c.eps1 >> c.eps2) ||
        version != "pre_norm_transformer_v1" || (file >> extra) ||
        c.batch <= 0 || c.sequence <= 0 || c.hidden <= 0 ||
        !std::isfinite(c.eps1) || !std::isfinite(c.eps2) || c.eps1 <= 0 || c.eps2 <= 0)
        throw std::runtime_error("Invalid or missing " + directory + "/config.txt; run the Python exporter first");
    // Kernels use int indexing; reject shapes that would overflow it.
    const int64_t limit = std::numeric_limits<int>::max();
    const int64_t rows = int64_t(c.batch) * c.sequence;
    if (c.hidden > limit / 4 || rows > limit / (4LL * c.hidden) ||
        4LL * c.hidden * c.hidden > limit || int64_t(c.sequence) * c.sequence > limit)
        throw std::runtime_error("Shape exceeds this example's 32-bit indexing limits");
    return c;
}

std::vector<float> load_float32(const std::string& path, size_t count) {
    static_assert(sizeof(float) == 4, "Expected 4-byte float");
    const uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
        throw std::runtime_error("The exported files require a little-endian host");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    if (file.tellg() != bytes)
        throw std::runtime_error(path + ": expected " + std::to_string(bytes) + " bytes");
    std::vector<float> values(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("Cannot read " + path);
    return values;
}

void save_float32(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float)))
        throw std::runtime_error("Cannot write " + path);
}

// Own GPU allocations so errors and normal returns both release the buffers.
struct DeviceTensor {
    float* data = nullptr;
    explicit DeviceTensor(size_t count) {
        CHECK_CUDA(cudaMalloc(&data, count * sizeof(float)));
    }
    explicit DeviceTensor(const std::vector<float>& values) : DeviceTensor(values.size()) {
        CHECK_CUDA(cudaMemcpy(data, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceTensor() { cudaFree(data); }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;
};

struct LinearWeights {
    DeviceTensor weight, bias;
    LinearWeights(const std::string& directory, const std::string& name, int in, int out)
        : weight(load_float32(directory + "/" + name + ".weight.bin", size_t(out) * in)),
          bias(load_float32(directory + "/" + name + ".bias.bin", out)) {}
};

struct NormWeights {
    DeviceTensor weight, bias;
    NormWeights(const std::string& directory, const std::string& name, int hidden)
        : weight(load_float32(directory + "/" + name + ".weight.bin", hidden)),
          bias(load_float32(directory + "/" + name + ".bias.bin", hidden)) {}
};

// Load the same named parameters as PreNormTransformer in Python once, then
// reuse them across forward calls. Input and output buffers belong to the caller.
struct TransformerWeights {
    const int hidden;
    NormWeights norm1;
    LinearWeights q_proj, k_proj, v_proj, out_proj;
    NormWeights norm2;
    LinearWeights ffn1, ffn2;

    TransformerWeights(const std::string& directory, int dim)
        : hidden(dim), norm1(directory, "norm1", dim),
          q_proj(directory, "q_proj", dim, dim),
          k_proj(directory, "k_proj", dim, dim),
          v_proj(directory, "v_proj", dim, dim),
          out_proj(directory, "out_proj", dim, dim),
          norm2(directory, "norm2", dim),
          ffn1(directory, "ffn.0", dim, 4 * dim),
          ffn2(directory, "ffn.2", 4 * dim, dim) {}
};

constexpr int blockSize = 256;  // Power of two for the shared-memory reductions.

__global__ void add_bias(float* output, const float* bias, int count, int columns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] += bias[i % columns];
}

// Row-major C[M,N] = A[M,K] @ B[K,N], or A @ B.T for B[N,K].
void matmul(cublasHandle_t handle, const float* a, const float* b, float* out, int m, int k, int n,
            bool transpose_b = false, const float* bias = nullptr) {
    // cuBLAS is column-major: compute C.T = B.T @ A.T by swapping operands.
    // The same memory holds the row-major result, with no explicit transpose.
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                            CUBLAS_OP_N, n, m, k, &alpha,
                            b, transpose_b ? k : n, a, k, &beta, out, n));
    if (bias) {
        add_bias<<<(m * n + blockSize - 1) / blockSize, blockSize>>>(out, bias, m * n, n);
        CHECK_CUDA(cudaGetLastError());
    }
}

void linear(cublasHandle_t handle, const float* input, const LinearWeights& weights, float* output,
            int rows, int in, int out) {
    matmul(handle, input, weights.weight.data, output, rows, in, out, true, weights.bias.data);
}

template<bool maximum>
__device__ float reduce(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            shared[tid] = maximum ? fmaxf(shared[tid], shared[tid + stride])
                                  : shared[tid] + shared[tid + stride];
        __syncthreads();
    }
    float result = shared[0];
    // All warps must read the result before the next reduction reuses shared.
    __syncthreads();
    return result;
}

__global__ void layer_norm(const float* input, float* output, const float* gamma,
                           const float* beta, int dim, float epsilon) {
    extern __shared__ float shared[];
    int offset = blockIdx.x * dim;
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += input[offset + i];
    float mean = reduce<false>(sum, shared) / dim;
    float square_sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = input[offset + i] - mean;
        square_sum += diff * diff;
    }
    float variance = reduce<false>(square_sum, shared) / dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        output[offset + i] = (input[offset + i] - mean) * rsqrtf(variance + epsilon) * gamma[i] + beta[i];
}

void normalize(const float* input, const NormWeights& weights, float* output,
               int rows, int dim, float epsilon) {
    layer_norm<<<rows, blockSize, blockSize * sizeof(float)>>>(
        input, output, weights.weight.data, weights.bias.data, dim, epsilon);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void softmaxKernel(const float* input, float* output, int dim) {
    extern __shared__ float shared[];
    int offset = blockIdx.x * dim;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_max = fmaxf(local_max, input[offset + i]);
    float maximum = reduce<true>(local_max, shared);
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float value = expf(input[offset + i] - maximum);
        output[offset + i] = value;
        sum += value;
    }
    float total = reduce<false>(sum, shared);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        output[offset + i] /= total;
}

__global__ void scale_and_causal_mask(float* scores, int sequence, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sequence * sequence)
        scores[i] = i % sequence > i / sequence ? -INFINITY : scores[i] * scale;
}

__global__ void relu(float* tensor, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) tensor[i] = fmaxf(tensor[i], 0.0f);
}

__global__ void matrix_elementwise_addition(const float* a, const float* b, float* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}

// Names and execution order mirror the Python model. Each sequence gets its
// own attention matrix; flattening batch and sequence is only used for Linear/LN.
void forward(const CublasHandle& blas, const Config& c, const TransformerWeights& model,
             const float* input, float* output) {
    if (c.hidden != model.hidden)
        throw std::runtime_error("Input hidden dimension must match the loaded model");
    int rows = c.batch * c.sequence;
    int elements = rows * c.hidden;
    int ffn_dim = 4 * c.hidden;
    DeviceTensor normalized(elements), q(elements), k(elements), v(elements);
    DeviceTensor context(elements), projected(elements), residual(elements);
    DeviceTensor scores(size_t(c.sequence) * c.sequence), attention(size_t(c.sequence) * c.sequence);
    DeviceTensor hidden(size_t(rows) * ffn_dim);

    // normalized = self.norm1(x); q/k/v = self.*_proj(normalized)
    normalize(input, model.norm1, normalized.data, rows, c.hidden, c.eps1);
    linear(blas.handle, normalized.data, model.q_proj, q.data, rows, c.hidden, c.hidden);
    linear(blas.handle, normalized.data, model.k_proj, k.data, rows, c.hidden, c.hidden);
    linear(blas.handle, normalized.data, model.v_proj, v.data, rows, c.hidden, c.hidden);
    for (int batch = 0; batch < c.batch; ++batch) {
        int offset = batch * c.sequence * c.hidden;
        // scores = (q @ k.T) * scale; scores.masked_fill_(causal_mask, -inf)
        matmul(blas.handle, q.data + offset, k.data + offset, scores.data, c.sequence, c.hidden, c.sequence, true);
        scale_and_causal_mask<<<(c.sequence * c.sequence + blockSize - 1) / blockSize, blockSize>>>(
            scores.data, c.sequence, 1.0f / sqrtf(static_cast<float>(c.hidden)));
        CHECK_CUDA(cudaGetLastError());
        // attention = softmax(scores); context = attention @ v
        softmaxKernel<<<c.sequence, blockSize, blockSize * sizeof(float)>>>(scores.data, attention.data, c.sequence);
        CHECK_CUDA(cudaGetLastError());
        matmul(blas.handle, attention.data, v.data + offset, context.data + offset, c.sequence, c.sequence, c.hidden);
    }
    // x = x + self.out_proj(context)
    linear(blas.handle, context.data, model.out_proj, projected.data, rows, c.hidden, c.hidden);
    matrix_elementwise_addition<<<(elements + blockSize - 1) / blockSize, blockSize>>>(
        input, projected.data, residual.data, elements);
    CHECK_CUDA(cudaGetLastError());

    // return x + self.ffn(self.norm2(x))
    normalize(residual.data, model.norm2, normalized.data, rows, c.hidden, c.eps2);
    linear(blas.handle, normalized.data, model.ffn1, hidden.data, rows, c.hidden, ffn_dim);
    relu<<<(rows * ffn_dim + blockSize - 1) / blockSize, blockSize>>>(hidden.data, rows * ffn_dim);
    CHECK_CUDA(cudaGetLastError());
    linear(blas.handle, hidden.data, model.ffn2, projected.data, rows, ffn_dim, c.hidden);
    matrix_elementwise_addition<<<(elements + blockSize - 1) / blockSize, blockSize>>>(
        residual.data, projected.data, output, elements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

bool compare(const std::vector<float>& output, const std::vector<float>& expected) {
    constexpr float atol = 1e-5f, rtol = 1e-4f;
    float max_error = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        float error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(output[i]) || !std::isfinite(expected[i]) ||
            error > atol + rtol * std::abs(expected[i])) {
            if (mismatches < 5)
                std::cerr << "Mismatch at " << i << ": CUDA=" << output[i] << ", PyTorch=" << expected[i] << '\n';
            ++mismatches;
        }
    }
    std::cout << "Max absolute error: " << max_error << '\n'
              << "Matching values: " << output.size() - mismatches << '/' << output.size()
              << " (atol=" << atol << ", rtol=" << rtol << ")\n"
              << (mismatches ? "FAIL" : "PASS") << ": CUDA vs PyTorch\n";
    return mismatches == 0;
}

int main(int argc, char** argv) {
    try {
        const std::string usage = "Usage: example_transformer_layer [tensor_directory] [--no-verify]";
        std::string directory = "tensors";
        bool verify = true, has_directory = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--no-verify") verify = false;
            else if (arg == "--help") { std::cout << usage << '\n'; return 0; }
            else if (arg.rfind("--", 0) == 0 || has_directory)
                throw std::runtime_error(usage);
            else { directory = arg; has_directory = true; }
        }
        Config c = load_config(directory);
        size_t count = size_t(c.batch) * c.sequence * c.hidden;
        auto input = load_float32(directory + "/input.bin", count);
        std::vector<float> expected;
        if (verify) expected = load_float32(directory + "/output.bin", count);
        TransformerWeights model(directory, c.hidden);
        CublasHandle blas;
        DeviceTensor d_input(input), d_output(count);
        forward(blas, c, model, d_input.data, d_output.data);
        std::vector<float> output(count);
        CHECK_CUDA(cudaMemcpy(output.data(), d_output.data, count * sizeof(float), cudaMemcpyDeviceToHost));
        if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("CUDA output contains non-finite values");
        save_float32(directory + "/computed_output.bin", output);
        std::cout << "Shape: [" << c.batch << ", " << c.sequence << ", " << c.hidden << "]\n";
        if (verify) return compare(output, expected) ? 0 : 1;
        std::cout << "Saved " << directory << "/computed_output.bin (reference comparison disabled)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
