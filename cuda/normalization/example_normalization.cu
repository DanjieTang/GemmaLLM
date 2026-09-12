// Layer Normalization in CUDA, mirroring PyTorch's:
//   torch.nn.LayerNorm(normalized_shape=N, eps=1e-5)
//   F.layer_norm(x, normalized_shape=(N,), weight=gamma, bias=beta, eps=eps)
//
// For each row x of shape (N,):
//   mean = sum(x) / N
//   var  = sum((x - mean)^2) / N          (biased, like PyTorch)
//   y_i  = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
//
// Strategy: one thread block per row, block-wide reductions (shared memory)
// for the mean and variance.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Block-wide sum reduction.
// Each thread passes its partial value; returns the total to every thread.
// Requires shared memory sized blockDim.x floats (passed dynamically).
// ---------------------------------------------------------------------------
__device__ float blockReduceSum(float val, float* shared) {
  int tid = threadIdx.x;
  shared[tid] = val;
  __syncthreads();

  // Tree reduction: stride halves each round.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  float total = shared[0];
  __syncthreads();  // avoid overwriting shared before everyone read it
  return total;
}

// ---------------------------------------------------------------------------
// LayerNorm kernel: grid = (rows) blocks, block = blockSize threads.
// Each block normalizes one row of `x` into `y`.
//   x, y   : (rows, N)
//   gamma  : (N,) scale    (PyTorch `weight`)
//   beta   : (N,) shift    (PyTorch `bias`)
// ---------------------------------------------------------------------------
__global__ void layerNormKernel(const float* x, float* y,
                                const float* gamma, const float* beta,
                                int N, float eps) {
  extern __shared__ float shared[];  // blockDim.x floats

  const int row = blockIdx.x;
  const float* x_row = x + (long)row * N;
  float* y_row = y + (long)row * N;

  // Pass 1: mean. Each thread accumulates a strided partial sum.
  float partial = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) partial += x_row[i];
  float mean = blockReduceSum(partial, shared) / N;

  // Pass 2: variance (biased, dividing by N like PyTorch).
  partial = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float d = x_row[i] - mean;
    partial += d * d;
  }
  float var = blockReduceSum(partial, shared) / N;

  // Pass 3: normalize + affine transform.
  float rstd = rsqrtf(var + eps);  // 1 / sqrt(var + eps)
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    y_row[i] = (x_row[i] - mean) * rstd * gamma[i] + beta[i];
  }
}

// ---------------------------------------------------------------------------
// CPU reference (what PyTorch would compute) for verification.
// ---------------------------------------------------------------------------
void layerNormCPU(const float* x, float* y,
                  const float* gamma, const float* beta,
                  int rows, int N, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (long)r * N;
    float* yr = y + (long)r * N;

    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += xr[i];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; ++i) {
      float d = xr[i] - mean;
      var += d * d;
    }
    var /= N;

    float rstd = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < N; ++i)
      yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
  }
}

std::vector<float> load_float32(const std::string& path, size_t count) {
  static_assert(sizeof(float) == 4, "Expected 4-byte float");
  const uint32_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
    throw std::runtime_error("The exported files require a little-endian host");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
  if (!file || file.tellg() != bytes)
    throw std::runtime_error(path + ": missing file or incorrect size (expected " + std::to_string(bytes) + " bytes)");
  std::vector<float> values(count);
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(values.data()), bytes))
    throw std::runtime_error("Cannot read " + path);
  return values;
}

// No arguments: original random CPU-reference demo.
// Exported PyTorch weights: ./example_normalization ../transformer/tensors [norm1|norm2]
int main(int argc, char** argv) {
 try {
  if (argc > 3)
    throw std::runtime_error("Usage: example_normalization [tensor_directory [norm1|norm2]]");
  int rows = 1024;
  int N = 512;
  float eps = 1e-5f;
  const int blockSize = 256;
  const bool from_files = argc >= 2;
  const std::string directory = from_files ? argv[1] : "";
  const std::string norm = argc == 3 ? argv[2] : "norm1";
  if (norm != "norm1" && norm != "norm2")
    throw std::runtime_error("LayerNorm name must be norm1 or norm2");
  if (from_files) {
    std::ifstream config(directory + "/config.txt");
    std::string version, extra;
    int batch, sequence;
    float eps1, eps2;
    if (!(config >> version >> batch >> sequence >> N >> eps1 >> eps2) ||
        version != "pre_norm_transformer_v1" || (config >> extra) ||
        batch <= 0 || sequence <= 0 || N <= 0 ||
        !std::isfinite(eps1) || !std::isfinite(eps2) || eps1 <= 0 || eps2 <= 0)
      throw std::runtime_error("Invalid or missing " + directory + "/config.txt");
    if (int64_t(batch) * sequence > std::numeric_limits<int>::max() / N)
      throw std::runtime_error("Shape exceeds this example's 32-bit indexing limits");
    rows = batch * sequence;
    eps = norm == "norm1" ? eps1 : eps2;
  }

  size_t matBytes = (size_t)rows * N * sizeof(float);
  size_t vecBytes = N * sizeof(float);

  std::vector<float> input(size_t(rows) * N), output(input.size()), reference(input.size());
  std::vector<float> gamma(N), beta(N);
  if (from_files) {
    input = load_float32(directory + "/" + norm + ".input.bin", input.size());
    reference = load_float32(directory + "/" + norm + ".output.bin", reference.size());
    gamma = load_float32(directory + "/" + norm + ".weight.bin", N);
    beta = load_float32(directory + "/" + norm + ".bias.bin", N);
    printf("Loaded %s weights and reference from %s\n", norm.c_str(), directory.c_str());
  } else {
    srand(42);
    for (float& value : input) value = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < N; ++i) {
      gamma[i] = 1.0f + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
      beta[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
  }
  float* h_x = input.data();
  float* h_y = output.data();
  float* h_ref = reference.data();
  float* h_gamma = gamma.data();
  float* h_beta = beta.data();

  // Device allocations + copies
  float *d_x, *d_y, *d_gamma, *d_beta;
  CUDA_CHECK(cudaMalloc(&d_x, matBytes));
  CUDA_CHECK(cudaMalloc(&d_y, matBytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, vecBytes));
  CUDA_CHECK(cudaMalloc(&d_beta, vecBytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, matBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, vecBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta, vecBytes, cudaMemcpyHostToDevice));

  // Launch: one block per row
  size_t sharedBytes = blockSize * sizeof(float);
  layerNormKernel<<<rows, blockSize, sharedBytes>>>(d_x, d_y, d_gamma, d_beta, N, eps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_y, d_y, matBytes, cudaMemcpyDeviceToHost));

  // Exported mode uses PyTorch's saved output; random mode uses the CPU reference.
  if (!from_files) layerNormCPU(h_x, h_ref, h_gamma, h_beta, rows, N, eps);

  float maxErr = 0.0f;
  int mismatches = 0;
  for (int i = 0; i < rows * N; ++i) {
    float err = fabsf(h_y[i] - h_ref[i]);
    maxErr = std::max(maxErr, err);
    if (!std::isfinite(h_y[i]) || !std::isfinite(h_ref[i]) ||
        err > 1e-5f + 1e-4f * fabsf(h_ref[i])) ++mismatches;
  }
  printf("max abs error vs %s reference: %g\n", from_files ? "PyTorch" : "CPU", maxErr);
  printf("%s (%d mismatches)\n", mismatches == 0 ? "PASS" : "FAIL", mismatches);

  // Show a few values from row 0 (like printing a tensor slice in PyTorch)
  for (int i = 0; i < std::min(N, 4); ++i)
    printf("x[0, %d] = % .4f, y[0, %d] = % .4f\n", i, h_x[i], i, h_y[i]);

  // Cleanup
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
  return mismatches == 0 ? 0 : 1;
 } catch (const std::exception& error) {
  fprintf(stderr, "Error: %s\n", error.what());
  return 1;
 }
}
