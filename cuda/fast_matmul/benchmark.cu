#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int TILE_SIZE = 16;
constexpr int ITERATIONS = 1000;
constexpr int WARMUP = 10;

#define CHECK_CUDA(call) do {                                          \
    cudaError_t _err = (call);                                         \
    if (_err != cudaSuccess) {                                         \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(_err));    \
        std::exit(EXIT_FAILURE);                                       \
    }                                                                  \
} while (0)

#define CHECK_CUBLAS(call) do {                                        \
    cublasStatus_t _status = (call);                                   \
    if (_status != CUBLAS_STATUS_SUCCESS) {                            \
        std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",     \
                     __FILE__, __LINE__, (int)_status);                \
        std::exit(EXIT_FAILURE);                                       \
    }                                                                  \
} while (0)

// Same tiling strategy as the `linear` kernel in small_nn/model.cu,
// minus the bias: C[y][x] = sum_k A[y][k] * B[k][x]  (all row-major).
__global__ void tiledMatmul(const float* A, const float* B, float* C,
                            int M, int K, int N) {
    int local_x_index = threadIdx.x;
    int local_y_index = threadIdx.y;
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int num_tiles = K / TILE_SIZE;

    float sum = 0.0f;
    for (int i = 0; i < num_tiles; i++) {
        tile_a[local_y_index][local_x_index] =
            A[global_y_index * K + i * TILE_SIZE + local_x_index];
        tile_b[local_y_index][local_x_index] =
            B[(local_y_index + i * TILE_SIZE) * N + global_x_index];

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_a[local_y_index][j] * tile_b[j][local_x_index];
        }
        __syncthreads();
    }

    C[global_y_index * N + global_x_index] = sum;
}

float randomNormal(std::mt19937& gen) {
    static std::normal_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

int main(int argc, char** argv) {
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;

    size_t a_size = (size_t)M * K * sizeof(float);
    size_t b_size = (size_t)K * N * sizeof(float);
    size_t c_size = (size_t)M * N * sizeof(float);

    std::mt19937 gen(42);
    std::vector<float> h_a((size_t)M * K);
    std::vector<float> h_b((size_t)K * N);
    for (auto& v : h_a) v = randomNormal(gen);
    for (auto& v : h_b) v = randomNormal(gen);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, a_size));
    CHECK_CUDA(cudaMalloc(&d_b, b_size));
    CHECK_CUDA(cudaMalloc(&d_c, c_size));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_size, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(N / TILE_SIZE, M / TILE_SIZE);

    // cuBLAS is column-major; this call computes the row-major C = A * B,
    // same convention as test.cu.
    auto runCublas = [&]() {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    };
    auto runTiled = [&]() {
        tiledMatmul<<<grid, block>>>(d_a, d_b, d_c, M, K, N);
    };

    // --- Correctness check: compare both kernels against each other ---
    runCublas();
    std::vector<float> h_cublas((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(h_cublas.data(), d_c, c_size, cudaMemcpyDeviceToHost));

    runTiled();
    CHECK_CUDA(cudaGetLastError());
    std::vector<float> h_tiled((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(h_tiled.data(), d_c, c_size, cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (size_t i = 0; i < h_cublas.size(); i++) {
        max_diff = std::max(max_diff,
                            (double)std::fabs(h_cublas[i] - h_tiled[i]));
    }
    std::printf("Max |cublas - tiled| = %.9g\n", max_diff);

    // CPU double-precision spot check on a few entries.
    double max_cpu_diff = 0.0;
    const int spots[][2] = {{0, 0}, {1, 4095}, {123, 456}, {2048, 2048}, {4095, 4095}};
    for (auto& s : spots) {
        int y = s[0], x = s[1];
        double ref = 0.0;
        for (int k = 0; k < K; k++) {
            ref += (double)h_a[(size_t)y * K + k] * (double)h_b[(size_t)k * N + x];
        }
        double d = std::fabs(ref - (double)h_tiled[(size_t)y * N + x]);
        max_cpu_diff = std::max(max_cpu_diff, d);
        std::printf("C[%d][%d]: cpu=%.6f tiled=%.6f cublas=%.6f\n",
                    y, x, ref, (double)h_tiled[(size_t)y * N + x],
                    (double)h_cublas[(size_t)y * N + x]);
    }
    std::printf("Max |cpu - tiled| (spot) = %.9g\n\n", max_cpu_diff);

    if (argc > 1) return 0;  // check-only mode: skip timing

    // --- Timing ---
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto benchmark = [&](const char* name, auto run) {
        for (int i = 0; i < WARMUP; i++) run();
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERATIONS; i++) run();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / ITERATIONS;
        double gflops = 2.0 * M * N * K / (avg_ms * 1e6);

        std::printf("%-25s total: %10.2f ms | avg: %8.4f ms | %8.2f GFLOP/s\n",
                    name, total_ms, avg_ms, gflops);
        return avg_ms;
    };

    float tiled_ms = benchmark("Tiled (custom)", runTiled);
    float cublas_ms = benchmark("cuBLAS sgemm", runCublas);

    std::printf("\ncuBLAS is %.2fx faster than the tiled kernel\n",
                tiled_ms / cublas_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return 0;
}
