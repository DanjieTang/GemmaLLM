#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

#define CHECK_KERNEL() do {                                            \
    cudaError_t _err = cudaGetLastError();                             \
    if (_err != cudaSuccess) {                                         \
        std::fprintf(stderr, "Kernel launch failed at %s:%d: %s\n",    \
                     __FILE__, __LINE__, cudaGetErrorString(_err));    \
        std::exit(EXIT_FAILURE);                                       \
    }                                                                  \
} while (0)

// Returns a random float in the half-open interval [0.0f, 1.0f).
float randomFloat() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

// The API: C = A * B using cuBLAS (what PyTorch calls under the hood).
// A, B, C are device pointers in row-major layout: A is MxK, B is KxN, C is MxN.
//
// cuBLAS assumes column-major storage. A row-major MxK matrix is the same
// memory as a column-major KxM matrix (its transpose), so instead of
// transposing data we compute C^T = B^T * A^T by swapping the operands.
void fast_mat_mul(cublasHandle_t handle,
                  const float* d_A, const float* d_B, float* d_C,
                  int M, int K, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,   // B^T viewed as column-major NxK
                             d_A, K,   // A^T viewed as column-major KxM
                             &beta,
                             d_C, N)); // C^T viewed as column-major NxM
}

constexpr int tileSize = 32;

// Hand-written tiled kernel for comparison.
__global__ void matrix_multiplication(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float tileA[tileSize][tileSize];
    __shared__ float tileB[tileSize][tileSize];

    int C_col = blockIdx.x * blockDim.x + threadIdx.x;
    int C_row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for(int i = 0; i < K; i += tileSize){
        int tileCol = i + threadIdx.x;
        int tileRow = i + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = A[C_row * K + tileCol];
        tileB[threadIdx.y][threadIdx.x] = B[tileRow * N + C_col];

        __syncthreads();

        for(int j = 0; j < tileSize; j++){
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[C_row * N + C_col] = sum;
}

void verify_result(const std::vector<float>& h_expected,
                   const std::vector<float>& h_actual,
                   int M, int N) {
    const float tolerance = 1e-1f;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            if (std::fabs(h_expected[row * N + col] - h_actual[row * N + col]) > tolerance) {
                std::cout << "Results DIFFER at (" << row << ", " << col
                          << ") - expected " << h_expected[row * N + col]
                          << ", got " << h_actual[row * N + col] << "\n";
                return;
            }
        }
    }
    std::cout << "cuBLAS and tiled kernel results MATCH\n";
}

int main(){
    int M = 8192;
    int K = 8192;
    int N = 8192;
    int iterations = 10;

    size_t matrix_a_size = M * K * sizeof(float);
    size_t matrix_b_size = K * N * sizeof(float);
    size_t matrix_c_size = M * N * sizeof(float);

    std::vector<float> h_matrix_a(M * K);
    std::vector<float> h_matrix_b(K * N);
    std::vector<float> h_matrix_c(M * N);
    std::vector<float> h_matrix_c_ref(M * N);

    for(int i = 0; i < M * K; i++){
        h_matrix_a[i] = randomFloat();
    }

    for(int i = 0; i < K * N; i++){
        h_matrix_b[i] = randomFloat();
    }

    float* d_matrix_a;
    float* d_matrix_b;
    float* d_matrix_c;

    CHECK_CUDA(cudaMalloc(&d_matrix_a, matrix_a_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_b, matrix_b_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_c, matrix_c_size));

    CHECK_CUDA(cudaMemcpy(d_matrix_a, h_matrix_a.data(), matrix_a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_matrix_b, h_matrix_b.data(), matrix_b_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // --- cuBLAS ---
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Warm-up run (also produces the result we verify against below).
    fast_mat_mul(handle, d_matrix_a, d_matrix_b, d_matrix_c, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_matrix_c_ref.data(), d_matrix_c, matrix_c_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < iterations; i++){
        fast_mat_mul(handle, d_matrix_a, d_matrix_b, d_matrix_c, M, K, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cublas_milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_milliseconds, start, stop));
    std::cout << "cuBLAS time: " << cublas_milliseconds << " ms ("
              << cublas_milliseconds / iterations << " ms per run)\n";

    // --- Hand-written tiled kernel ---
    dim3 blockSize(tileSize, tileSize);
    dim3 gridSize(N / tileSize, M / tileSize);

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < iterations; i++){
        matrix_multiplication<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, K, N);
        CHECK_KERNEL();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float tiled_milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&tiled_milliseconds, start, stop));
    std::cout << "Tiled kernel time: " << tiled_milliseconds << " ms ("
              << tiled_milliseconds / iterations << " ms per run)\n";

    std::cout << "cuBLAS speedup: " << tiled_milliseconds / cublas_milliseconds << "x\n";

    CHECK_CUDA(cudaMemcpy(h_matrix_c.data(), d_matrix_c, matrix_c_size, cudaMemcpyDeviceToHost));
    verify_result(h_matrix_c_ref, h_matrix_c, M, N);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_matrix_a));
    CHECK_CUDA(cudaFree(d_matrix_b));
    CHECK_CUDA(cudaFree(d_matrix_c));

    return 0;
}
