// Minimal cuBLAS matrix multiplication example.
// Computes C = A * B where A is MxK, B is KxN, C is MxN (all row-major).
//
// Build: nvcc -O2 -o example example.cu -lcublas
// Run:   ./example
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

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

// Returns a random float from the standard normal distribution
// (mean 0, standard deviation 1).
float randomNormal() {
    static std::mt19937 gen(42);
    static std::normal_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

int main() {
    // Matrix dimensions: C(MxN) = A(MxK) * B(KxN)
    int M = 1024;
    int K = 512;
    int N = 2048;

    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    // Fill host matrices with random values from a standard normal
    // distribution (mean 0, standard deviation 1).
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = randomNormal();
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = randomNormal();
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
    }

    // Copy inputs to the device.
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, a_size));
    CHECK_CUDA(cudaMalloc(&d_B, b_size));
    CHECK_CUDA(cudaMalloc(&d_C, c_size));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), b_size, cudaMemcpyHostToDevice));

    // cuBLAS entry point: one handle, reused for every call.
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // cuBLAS expects column-major storage. Our row-major MxK matrix is the
    // same memory as a column-major KxM matrix (its transpose), so we compute
    // C^T = B^T * A^T by swapping operands instead of transposing data.
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

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, c_size, cudaMemcpyDeviceToHost));


    // Spot-check one element against a CPU dot product.
    int row = M / 2, col = N / 3;
    float expected = 0.0f;
    for (int i = 0; i < K; i++) {
        expected += h_A[row * K + i] * h_B[i * N + col];
    }
    float actual = h_C[row * N + col];
    if (std::fabs(expected - actual) < 1e-2f) {
        std::printf("C[%d][%d] = %f (CPU: %f) -- MATCH\n", row, col, actual, expected);
    } else {
        std::printf("C[%d][%d] = %f (CPU: %f) -- MISMATCH\n", row, col, actual, expected);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
