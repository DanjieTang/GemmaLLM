#include <cuda_runtime.h>
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

constexpr int tileSize = 32;

__global__ void matrix_multiplication(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float tileA[tileSize][tileSize];
    __shared__ float tileB[tileSize][tileSize];

    int C_col = blockIdx.x * blockDim.x + threadIdx.x;
    int C_row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for(int i = 0; i < K; i += tileSize){
        // Step 1, load in both tile A & B
        int tileCol = i + threadIdx.x;
        int tileRow = i + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = A[C_row * K + tileCol]; // Load in tile A
        tileB[threadIdx.y][threadIdx.x] = B[tileRow * N + C_col]; // Load in tile B

        __syncthreads();

        for(int j = 0; j < tileSize; j++){
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[C_row * N + C_col] = sum;
}

void verify_result(const std::vector<float>& h_matrix_a,
                   const std::vector<float>& h_matrix_b,
                   const std::vector<float>& h_matrix_c,
                   int M, int K, int N) {
    const float tolerance = 1e-2f;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += h_matrix_a[row * K + k] * h_matrix_b[k * N + col];
            }
            float actual = h_matrix_c[row * N + col];
            if (std::fabs(expected - actual) > tolerance) {
                std::cout << "Matrix multiplication is WRONG at (" << row
                          << ", " << col << ") - expected " << expected
                          << ", got " << actual << "\n";
                return;
            }
        }
    }
    std::cout << "Matrix multiplication is CORRECT\n";
}

int main(){
    int M = 8192;
    int K = 8192;
    int N = 8192;

    size_t matrix_a_size = M * K * sizeof(float);
    size_t matrix_b_size = K * N * sizeof(float);
    size_t matrix_c_size = M * N * sizeof(float);

    std::vector<float> h_matrix_a(M * K);
    std::vector<float> h_matrix_b(K * N);
    std::vector<float> h_matrix_c(M * N);

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

    dim3 blockSize(tileSize, tileSize);
    dim3 gridSize(N / tileSize, M / tileSize);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < 10; i++){
        matrix_multiplication<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, K, N);
        CHECK_KERNEL();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU kernel time: " << milliseconds << " ms\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaMemcpy(h_matrix_c.data(), d_matrix_c, matrix_c_size, cudaMemcpyDeviceToHost));

    // verify_result(h_matrix_a, h_matrix_b, h_matrix_c, M, K, N);

    CHECK_CUDA(cudaFree(d_matrix_a));
    CHECK_CUDA(cudaFree(d_matrix_b));
    CHECK_CUDA(cudaFree(d_matrix_c));

    return 0;
}