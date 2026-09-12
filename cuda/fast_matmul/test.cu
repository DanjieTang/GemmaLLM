#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

#include <cmath>
#include <random>

#include <stdio.h>

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

float randomNormal() {
    static std::mt19937 gen(42);
    static std::normal_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

int main(){
    int M = 8192;
    int K = 4096;
    int N = 2048;

    int matrix_a_element_count = M * K;
    int matrix_b_element_count = K * N;
    int matrix_c_element_count = M * N;

    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    std::vector<float> h_matrix_a(matrix_a_element_count);
    std::vector<float> h_matrix_b(matrix_b_element_count);
    std::vector<float> h_matrix_c(matrix_c_element_count);

    for (int i = 0; i < matrix_a_element_count; i++){
        h_matrix_a[i] = randomNormal();
    }
    for (int i = 0; i < matrix_b_element_count; i++){
        h_matrix_b[i] = randomNormal();
    }
    for (int i = 0; i < matrix_c_element_count; i++){
        h_matrix_c[i] = randomNormal();
    }

    float *d_matrix_a, *d_matrix_b, *d_matrix_c;

    CHECK_CUDA(cudaMalloc(&d_matrix_a, a_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_b, b_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_c, c_size));

    CHECK_CUDA(cudaMemcpy(d_matrix_a, h_matrix_a.data(), a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_matrix_b, h_matrix_b.data(), b_size, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1;
    float beta = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_matrix_b, N, d_matrix_a, K, &beta, d_matrix_c, N);

    CHECK_CUDA(cudaMemcpy(h_matrix_c.data(), d_matrix_c, c_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_matrix_a));
    CHECK_CUDA(cudaFree(d_matrix_b));
    CHECK_CUDA(cudaFree(d_matrix_c));

    printf("Finished execution");

    return 0;
}