#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <stdio.h>
#include <string.h>
#include <math_constants.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

float random_normal() {
    static std::mt19937 gen(std::random_device{}());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

void random_init_tensor(std::vector<float>& tensor){
    for (size_t i = 0; i < tensor.size(); i++){
        tensor[i] = random_normal();
    }
}

__global__ void matrix_scaler_multiplication(float* matrix_a, float* matrix_b, int N){
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_x_index > global_y_index){
        matrix_b[global_y_index * N + global_x_index] = -CUDART_INF_F;
    } else {
        matrix_b[global_y_index * N + global_x_index] = matrix_a[global_y_index * N + global_x_index];
    }
}

bool verify_causal_mask(const std::vector<float>& h_matrix_a,
                        const std::vector<float>& h_matrix_b,
                        int N, int M) {
    int mismatch_count = 0;
    int first_y = 0, first_x = 0;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            float expected = (x > y) ? -INFINITY : h_matrix_a[y * N + x];
            float actual = h_matrix_b[y * N + x];
            if (memcmp(&expected, &actual, sizeof(float)) != 0) {
                if (mismatch_count < 10) {
                    printf("Mismatch at (y=%d, x=%d): expected=%f (%a), actual=%f (%a)\n",
                           y, x, expected, expected, actual, actual);
                }
                if (mismatch_count == 0) { first_y = y; first_x = x; }
                mismatch_count++;
            }
        }
    }
    if (mismatch_count == 0) {
        printf("PASS: causal mask matches reference\n");
        return true;
    }
    printf("FAIL: %d mismatches, first at (y=%d, x=%d)\n", mismatch_count, first_y, first_x);
    return false;
}

int main(){
    int M = 1024;
    int N = 1024;
    int num_elements = M * N;
    size_t total_size = num_elements * sizeof(float);

    std::vector<float> h_matrix_a(num_elements);
    std::vector<float> h_matrix_b(num_elements);

    random_init_tensor(h_matrix_a);

    float *d_matrix_a, *d_matrix_b;

    CHECK_CUDA(cudaMalloc(&d_matrix_a, total_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_b, total_size));

    CHECK_CUDA(cudaMemcpy(d_matrix_a, h_matrix_a.data(), total_size, cudaMemcpyHostToDevice));

    dim3 block_size(16, 16);
    dim3 grid_size(N / 16, M / 16);

    matrix_scaler_multiplication<<<grid_size, block_size>>>(d_matrix_a, d_matrix_b, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_matrix_b.data(), d_matrix_b, total_size, cudaMemcpyDeviceToHost));

    verify_causal_mask(h_matrix_a, h_matrix_b, N, M);

    CHECK_CUDA(cudaFree(d_matrix_a));
    CHECK_CUDA(cudaFree(d_matrix_b));
    return 0;
}