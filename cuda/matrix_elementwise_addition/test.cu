#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <stdio.h>

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

__global__ void matrix_elementwise_addition(float* matrix_a, float* matrix_b, float* matrix_c, int N){
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    matrix_c[global_y_index * N + global_x_index] = matrix_a[global_y_index * N + global_x_index] + matrix_b[global_y_index * N + global_x_index];
}

int main(){
    int M = 1024;
    int N = 2048;
    int num_elements = M * N;
    size_t total_size = num_elements * sizeof(float);

    std::vector<float> h_matrix_a(num_elements);
    std::vector<float> h_matrix_b(num_elements);
    std::vector<float> h_matrix_c(num_elements);

    random_init_tensor(h_matrix_a);
    random_init_tensor(h_matrix_b);

    float *d_matrix_a, *d_matrix_b, *d_matrix_c;

    CHECK_CUDA(cudaMalloc(&d_matrix_a, total_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_b, total_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_c, total_size));

    CHECK_CUDA(cudaMemcpy(d_matrix_a, h_matrix_a.data(), total_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_matrix_b, h_matrix_b.data(), total_size, cudaMemcpyHostToDevice));

    dim3 block_size(16, 16);
    dim3 grid_size(N / 16, M / 16);

    matrix_elementwise_addition<<<grid_size, block_size>>>(d_matrix_a, d_matrix_b, d_matrix_c, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_matrix_c.data(), d_matrix_c, total_size, cudaMemcpyDeviceToHost));
}