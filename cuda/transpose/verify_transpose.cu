#include <cuda_runtime.h>
#include <vector>
#include <random>

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

__global__ void transpose(float* original_tensor, float* transposed_tensor, int M, int N){
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    transposed_tensor[global_y_index * M + global_x_index] = original_tensor[global_x_index * N + global_y_index];
}

int verify_transpose(const std::vector<float>& original, const std::vector<float>& transposed, int M, int N) {
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float expected = original[j * N + i];
            float actual = transposed[i * M + j];
            if (actual != expected) {
                if (mismatches < 10) {
                    fprintf(stderr, "Mismatch at (i=%d, j=%d): expected %f, got %f\n", i, j, expected, actual);
                }
                mismatches++;
            }
        }
    }
    return mismatches;
}

int main(){
    int M = 513;
    int N = 1024;

    size_t tensor_size = M * N * sizeof(float);

    std::vector<float> h_original_tensor(M * N);
    std::vector<float> h_transposed_tensor(N * M);

    random_init_tensor(h_original_tensor);

    float *d_original_tensor, *d_transposed_tensor;

    CHECK_CUDA(cudaMalloc(&d_original_tensor, tensor_size));
    CHECK_CUDA(cudaMalloc(&d_transposed_tensor, tensor_size));

    CHECK_CUDA(cudaMemcpy(d_original_tensor, h_original_tensor.data(), tensor_size, cudaMemcpyHostToDevice));

    int block_length = 16;
    dim3 block_size(block_length, block_length);
    dim3 grid_size(M / block_length, N / block_length);

    transpose<<<grid_size, block_size>>>(d_original_tensor, d_transposed_tensor, M, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_transposed_tensor.data(), d_transposed_tensor, tensor_size, cudaMemcpyDeviceToHost));

    int mismatches = verify_transpose(h_original_tensor, h_transposed_tensor, M, N);
    if (mismatches > 0) {
        printf("FAILED: %d errors out of %d elements\n", mismatches, M * N);
        return EXIT_FAILURE;
    }
    printf("PASSED: transpose verified (%d x %d)\n", M, N);
    return EXIT_SUCCESS;
}