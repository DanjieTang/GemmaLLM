#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <string.h>
#include <math.h>
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

__global__ void matrix_scaler_multiplication(float* matrix_a, float* matrix_b, float scaler, int N){
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    matrix_b[global_y_index * N + global_x_index] = matrix_a[global_y_index * N + global_x_index] * scaler;
}

bool verify_result(const std::vector<float>& h_matrix_a,
                   const std::vector<float>& h_matrix_b,
                   float scaler, float tolerance = 1e-4f) {
    if (h_matrix_a.size() != h_matrix_b.size()) {
        fprintf(stderr, "Size mismatch: %zu vs %zu\n", h_matrix_a.size(), h_matrix_b.size());
        return false;
    }
    for (size_t i = 0; i < h_matrix_a.size(); i++) {
        float expected = h_matrix_a[i] * scaler;
        if (fabsf(expected - h_matrix_b[i]) > tolerance) {
            fprintf(stderr, "Mismatch at index %zu: expected %f, got %f\n",
                    i, expected, h_matrix_b[i]);
            return false;
        }
    }
    printf("Verification passed: all %zu elements match\n", h_matrix_a.size());
    return true;
}

bool verify_result_advanced(const std::vector<float>& h_matrix_a,
                            const std::vector<float>& h_matrix_b,
                            float scaler, float tolerance = 1e-4f,
                            float rel_tolerance = 1e-4f) {
    if (h_matrix_a.size() != h_matrix_b.size()) {
        fprintf(stderr, "Size mismatch: %zu vs %zu\n", h_matrix_a.size(), h_matrix_b.size());
        return false;
    }

    size_t num_elements = h_matrix_a.size();
    size_t mismatch_count = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float sum_abs_error = 0.0f;

    for (size_t i = 0; i < num_elements; i++) {
        float expected = h_matrix_a[i] * scaler;
        float actual = h_matrix_b[i];
        float abs_error = fabsf(expected - actual);
        float rel_error = fabsf(expected) > 0.0f ? abs_error / fabsf(expected) : 0.0f;

        sum_abs_error += abs_error;
        max_abs_error = fmaxf(max_abs_error, abs_error);
        max_rel_error = fmaxf(max_rel_error, rel_error);

        if (abs_error > tolerance && rel_error > rel_tolerance) {
            if (mismatch_count < 5) {
                fprintf(stderr, "Mismatch at index %zu: expected %f, got %f\n",
                        i, expected, actual);
            }
            mismatch_count++;
        }
    }

    printf("Elements: %zu | mismatches: %zu | max abs error: %e | max rel error: %e | mean abs error: %e\n",
           num_elements, mismatch_count, max_abs_error, max_rel_error, sum_abs_error / num_elements);

    if (mismatch_count == 0) {
        printf("Verification passed\n");
        return true;
    }
    fprintf(stderr, "%zu mismatches detected\n", mismatch_count);
    return false;
}

int main(){
    int M = 1024;
    int N = 2048;
    int num_elements = M * N;
    size_t total_size = num_elements * sizeof(float);

    float scaler = 2;

    std::vector<float> h_matrix_a(num_elements);
    std::vector<float> h_matrix_b(num_elements);

    random_init_tensor(h_matrix_a);

    float *d_matrix_a, *d_matrix_b;

    CHECK_CUDA(cudaMalloc(&d_matrix_a, total_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_b, total_size));

    CHECK_CUDA(cudaMemcpy(d_matrix_a, h_matrix_a.data(), total_size, cudaMemcpyHostToDevice));

    dim3 block_size(16, 16);
    dim3 grid_size(N / 16, M / 16);

    matrix_scaler_multiplication<<<grid_size, block_size>>>(d_matrix_a, d_matrix_b, scaler, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_matrix_b.data(), d_matrix_b, total_size, cudaMemcpyDeviceToHost));

    bool success = verify_result(h_matrix_a, h_matrix_b, scaler);
    if (!success) {
        fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}