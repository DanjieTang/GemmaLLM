#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__device__ float reduce_sum(float local_sum, float* shared){
    int local_index = threadIdx.x;
    shared[local_index] = local_sum;
    __syncthreads();

    int stride = blockDim.x / 2;

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] += shared[local_index + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    float total = shared[0];
    __syncthreads();
    return total;
}

__global__ void layer_norm(float* input_tensor, float* output_tensor, int dim){
    extern __shared__ float shared[];
    int local_index = threadIdx.x;
    int global_index = blockIdx.x * dim;
    float epsilon = 1e-5f;

    float local_sum = 0;
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        local_sum += input_tensor[i];
    }
    float dim_sum = reduce_sum(local_sum, shared);
    float dim_avg = dim_sum / dim;

    float local_sum_square = 0;
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        float diff_square = input_tensor[i] - dim_avg;
        local_sum_square += diff_square * diff_square;
    }
    float dim_sum_square = reduce_sum(local_sum_square, shared);
    float variance = dim_sum_square / dim;

    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        float denominator = rsqrtf(variance + epsilon);
        output_tensor[i] = (input_tensor[i] - dim_avg) * denominator;
    }
}

void layer_norm_cpu(const float* input_tensor, float* output_tensor, int rows, int dim){
    float epsilon = 1e-5f;

    for (int row = 0; row < rows; row++){
        const float* input_row = input_tensor + (size_t)row * dim;
        float* output_row = output_tensor + (size_t)row * dim;

        float dim_sum = 0;
        for (int i = 0; i < dim; i++){
            dim_sum += input_row[i];
        }
        float dim_avg = dim_sum / dim;

        float dim_sum_square = 0;
        for (int i = 0; i < dim; i++){
            float diff_square = input_row[i] - dim_avg;
            dim_sum_square += diff_square * diff_square;
        }
        float variance = dim_sum_square / dim;

        float denominator = 1.0f / sqrtf(variance + epsilon);
        for (int i = 0; i < dim; i++){
            output_row[i] = (input_row[i] - dim_avg) * denominator;
        }
    }
}

float random_normal() {
    static std::mt19937 gen(42);
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

bool write_bin(const char* path, const std::vector<float>& data){
    FILE* file = fopen(path, "wb");
    if (!file){
        fprintf(stderr, "Failed to open %s for writing\n", path);
        return false;
    }
    size_t written = fwrite(data.data(), sizeof(float), data.size(), file);
    fclose(file);
    if (written != data.size()){
        fprintf(stderr, "Short write to %s\n", path);
        return false;
    }
    return true;
}

int main(){
    int batch_size = 2;
    int sequence_size = 512;
    int dim = 2048;
    int num_rows = batch_size * sequence_size;
    int num_elements = num_rows * dim;
    size_t tensor_size = num_elements * sizeof(float);

    std::vector<float> h_input(num_elements);
    for (int i = 0; i < num_elements; i++){
        h_input[i] = random_normal();
    }

    float *d_input_tensor, *d_output_tensor;

    CHECK_CUDA(cudaMalloc(&d_input_tensor, tensor_size));
    CHECK_CUDA(cudaMalloc(&d_output_tensor, tensor_size));

    CHECK_CUDA(cudaMemcpy(d_input_tensor, h_input.data(), tensor_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = num_rows;

    layer_norm<<<grid_size, block_size, dim * sizeof(float)>>>(d_input_tensor, d_output_tensor, dim);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++){
        layer_norm<<<grid_size, block_size, dim * sizeof(float)>>>(d_input_tensor, d_output_tensor, dim);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    gpu_ms /= iterations;

    std::vector<float> h_gpu_output(num_elements);
    CHECK_CUDA(cudaMemcpy(h_gpu_output.data(), d_output_tensor, tensor_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input_tensor));
    CHECK_CUDA(cudaFree(d_output_tensor));

    std::vector<float> h_cpu_output(num_elements);

    auto cpu_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++){
        layer_norm_cpu(h_input.data(), h_cpu_output.data(), num_rows, dim);
    }
    auto cpu_stop = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count() / iterations;

    float max_diff = 0;
    for (int i = 0; i < num_elements; i++){
        float diff = fabsf(h_gpu_output[i] - h_cpu_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("CUDA vs CPU:  max abs diff = %.3e  %s\n", max_diff, max_diff < 1e-4f ? "PASS" : "FAIL");

    double speedup = cpu_ms / gpu_ms;
    printf("GPU kernel:            %8.3f ms\n", gpu_ms);
    printf("CPU (single-threaded): %8.3f ms\n", cpu_ms);
    printf("Speedup (CPU/GPU):     %8.1fx\n", speedup);

    if (!write_bin("input.bin", h_input)) return 1;
    if (!write_bin("gpu_output.bin", h_gpu_output)) return 1;
    if (!write_bin("cpu_output.bin", h_cpu_output)) return 1;

    return 0;
}
