#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <random>
// #include <cmath>

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
    __syncthreads();  // make sure every thread has read shared[0] before shared is reused
    return total;
}

__global__ void layer_norm(float* input_tensor, float* output_tensor, int dim){
    extern __shared__ float shared[];
    int local_index = threadIdx.x;
    int global_index = blockIdx.x * dim;
    float epsilon = 1e-5f;
    
    // Step 1, find the mean
    float local_sum = 0;
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        local_sum += input_tensor[i];
    }
    float dim_sum = reduce_sum(local_sum, shared, dim);
    float dim_avg = dim_sum / dim;

    // Step 2, find the variance
    float local_sum_square = 0;
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        float diff_square = input_tensor[i] - dim_avg;
        local_sum_square += diff_square * diff_square;
    }
    float dim_sum_square = reduce_sum(local_sum_square, shared, dim);
    float variance = dim_sum_square / dim;

    // Step 3, do the layernorm
    for (int i = global_index + local_index; i < global_index + dim; i += blockDim.x){
        float denominator = rsqrtf(variance + epsilon);
        output_tensor[i] = (input_tensor[i] - dim_avg) * denominator;
    }
}

// Returns one random float from a standard normal distribution N(0, 1).
float random_normal() {
    static std::mt19937 gen(std::random_device{}());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

int main(){
    int batch_size = 2;
    int sequence_size = 512;
    int dim = 2048;
    int num_elements = batch_size * sequence_size * dim;
    size_t tensor_size = num_elements * sizeof(float);

    std::vector<float> h_tensor(batch_size * sequence_size * dim);
    for (int i = 0; i < h_tensor.size(); i++){
        h_tensor[i] = random_normal();
    }

    float *d_input_tensor, *d_output_tensor;

    CHECK_CUDA(cudaMalloc(&d_input_tensor, tensor_size));
    CHECK_CUDA(cudaMalloc(&d_output_tensor, tensor_size));

    CHECK_CUDA(cudaMemcpy(d_input_tensor, h_tensor.data(), tensor_size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = batch_size * sequence_size;
    
    layer_norm<<<grid_size, block_size, dim * sizeof(float)>>>(d_input_tensor, d_output_tensor, dim);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaMemcpy(h_tensor.data(), d_output_tensor, tensor_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input_tensor));
    CHECK_CUDA(cudaFree(d_output_tensor));
}