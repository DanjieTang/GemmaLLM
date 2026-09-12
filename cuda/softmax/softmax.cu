#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <random>
#include <cmath>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Returns one random float from a standard normal distribution N(0, 1).
float random_normal() {
    static std::mt19937 gen(std::random_device{}());
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

__device__ float maxReduce(float local_max, float* shared){
    int local_index = threadIdx.x;
    int stride = blockDim.x / 2;
    
    // Load shared memory
    shared[local_index] = local_max;
    __syncthreads();

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] = fmaxf(shared[local_index], shared[local_index + stride]);
        }
        stride /= 2;
        __syncthreads();
    }

    return shared[0];
}

__device__ float sumReduce(float local_sum, float* shared){
    int local_index = threadIdx.x;
    int stride = blockDim.x / 2;
    
    // Load shared memory
    shared[local_index] = local_sum;
    __syncthreads();

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] += shared[local_index + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    return shared[0];
}

__global__ void softmaxKernel(float* input_tensor, float* output_tensor, int dim){
    int batch_index = blockIdx.x * dim;
    extern __shared__ float shared[];

    // Step 1, find the max of each datapoint(each slice in the batch)
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        local_max = fmaxf(local_max, input_tensor[batch_index + i]);
    }
    float batch_max = maxReduce(local_max, shared);

    // Step 2
    float local_sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        output_tensor[batch_index + i] = expf(input_tensor[batch_index + i] - batch_max);
        local_sum += output_tensor[batch_index + i];
    }
    float batch_sum = sumReduce(local_sum, shared);

    // Step 3
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        output_tensor[batch_index + i] /= batch_sum;
    }
}

int main(){
    int batch_size = 20;
    int dim = 2048;
    size_t size = batch_size * dim * sizeof(float);

    std::vector<float> h_input_tensor(batch_size * dim);
    std::vector<float> h_output_tensor(batch_size * dim);

    for (int i = 0; i < batch_size * dim; i++){
        h_input_tensor[i] = random_normal();
    }

    float *d_input_tensor, *d_output_tensor;

    CHECK_CUDA(cudaMalloc(&d_input_tensor, size));
    CHECK_CUDA(cudaMalloc(&d_output_tensor, size));

    CHECK_CUDA(cudaMemcpy(d_input_tensor, h_input_tensor.data(), size, cudaMemcpyHostToDevice));

    int block_size = 256;
    
    softmaxKernel<<<batch_size, block_size, block_size*sizeof(float)>>>(d_input_tensor, d_output_tensor, dim);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_output_tensor.data(), d_output_tensor, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input_tensor));
    CHECK_CUDA(cudaFree(d_output_tensor));
}