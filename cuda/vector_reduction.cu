#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdio.h>

__global__ void sum_array_fun(int* d_array, int* sum_array, int vector_length){
    int local_index = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + local_index;
    
    extern __shared__ int shared_memory[];

    // Begin with copying memory to shared memory
    if(global_index < vector_length){
        shared_memory[local_index] = d_array[global_index];
    }else{
        shared_memory[local_index] = 0;
    }
    __syncthreads();
    

    // Actually summing things up
    int stride = blockDim.x / 2;
    while(stride > 0){
        if(local_index < stride){
            shared_memory[local_index] += shared_memory[local_index + stride];
        }
        stride /= 2;
    }

    sum_array[blockIdx.x] = shared_memory[0];
}

int main(){
    int vector_length = 10000;
    size_t array_size = vector_length * sizeof(int);

    std::vector<int> h_array(vector_length);

    for(int i = 0; i < vector_length; i++){
        h_array[i] = i;
    }

    int block_size = 256;
    int grid_size = (vector_length + block_size - 1) / block_size;

    int* d_array;
    int* d_sum_array;

    cudaMalloc(&d_array, array_size);
    cudaMemcpy(d_array, h_array.data(), array_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_sum_array, grid_size * sizeof(int));
    
    sum_array_fun<<<grid_size, block_size, block_size * sizeof(int)>>>(d_array, d_sum_array, vector_length);
    cudaDeviceSynchronize();

    std::vector<int> sum_array_results(grid_size);

    cudaMemcpy(sum_array_results.data(), d_sum_array, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for(int i = 0; i < sum_array_results.size(); i++){
        // std::cout << sum_array_results[i] << "\n";
        sum += sum_array_results[i];
    }

    std::cout << "The sum of the array is: " << sum;
}