#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdlib>

__global__ void sum_array_fun(const int* d_array, int* d_block_sums, int vector_length){
    int local_index = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + local_index;

    extern __shared__ int shared_memory[];

    if(global_index < vector_length){
        shared_memory[local_index] = d_array[global_index];
    }else{
        shared_memory[local_index] = 0;
    }
    __syncthreads();

    int stride = blockDim.x / 2;
    while(stride > 0){
        if(local_index < stride){
            shared_memory[local_index] += shared_memory[local_index + stride];
        }
        __syncthreads();
        stride /= 2;
    }

    if(local_index == 0){
        d_block_sums[blockIdx.x] = shared_memory[0];
    }
}

long long cpu_loop_sum(const std::vector<int>& h_array){
    long long sum = 0;
    for(size_t i = 0; i < h_array.size(); i++){
        sum += h_array[i];
    }
    return sum;
}

int main(int argc, char** argv){
    int vector_length = 100000000;
    if(argc > 1){
        vector_length = std::atoi(argv[1]);
    }
    size_t array_size = vector_length * sizeof(int);
    int cpu_runs = vector_length < 1000000 ? 100 : 5;
    int gpu_runs = vector_length < 1000000 ? 100 : 10;

    std::vector<int> h_array(vector_length);
    for(int i = 0; i < vector_length; i++){
        h_array[i] = i % 10;
    }

    long long cpu_sum = 0;
    double cpu_total_ms = 0.0;
    double cpu_best_ms = 1e18;
    for(int run = 0; run < cpu_runs; run++){
        auto start = std::chrono::high_resolution_clock::now();
        cpu_sum = cpu_loop_sum(h_array);
        auto stop = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        cpu_total_ms += ms;
        if(ms < cpu_best_ms) cpu_best_ms = ms;
    }
    double cpu_avg_ms = cpu_total_ms / cpu_runs;

    int block_size = 256;
    int grid_size = (vector_length + block_size - 1) / block_size;

    int* d_array;
    int* d_block_sums;
    cudaMalloc(&d_array, array_size);
    cudaMalloc(&d_block_sums, grid_size * sizeof(int));

    auto copy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_array, h_array.data(), array_size, cudaMemcpyHostToDevice);
    auto copy_stop = std::chrono::high_resolution_clock::now();
    double h2d_ms = std::chrono::duration<double, std::milli>(copy_stop - copy_start).count();

    sum_array_fun<<<grid_size, block_size, block_size * sizeof(int)>>>(d_array, d_block_sums, vector_length);
    cudaDeviceSynchronize();

    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    double kernel_total_ms = 0.0;
    double kernel_best_ms = 1e18;
    for(int run = 0; run < gpu_runs; run++){
        cudaEventRecord(event_start);
        sum_array_fun<<<grid_size, block_size, block_size * sizeof(int)>>>(d_array, d_block_sums, vector_length);
        cudaEventRecord(event_stop);
        cudaEventSynchronize(event_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, event_start, event_stop);
        kernel_total_ms += ms;
        if(ms < kernel_best_ms) kernel_best_ms = ms;
    }
    double kernel_avg_ms = kernel_total_ms / gpu_runs;

    std::vector<int> h_block_sums(grid_size);
    cudaMemcpy(h_block_sums.data(), d_block_sums, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    long long gpu_sum = 0;
    for(int i = 0; i < grid_size; i++){
        gpu_sum += h_block_sums[i];
    }

    std::cout << "Vector length: " << vector_length << " ints ("
              << array_size / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "CPU sum: " << cpu_sum << " | GPU sum: " << gpu_sum
              << " | match: " << (cpu_sum == gpu_sum ? "yes" : "NO") << std::endl;
    std::cout << std::endl;
    std::cout << "CPU loop:        avg " << cpu_avg_ms << " ms | best " << cpu_best_ms << " ms" << std::endl;
    std::cout << "GPU kernel:      avg " << kernel_avg_ms << " ms | best " << kernel_best_ms << " ms" << std::endl;
    std::cout << "GPU H2D copy:    " << h2d_ms << " ms" << std::endl;
    std::cout << "GPU total:       " << (h2d_ms + kernel_avg_ms) << " ms (copy + kernel)" << std::endl;
    std::cout << std::endl;
    std::cout << "Speedup (kernel only):   " << cpu_avg_ms / kernel_avg_ms << "x" << std::endl;
    std::cout << "Speedup (copy + kernel): " << cpu_avg_ms / (h2d_ms + kernel_avg_ms) << "x" << std::endl;

    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
    cudaFree(d_array);
    cudaFree(d_block_sums);
}
