#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matmul(int* d_vector_a, int* d_vector_b, int* d_vector_c, int vector_size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < vector_size){
        d_vector_c[index] = d_vector_a[index] * d_vector_b[index];
    }
}

int main(){
    int vector_size = 10000;
    size_t size = vector_size * sizeof(int);

    std::vector<int> h_vector_a(vector_size);
    std::vector<int> h_vector_b(vector_size);
    std::vector<int> h_vector_c(vector_size);

    for(int i = 0; i < vector_size; i++){
        h_vector_a[i] = i;
        h_vector_b[i] = i;
    }

    int* d_vector_a;
    int* d_vector_b;
    int* d_vector_c;

    cudaMalloc(&d_vector_a, size);
    cudaMalloc(&d_vector_b, size);
    cudaMalloc(&d_vector_c, size);

    cudaMemcpy(d_vector_a, h_vector_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b.data(), size, cudaMemcpyHostToDevice);

    int threadNum = 256;
    int blockNum = (vector_size + threadNum - 1) / threadNum;

    matmul<<<blockNum, threadNum>>>(d_vector_a, d_vector_b, d_vector_c, vector_size);

    cudaMemcpy(h_vector_c.data(), d_vector_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++){
        std::cout << h_vector_c[i];
    }

    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_vector_c);
}