#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

__global__ void element_wise_multiplication(int* d_vector_a, int* d_vector_b, int* d_vector_c, int length_of_vector){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < length_of_vector){
        d_vector_c[index] = d_vector_a[index] * d_vector_b[index];
    }
}

int main(){
    int length_of_vector = 10000;
    size_t size = sizeof(int) * length_of_vector;
    int* h_vector_a = (int*)malloc(size);
    int* h_vector_b = (int*)malloc(size);
    int* h_vector_c = (int*)malloc(size);

    for (int i = 0; i < 10000; i++){
        *(h_vector_a + i) = i;
        *(h_vector_b + i) = i;
    }

    int* d_vector_a;
    int* d_vector_b;
    int* d_vector_c;

    cudaMalloc(&d_vector_a, size);
    cudaMalloc(&d_vector_b, size);
    cudaMalloc(&d_vector_c, size);

    cudaMemcpy(d_vector_a, h_vector_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b, size, cudaMemcpyHostToDevice);

    int threadNum = 256;
    int blockNum = (length_of_vector + threadNum - 1) / threadNum;

    element_wise_multiplication<<<blockNum, threadNum>>>(d_vector_a, d_vector_b, d_vector_c, length_of_vector);

    cudaMemcpy(h_vector_c, d_vector_c, size, cudaMemcpyDeviceToHost);

    std::cout << h_vector_c[100];

    free(h_vector_a);
    free(h_vector_b);
    free(h_vector_c);

    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_vector_c);
}