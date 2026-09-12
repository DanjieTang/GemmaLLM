#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_element(float* d_matrix_a, float* d_matrix_b, float* d_matrix_c, int total_element){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < total_element){
        d_matrix_c[index] = d_matrix_a[index] * d_matrix_b[index];
    }
}

int main(){
    int M = 1000;
    int N = 1000;
    int total_element = M * N;
    size_t size = total_element * sizeof(float);

    std::vector<float> h_matrix_a(total_element);
    std::vector<float> h_matrix_b(total_element);
    std::vector<float> h_matrix_c(total_element);

    for(int i = 0; i < total_element; i++){
        h_matrix_a[i] = i;
        h_matrix_b[i] = i;
    }

    float* d_matrix_a;
    float* d_matrix_b;
    float* d_matrix_c;

    cudaMalloc(&d_matrix_a, size);
    cudaMalloc(&d_matrix_b, size);
    cudaMalloc(&d_matrix_c, size);

    cudaMemcpy(d_matrix_a, h_matrix_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b.data(), size, cudaMemcpyHostToDevice);

    int threadNum = 256;
    int blockNum = (total_element + threadNum - 1) / threadNum;

    matrix_element<<<blockNum, threadNum>>>(d_matrix_a, d_matrix_b, d_matrix_c, total_element);

    cudaMemcpy(h_matrix_c.data(), d_matrix_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++){
        std::cout << h_matrix_c[i];
    }

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
}