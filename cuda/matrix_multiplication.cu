#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>

// Returns a random float in the half-open interval [0.0f, 1.0f).
float randomFloat() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

__global__ void mat_mul(float* d_matrix_a, float* d_matrix_b, float* d_matrix_c, int M, int K, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index / N;
    int column = index % N;

    if (index < M * N){
        float sum = 0;

        for(int i = 0; i < K; i++){
            sum += d_matrix_a[row*K+i] * d_matrix_b[i*N+column];
        }

        d_matrix_c[index] = sum;
    }
}

int main(int argc, char** argv){
    int M = 8192;
    int K = 8192;
    int N = 8192;
    if (argc == 4){
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }
    std::cout << "CUDA naive: C[" << M << "x" << N << "] = A[" << M << "x" << K
              << "] * B[" << K << "x" << N << "]" << std::endl;

    std::vector<float> h_matrix_a(M * K);
    std::vector<float> h_matrix_b(K * N);
    std::vector<float> h_matrix_c(M * N);

    for (int i = 0; i < M * K; i++) h_matrix_a[i] = randomFloat();
    for (int i = 0; i < K * N; i++) h_matrix_b[i] = randomFloat();

    float* d_matrix_a;
    float* d_matrix_b;
    float* d_matrix_c;

    cudaMalloc(&d_matrix_a, M * K * sizeof(float));
    cudaMalloc(&d_matrix_b, K * N * sizeof(float));
    cudaMalloc(&d_matrix_c, M * N * sizeof(float));

    cudaMemcpy(d_matrix_a, h_matrix_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    int threadNum = 256;
    int blockNum = (M * N + threadNum - 1) / threadNum;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul<<<blockNum, threadNum>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, K, N);
    cudaEventRecord(stop);

    cudaMemcpy(h_matrix_c.data(), d_matrix_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double gflops = 2.0 * M * N * K / (milliseconds * 1e6);
    std::cout << "Kernel time: " << milliseconds << " ms (" << gflops << " GFLOPS)" << std::endl;

    // Verify a few random entries against a direct CPU dot product.
    float max_err = 0.0f;
    for (int s = 0; s < 5; s++){
        int row = rand() % M;
        int col = rand() % N;
        float ref = 0.0f;
        for (int i = 0; i < K; i++){
            ref += h_matrix_a[row*K+i] * h_matrix_b[i*N+col];
        }
        max_err = fmaxf(max_err, fabsf(ref - h_matrix_c[row*N+col]));
    }
    std::cout << "Max sample error: " << max_err << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
}
