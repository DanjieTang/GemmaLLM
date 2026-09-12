#include <cuda_runtime.h>
#include <vector>
#include <random>

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

constexpr int tileSize = 32;

__global__ void matrix_multiplication(float* A, float* B, float* C, float* bias, int M, int K, int N){
    __shared__ float tileA[tileSize][tileSize];
    __shared__ float tileB[tileSize][tileSize];

    int C_col = blockIdx.x * blockDim.x + threadIdx.x;
    int C_row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for(int i = 0; i < K; i += tileSize){
        // Step 1, load in both tile A & B
        int tileCol = i + threadIdx.x;
        int tileRow = i + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = A[C_row * K + tileCol]; // Load in tile A
        tileB[threadIdx.y][threadIdx.x] = B[tileRow * N + C_col]; // Load in tile B

        __syncthreads();

        for(int j = 0; j < tileSize; j++){
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[C_row * N + C_col] = sum + bias[C_row];
}

int main(){
    int M = 2048;
    int K = 2048;
    int N = 2048;

    int matrix_a_elements = M * K;
    int matrix_b_elements = K * N;
    int matrix_c_elements = M * N;

    size_t matrix_a_size = matrix_a_elements * sizeof(float);
    size_t matrix_b_size = matrix_b_elements * sizeof(float);
    size_t matrix_c_size = matrix_c_elements * sizeof(float);
    size_t bias_size = M * sizeof(float);

    std::vector<float> h_matrix_a(matrix_a_elements);
    std::vector<float> h_matrix_b(matrix_b_elements);
    std::vector<float> h_matrix_c(matrix_c_elements);
    std::vector<float> h_bias(M);

    random_init_tensor(h_matrix_a);
    random_init_tensor(h_matrix_b);
    random_init_tensor(h_bias);

    float *d_matrix_a, *d_matrix_b, *d_matrix_c, *d_bias;

    CHECK_CUDA(cudaMalloc(&d_matrix_a, matrix_a_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_b, matrix_b_size));
    CHECK_CUDA(cudaMalloc(&d_matrix_c, matrix_c_size));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_size));

    CHECK_CUDA(cudaMemcpy(d_matrix_a, h_matrix_a.data(), matrix_a_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_matrix_b, h_matrix_b.data(), matrix_b_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), bias_size, cudaMemcpyHostToDevice));

    dim3 blockSize(tileSize, tileSize);
    dim3 gridSize(N / tileSize, M / tileSize);

    matrix_multiplication<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_matrix_c, d_bias, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_matrix_c.data(), d_matrix_c, matrix_c_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_matrix_a));
    CHECK_CUDA(cudaFree(d_matrix_b));
    CHECK_CUDA(cudaFree(d_matrix_c));

    return 0;
}