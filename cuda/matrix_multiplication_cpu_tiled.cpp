#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// Returns a random float in the half-open interval [0.0f, 1.0f).
float randomFloat() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

// Cache-blocked (tiled) matrix multiplication: process TILE x TILE blocks of C,
// streaming through K in TILE-sized chunks so each block of A and B stays in cache.
void mat_mul_cpu_tiled(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b,
                       std::vector<float>& matrix_c, int M, int K, int N){
    const int TILE = 64;
    for (int i0 = 0; i0 < M; i0 += TILE){
        for (int k0 = 0; k0 < K; k0 += TILE){
            for (int j0 = 0; j0 < N; j0 += TILE){
                int i_max = std::min(i0 + TILE, M);
                int k_max = std::min(k0 + TILE, K);
                int j_max = std::min(j0 + TILE, N);
                for (int i = i0; i < i_max; i++){
                    for (int k = k0; k < k_max; k++){
                        float a = matrix_a[i*K+k];
                        for (int j = j0; j < j_max; j++){
                            matrix_c[i*N+j] += a * matrix_b[k*N+j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv){
    int M = 4096;
    int K = 4096;
    int N = 4096;
    if (argc == 4){
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }
    std::cout << "CPU tiled: C[" << M << "x" << N << "] = A[" << M << "x" << K
              << "] * B[" << K << "x" << N << "]" << std::endl;

    std::vector<float> h_matrix_a(M * K);
    std::vector<float> h_matrix_b(K * N);
    std::vector<float> h_matrix_c(M * N, 0.0f);

    for (int i = 0; i < M * K; i++) h_matrix_a[i] = randomFloat();
    for (int i = 0; i < K * N; i++) h_matrix_b[i] = randomFloat();

    auto start = std::chrono::high_resolution_clock::now();
    mat_mul_cpu_tiled(h_matrix_a, h_matrix_b, h_matrix_c, M, K, N);
    auto stop = std::chrono::high_resolution_clock::now();

    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
    double gflops = 2.0 * M * N * K / (milliseconds * 1e6);
    std::cout << "Compute time: " << milliseconds << " ms (" << gflops << " GFLOPS)" << std::endl;

    // Verify a few random entries against a direct dot product.
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
}
