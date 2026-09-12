#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>

// Returns a random float in the half-open interval [0.0f, 1.0f).
float randomFloat() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

void mat_mul_cpu_naive(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b,
                       std::vector<float>& matrix_c, int M, int K, int N){
    for (int row = 0; row < M; row++){
        for (int col = 0; col < N; col++){
            float sum = 0;
            for (int i = 0; i < K; i++){
                sum += matrix_a[row*K+i] * matrix_b[i*N+col];
            }
            matrix_c[row*N+col] = sum;
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
    std::cout << "CPU naive: C[" << M << "x" << N << "] = A[" << M << "x" << K
              << "] * B[" << K << "x" << N << "]" << std::endl;

    std::vector<float> h_matrix_a(M * K);
    std::vector<float> h_matrix_b(K * N);
    std::vector<float> h_matrix_c(M * N);

    for (int i = 0; i < M * K; i++) h_matrix_a[i] = randomFloat();
    for (int i = 0; i < K * N; i++) h_matrix_b[i] = randomFloat();

    auto start = std::chrono::high_resolution_clock::now();
    mat_mul_cpu_naive(h_matrix_a, h_matrix_b, h_matrix_c, M, K, N);
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
