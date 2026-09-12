// Example: __device__ functions in CUDA
//
// A __device__ function runs on the GPU and can ONLY be called from
// GPU code (a __global__ kernel or another __device__ function).
// It cannot be called from host (CPU) code.
//
// Build:  nvcc device_function.cu -o device_function
// Run:    ./device_function

#include <cstdio>

// A __device__ function: callable only from device code.
// Marking it __inline__ is optional; the compiler usually inlines
// small device functions automatically.
__device__ float square(float x)
{
    return x * x;
}

// Device functions can call other device functions.
__device__ float sum_of_squares(float a, float b)
{
    return square(a) + square(b);
}

// __host__ __device__ means the compiler generates TWO versions:
// one callable from CPU code, one from GPU code.
__host__ __device__ float clamp01(float x)
{
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

__global__ void kernel(const float *a, const float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = clamp01(sum_of_squares(a[i], b[i]));
}

int main()
{
    const int n = 8;
    float h_a[n], h_b[n], h_out[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = 0.1f * i;
        h_b[i] = 0.2f * i;
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);

    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("a=%.2f b=%.2f -> clamp01(a^2 + b^2) = %.4f\n",
               h_a[i], h_b[i], h_out[i]);

    // clamp01 also works on the host because of __host__ __device__:
    printf("host call: clamp01(1.5) = %.1f\n", clamp01(1.5f));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
