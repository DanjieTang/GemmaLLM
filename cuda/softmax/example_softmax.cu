// example_softmax.cu
//
// A minimal, numerically-stable softmax in CUDA.
//
// softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
//
// Strategy: one thread block per row.
//   1. Block reduction to find the row max (for numerical stability).
//   2. Each thread computes exp(x_i - max) for its elements.
//   3. Block reduction to sum those exponentials.
//   4. Each thread divides its elements by the sum.
//
// Build:  nvcc -O2 example_softmax.cu -o example_softmax
// Run:    ./example_softmax

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Block-wide reduction using shared memory. The threads of a block each
// contribute one value; after the call, shared[0] holds the reduced result.
// `op` is either max or sum, selected by the template parameter.
enum class ReduceOp { MAX, SUM };

template <ReduceOp op>
__device__ float blockReduce(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Tree reduction: each round halves the number of active threads.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (op == ReduceOp::MAX)
                shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
            else
                shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];  // valid for all threads after final __syncthreads
}

// One block handles one row of length n. Rows may be longer than the block,
// so each thread strides over the row with step = blockDim.x.
__global__ void softmaxKernel(const float* __restrict__ input,
                              float* __restrict__ output, int n) {
    extern __shared__ float shared[];  // blockDim.x floats for reductions

    const float* row_in = input + blockIdx.x * (size_t)n;
    float* row_out = output + blockIdx.x * (size_t)n;
    int tid = threadIdx.x;

    // 1. Find the row maximum. Threads with no element contribute -inf.
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);
    float row_max = blockReduce<ReduceOp::MAX>(local_max, shared);

    // 2. Compute exp(x - max) and accumulate the local partial sum.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;  // stash un-normalized value; normalize in step 4
        local_sum += e;
    }
    float row_sum = blockReduce<ReduceOp::SUM>(local_sum, shared);

    // 3. Normalize.
    for (int i = tid; i < n; i += blockDim.x)
        row_out[i] /= row_sum;
}

int main() {
    const int rows = 4;
    const int cols = 8;  // try 1024+ to see the strided loops kick in
    const size_t bytes = (size_t)rows * cols * sizeof(float);

    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out(rows * cols);

    srand(42);
    for (int i = 0; i < rows * cols; ++i)
        h_in[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;  // values in [-5, 5]

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    softmaxKernel<<<rows, threads, shared_bytes>>>(d_in, d_out, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Print results and check that each row sums to 1.
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        printf("row %d: ", r);
        for (int c = 0; c < cols; ++c) {
            printf("%.4f ", h_out[r * cols + c]);
            sum += h_out[r * cols + c];
        }
        printf("| sum = %.6f\n", sum);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
