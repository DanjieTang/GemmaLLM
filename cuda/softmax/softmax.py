import torch
from torch.utils.cpp_extension import load_inline

# Shapes to benchmark: (batch_size, dim)
SHAPES = [
    (20, 2048),      # original size from softmax.cu
    (4096, 2048),
    (1024, 8192),
    (256, 32768),
]
NUM_RUNS = 1000
WARMUP_RUNS = 100

cuda_src = r"""
#include <cuda_runtime.h>

__device__ float maxReduce(float local_max, float* shared){
    int local_index = threadIdx.x;
    int stride = blockDim.x / 2;

    shared[local_index] = local_max;
    __syncthreads();

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] = fmaxf(shared[local_index], shared[local_index + stride]);
        }
        stride /= 2;
        __syncthreads();
    }

    return shared[0];
}

__device__ float sumReduce(float local_sum, float* shared){
    int local_index = threadIdx.x;
    int stride = blockDim.x / 2;

    shared[local_index] = local_sum;
    __syncthreads();

    while(stride > 0){
        if (local_index < stride){
            shared[local_index] += shared[local_index + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    return shared[0];
}

__global__ void softmaxKernel(float* input_tensor, float* output_tensor, int dim){
    int batch_index = blockIdx.x * dim;
    extern __shared__ float shared[];

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        local_max = fmaxf(local_max, input_tensor[batch_index + i]);
    }
    float batch_max = maxReduce(local_max, shared);

    float local_sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        output_tensor[batch_index + i] = expf(input_tensor[batch_index + i] - batch_max);
        local_sum += output_tensor[batch_index + i];
    }
    float batch_sum = sumReduce(local_sum, shared);

    for (int i = threadIdx.x; i < dim; i += blockDim.x){
        output_tensor[batch_index + i] /= batch_sum;
    }
}

void custom_softmax(torch::Tensor input, torch::Tensor output, int batch_size, int dim){
    int block_size = 256;
    softmaxKernel<<<batch_size, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), dim);
}
"""

cpp_src = "void custom_softmax(torch::Tensor input, torch::Tensor output, int batch_size, int dim);"

module = load_inline(
    name="custom_softmax",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["custom_softmax"],
    verbose=False,
)


def benchmark(fn, num_runs):
    # Warmup
    for _ in range(WARMUP_RUNS):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_runs  # average ms per run


def main():
    device = "cuda"

    print(f"{NUM_RUNS} runs each (+{WARMUP_RUNS} warmup)\n")
    print(f"{'Shape':>15} | {'Custom (us)':>12} | {'PyTorch (us)':>12} | {'Ratio':>6}")
    print("-" * 55)

    for batch_size, dim in SHAPES:
        x = torch.randn(batch_size, dim, device=device)
        out = torch.empty_like(x)

        # Correctness sanity check before timing
        module.custom_softmax(x, out, batch_size, dim)
        ref = torch.softmax(x, dim=-1)
        assert torch.allclose(out, ref, atol=1e-5), \
            f"results differ at shape [{batch_size}, {dim}]"

        custom_ms = benchmark(lambda: module.custom_softmax(x, out, batch_size, dim), NUM_RUNS)
        pytorch_ms = benchmark(lambda: torch.softmax(x, dim=-1), NUM_RUNS)

        shape = f"[{batch_size}, {dim}]"
        print(f"{shape:>15} | {custom_ms * 1000:12.2f} | {pytorch_ms * 1000:12.2f} | "
              f"{pytorch_ms / custom_ms:5.2f}x")


if __name__ == "__main__":
    main()
