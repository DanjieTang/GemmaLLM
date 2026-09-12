import torch

M, K, N = 8192, 2048, 4096
device = torch.device("cuda")
dtype = torch.float32  # match your CUDA kernel

# Move to GPU. pin_memory + non_blocking makes H2D transfer faster, but for
# pure matmul timing we want the data already on device before we start timing.
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)

# Warm up: PyTorch (cuBLAS) selects the best kernel on first call and caches it.
# If you skip this, the first timed iteration includes the autotuning overhead.
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()

# Timed run: average over many iterations for a stable number.
iters = 50
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(iters):
    c = a @ b
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end) / iters
flops = 2 * M * K * N  # 2 FLOPs per MAC
tflops = flops / (elapsed_ms / 1000) / 1e12

print(f"PyTorch matmul (FP32, cuBLAS):")
print(f"  shape: ({M}, {K}) @ ({K}, {N}) -> ({M}, {N})")
print(f"  avg time: {elapsed_ms:.3f} ms over {iters} iters")
print(f"  throughput: {tflops:.2f} TFLOP/s")
