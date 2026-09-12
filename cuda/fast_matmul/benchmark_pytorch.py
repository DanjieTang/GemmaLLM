import torch
import torch.nn as nn

M = K = N = 4096
ITERATIONS = 1000
WARMUP = 10


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"allow_tf32: {torch.backends.cuda.matmul.allow_tf32}\n")

    torch.manual_seed(42)
    # nn.Linear(K, N): y = x @ W^T + b, with x of shape (M, K) -> y (M, N)
    x = torch.randn(M, K, device=device)
    linear = nn.Linear(K, N, device=device)

    # --- Correctness spot check against CPU float64 ---
    with torch.no_grad():
        y = linear(x)
    x64 = x.double().cpu()
    w64 = linear.weight.double().cpu()
    b64 = linear.bias.double().cpu()
    y64 = x64 @ w64.T + b64
    spots = [(0, 0), (1, 4095), (123, 456), (2048, 2048), (4095, 4095)]
    max_diff = 0.0
    for i, j in spots:
        ref = y64[i, j].item()
        got = y[i, j].item()
        max_diff = max(max_diff, abs(ref - got))
        print(f"y[{i}][{j}]: cpu64={ref:.6f} torch={got:.6f}")
    print(f"Max |cpu64 - torch| (spot) = {max_diff:.9g}\n")

    # --- Timing ---
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(WARMUP):
            linear(x)
        torch.cuda.synchronize()

        start.record()
        for _ in range(ITERATIONS):
            linear(x)
        stop.record()
        torch.cuda.synchronize()

    total_ms = start.elapsed_time(stop)
    avg_ms = total_ms / ITERATIONS
    gflops = 2.0 * M * N * K / (avg_ms * 1e6)

    print(f"{'PyTorch nn.Linear':<25s} total: {total_ms:10.2f} ms | "
          f"avg: {avg_ms:8.4f} ms | {gflops:8.2f} GFLOP/s")


if __name__ == "__main__":
    main()
