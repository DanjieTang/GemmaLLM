import torch
import torch.nn.functional as F

BATCH, SEQ, DIM = 2, 512, 2048
EPS = 1e-5


def load(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    return torch.frombuffer(buf, dtype=torch.float32).reshape(BATCH, SEQ, DIM)


def bench(fn, iters: int = 100) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    x = load("input.bin")
    gpu_out = load("gpu_output.bin")
    cpu_out = load("cpu_output.bin")

    ref = F.layer_norm(x, (DIM,), eps=EPS)

    for name, out in [("CUDA", gpu_out), ("CPU", cpu_out)]:
        diff = (out - ref).abs().max().item()
        match = torch.allclose(out, ref, atol=1e-4)
        print(f"{name} vs PyTorch: max abs diff = {diff:.3e}  match = {match}")

    if torch.cuda.is_available():
        x_cuda = x.to("cuda")
        torch_ms = bench(lambda: F.layer_norm(x_cuda, (DIM,), eps=EPS))
        print(f"PyTorch GPU layer_norm: {torch_ms:.3f} ms")


if __name__ == "__main__":
    main()
