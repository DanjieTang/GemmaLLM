"""Benchmark the PyTorch model from model.py over repeated forward passes.

Loads the same input/weights that model.py saved to tensors/, runs N forward
passes, and reports GPU time (CUDA events) and wall-clock time per iteration.

Usage:
    uv run benchmark_pytorch.py [--iterations 1000] [--warmup 100] [--device cuda|cpu]
"""

import argparse
import contextlib
import io
import time

import numpy as np
import torch
import torch.nn.functional as F

# model.py runs its forward pass and saves tensors at import time; silence it.
with contextlib.redirect_stdout(io.StringIO()):
    from model import Classifier

BATCH_SIZE = 32
INPUT_SIZE = 4096
OUTPUT_SIZE = 16
HIDDEN_DIM = 256


def load_input(path: str) -> torch.Tensor:
    array = np.fromfile(path, dtype=np.float32)
    # Saved as [features, batch] to match model.cu's layout; flip back.
    return torch.from_numpy(array.reshape(INPUT_SIZE, BATCH_SIZE).T.copy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    # Same seed as model.py so the weights are identical.
    torch.manual_seed(42)
    model = Classifier(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_DIM).to(device).eval()
    input_tensor = load_input("tensors/input.bin").to(device)

    def sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.warmup):
            model(input_tensor)
        sync()

        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(args.iterations):
                model(input_tensor)
            end_event.record()
            sync()
            gpu_ms = start_event.elapsed_time(end_event)
        else:
            gpu_ms = None

        start = time.perf_counter()
        for _ in range(args.iterations):
            model(input_tensor)
        sync()
        wall_ms = (time.perf_counter() - start) * 1000

    print(f"device:              {device}")
    print(f"iterations:          {args.iterations} (warmup: {args.warmup})")
    if gpu_ms is not None:
        print(f"total GPU time:      {gpu_ms:.2f} ms")
        print(f"per-iteration GPU:   {gpu_ms / args.iterations * 1000:.2f} us")
    print(f"total wall time:     {wall_ms:.2f} ms")
    print(f"per-iteration wall:  {wall_ms / args.iterations * 1000:.2f} us")


if __name__ == "__main__":
    main()
