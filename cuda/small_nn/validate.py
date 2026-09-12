"""Validate the CUDA model output against the expected tensor.

Runs ./model (built from model.cu), which writes its GPU-computed output to
tensors/computed_output.bin, then compares it element-wise against the
expected values in tensors/output.bin (produced by model.py / PyTorch).

Usage:
    uv run validate.py            # run ./model, then compare
    uv run validate.py --no-run   # compare existing computed_output.bin
"""

import argparse
import subprocess
import sys

import numpy as np

BATCH_SIZE = 32
OUTPUT_DIM = 16
NUM_ELEMENTS = BATCH_SIZE * OUTPUT_DIM

EXPECTED_PATH = "tensors/output.bin"
COMPUTED_PATH = "tensors/computed_output.bin"

# Tolerances for float32: the tiled CUDA kernel accumulates in a different
# order than PyTorch's GEMM, so tiny rounding differences are expected.
RTOL = 1e-4
ATOL = 1e-5


def load_tensor(path: str) -> np.ndarray:
    array = np.fromfile(path, dtype=np.float32)
    if array.size != NUM_ELEMENTS:
        raise ValueError(
            f"{path} has {array.size} elements; expected {NUM_ELEMENTS}"
        )
    # Activations are stored as [features, batch] to match model.cu's layout.
    return array.reshape(OUTPUT_DIM, BATCH_SIZE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="skip running ./model and compare the existing computed_output.bin",
    )
    args = parser.parse_args()

    if not args.no_run:
        result = subprocess.run(["./model"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"./model failed with exit code {result.returncode}")
            print(result.stderr)
            return 1

    expected = load_tensor(EXPECTED_PATH)
    computed = load_tensor(COMPUTED_PATH)

    abs_diff = np.abs(expected - computed)
    max_abs_diff = abs_diff.max()
    matches = np.isclose(expected, computed, rtol=RTOL, atol=ATOL)

    print(f"max absolute difference: {max_abs_diff:.3e}")
    print(f"matching elements: {matches.sum()}/{NUM_ELEMENTS} (rtol={RTOL}, atol={ATOL})")

    if matches.all():
        print("PASS: computed output matches expected output")
        return 0

    worst = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
    print(f"worst mismatch at {worst}: expected {expected[worst]:.6f}, got {computed[worst]:.6f}")
    print("FAIL: computed output does not match expected output")
    return 1


if __name__ == "__main__":
    sys.exit(main())
