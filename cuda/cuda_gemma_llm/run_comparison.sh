#!/usr/bin/env bash
# Build the CUDA program, run both benchmarks with the same options, and print
# the speedup. Every argument is passed to both programs, so use only shared
# options: --total_tokens, --prompt_length, --warmup, --runs, --seed,
# --vocab_size, --text_dim.
#
#   ./run_comparison.sh                      # 1024 tokens, sweep_config.yaml
#   ./run_comparison.sh --total_tokens 2048
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

make -s -C "$here"

result() { awk -F'mean_seconds=' '/^RESULT/ { print $2 }'; }

echo "=== PyTorch ==="
pytorch=$(cd "$repo" && uv run python "$here/benchmark_pytorch.py" "$@" | tee /dev/stderr | result)
echo "=== CUDA ==="
cuda=$(cd "$here" && ./gemma_llm "$@" | tee /dev/stderr | result)

awk -v p="$pytorch" -v c="$cuda" 'BEGIN {
    printf "\nPyTorch %.1f ms, CUDA %.1f ms -> %.2fx speedup\n", p * 1e3, c * 1e3, p / c
}'
