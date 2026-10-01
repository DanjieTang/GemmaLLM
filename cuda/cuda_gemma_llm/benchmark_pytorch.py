"""Time PyTorch greedy generation until the sequence reaches --total_tokens.

Uses the repository's LLM class with the sweep_config.yaml architecture and
random weights (or a trained checkpoint), following generate.py's cached loop.
Pass --export_dir to also write these exact weights, plus a reference
generation, for `gemma_llm --weights DIR --verify`.

    uv run python cuda/cuda_gemma_llm/benchmark_pytorch.py
"""

import argparse
from pathlib import Path
import time

import torch

from export_weights import export_for_cuda
from text_model import (GEMMA_TEXT_DIM, GEMMA_VOCAB_SIZE, REPO_ROOT, TextGenerator,
                        architecture_from_yaml, generate_greedy, load_checkpoint,
                        make_prompt)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "sweep_config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Benchmark a train.py checkpoint instead of random weights.")
    parser.add_argument("--embeddings_path", type=Path, default=None)
    parser.add_argument("--vocab_size", type=int, default=GEMMA_VOCAB_SIZE,
                        help="Random embedding rows (ignored with --checkpoint).")
    parser.add_argument("--text_dim", type=int, default=GEMMA_TEXT_DIM,
                        help="Random embedding width (ignored with --checkpoint).")
    parser.add_argument("--total_tokens", type=int, default=1024,
                        help="Sequence length to reach, prompt included.")
    parser.add_argument("--prompt_length", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export_dir", type=Path, default=None,
                        help="Write weights and a reference generation for the CUDA program.")
    args = parser.parse_args()
    if args.prompt_length < 1 or args.total_tokens <= args.prompt_length:
        parser.error("Require 1 <= prompt_length < total_tokens.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.checkpoint is not None:
        model, architecture = load_checkpoint(args.checkpoint, args.embeddings_path,
                                              max_context_length=args.total_tokens,
                                              device=args.device)
        source = str(args.checkpoint)
    else:
        architecture = architecture_from_yaml(args.config)
        # RoPE table size only; positions beyond the training length still run.
        architecture["max_context_length"] = max(architecture["max_context_length"],
                                                 args.total_tokens)
        model = TextGenerator(**architecture, vocab_size=args.vocab_size,
                              text_dim=args.text_dim, device=args.device).eval()
        source = f"random (seed {args.seed})"
    parameters = model.word_embeddings_tensor.numel() + sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if "lora" not in name and "pos_emb" not in name)
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Config: {architecture}")
    print(f"Parameters: {parameters / 1e6:.1f}M, weights: {source}")

    prompt = make_prompt(args.prompt_length, model.vocabulary_size)
    for _ in range(args.warmup):
        generate_greedy(model, prompt, args.total_tokens)
    seconds = []
    for run in range(args.runs):
        synchronize(device)
        start = time.perf_counter()
        generate_greedy(model, prompt, args.total_tokens)  # .item() waits for each token.
        synchronize(device)
        seconds.append(time.perf_counter() - start)
        print(f"Run {run + 1}/{args.runs}: {args.total_tokens} tokens in {seconds[-1] * 1e3:.1f} ms")
    if seconds:
        mean = sum(seconds) / len(seconds)
        generated = args.total_tokens - args.prompt_length
        print(f"Mean: {mean * 1e3:.1f} ms to reach {args.total_tokens} tokens "
              f"({generated / mean:.1f} generated tokens/s, {mean * 1e3 / generated:.3f} ms/token)")
        print(f"RESULT implementation=pytorch total_tokens={args.total_tokens} "
              f"mean_seconds={mean:.6f}")

    if args.export_dir is not None:
        tokens, logits = generate_greedy(model, prompt, args.total_tokens,
                                         return_first_logits=True)
        export_for_cuda(model, args.export_dir, architecture.get("theta", 10000),
                        reference=(len(prompt), tokens, logits))
        print(f"Exported weights and reference generation to {args.export_dir.resolve()}")


if __name__ == "__main__":
    main()
