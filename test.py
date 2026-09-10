"""Benchmark generation with a randomly initialized ~200M LLM and a KV cache.

Run: uv run python test.py
Optional: uv run python test.py --device cuda --dtype bfloat16 --runs 3
Baseline: uv run python test.py --no_kv_cache

No checkpoint, tokenizer, dataset, or downloaded embeddings are needed. This
processes the prompt once, then only the newest token at each decoding step.
Use --no_kv_cache to measure the original full-prefix inference path.
"""

import argparse
import statistics
import time

import torch
from torch import nn

from model import LLM


VOCABULARY_SIZE = 29_000
HIDDEN_DIM = 768
NUM_LAYERS = 16


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading the wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def generate(
    llm: LLM,
    token_embeddings: nn.Embedding,
    prompt: torch.Tensor,
    max_new_tokens: int,
    causal_mask: torch.Tensor,
    use_cache: bool = True,
) -> torch.Tensor:
    """Return token IDs [1, prompt_length + max_new_tokens], using argmax.

    Always generate the requested number of new tokens, without EOS stopping.
    Keep token selection on the device to avoid a CPU/GPU sync at each step.
    """
    prompt_length = prompt.shape[1]
    tokens = prompt.new_empty((1, prompt_length + max_new_tokens))
    tokens[:, :prompt_length] = prompt
    past_key_values = None
    for position in range(prompt_length, tokens.shape[1]):
        # Prefill the prompt, then project and attend only for the newest token.
        start = position - 1 if past_key_values is not None else 0
        output = llm(
            token_embeddings(tokens[:, start:position]),
            causal_mask=causal_mask[start:position, :position],
            fine_tuning=False,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            logits, _, past_key_values = output
        else:
            logits, _ = output
        tokens[:, position] = logits[:, -1, :].argmax(dim=-1)
        del logits, output
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, or mps (default: auto).")
    parser.add_argument("--dtype", default="float32",
                        choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--prompt_length", type=int, default=1)
    parser.add_argument("--warmup_tokens", type=int, default=8,
                        help="Untimed generation steps; zero disables warmup.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_tokens", action="store_true",
                        help="Print generated token IDs after timing.")
    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Recompute the full prefix for the uncached baseline.")
    args = parser.parse_args()
    for name in ("max_new_tokens", "prompt_length", "runs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    if args.warmup_tokens < 0:
        parser.error("--warmup_tokens must be nonnegative.")

    if args.device == "auto":
        args.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda", "mps"):
        parser.error("--device must select cpu, cuda, or mps.")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    context_length = args.prompt_length + args.max_new_tokens

    # LLM expects [batch, sequence, hidden_dim], rather than integer token IDs.
    # Include a separate, untied input embedding table in the parameter budget.
    token_embeddings = nn.Embedding(VOCABULARY_SIZE, HIDDEN_DIM)
    llm = LLM(
        num_layer=NUM_LAYERS,
        vocabulary_size=VOCABULARY_SIZE,
        max_context_length=context_length,
        hidden_dim=HIDDEN_DIM,
        expansion_factor=4,
        head_dim=64,
        dropout_ratio=0.0,
        use_moe=False,
        device=str(device),
    )
    token_embeddings = token_embeddings.to(device=device, dtype=dtype).eval()
    llm = llm.to(device=device, dtype=dtype).eval()

    llm_count = sum(parameter.numel() for parameter in llm.parameters())
    embedding_count = sum(p.numel() for p in token_embeddings.parameters())
    lora_count = sum(p.numel() for name, p in llm.named_parameters()
                     if "lora_" in name)
    rope_count = llm.embedding.pos_emb.numel()
    total_count = llm_count + embedding_count
    print(f"Parameters: {total_count:,} ({total_count / 1e6:.3f}M total)")
    print(f"  LLM: {llm_count:,}; input embeddings: {embedding_count:,}")
    print(f"  Includes {lora_count:,} inactive LoRA parameters and "
          f"{rope_count:,} fixed RoPE values.")
    hardware = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else str(device))
    print(f"Device: {device} ({hardware}); dtype: {args.dtype}; "
          f"PyTorch: {torch.__version__}")
    print(f"Batch size: 1; prompt: {args.prompt_length} token(s); "
          f"new tokens: {args.max_new_tokens}; decoding: greedy; "
          f"KV cache: {'disabled' if args.no_kv_cache else 'enabled'}")
    forward_description = ("full-prefix forward passes" if args.no_kv_cache
                           else "prompt prefill and cached decoding")
    print(f"Timing includes embedding lookup, {forward_description}, and "
          "token selection; excludes model setup and warmup.", flush=True)

    prompt = torch.randint(VOCABULARY_SIZE, (1, args.prompt_length), device=device)
    causal_mask = torch.full(
        (context_length, context_length), float("-inf"), device=device, dtype=dtype,
    ).triu(diagonal=1)
    warmup_tokens = min(args.warmup_tokens, args.max_new_tokens)
    if warmup_tokens:
        print(f"Warming up with {warmup_tokens} new tokens...", flush=True)
        generate(llm, token_embeddings, prompt, warmup_tokens, causal_mask,
                 use_cache=not args.no_kv_cache)
    synchronize(device)

    durations = []
    for run in range(args.runs):
        print(f"Generating (run {run + 1}/{args.runs})...", flush=True)
        synchronize(device)
        start = time.perf_counter()
        tokens = generate(
            llm, token_embeddings, prompt, args.max_new_tokens, causal_mask,
            use_cache=not args.no_kv_cache,
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"  {args.max_new_tokens} new tokens in {elapsed:.3f} s | "
              f"{args.max_new_tokens / elapsed:.2f} tokens/s | "
              f"{1000 * elapsed / args.max_new_tokens:.2f} ms/token", flush=True)

    if args.runs > 1:
        median = statistics.median(durations)
        print(f"Median: {median:.3f} s | "
              f"{args.max_new_tokens / median:.2f} tokens/s")
    print(f"Output length: {tokens.shape[1]} tokens "
          f"({args.prompt_length} prompt + {args.max_new_tokens} generated).")
    if args.print_tokens:
        print("Generated token IDs:", tokens[0, args.prompt_length:].tolist())


if __name__ == "__main__":
    main()
