# CUDA Gemma text decoder

A CUDA/cuBLAS implementation of this repository's model (the `VLM` text path:
Gemma token embeddings → `text_token_projection` → `LLM`) that generates tokens
greedily with a KV cache. It exists to measure how long PyTorch and a hand-written
CUDA implementation take to reach a 1024-token sequence with the same architecture.

By default both programs read the architecture from `sweep_config.yaml` and use
random weights, which do not change inference speed. With `use_moe: false`, the
YAML settings are 4 layers, hidden 512, FFN 512×18, 8 query / 4 KV heads × 64.
MTP is not used. Trained weights can be exported from PyTorch into `.bin` files
and loaded instead (see below).

## Quick start (on a machine with an NVIDIA GPU)

From this directory:

```bash
make                       # nvcc -O3, builds for the local GPU (make ARCH=sm_90 to pick one)
./run_comparison.sh        # both benchmarks, 1024 tokens, prints the speedup
```

Or run the two benchmarks separately:

```bash
./gemma_llm                                                        # CUDA
(cd ../.. && uv run python cuda/cuda_gemma_llm/benchmark_pytorch.py)  # PyTorch
```

Each program starts from one BOS token, generates until the sequence holds
`--total_tokens` tokens (default 1024, so 1023 forward passes), does one untimed
warmup generation, then reports the mean of 3 timed runs and a `RESULT ...
mean_seconds=` line. Time is measured from submitting the prompt until the final
token is back on the host. Useful options for both: `--total_tokens`,
`--prompt_length`, `--runs`, `--warmup`, `--seed`. `./gemma_llm --help` lists the
rest, including `--no_graph` and architecture overrides such as `--num_layer 8`.

`max_context_length: 256` in the YAML is raised to `--total_tokens` automatically.
In this model it only sizes the RoPE table and KV cache.

## What is compared

Both sides do the same work per token:

- one embedding row lookup (bf16 table, like `data/gemma-4-31B-it-embeddings.pt`)
  and the 5376→512 projection
- 4 decoder layers
- the final LayerNorm, the 512→262144 classifier, and argmax

Both run in float32 (PyTorch's default; TF32 is off in both), with batch size 1.
Neither stops at EOS, so every run does the same amount of work.

The PyTorch side (`text_model.py`) calls the repository's `LLM` directly.
`generate_greedy` follows `generate.py`'s cached loop, including the per-token
`.item()`. A test checks that its logits match `VLM` on text-only input. CLIP and
the image prefix are not included: the image is encoded once before decoding,
so it does not change per-token speed.

## How the CUDA version maps to PyTorch

| PyTorch | CUDA (`model.cu`, `kernels.cu`) |
| --- | --- |
| `word_embeddings_tensor[ids].float()` | `embed_tokens` kernel |
| every `nn.Linear` | cuBLAS: `cublasSgemv` for one token, `cublasSgemm` for the prompt |
| `mqa.qkv` and `mqa.gate` | one GEMM on concatenated weights, giving `[q, k, v, gate]` per token |
| qkv bias, RoPE, KV-cache `torch.cat`, `sigmoid(gate)` | `qkv_rope_cache` kernel (writes into a preallocated cache) |
| `repeat_interleave` of KV heads, `QK^T`, mask, softmax, `@V`, gate multiply | `attention` kernel: online softmax, reads the shared KV head directly |
| `o(...) + skip`, `down(...) + skip` | GEMM with `beta = 1` accumulates into the residual stream |
| Linear bias, then `norm1` / `norm2` / `output_norm` | `add_bias_layer_norm` kernel (the bias of the previous Linear is added here) |
| `gelu_tanh(gate) * up` | `gelu_mul` kernel |
| classifier bias + `argmax().item()` | two-pass `argmax` kernels that write the token to device memory |

The kernels read the next token and its position from GPU memory, so the host
never waits for a token between steps. Every decode step also has identical
launch parameters, so it is captured once as a CUDA graph and replayed per token.
`--no_graph` launches the kernels individually instead, which shows how much the
graph saves.

GPU memory is about 3.6 GB with the Gemma vocabulary: a 2.8 GB bf16 embedding
table, a 0.54 GB classifier, and 0.23 GB of decoder layers.

## Checking the CUDA output against PyTorch

`benchmark_pytorch.py --export_dir DIR` writes the exact (random) PyTorch weights
and a reference generation. `gemma_llm --verify` compares against them:

```bash
(cd ../.. && uv run python cuda/cuda_gemma_llm/benchmark_pytorch.py --runs 0 --export_dir cuda/cuda_gemma_llm/exported)
./gemma_llm --weights exported --verify
```

The check compares the post-prompt logits (atol/rtol 1e-3). It passes or fails
on those alone. It also reports how many greedy tokens match before the first
difference. Float32 summation order differs between cuBLAS and PyTorch, so a
near-tie in argmax can eventually send the sequences apart; that is reported but
does not fail the run. Add `--vocab_size 32000 --text_dim 512` to both commands
(or `--total_tokens 128`) for a smaller, faster export.

## Loading trained weights

Export a `train.py` checkpoint once, then point the CUDA program at the folder:

```bash
cd ../..
uv run python cuda/cuda_gemma_llm/export_weights.py \
    --checkpoint checkpoints/mixed_sweep/<run>/latest.pt \
    --output_dir cuda/cuda_gemma_llm/exported --max_context_length 1024 \
    --reference_tokens 64          # optional: also save a PyTorch reference for --verify
cd cuda/cuda_gemma_llm
./gemma_llm --weights exported [--verify]
```

The export reads the embedding table path saved in the checkpoint (override it
with `--embeddings_path`). LoRA adapters from fine-tuned checkpoints are merged
into the base weights, which gives the same outputs.
`benchmark_pytorch.py --checkpoint ...` times PyTorch on the same trained weights.

### File format

The folder holds `config.yaml` (flat `key: value`, using `sweep_config.yaml`
names plus `vocab_size`, `text_dim`, `norm_eps`, and `embedding_dtype`) and one
headerless little-endian file per tensor. Each file is named after its
`state_dict` key:

| File | Shape, dtype |
| --- | --- |
| `embeddings.bin` | `[vocab, text_dim]`, bfloat16 (or float32) |
| `text_token_projection.{weight,bias}.bin` | `[hidden, text_dim]`, `[hidden]` (absent when `text_dim == hidden`) |
| `llm.transformer.{i}.norm{1,2}.{weight,bias}.bin` | `[hidden]` |
| `llm.transformer.{i}.mqa.qkv.{weight,bias}.bin` | `[(q_head + 2 kv_head) head_dim, hidden]`, `[...]` |
| `llm.transformer.{i}.mqa.gate.{weight,bias}.bin` | `[q_head, hidden]`, `[q_head]` |
| `llm.transformer.{i}.mqa.o.{weight,bias}.bin` | `[hidden, q_head head_dim]`, `[hidden]` |
| `llm.transformer.{i}.ffn.gate_and_up.{weight,bias}.bin` | `[2 ffn, hidden]`, `[2 ffn]` |
| `llm.transformer.{i}.ffn.down.{weight,bias}.bin` | `[hidden, ffn]`, `[hidden]` |
| `llm.output_norm.{weight,bias}.bin` | `[hidden]` |
| `llm.classifier.{weight,bias}.bin` | `[vocab, hidden]`, `[vocab]` |

Tensors other than the embeddings are float32. Linear weights keep PyTorch's
`[out_features, in_features]` layout.

## Limits

- Text-only decoding, batch size 1, greedy. There is no CLIP image prefix, no MoE
  (`use_moe: true` is rejected), and no MTP.
- `head_dim` must be 32, 64, 128, or 256, and `q_head * head_dim` must equal the
  hidden size (the PyTorch attention requires the same).
- Float32 weights. Storing them in bf16 would roughly halve the memory traffic
  per token. Batch-1 decoding is memory-bound, so that is the next speedup
  available, but PyTorch would need the same change for a fair comparison.
