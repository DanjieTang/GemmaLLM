"""Export text-decoder weights as raw .bin files for the CUDA implementation.

Each tensor is written to <output_dir>/<state_dict name>.bin as contiguous
little-endian values without a header. Linear weights keep PyTorch's
[out_features, in_features] layout. The embedding table keeps bfloat16 when
stored that way (other dtypes become float32). LoRA adapters of fine-tuned
checkpoints are merged into the base weights, which is exact for inference.

    uv run python cuda/cuda_gemma_llm/export_weights.py \
        --checkpoint checkpoints/mixed_sweep/<run>/latest.pt --output_dir exported
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

from text_model import TextGenerator, generate_greedy, load_checkpoint, make_prompt


def merge_lora(base: nn.Linear, lora_a: nn.Linear, lora_b: nn.Linear,
               scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold base(x) + scale * lora_b(lora_a(x)) into one weight and bias."""
    weight = base.weight + scale * lora_b.weight @ lora_a.weight
    bias = base.bias + scale * (lora_b.weight @ lora_a.bias + lora_b.bias)
    return weight, bias


def decoder_tensors(model: TextGenerator) -> dict[str, torch.Tensor]:
    """Every tensor the CUDA decoder loads, keyed by its state_dict name."""
    tensors = {}
    if model.text_projection:
        tensors["text_token_projection.weight"] = model.text_token_projection.weight
        tensors["text_token_projection.bias"] = model.text_token_projection.bias
    for index, layer in enumerate(model.llm.transformer):
        prefix = f"llm.transformer.{index}."
        attention, ffn = layer.mqa, layer.ffn
        linears = {
            "mqa.qkv": attention.qkv, "mqa.gate": attention.gate, "mqa.o": attention.o,
            "ffn.gate_and_up": ffn.gate_and_up, "ffn.down": ffn.down,
        }
        weights = {name: (linear.weight, linear.bias) for name, linear in linears.items()}
        if model.fine_tuning:
            weights["mqa.qkv"] = merge_lora(attention.qkv, attention.lora_qkv_a,
                                            attention.lora_qkv_b, attention.lora_scale)
            weights["mqa.o"] = merge_lora(attention.o, attention.lora_o_a,
                                          attention.lora_o_b, attention.lora_scale)
            weights["ffn.gate_and_up"] = merge_lora(ffn.gate_and_up, ffn.lora_gate_and_up_a,
                                                    ffn.lora_gate_and_up_b, ffn.lora_scale)
            weights["ffn.down"] = merge_lora(ffn.down, ffn.lora_down_a,
                                             ffn.lora_down_b, ffn.lora_scale)
        for norm in ("norm1", "norm2"):
            tensors[f"{prefix}{norm}.weight"] = getattr(layer, norm).weight
            tensors[f"{prefix}{norm}.bias"] = getattr(layer, norm).bias
        for name, (weight, bias) in weights.items():
            tensors[f"{prefix}{name}.weight"] = weight
            tensors[f"{prefix}{name}.bias"] = bias
    tensors["llm.output_norm.weight"] = model.llm.output_norm.weight
    tensors["llm.output_norm.bias"] = model.llm.output_norm.bias
    tensors["llm.classifier.weight"] = model.llm.classifier.weight
    tensors["llm.classifier.bias"] = model.llm.classifier.bias
    return tensors


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write raw values; bfloat16 is stored as its 16-bit pattern."""
    tensor = tensor.detach().to("cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    elif tensor.dtype != torch.float32:
        tensor = tensor.float()
    tensor.numpy().tofile(path)


def cuda_config(model: TextGenerator, theta: float) -> dict:
    """Architecture keys read by the CUDA program (sweep_config.yaml names)."""
    llm = model.llm
    layer = llm.transformer[0]
    vocab_size, text_dim = model.word_embeddings_tensor.shape
    hidden_dim = llm.output_norm.normalized_shape[0]
    epsilons = {module.eps for module in model.modules() if isinstance(module, nn.LayerNorm)}
    if len(epsilons) != 1 or layer.ffn.down.in_features % hidden_dim:
        raise ValueError("Expected one LayerNorm epsilon and an integer expansion factor.")
    return {
        "format": "cuda_gemma_llm_v1",
        "num_layer": len(llm.transformer),
        "vocab_size": vocab_size,
        "text_dim": text_dim,
        "projection_dim": hidden_dim,
        "expansion_factor": layer.ffn.down.in_features // hidden_dim,
        "head_dim": layer.mqa.head_dim,
        "q_head": layer.mqa.q_head,
        "kv_head": layer.mqa.kv_head,
        "theta": theta,
        "max_context_length": model.max_context_length,
        "norm_eps": repr(epsilons.pop()),
        "embedding_dtype": ("bfloat16" if model.word_embeddings_tensor.dtype == torch.bfloat16
                            else "float32"),
    }


@torch.no_grad()
def export_for_cuda(model: TextGenerator, directory: Path, theta: float = 10000,
                    reference: tuple[int, list[int], torch.Tensor] | None = None) -> None:
    """Write config.yaml, embeddings.bin, and one .bin per decoder tensor.

    reference = (prompt_length, full token sequence, logits after the prompt)
    adds reference_tokens.txt and reference_logits.bin for `gemma_llm --verify`.
    """
    if sys.byteorder != "little":
        raise RuntimeError("The CUDA loader expects little-endian files.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = cuda_config(model, theta)
    save_tensor(directory / "embeddings.bin", model.word_embeddings_tensor)
    for name, tensor in decoder_tensors(model).items():
        save_tensor(directory / f"{name}.bin", tensor.float())
    (directory / "config.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in config.items()))
    if reference is not None:
        prompt_length, tokens, logits = reference
        (directory / "reference_tokens.txt").write_text(
            f"{prompt_length}\n{' '.join(map(str, tokens))}\n")
        save_tensor(directory / "reference_logits.bin", logits.float())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="train.py checkpoint with model_config and model_state_dict.")
    parser.add_argument("--embeddings_path", type=Path,
                        help="Override the embedding path saved in the checkpoint.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_context_length", type=int, default=None,
                        help="Raise the exported context length (RoPE/KV cache size).")
    parser.add_argument("--reference_tokens", type=int, default=0,
                        help="Also save a PyTorch greedy reference of this total length.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, architecture = load_checkpoint(
        args.checkpoint, args.embeddings_path,
        max_context_length=max(args.max_context_length or 0, args.reference_tokens) or None,
        device=args.device,
    )
    reference = None
    if args.reference_tokens:
        prompt = make_prompt(1, model.vocabulary_size)
        tokens, logits = generate_greedy(model, prompt, args.reference_tokens,
                                         return_first_logits=True)
        reference = (len(prompt), tokens, logits)
    export_for_cuda(model, args.output_dir, architecture.get("theta", 10000), reference)
    print(f"Exported {args.checkpoint} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
