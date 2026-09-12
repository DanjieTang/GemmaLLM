"""Export this single-head pre-norm layer for example_transformer_layer.cu."""

import argparse
from array import array
from pathlib import Path
import sys

import torch
import torch.nn as nn


class PreNormTransformer(nn.Module):
    """Single-head transformer layer with causal self-attention.

    Inputs and outputs have shape (batch, seq_len, hidden_dim), or
    (seq_len, hidden_dim) for an unbatched sequence.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.scale = hidden_dim ** -0.5

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention: x + Attention(LayerNorm(x)).
        normalized = self.norm1(x)
        q = self.q_proj(normalized)
        k = self.k_proj(normalized)
        v = self.v_proj(normalized)

        # One head uses the entire hidden dimension for scaled dot products.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        seq_len = x.size(-2)
        mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        x = x + self.out_proj(torch.matmul(attention, v))

        # Pre-norm feed-forward: x + FFN(LayerNorm(x)).
        return x + self.ffn(self.norm2(x))


def save_float32(path: Path, tensor: torch.Tensor) -> None:
    """Write contiguous, little-endian float32 values without a header or NumPy."""
    values = array("f", tensor.detach().to(device="cpu", dtype=torch.float32).flatten().tolist())
    if values.itemsize != 4:
        raise RuntimeError("The binary format requires 4-byte floats")
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("wb") as file:
        values.tofile(file)


def export_weights(model: PreNormTransformer, directory: Path) -> None:
    """Save the model's current parameters, including any trained parameters.

    Each state_dict entry becomes <name>.bin. Linear weights remain in
    [out_features, in_features] order; do not transpose them before saving.
    Use export_model when also exporting an input and CUDA run configuration.
    """
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Export a float32 model for the float32 CUDA example")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, parameter in model.state_dict().items():
        save_float32(directory / f"{name}.bin", parameter)


@torch.no_grad()
def export_model(model: PreNormTransformer, x: torch.Tensor, directory: Path) -> torch.Tensor:
    """Export current weights (including trained weights), input, and references.

    Linear weights retain PyTorch's [out_features, in_features] layout.
    Activations retain [batch, sequence, hidden] layout. An unbatched input
    is recorded as batch size 1. CUDA computes each linear as x @ weight.T + bias.
    """
    if x.ndim not in (2, 3) or any(size <= 0 for size in x.shape):
        raise ValueError("Expected a nonempty [sequence, hidden] or [batch, sequence, hidden] input")
    if x.dtype != torch.float32 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Export a float32 model and input for the float32 CUDA example")
    if x.shape[-1] != model.q_proj.in_features:
        raise ValueError("Input hidden dimension must match the model")
    if any(p.device != x.device for p in model.parameters()):
        raise ValueError("The model and input must be on the same device")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Also export isolated LayerNorm inputs/outputs for the normalization example.
    norm_references = {}

    def capture_norm(name):
        def hook(module, inputs, output):
            norm_references[f"{name}.input"] = inputs[0].detach().clone()
            norm_references[f"{name}.output"] = output.detach().clone()
        return hook

    handles = [getattr(model, name).register_forward_hook(capture_norm(name))
               for name in ("norm1", "norm2")]
    # Match the CUDA kernels even when the caller normally uses TF32.
    previous_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        output = model(x)
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous_precision
        for handle in handles:
            handle.remove()

    export_weights(model, directory)
    save_float32(directory / "input.bin", x)
    save_float32(directory / "output.bin", output)
    for name, tensor in norm_references.items():
        save_float32(directory / f"{name}.bin", tensor)

    batch = x.shape[0] if x.ndim == 3 else 1
    sequence, hidden = x.shape[-2:]
    (directory / "config.txt").write_text(
        f"pre_norm_transformer_v1\n{batch} {sequence} {hidden}\n"
        f"{model.norm1.eps:.17g} {model.norm2.eps:.17g}\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "tensors")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.hidden_dim) <= 0:
        parser.error("batch-size, seq-len, and hidden-dim must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = PreNormTransformer(hidden_dim=args.hidden_dim).to(device).eval()
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_dim, device=device)
    output = export_model(model, x, args.output_dir)
    print(f"Device: {device}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Exported {len(model.state_dict())} parameter tensors to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
