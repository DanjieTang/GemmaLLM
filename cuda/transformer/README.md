# PyTorch to CUDA Transformer example

`example_transformer_layer.cu` matches `PreNormTransformer` in
`example_transformer_layer.py`: one causal attention head, two pre-norm
residual branches, and a `hidden_dim → 4 * hidden_dim → hidden_dim` ReLU FFN.
This is a single layer's float32 forward pass, without dropout or a final
stack-level normalization.

From this directory:

```bash
uv run example_transformer_layer.py
nvcc -O2 -std=c++17 example_transformer_layer.cu -lcublas -o example_transformer_layer
./example_transformer_layer tensors
```

The Python script exports initialized model weights, input, and expected output
to `tensors/`. CUDA loads them, writes `computed_output.bin`, and compares every
output value with PyTorch using `atol=1e-5, rtol=1e-4`. Missing files, incorrect
file sizes, non-finite outputs, and comparison failures return a nonzero exit code.
The exporter uses Python's standard `array` module; NumPy is not required.

All matrix products use cuBLAS `cublasSgemm` with full float32 precision.
The wrapper adapts the row-major tensors to cuBLAS's column-major interface
without copying or transposing buffers; a separate CUDA kernel adds linear biases.

To export a different example shape:

```bash
uv run example_transformer_layer.py --batch-size 2 --seq-len 17 --hidden-dim 33 --output-dir /tmp/transformer-small
./example_transformer_layer /tmp/transformer-small
```

To export your own trained model, call the existing helper after loading its
state dict. `x` and the model must be float32 and on the same device:

```python
from pathlib import Path
from example_transformer_layer import export_model

model.eval()
export_model(model, x, Path("tensors"))
```

To update just the binary weights after training, keeping the same architecture
and normalization epsilons, use `export_weights(model, Path("tensors"))`.
The CUDA `TransformerWeights` object loads all parameters once and can be reused
for multiple `forward` calls. Create a `CublasHandle` once and pass it as the first
argument to `forward(blas, config, model, input, output)` to reuse it as well.
The forward function follows the Python layer's
normalization, Q/K/V projections, causal attention, output projection, and FFN.

To run CUDA without `output.bin` (for example after updating weights or input):

```bash
./example_transformer_layer tensors --no-verify
```

This still requires `config.txt`, `input.bin`, and all parameter files, and writes
`computed_output.bin`. Run `export_model` again to regenerate a matching PyTorch
reference for verification after changing weights or inputs.

## Binary format

Each `.bin` contains contiguous little-endian float32 values with no header.
Names follow `model.state_dict()` directly. Linear weights are saved in their
original PyTorch layout; CUDA reads them as a transposed right-hand matrix.

| Files | Shape |
| --- | --- |
| `norm1.weight.bin`, `norm1.bias.bin`, `norm2.weight.bin`, `norm2.bias.bin` | `[hidden]` |
| `{q_proj,k_proj,v_proj,out_proj}.weight.bin` | `[hidden, hidden]` |
| `{q_proj,k_proj,v_proj,out_proj}.bias.bin` | `[hidden]` |
| `ffn.0.weight.bin`, `ffn.0.bias.bin` | `[4*hidden, hidden]`, `[4*hidden]` |
| `ffn.2.weight.bin`, `ffn.2.bias.bin` | `[hidden, 4*hidden]`, `[hidden]` |
| `input.bin`, `output.bin`, `computed_output.bin` | `[batch, sequence, hidden]` |
| `{norm1,norm2}.input.bin`, `{norm1,norm2}.output.bin` | `[batch, sequence, hidden]` |

`config.txt` records the version tag `pre_norm_transformer_v1`, then
`batch sequence hidden`, then `norm1_epsilon norm2_epsilon`. An unbatched Python
input is represented as batch size 1. Attention runs separately per batch item.

## Check the standalone LayerNorm kernel

From this directory, use the same export to test either normalization:

```bash
nvcc -O2 -std=c++17 ../normalization/example_normalization.cu -o ../normalization/example_normalization
../normalization/example_normalization tensors norm1
../normalization/example_normalization tensors norm2
```

Each run loads that LayerNorm's scale, shift, input, epsilon, and expected
PyTorch output. With no arguments, the normalization program retains its
original random-data demo and CPU reference.
