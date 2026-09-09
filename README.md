![image](https://github.com/DanjieTang/FoundationLLM/assets/37476565/1d0dfa5a-89dd-4cfd-80af-06db247f2720)

# My implementation of the Gemma LLM.

## Setup

Use [uv](https://docs.astral.sh/uv/) to manage Python and dependencies:

```bash
uv sync --locked
```

Python 3.12 is selected by `.python-version`. Dependencies are declared in
`pyproject.toml`, and `uv.lock` records their resolved versions. Run commands
with `uv run`; manual environment activation is unnecessary. uv creates its
own ignored `.venv` directory when you sync or run the project.

PyTorch 2.10.0 and Transformers 5.2.0 preserve the versions used during the
tokenizer validation. PyTorch uses the CUDA 13.0 index on Linux and Windows,
and PyPI on macOS. See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
if you need a different accelerator build.

```bash
uv run python train.py --train_path data/train.npy --val_path data/eval.npy \
  --embeddings_path data/embeddings.pt --device cpu
uv run python run_sweep.py --config sweep_config.yaml --dry-run
```

For W&B logging, use `uv sync --locked --extra wandb` and
`uv run --extra wandb python train.py ...`, supplying `--project` and `--entity`.

## Training data.

    a) All English Wikipedia pages(6.5 million).

    b) ~2 billion tokens.

## Key insights from this implementation.

    a)RMS Normalization

    b)ROPE Embedding

    c)MultiQueryAttention

    d)GeGLU Activations

    e)Pre-Norm Transformers

    f)Mixtral of Experts

    g)LoRA: Low-Rank Adaptation of Large Language Models

    h)Gated Attention for Large Language Models

    i)Late fusion for LLM image capability

## Training detail.

    a) 665 Million parameters Mixture of Experts Architecture

    b) Contextual length of 64 tokens.

## Image inputs

`VLM.forward` uses the first hidden state from CLIP ViT—the CLS token—as
one image token. The model input is arranged as:

    [CLIP CLS] [learned separator] [text tokens]

For multimodal training, pass `--train_image_paths` and
`--val_image_paths` to `train.py`. Each file must be a one-dimensional
NumPy string array with one image path per tokenized sample. Paths may be
absolute or relative to the image-path array. Use an empty string for a
text-only sample.

Run `uv run python -m pytest` to verify CLS-token fusion and image-path loading
without downloading CLIP weights.

## Tokenize image annotations

Run from the repository root, with the dependencies installed:

```bash
uv run python data_preprocessing/tokenize_annotations.py \
  --tokenizer_path /home/danjie_tang/models/gemma-4-E2B-it
```

This processes all four datasets under `data/`: `train`, `val`,
`OpenImageV7_train`, and `OpenImageV7_val`. It accepts `annotations`,
`images_annotation`, or `annotation` as the annotation directory name:

```text
data/train/
  images/01234567.png
  annotations/01234567.txt
  input_ids/01234567.npy
```

Subdirectories within the annotation folder are preserved under `input_ids`;
images must have the same relative path and filename stem. Common image
extensions are supported, including PNG and JPEG. Annotations without a
matching image are skipped and counted in the summary.

Each `.npy` is a one-dimensional `int32` array containing
`[BOS] annotation_tokens [EOS]`, with no padding, truncation, or chat template.
The full UTF-8 annotation is tokenized, including its whitespace. Only local
tokenizer files are loaded; this step needs neither GPU nor model weights.

Existing arrays are skipped so interrupted runs can resume. Use `--overwrite`
after changing annotations or the tokenizer. To select datasets or another
root, use `--folders train val --data_root /path/to/data`. The default
`--batch_size 256` controls how many annotations are tokenized at once.

```python
import numpy as np

input_ids = np.load("data/train/input_ids/01234567.npy", allow_pickle=False)
```

These individual, variable-length files are preparation for image-conditioned
training. The current `train.py` loader expects a single rectangular token
array, so it will need a paired-file loader and padding/loss masking before
using these outputs directly. Training must also use word embeddings matching
the Gemma tokenizer's vocabulary and token IDs.
