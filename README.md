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
`--batch_size 1024` controls how many annotations are tokenized at once.
Progress output identifies the current folder (for example, `Folder 1/4`),
shows image indexing and annotation counting, then displays tokenization
percentage, counts, speed, and estimated time remaining. The annotation count
uses an extra directory scan without loading the annotation contents.

```python
import numpy as np

input_ids = np.load("data/train/input_ids/01234567.npy", allow_pickle=False)
```

These individual, variable-length files can be used directly by `train.py
--data_root data`, as described below. Regenerate token files with `--overwrite`
if their annotations change; training uses the saved IDs, without re-reading
or retokenizing annotation contents on every epoch.

## Download Gemma 4 31B IT token embeddings

```bash
uv run python data_preprocessing/download_embeddings.py \
  --tokenizer_path /home/danjie_tang/models/gemma-4-E2B-it \
  --output data/gemma-4-31B-it-embeddings.pt
```

The downloader pins a Hugging Face revision, checks that the complete tokenizer
JSON and BOS/EOS/PAD IDs match your local tokenizer, and locates
`model.language_model.embed_tokens.weight` in the safetensors index. It reads
only that tensor using HTTP byte ranges. The result is a plain PyTorch tensor
of shape `[262144, 5376]`, stored in bfloat16 (2.625 GiB), plus a JSON file
recording its provenance. No Gemma transformer weights are loaded. Allow about
5.3 GiB of free disk space during extraction; the temporary tensor is removed
after saving. Interrupted transfers resume when the command is rerun with the
same revision and output. An existing finished output is never overwritten.

The E2B IT and 31B IT `tokenizer.json` files were verified identical for 31B
revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. This check is repeated by the
downloader, rather than assuming that vocabulary size alone proves compatibility.
The script uses your existing Hugging Face credentials; if access is denied,
accept the model's access terms on Hugging Face and authenticate with `uv run hf auth login`.
If your server/proxy cannot serve byte ranges, add `--download_shard`; this
fallback downloads the containing shard (about 46.4 GiB), not the entire model.
The cached shard remains in the Hugging Face cache. The downloader uses
safetensors directly, so loading the Gemma 4 architecture in Transformers is
unnecessary.

## Train image annotations

```bash
uv run python train.py --data_root data \
  --embeddings_path data/gemma-4-31B-it-embeddings.pt \
  --tokenizer_path /home/danjie_tang/models/gemma-4-E2B-it \
  --batch_size 4 --max_context_length 512 --epochs 10 \
  --device cuda --output_dir checkpoints/annotations
```

By default, `train` and `OpenImageV7_train` are combined for training; `val`
and `OpenImageV7_val` are combined for validation. Override these with
`--train_folders train --val_folders val`, or provide several folder names
after each flag. `--data_root` can point to any parent directory containing
the splits. Each image must have a `.txt` annotation and `.npy` token array
with the same relative stem inside its own split. Missing annotations or token
files are skipped and counted explicitly; an empty dataset fails. Duplicate
image stems within a split and overlapping train/validation folders fail.
Arrays must be 1D integers with the tokenizer's BOS/EOS IDs and IDs within the
embedding vocabulary. Token arrays are loaded lazily, one sample at a time;
only paired file paths are indexed in memory.

The prediction task is:

```text
Input:  [CLIP CLS] [learned separator] [BOS] word1 word2 ...
Target:                               word1 word2 ... [EOS]
```

CLIP and Gemma token embeddings are frozen. The image/text projections,
separator, custom transformer decoder, and vocabulary classifier are trained.
Causal attention prevents looking ahead. Each batch is right-padded to its
longest sequence; padding is excluded from cross-entropy and MoE routing
statistics. Loss is calculated only on annotation tokens and EOS, not on image
prefix positions. Validation loss is averaged by the number of target tokens.

`--max_context_length` counts text input tokens, excluding the two image-prefix
positions. Longer annotations are truncated to fit, keeping BOS and forcing
EOS at the end. Choose a larger limit to preserve long annotations. The full
262,144-token output vocabulary makes logits expensive: reduce `--batch_size`
and/or the context length if you run out of memory. The default decoder width
is 512; `--projection_dim` must equal `--q_head * --head_dim`.

This trains this repository's custom decoder from scratch using pretrained
CLIP and Gemma embeddings. It does not fine-tune the pretrained Gemma 31B
transformer or inherit its instruction-following behavior.

Each epoch writes `OUTPUT_DIR/latest.pt` atomically, including the model,
optimizer, scheduler, architecture, tokenizer path, and special-token IDs.
Frozen CLIP weights are included; the large frozen token matrix stays in its
external file and must remain available for inference. Use a distinct
`--output_dir` for each experiment or sweep to preserve checkpoints. Losses
are printed without opening an interactive plot; W&B remains optional.

For a short end-to-end check before launching a full run:

```bash
uv run python train.py --data_root data \
  --train_folders train --val_folders val \
  --max_samples 2 --max_steps 1 --batch_size 2 \
  --num_layer 1 --projection_dim 128 --q_head 2 --kv_head 1 \
  --expansion_factor 2 --max_context_length 32 \
  --device cuda --output_dir data/smoke_checkpoint
```

`--max_samples` limits the indexed training and validation datasets separately;
`--max_steps` limits batches in each training and validation epoch. A smoke-test
checkpoint verifies execution only and will not produce useful annotations.
Legacy rectangular `.npy` datasets and optional image-path manifests still work
when `--data_root` is omitted; specify their matching tokenizer and embeddings.

## Generate an annotation

```bash
uv run python generate.py \
  --checkpoint checkpoints/annotations/latest.pt \
  --image /path/to/image.png --max_new_tokens 512 --device cuda
```

Inference restores the architecture from the checkpoint, encodes the image once,
and autoregressively predicts from BOS until EOS, the requested token limit, or
the trained context limit. Greedy decoding is the default; `--temperature 0.7`
enables sampling. BOS and PAD are suppressed during generation. The decoder
currently recomputes the text prefix on each step (no KV cache). Use
`--embeddings_path` and `--tokenizer_path` if the artifacts have moved. The
original tokenizer mapping must be preserved.

## Validation

```bash
uv run python -m pytest -q
uv run python -m compileall model.py train.py run_sweep.py lazy_dataloader.py generate.py data_preprocessing/download_embeddings.py
uv run python run_sweep.py --config sweep_config.yaml --dry-run
```

Tests use synthetic images, small tensors, and mocked CLIP weights. They cover
pair matching, truncation, shifted targets, padding, causal attention,
backpropagation, EOS termination, and checkpoint loading without network access.
Pytest searches only `tests/`, avoiding scans of the large data directories.
