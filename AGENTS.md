# Repository Guidelines

## Project Structure & Module Organization

The `model/` package defines the transformer/VLM architecture, with separate modules for rotary embeddings, attention, feed-forward networks, mixture of experts, decoder layers, the LLM, the VLM, and cache types. Its `__init__.py` preserves imports such as `from model import LLM, VLM`. `train.py` is the main training entry point, `lazy_dataloader.py` memory-maps tokenized NumPy datasets, and `checkpoint.py` contains checkpoint helpers. `run_sweep.py` expands `sweep_config.yaml` into training runs and records progress in `sweep_progress.json`. Data preparation and fine-tuning experiments live in `data_preprocessing/` and `fine-tuning/`; old notebooks are retained under `deprecated/`.

Large generated artifacts (`*.npy`, `*.pt`, W&B runs, datasets, and sweep state) are intentionally ignored. Do not commit model weights or local corpora.

## Setup, Training, and Development Commands

Use uv to install the locked dependencies (including development tools):

```bash
uv sync --locked
```

Run training with explicit local artifact paths:

```bash
uv run python train.py --train_path data/train.npy --val_path data/eval.npy \
  --embeddings_path data/embeddings.pt --device cpu
```

Preview a hyperparameter sweep without launching jobs:

```bash
uv run python run_sweep.py --config sweep_config.yaml --dry-run
```

Use `uv run python -m compileall model train.py run_sweep.py` for a quick syntax check. Dependencies live in `pyproject.toml`; commit `uv.lock` after dependency changes. W&B logging is enabled only when both `--project` and `--entity` are supplied; install its optional dependency with `uv sync --locked --extra wandb` and run training with `uv run --extra wandb python train.py ...`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, files, and CLI flags; use `PascalCase` for model and dataset classes. Preserve type hints on public helpers and document tensor shapes when adding non-obvious operations. Keep device placement explicit and avoid hidden global configuration. No formatter or linter is currently enforced, so keep imports grouped and changes consistent with nearby code.

## Testing Guidelines

There is no committed automated test suite or coverage threshold. For model changes, add focused `pytest` tests under `tests/` named `test_<module>.py`, using small synthetic tensors rather than external datasets. At minimum, run the syntax check, exercise the affected forward pass, and use sweep `--dry-run` when changing argument or configuration handling.

## Commit & Pull Request Guidelines

History uses short, imperative summaries such as `Created sweep python code` and `Added vision capability`. Keep commits narrowly scoped and describe the outcome; avoid vague messages like “updates.” Pull requests should explain architectural or training changes, list validation commands, note required data shapes or hardware, link relevant issues, and include metric plots or notebook screenshots when results change.
