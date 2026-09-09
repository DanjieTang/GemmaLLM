"""Save one Gemma token array per image annotation, using only local files."""

import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterator

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_TOKENIZER = Path.home() / "models" / "gemma-4-E2B-it"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ANNOTATION_FOLDERS = ("annotations", "images_annotation", "annotation")


def iter_files(root: Path, suffixes: set[str]) -> Iterator[Path]:
    """Walk one directory at a time instead of retaining millions of Paths."""
    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, subdirectories, filenames in os.walk(
        root, onerror=raise_walk_error,
    ):
        subdirectories.sort()
        for name in sorted(filenames):
            if Path(name).suffix.lower() in suffixes:
                yield Path(directory) / name


def annotation_directory(split: Path) -> Path:
    candidates = [split / name for name in ANNOTATION_FOLDERS
                  if (split / name).is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f"{split}: expected exactly one annotation directory named "
            f"{', '.join(ANNOTATION_FOLDERS)}; found {len(candidates)}."
        )
    if not (split / "images").is_dir():
        raise ValueError(f"{split}: missing images directory.")
    return candidates[0]


def save_input_ids(destination: Path, token_ids: list[int]) -> None:
    """Atomically save an unpadded (sequence_length,) int32 array."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, np.asarray(token_ids, dtype=np.int32),
                    allow_pickle=False)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def tokenize_split(
    split: Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 256,
    overwrite: bool = False,
) -> dict[str, int]:
    """Match images by relative stem and tokenize annotations in bounded batches."""
    annotations = annotation_directory(split)
    images = split / "images"
    output = split / "input_ids"
    print(f"{split.name}: indexing image/annotation pairs...", flush=True)
    image_stems = {
        str(path.relative_to(images).with_suffix(""))
        for path in iter_files(images, IMAGE_SUFFIXES)
    }
    annotation_paths = iter_files(annotations, {".txt"})

    stats = {"written": 0, "existing": 0, "missing_image": 0}
    pending_texts = []
    pending_outputs = []

    def flush_batch() -> None:
        if not pending_texts:
            return
        encoded = tokenizer(
            pending_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for destination, tokens in zip(pending_outputs, encoded):
            # Explicit boundaries avoid depending on the tokenizer's defaults.
            save_input_ids(
                destination,
                [tokenizer.bos_token_id, *tokens, tokenizer.eos_token_id],
            )
            stats["written"] += 1
        pending_texts.clear()
        pending_outputs.clear()

    annotation_count = 0
    for annotation in tqdm(annotation_paths, desc=split.name, unit="annotation"):
        annotation_count += 1
        relative = annotation.relative_to(annotations)
        if str(relative.with_suffix("")) not in image_stems:
            stats["missing_image"] += 1
            continue
        destination = output / relative.with_suffix(".npy")
        if destination.exists() and not overwrite:
            stats["existing"] += 1
            continue
        # Preserve whitespace and newlines; an empty file becomes [BOS, EOS].
        pending_texts.append(annotation.read_text(encoding="utf-8"))
        pending_outputs.append(destination)
        if len(pending_texts) >= batch_size:
            flush_batch()
    flush_batch()
    if annotation_count == 0:
        raise ValueError(f"{annotations}: no .txt annotations found.")
    print(
        f"{split.name}: wrote {stats['written']:,}, "
        f"skipped {stats['existing']:,} existing, "
        f"skipped {stats['missing_image']:,} without matching images.",
        flush=True,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing the dataset folders (default: repo/data).",
    )
    parser.add_argument(
        "--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER,
        help="Local Gemma tokenizer directory; no model weights are loaded.",
    )
    parser.add_argument(
        "--folders", nargs="+",
        help="Dataset folder names to process (default: all paired datasets).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate existing arrays, e.g. after changing annotations/tokenizer.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive.")
    if not args.data_root.is_dir():
        parser.error(f"Data directory does not exist: {args.data_root}")
    if args.folders:
        splits = [args.data_root / name for name in args.folders]
    else:
        splits = sorted(
            path for path in args.data_root.iterdir()
            if path.is_dir() and any(
                (path / name).is_dir() for name in ANNOTATION_FOLDERS
            )
        )
    if not splits:
        parser.error("No dataset folders found.")
    for split in splits:
        annotation_directory(split)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path.expanduser(), local_files_only=True,
    )
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        parser.error("The tokenizer must define BOS and EOS token IDs.")
    print(
        f"Tokenizer: {args.tokenizer_path}, vocabulary={len(tokenizer):,}, "
        f"BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}",
        flush=True,
    )
    for split in splits:
        tokenize_split(split, tokenizer, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
