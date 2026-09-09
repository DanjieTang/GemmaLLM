from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_preprocessing.tokenize_annotations import (
    IMAGE_SUFFIXES, annotation_directory, iter_files,
)


class AnnotationDataset(Dataset):
    """Lazy token loading for images/annotations/input_ids in one or more splits."""

    def __init__(self, folders: list[str | Path], max_context_length: int,
                 vocabulary_size: int, bos_token_id: int, eos_token_id: int,
                 max_samples: int | None = None):
        if max_context_length < 1:
            raise ValueError("max_context_length must be positive.")
        self.max_context_length = max_context_length
        self.vocabulary_size = vocabulary_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.samples = []
        self.missing_annotation = 0
        self.missing_tokens = 0
        for folder in folders:
            split = Path(folder).resolve()
            annotations = annotation_directory(split)
            if not (split / "input_ids").is_dir():
                raise ValueError(f"{split}: missing input_ids directory; tokenize first.")
            seen = set()
            for image in iter_files(split / "images", IMAGE_SUFFIXES):
                relative = image.relative_to(split / "images").with_suffix("")
                if relative in seen:
                    raise ValueError(f"Ambiguous image stem in {split}: {relative}")
                seen.add(relative)
                # Append extensions: with_suffix would corrupt stems containing dots.
                annotation = annotations / (str(relative) + ".txt")
                tokens = split / "input_ids" / (str(relative) + ".npy")
                if not annotation.is_file():
                    self.missing_annotation += 1
                    continue
                if not tokens.is_file():
                    self.missing_tokens += 1
                    continue
                self.samples.append((str(tokens), str(image)))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                break
        print(f"Indexed {len(self.samples):,} pairs; skipped "
              f"{self.missing_annotation:,} images without annotations and "
              f"{self.missing_tokens:,} without token files.", flush=True)
        if not self.samples:
            raise ValueError(f"No image/annotation/token triples found in {folders}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        token_path, image_path = self.samples[index]
        ids = np.load(token_path, allow_pickle=False)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) < 2:
            raise ValueError(f"{token_path}: expected a 1D integer [BOS, ..., EOS] array.")
        if ids.min() < 0 or ids.max() >= self.vocabulary_size:
            raise ValueError(f"{token_path}: token ID outside the embedding vocabulary.")
        if ids[0] != self.bos_token_id or ids[-1] != self.eos_token_id:
            raise ValueError(f"{token_path}: missing expected BOS/EOS boundaries.")
        # Context length counts decoder inputs; one more token supplies the target.
        ids = ids[:self.max_context_length + 1].astype(np.int64, copy=True)
        ids[-1] = self.eos_token_id
        return torch.from_numpy(ids), image_path


def collate_annotations(batch, pad_token_id: int) -> dict:
    """Right-pad shifted inputs and use -100 only for ignored target positions."""
    lengths = torch.tensor([len(tokens) - 1 for tokens, _ in batch])
    shape = (len(batch), int(lengths.max()))
    inputs = torch.full(shape, pad_token_id, dtype=torch.long)
    labels = torch.full(shape, -100, dtype=torch.long)
    mask = torch.arange(shape[1])[None, :] < lengths[:, None]
    for row, (tokens, _) in enumerate(batch):
        length = len(tokens) - 1
        inputs[row, :length] = tokens[:-1]
        labels[row, :length] = tokens[1:]
    return {"input_ids": inputs, "labels": labels, "attention_mask": mask,
            "image_paths": [image for _, image in batch]}


def prepare_annotation_dataset(
    data_root: str, train_folders: list[str], val_folders: list[str],
    batch_size: int, max_context_length: int, vocabulary_size: int,
    bos_token_id: int, eos_token_id: int, pad_token_id: int,
    num_workers: int = 0, max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_paths = [Path(data_root) / name for name in train_folders]
    val_paths = [Path(data_root) / name for name in val_folders]
    if {p.resolve() for p in train_paths} & {p.resolve() for p in val_paths}:
        raise ValueError("Training and validation folders must be disjoint.")
    kwargs = dict(max_context_length=max_context_length,
                  vocabulary_size=vocabulary_size, bos_token_id=bos_token_id,
                  eos_token_id=eos_token_id, max_samples=max_samples)
    train = AnnotationDataset(train_paths, **kwargs)
    val = AnnotationDataset(val_paths, **kwargs)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         collate_fn=partial(collate_annotations,
                                            pad_token_id=pad_token_id))
    return (DataLoader(train, shuffle=True, **loader_kwargs),
            DataLoader(val, shuffle=False, **loader_kwargs))


class LazyLoadDataset(Dataset):
    def __init__(
        self,
        filename: str,
        image_paths_filename: str | None = None,
    ):
        # Create memory-mapped array
        self.mmap_data = np.load(filename, mmap_mode='r')
        self.length = self.mmap_data.shape[0]
        self.image_paths = None
        self.image_paths_root = None

        if image_paths_filename is not None:
            try:
                self.image_paths = np.load(
                    image_paths_filename,
                    mmap_mode="r",
                    allow_pickle=False,
                )
            except ValueError as exc:
                raise ValueError(
                    "Image paths must be stored in a non-object NumPy "
                    "string array."
                ) from exc

            if self.image_paths.ndim != 1:
                raise ValueError(
                    "Image paths must be a one-dimensional NumPy array."
                )
            if len(self.image_paths) != self.length:
                raise ValueError(
                    "Token data and image paths must contain the same "
                    "number of samples."
                )
            self.image_paths_root = Path(image_paths_filename).resolve().parent
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.mmap_data[idx])
        if self.image_paths is None:
            return tokens

        image_path = self.image_paths[idx]
        if isinstance(image_path, bytes):
            image_path = image_path.decode("utf-8")
        image_path = str(image_path)

        if image_path and not Path(image_path).is_absolute():
            image_path = str(self.image_paths_root / image_path)

        # Empty strings represent text-only samples and collate cleanly.
        return tokens, image_path


def prepare_dataset(
    train_data_path: str,
    val_data_path: str,
    train_batch_size: int,
    val_batch_size: int,
    train_image_paths: str | None = None,
    val_image_paths: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = LazyLoadDataset(train_data_path, train_image_paths)
    val_dataset = LazyLoadDataset(val_data_path, val_image_paths)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
