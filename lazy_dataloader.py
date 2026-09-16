from bisect import bisect_right
from collections import OrderedDict
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

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


class WikipediaDataset(Dataset):
    """Memory-map completed Wikipedia shards; return unpadded token windows.

    Only a bounded set of shard mappings is kept per worker. Row offsets index
    the manifests without allocating an entry for every training example.
    """

    def __init__(self, directory: str | Path, languages: list[str], split: str,
                 max_context_length: int, vocabulary_size: int,
                 special_token_ids: dict, tokenizer_sha256: str,
                 max_samples: int | None = None):
        self.vocabulary_size = vocabulary_size
        self.shards = []
        self.offsets = [0]
        self._mappings = OrderedDict()
        for language in languages:
            folder = Path(directory).expanduser() / language
            manifest_path = folder / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"{folder}: no completed Wikipedia manifest.")
            manifest = json.loads(manifest_path.read_text())
            config = manifest["config"]
            if config["format_version"] != 1 or config["language"] != language:
                raise ValueError(f"{manifest_path}: unsupported format or wrong language.")
            width = config["context_length"] + 1
            if not 2 <= width <= max_context_length + 1:
                raise ValueError(f"{manifest_path}: shard context exceeds --max_context_length.")
            if (config["special_token_ids"] != special_token_ids
                    or config["tokenizer_sha256"] != tokenizer_sha256):
                raise ValueError(f"{manifest_path}: tokenizer differs from preprocessing.")
            stats = manifest["splits"][split]
            if sum(shard["rows"] for shard in stats["shards"]) != stats["rows"]:
                raise ValueError(f"{manifest_path}: inconsistent shard row counts.")
            for shard in stats["shards"]:
                paths = (folder / split / shard["tokens"],
                         folder / split / shard["lengths"])
                tokens, lengths = (np.load(path, mmap_mode="r", allow_pickle=False)
                                   for path in paths)
                rows = shard["rows"]
                if (rows < 1 or tokens.shape != (rows, width)
                        or lengths.shape != (rows,)
                        or tokens.dtype != np.int32 or lengths.dtype != np.int32):
                    raise ValueError(f"{paths[0]}: invalid token/length shapes or dtypes.")
                self.shards.append(paths)
                self.offsets.append(self.offsets[-1] + rows)
        self.length = self.offsets[-1]
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive.")
            self.length = min(self.length, max_samples)
        if not self.length:
            raise ValueError(f"{directory}: no Wikipedia {split} windows.")
        print(f"Wikipedia {split}: {self.length:,} windows across "
              f"{', '.join(languages)}", flush=True)

    def __len__(self):
        return self.length

    def __getstate__(self):
        # Spawned DataLoader workers reopen mappings rather than pickle arrays.
        return {**self.__dict__, "_mappings": OrderedDict()}

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        shard_index = bisect_right(self.offsets, index) - 1
        if shard_index not in self._mappings:
            self._mappings[shard_index] = tuple(
                np.load(path, mmap_mode="r", allow_pickle=False)
                for path in self.shards[shard_index]
            )
            if len(self._mappings) > 8:
                self._mappings.popitem(last=False)
        self._mappings.move_to_end(shard_index)
        tokens, lengths = self._mappings[shard_index]
        row = index - self.offsets[shard_index]
        length = int(lengths[row])
        if not 2 <= length <= tokens.shape[1]:
            raise ValueError(f"{self.shards[shard_index][1]}: invalid length at row {row}.")
        ids = tokens[row, :length].astype(np.int64, copy=True)
        if ids.min() < 0 or ids.max() >= self.vocabulary_size:
            raise ValueError(f"{self.shards[shard_index][0]}: token ID outside vocabulary.")
        # Chunks may start/end mid-article: never insert BOS/EOS or truncate.
        return torch.from_numpy(ids), None


class WikipediaSampler(Sampler[int]):
    """Shuffle shards and then their rows, using memory proportional to a shard."""

    def __init__(self, dataset: WikipediaDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        offsets = self.dataset.offsets
        shard_count = bisect_right(offsets, len(self.dataset) - 1)
        for shard in torch.randperm(shard_count).tolist():
            start = offsets[shard]
            count = min(offsets[shard + 1], len(self.dataset)) - start
            for row in torch.randperm(count).tolist():
                yield start + row


def prepare_wikipedia_dataset(
    directory: str | Path, languages: list[str] | None, batch_size: int,
    max_context_length: int, vocabulary_size: int, tokenizer,
    num_workers: int = 0, max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load existing article splits; all completed languages are used by default."""
    directory = Path(directory).expanduser()
    if languages is None or languages == ["all"]:
        languages = sorted(path.parent.name for path in directory.glob("*/manifest.json")
                           if not path.parent.name.startswith("."))
    elif "all" in languages:
        raise ValueError("Use all alone, or list individual Wikipedia languages.")
    if (not languages or len(set(languages)) != len(languages)
            or any(Path(language).name != language or language.startswith(".")
                   for language in languages)):
        raise ValueError("Select distinct, completed Wikipedia language folders.")
    if not tokenizer.is_fast:
        raise ValueError("Wikipedia requires the fast tokenizer used for preprocessing.")
    fingerprint = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    special_ids = {name: getattr(tokenizer, name) for name in
                   ("bos_token_id", "eos_token_id", "pad_token_id")}
    datasets = [WikipediaDataset(
        directory, languages, split, max_context_length, vocabulary_size,
        special_ids, fingerprint, max_samples,
    ) for split in ("train", "val")]
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  collate_fn=partial(collate_annotations,
                                     pad_token_id=tokenizer.pad_token_id))
    return (DataLoader(datasets[0], sampler=WikipediaSampler(datasets[0]), **kwargs),
            DataLoader(datasets[1], shuffle=False, **kwargs))


class GlobalShuffleSampler(Sampler[int]):
    """Shuffle every example once without building a corpus-sized Python list.

    The permutation uses four bytes per example (eight for >= 2**31 examples).
    Only small slices are converted to Python indices for DataLoader workers.
    """

    def __init__(self, dataset: Dataset):
        self.length = len(dataset)

    def __len__(self):
        return self.length

    def __iter__(self):
        dtype = torch.int32 if self.length < 2**31 else torch.int64
        order = torch.randperm(self.length, dtype=dtype, device="cpu")
        for chunk in order.split(65536):
            yield from chunk.tolist()


def prepare_mixed_dataset(
    data_root: str, train_folders: list[str], val_folders: list[str],
    wikipedia_dir: str | Path, wikipedia_languages: list[str] | None,
    batch_size: int, max_context_length: int, vocabulary_size: int, tokenizer,
    num_workers: int = 0, max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Combine annotations and Wikipedia within each split, shuffling training.

    Each image/annotation pair and each Wikipedia window is one example. There
    is no oversampling; max_samples, when set, caps each source in each split.
    """
    special_ids = {name: getattr(tokenizer, name) for name in
                   ("bos_token_id", "eos_token_id", "pad_token_id")}
    annotation_loaders = prepare_annotation_dataset(
        data_root, train_folders, val_folders, batch_size, max_context_length,
        vocabulary_size, **special_ids, max_samples=max_samples,
    )
    wikipedia_loaders = prepare_wikipedia_dataset(
        wikipedia_dir, wikipedia_languages, batch_size, max_context_length,
        vocabulary_size, tokenizer, max_samples=max_samples,
    )
    datasets = [ConcatDataset([annotations.dataset, wikipedia.dataset])
                for annotations, wikipedia in zip(annotation_loaders, wikipedia_loaders)]
    for split, dataset in zip(("train", "val"), datasets):
        print(f"Mixed {split}: {len(dataset.datasets[0]):,} image/annotation pairs + "
              f"{len(dataset.datasets[1]):,} Wikipedia windows", flush=True)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  collate_fn=partial(collate_annotations,
                                     pad_token_id=tokenizer.pad_token_id))
    return (DataLoader(datasets[0], sampler=GlobalShuffleSampler(datasets[0]), **kwargs),
            DataLoader(datasets[1], shuffle=False, **kwargs))
