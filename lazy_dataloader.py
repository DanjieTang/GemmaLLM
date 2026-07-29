from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
