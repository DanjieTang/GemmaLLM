import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LazyLoadDataset(Dataset):
    def __init__(self, filename):
        # Create memory-mapped array
        self.mmap_data = np.load(filename, mmap_mode='r')
        self.length = self.mmap_data.shape[0]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.tensor(self.mmap_data[idx])

def prepare_dataset(train_data_path: str, val_data_path: str, train_batch_size: int, val_batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset: LazyLoadDataset = LazyLoadDataset(train_data_path)
    val_dataset: LazyLoadDataset = LazyLoadDataset(val_data_path)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader