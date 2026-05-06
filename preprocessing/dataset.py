"""
dataset.py
Memory-safe PyTorch Dataset that generates sliding windows lazily via __getitem__.
The normalized data is memory-mapped so it never fully loads into RAM.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LazyWindowDataset(Dataset):
    """
    Lazily generates (input, target) sliding windows from memory-mapped data.
    Each window position was precomputed by preprocess.py.
    """

    def __init__(self, split: str):
        """
        Args:
            split: one of 'train', 'val', 'test'
        """
        proc = config.PROCESSED_DATA_DIR
        data_path = os.path.join(proc, "normalized_data.npy")
        win_path = os.path.join(proc, f"{split}_windows.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Normalized data not found at {data_path}. Run preprocess.py first."
            )
        if not os.path.exists(win_path):
            raise FileNotFoundError(
                f"Window map not found at {win_path}. Run preprocess.py first."
            )

        # Memory-map the full normalized array (never fully loaded into RAM)
        self.data = np.load(data_path, mmap_mode="r")
        self.window_starts = np.load(win_path)  # shape: (N, 1)
        self.input_window = config.INPUT_WINDOW
        self.output_horizon = config.OUTPUT_HORIZON

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int):
        start = int(self.window_starts[idx, 0])
        x_end = start + self.input_window
        y_end = x_end + self.output_horizon

        # .copy() required because mmap slices can't convert to tensors directly
        x = self.data[start:x_end].copy()       # (input_window, 3)
        y = self.data[x_end:y_end].copy()        # (output_horizon, 3)

        return torch.from_numpy(x), torch.from_numpy(y)


def create_dataloaders() -> tuple:
    """
    Returns (train_loader, val_loader, test_loader).
    """
    train_ds = LazyWindowDataset("train")
    val_ds = LazyWindowDataset("val")
    test_ds = LazyWindowDataset("test")

    common = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
        drop_last=False,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    print(f"[DataLoader] train: {len(train_ds):,}  val: {len(val_ds):,}  "
          f"test: {len(test_ds):,}  batch_size: {config.BATCH_SIZE}")
    return train_loader, val_loader, test_loader
