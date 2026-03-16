"""Memory-mapped dataset for pre-tokenized binary corpus files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemoryMappedDataset(Dataset):
    """Reads token sequences directly from a uint16 binary file without loading into RAM.

    The binary file is a flat array of uint16 token IDs produced by prepare_data.py.
    Each sample is a (block_size,) input sequence x and a (block_size,) target
    sequence y where y = x shifted right by one position.

    Args:
        file_path: Path to the .bin file (uint16 numpy memmap).
        block_size: Number of tokens per sample (context length).
    """

    def __init__(self, file_path: str | Path, block_size: int) -> None:
        self.block_size = block_size
        self.data = np.memmap(str(file_path), dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a (x, y) pair for causal language modeling.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x, y) where both are int64 tensors of shape (block_size,)
            and y = x shifted right by one token.
        """
        start = idx * self.block_size
        # Cast uint16 → int64 before converting to tensor.
        # torch.from_numpy on uint16 produces int16 (signed), which would corrupt
        # token IDs above 32767. int64 is required by nn.Embedding.
        chunk = torch.from_numpy(
            self.data[start : start + self.block_size + 1].astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
