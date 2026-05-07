"""
PyTorch Dataset and DataLoader factory for the preprocessed PTB-XL splits.

Signals are already normalized on disk. Tokenization happens here at sample
load time using a tokenizer instance passed in at construction.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ECGTextDataset(Dataset):
    def __init__(self, split_dir: str, tokenizer, max_text_length: int = 128):
        """
        Args:
            split_dir: Path to one of data/processed/{train,val,test}.
            tokenizer: Initialized HuggingFace tokenizer — loaded once externally.
            max_text_length: Token sequence length passed to the tokenizer.
        """
        split_dir = Path(split_dir)

        self.signals = np.load(split_dir / "signals.npy")   # [N, 1000, 12]
        self.labels  = np.load(split_dir / "labels.npy")    # [N]

        with open(split_dir / "reports.txt", encoding="utf-8") as f:
            self.reports = f.read().splitlines()

        assert len(self.signals) == len(self.labels) == len(self.reports), (
            "Mismatch between signals, labels, and reports — re-run preprocessing."
        )

        self.tokenizer     = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        signal = torch.from_numpy(self.signals[idx]).float()  # [1000, 12]
        label  = torch.tensor(self.labels[idx], dtype=torch.long)

        encoded = self.tokenizer(
            self.reports[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "signal":         signal,
            "input_ids":      encoded["input_ids"].squeeze(0),       # [max_text_length]
            "attention_mask": encoded["attention_mask"].squeeze(0),  # [max_text_length]
            "label":          label,
        }


def make_dataloader(
    split_dir: str,
    tokenizer,
    batch_size: int = 32,
    max_text_length: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    dataset = ECGTextDataset(split_dir, tokenizer, max_text_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )