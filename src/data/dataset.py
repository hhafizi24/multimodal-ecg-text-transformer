"""
Datasets and DataLoader factories for preprocessed PTB-XL splits.

Supports raw multimodal inputs and cached encoder features.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ECGAugmentation:
    """
    Apply stochastic noise, amplitude scaling, and time shifts to ECG signals.

    Transforms operate independently on arrays with shape [1000, 12].
    """

    def __init__(
        self,
        augment_classes: set[int] = {1, 4},
        noise_std: float = 0.05,
        amplitude_scale_range: tuple[float, float] = (0.8, 1.2),
        time_shift_max: int = 50,
        p: float = 0.5,
    ):
        """
        Args:
            augment_classes: Labels eligible for augmentation.
            noise_std: Gaussian noise scale relative to signal variation.
            amplitude_scale_range: Range used for amplitude scaling.
            time_shift_max: Maximum temporal shift in either direction.
            p: Independent probability of applying each transform.
        """
        self.augment_classes = augment_classes
        self.noise_std             = noise_std
        self.amplitude_scale_range = amplitude_scale_range
        self.time_shift_max        = time_shift_max
        self.p                     = p

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply each transform independently with probability p."""
        if random.random() < self.p:
            signal = self._add_noise(signal)
        if random.random() < self.p:
            signal = self._scale_amplitude(signal)
        if random.random() < self.p:
            signal = self._time_shift(signal)
        return signal

    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        std = signal.std() + 1e-8
        noise = np.random.normal(0, self.noise_std * std, signal.shape).astype(np.float32)
        return signal + noise

    def _scale_amplitude(self, signal: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.amplitude_scale_range)
        return (signal * scale).astype(np.float32)

    def _time_shift(self, signal: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max)
        return np.roll(signal, shift, axis=0).astype(np.float32)


class ECGTextDataset(Dataset):
    def __init__(
        self,
        split_dir: str,
        tokenizer,
        max_text_length: int = 128,
        augmentation: ECGAugmentation | None = None,
    ):
        """
        Args:
            split_dir: Directory containing a processed dataset split.
            tokenizer: Initialized Hugging Face tokenizer.
            max_text_length: Maximum tokenized report length.
            augmentation: Optional ECG augmentation pipeline.
        """
        split_dir = Path(split_dir)

        self.signals = np.load(split_dir / "signals.npy")   # [N, 1000, 12]
        self.labels  = np.load(split_dir / "labels.npy")    # [N]

        with open(split_dir / "reports.json", encoding="utf-8") as f:
            self.reports = json.load(f)

        assert len(self.signals) == len(self.labels) == len(self.reports), (
            "Mismatch between signals, labels, and reports — re-run preprocessing."
        )

        self.tokenizer       = tokenizer
        self.max_text_length = max_text_length
        self.augmentation    = augmentation

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        signal = self.signals[idx].copy()  # Avoid modifying the loaded array
        label  = int(self.labels[idx])

        if (
            self.augmentation is not None
            and label in self.augmentation.augment_classes
        ):
            signal = self.augmentation(signal)

        signal_tensor = torch.from_numpy(signal).float()  # [1000, 12]
        label_tensor  = torch.tensor(label, dtype=torch.long)

        encoded = self.tokenizer(
            self.reports[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "signal":         signal_tensor,
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          label_tensor,
        }
    
    
class CachedEmbeddingDataset(Dataset):
    """
    Load raw ECG signals with precomputed text-backbone features.

    Cached features are valid only while the text backbone remains frozen.
    """

    def __init__(
        self,
        split_dir: str,
        embedding_path: str,
        augmentation: ECGAugmentation | None = None,
    ):
        split_dir = Path(split_dir)

        self.signals    = np.load(split_dir / "signals.npy")
        self.labels     = np.load(split_dir / "labels.npy")
        self.embeddings = np.load(embedding_path)  # [N, hidden_size], pre-projection

        assert len(self.signals) == len(self.labels) == len(self.embeddings), (
            "Mismatch between signals, labels, and cached embeddings — re-run precompute script."
        )

        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        signal = self.signals[idx].copy()
        label  = int(self.labels[idx])

        if (
            self.augmentation is not None
            and label in self.augmentation.augment_classes
        ):
            signal = self.augmentation(signal)

        return {
            "signal":         torch.from_numpy(signal).float(),
            "text_embedding": torch.from_numpy(self.embeddings[idx]).float(),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def make_cached_dataloader(
    split_dir: str,
    embedding_path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    augmentation: ECGAugmentation | None = None,
    seed: int = 42,
) -> DataLoader:
    dataset = CachedEmbeddingDataset(split_dir, embedding_path, augmentation=augmentation)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


class CachedFusionDataset(Dataset):
    """
    Load paired signal and text embeddings for frozen-encoder fusion training.
    """

    def __init__(
        self,
        labels_dir: str,
        signal_embedding_path: str,
        text_embedding_path: str,
    ):
        self.labels = np.load(Path(labels_dir) / "labels.npy")
        self.signal_embeddings = np.load(signal_embedding_path)
        self.text_embeddings = np.load(text_embedding_path)

        assert len(self.labels) == len(self.signal_embeddings) == len(self.text_embeddings), (
            "Mismatch between labels, signal embeddings, and text embeddings — re-run precompute script."
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "signal_embedding": torch.from_numpy(self.signal_embeddings[idx]).float(),
            "text_embedding":   torch.from_numpy(self.text_embeddings[idx]).float(),
            "label":            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def make_cached_fusion_dataloader(
    labels_dir: str,
    signal_embedding_path: str,
    text_embedding_path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    dataset = CachedFusionDataset(labels_dir, signal_embedding_path, text_embedding_path)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


def _worker_init_fn(worker_id: int) -> None:
    """
    Seed NumPy and Python RNGs for a DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    split_dir: str,
    tokenizer,
    batch_size: int = 32,
    max_text_length: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
    augmentation: ECGAugmentation | None = None,
    seed: int = 42,
) -> DataLoader:
    dataset = ECGTextDataset(
        split_dir,
        tokenizer,
        max_text_length,
        augmentation=augmentation,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )