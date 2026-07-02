"""
PyTorch Dataset and DataLoader factory for the preprocessed PTB-XL splits.

Signals are already normalized on disk. Tokenization happens here at sample
load time using a tokenizer instance passed in at construction.

Augmentation is applied at the sample level during training only — val and
test loaders always receive unmodified signals.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Apply augmentation only to selected minority classes.
# The target classes can be adjusted as experiments evolve.

class ECGAugmentation:
    """
    Stochastic augmentation pipeline for normalized ECG signals.

    Each transform is applied independently with probability p, so the model
    sees a random mix of original and augmented versions across epochs rather
    than always seeing the same transformed version of every minority sample.

    All transforms operate on numpy arrays of shape [1000, 12].
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
            noise_std:             Std of Gaussian noise relative to signal std.
                                   0.05 means noise is 5% of the signal's own std.
            amplitude_scale_range: Uniform range for amplitude scaling factor.
            time_shift_max:        Maximum timesteps to shift in either direction.
            p:                     Probability of applying each individual transform.
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
        # Scale noise to the signal's own std so it's meaningful regardless
        # of amplitude. Simulates electrode contact noise and muscle artifact.
        std = signal.std() + 1e-8
        noise = np.random.normal(0, self.noise_std * std, signal.shape).astype(np.float32)
        return signal + noise

    def _scale_amplitude(self, signal: np.ndarray) -> np.ndarray:
        # Simulates variability in lead placement and patient body habitus.
        # Morphology is preserved — only the scale changes.
        scale = np.random.uniform(*self.amplitude_scale_range)
        return (signal * scale).astype(np.float32)

    def _time_shift(self, signal: np.ndarray) -> np.ndarray:
        # Rolls the signal along the time axis by a random number of steps.
        # Wrap-around is acceptable here because small shifts preserve
        # waveform morphology and the encoder aggregates over the sequence.
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
            split_dir:     Path to one of data/processed/{train,val,test}.
            tokenizer:     Initialized HuggingFace tokenizer.
            max_text_length: Token sequence length passed to the tokenizer.
            augmentation:  ECGAugmentation instance or None. When provided,
                           augmentation is applied only to samples whose label
                           is in AUGMENT_CLASSES. Pass None for val/test.
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
        signal = self.signals[idx].copy()  # copy so augmentation doesn't mutate the cache
        label  = int(self.labels[idx])

        # Augmentation applies only to minority classes and only during training.
        # Val/test datasets are constructed with augmentation=None.
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

def _worker_init_fn(worker_id: int) -> None:
    """
    Initialize RNG state for a DataLoader worker.

    PyTorch derives each worker seed from the DataLoader generator. The same
    seed is applied to NumPy and Python's random module so sample-level
    augmentation remains deterministic across worker processes.
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