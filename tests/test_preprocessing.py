"""Tests for persisted preprocessing artifacts."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

PROCESSED_DIR = Path(os.environ.get("PROCESSED_DATA_DIR", "data/processed"))

# Dataset-backed checks are skipped when preprocessing artifacts are unavailable
required_artifact = PROCESSED_DIR / "train" / "signals.npy"
if not required_artifact.exists():
    pytest.skip(
        "Processed PTB-XL artifacts are not available.",
        allow_module_level=True,
    )

SPLITS = ["train", "val", "test"]
NUM_LEADS = 12
TIMESTEPS = 1000
NUM_CLASSES = 5


def _split_dir(split: str) -> Path:
    return PROCESSED_DIR / split


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = _split_dir(split)
    signals = np.load(d / "signals.npy")
    labels = np.load(d / "labels.npy")
    with open(d / "reports.json", encoding="utf-8") as f:
        reports = json.load(f)
    return signals, labels, reports


@pytest.mark.parametrize("split", SPLITS)
def test_signal_shape(split):
    signals, _, _ = load_split(split)
    assert signals.ndim == 3
    assert signals.shape[1] == TIMESTEPS
    assert signals.shape[2] == NUM_LEADS


@pytest.mark.parametrize("split", SPLITS)
def test_array_lengths_consistent(split):
    signals, labels, reports = load_split(split)
    assert signals.shape[0] == len(labels) == len(reports)


@pytest.mark.parametrize("split", SPLITS)
def test_labels_valid_range(split):
    _, labels, _ = load_split(split)
    assert labels.dtype == np.int64
    assert labels.min() >= 0
    assert labels.max() < NUM_CLASSES


def test_train_signals_approximately_normalized():
    """Validate per-lead normalization of the training signals."""
    signals, _, _ = load_split("train")
    # Collapse record and time axes while preserving the lead dimension
    flat = signals.reshape(-1, NUM_LEADS)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    np.testing.assert_allclose(means, np.zeros(NUM_LEADS), atol=1e-2)
    np.testing.assert_allclose(stds, np.ones(NUM_LEADS), atol=3e-2)


def test_norm_stats_file_exists():
    path = PROCESSED_DIR / "norm_stats.json"
    assert path.exists()
    with open(path) as f:
        stats = json.load(f)
    assert "mean" in stats and "std" in stats
    assert len(stats["mean"]) == NUM_LEADS
    assert len(stats["std"]) == NUM_LEADS


def test_no_record_overlap_across_splits():
    """Verify that ECG records are unique across dataset splits."""
    id_sets = {}
    for split in SPLITS:
        ids = np.load(_split_dir(split) / "ecg_ids.npy")
        id_sets[split] = set(ids.tolist())

    assert id_sets["train"].isdisjoint(id_sets["val"]), "Train/val overlap"
    assert id_sets["train"].isdisjoint(id_sets["test"]), "Train/test overlap"
    assert id_sets["val"].isdisjoint(id_sets["test"]), "Val/test overlap"


def test_config_snapshot_exists_and_valid():
    path = PROCESSED_DIR / "config_snapshot.json"
    assert path.exists()
    with open(path) as f:
        snap = json.load(f)

    assert "label_map" in snap
    assert "class_weights" in snap
    assert "likelihood_threshold" in snap

    assert "filter" in snap
    assert "apply_bandpass" in snap["filter"]

    assert len(snap["class_weights"]) == NUM_CLASSES


@pytest.mark.parametrize("split", SPLITS)
def test_reports_are_strings(split):
    _, _, reports = load_split(split)
    assert all(isinstance(r, str) for r in reports)
