"""
Unit tests for the preprocessing pipeline.

These run against the processed output files, so preprocess_data.py must have
been run before executing this test module.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

PROCESSED_DIR = Path(os.environ.get("PROCESSED_DATA_DIR", "data/processed"))
SPLITS = ["train", "val", "test"]
NUM_LEADS = 12
TIMESTEPS = 1000
NUM_CLASSES = 5


def _split_dir(split: str) -> Path:
    return PROCESSED_DIR / split


# --- Helpers ---

def load_split(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = _split_dir(split)
    signals = np.load(d / "signals.npy")
    labels  = np.load(d / "labels.npy")
    with open(d / "reports.json", encoding="utf-8") as f:
        reports = json.load(f)
    return signals, labels, reports


# --- Shape and dtype tests ---

@pytest.mark.parametrize("split", SPLITS)
def test_signal_shape(split):
    signals, labels, reports = load_split(split)
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


# --- Normalization ---

def test_train_signals_approximately_normalized():
    """Training signals should be close to zero mean, unit variance per channel."""
    signals, _, _ = load_split("train")
    # signals shape: [N, 1000, 12] — compute stats over N and timesteps
    flat = signals.reshape(-1, NUM_LEADS)
    means = flat.mean(axis=0)
    stds  = flat.std(axis=0)
    np.testing.assert_allclose(means, np.zeros(NUM_LEADS), atol=1e-2)
    np.testing.assert_allclose(stds,  np.ones(NUM_LEADS),  atol=1e-2)


def test_norm_stats_file_exists():
    path = PROCESSED_DIR / "norm_stats.json"
    assert path.exists()
    with open(path) as f:
        stats = json.load(f)
    assert "mean" in stats and "std" in stats
    assert len(stats["mean"]) == NUM_LEADS
    assert len(stats["std"])  == NUM_LEADS


# --- No data leakage between splits ---

def test_no_patient_overlap_across_splits():
    """
    PTB-XL's strat_fold is designed to keep patients in a single split.
    We verify this via ecg_ids — not patient IDs directly — but the fold
    assignment guarantees it at the source.

    This test checks that the ecg_id sets are disjoint across splits,
    which confirms no record appears in more than one split.
    """
    id_sets = {}
    for split in SPLITS:
        ids = np.load(_split_dir(split) / "ecg_ids.npy")
        id_sets[split] = set(ids.tolist())

    assert id_sets["train"].isdisjoint(id_sets["val"]),  "Train/val overlap"
    assert id_sets["train"].isdisjoint(id_sets["test"]), "Train/test overlap"
    assert id_sets["val"].isdisjoint(id_sets["test"]),   "Val/test overlap"


# --- Config snapshot ---

def test_config_snapshot_exists_and_valid():
    path = PROCESSED_DIR / "config_snapshot.json"
    assert path.exists()
    with open(path) as f:
        snap = json.load(f)
    assert "label_map"          in snap
    assert "class_weights"      in snap
    assert "likelihood_threshold" in snap
    assert len(snap["class_weights"]) == NUM_CLASSES


# --- Reports ---

@pytest.mark.parametrize("split", SPLITS)
def test_reports_are_strings(split):
    _, _, reports = load_split(split)
