"""
Preprocessing pipeline for PTB-XL.

Derives single-label diagnostic superclasses from scp_codes, assigns train/val/test
splits using the dataset's strat_fold column, loads and normalizes waveforms, and
saves processed splits to disk alongside normalization stats and a config snapshot.
"""

import ast
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Diagnostic superclass → integer label
LABEL_MAP = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}

# strat_fold assignments per the PTB-XL benchmark protocol
FOLD_SPLITS = {
    "train": list(range(1, 9)),   # folds 1–8
    "val":   [9],
    "test":  [10],
}


def load_scp_lookup(scp_path: str) -> pd.DataFrame:
    """Return scp_statements filtered to diagnostic codes only."""
    scp = pd.read_csv(scp_path, index_col=0)
    return scp[scp["diagnostic"] == 1.0]


def derive_label(scp_codes_str: str, scp_lookup: pd.DataFrame, threshold: float = 0.5) -> str | None:
    """
    Map a raw scp_codes string to a single diagnostic superclass label.

    Accumulates likelihood scores per superclass, then returns the dominant class
    only if it clears the threshold. Returns None if the record is ambiguous or
    contains no diagnostic codes.
    """
    try:
        codes = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return None

    scores: dict[str, float] = {}
    for code, likelihood in codes.items():
        if code in scp_lookup.index:
            sc = scp_lookup.loc[code, "diagnostic_class"]
            if isinstance(sc, str):
                scores[sc] = scores.get(sc, 0.0) + likelihood

    if not scores:
        return None

    dominant = max(scores, key=scores.get)
    total_score = sum(scores.values())
    if total_score == 0:
        return None
    
    dominant_fraction = scores[dominant] / total_score

    if dominant_fraction >= threshold and dominant in LABEL_MAP:
        return dominant
    
    return None


def load_waveform(filename_lr: str, raw_data_dir: str) -> np.ndarray | None:
    """Load a 100 Hz waveform via wfdb. Returns float32 array [1000, 12] or None on error."""
    path = os.path.join(raw_data_dir, filename_lr)
    try:
        signal, _ = wfdb.rdsamp(path)

        if signal.shape != (1000, 12):
            log.warning("Unexpected waveform shape for %s: %s", filename_lr, signal.shape)
            return None

        return signal.astype(np.float32)
    except Exception as e:
        log.warning("Failed to load %s: %s", filename_lr, e)
        return None


def compute_norm_stats(signals: list[np.ndarray]) -> dict:
    """
    Compute per-channel mean and std across all training waveforms.

    Stats are fit on the training split only and applied to all splits.
    Each channel is treated independently — shape of each signal is [1000, 12].
    """
    stacked = np.concatenate(signals, axis=0)  # [N*1000, 12]
    means = stacked.mean(axis=0).tolist()       # [12]
    stds  = stacked.std(axis=0).tolist()        # [12]
    return {"mean": means, "std": stds}


def normalize(signal: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    # Avoid division by zero for flat channels
    std_arr  = np.where(std_arr < 1e-8, 1.0, std_arr)
    return (signal - mean_arr) / std_arr


def run(
    raw_data_dir: str,
    processed_data_dir: str,
    likelihood_threshold: float = 0.5,
) -> None:
    raw_data_dir       = str(Path(raw_data_dir).resolve())
    processed_data_dir = Path(processed_data_dir).resolve()
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Load metadata ---
    log.info("Loading metadata...")
    df = pd.read_csv(
        os.path.join(raw_data_dir, "ptbxl_database.csv"),
        index_col="ecg_id",
    )
    scp_lookup = load_scp_lookup(os.path.join(raw_data_dir, "scp_statements.csv"))

    # --- Derive labels ---
    log.info("Deriving diagnostic superclass labels...")
    df["label_str"] = df["scp_codes"].apply(
        lambda x: derive_label(x, scp_lookup, likelihood_threshold)
    )
    df["label"] = df["label_str"].map(LABEL_MAP)

    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    log.info("Retained %d / %d records after label filtering.", len(df), before)

    label_counts = df["label_str"].value_counts().to_dict()
    log.info("Label distribution: %s", label_counts)

    # --- Assign splits ---
    split_mask: dict[str, pd.Series] = {
        split: df["strat_fold"].isin(folds)
        for split, folds in FOLD_SPLITS.items()
    }
    splits: dict[str, pd.DataFrame] = {
        split: df[mask] for split, mask in split_mask.items()
    }
    for split, sdf in splits.items():
        log.info("Split '%s': %d records", split, len(sdf))

    # --- Load waveforms and reports per split ---
    processed: dict[str, dict] = {}
    train_signals: list[np.ndarray] = []

    for split, sdf in splits.items():
        log.info("Loading waveforms for split '%s'...", split)
        signals, reports, labels, ecg_ids = [], [], [], []

        for ecg_id, row in sdf.iterrows():
            sig = load_waveform(row["filename_lr"], raw_data_dir)
            if sig is None:
                continue
            signals.append(sig)
            reports.append(row["report"] if isinstance(row["report"], str) else "")
            labels.append(row["label"])
            ecg_ids.append(ecg_id)

        processed[split] = {
            "signals":  signals,
            "reports":  reports,
            "labels":   labels,
            "ecg_ids":  ecg_ids,
        }

        if split == "train":
            train_signals = signals

    # --- Normalization stats from training split only ---
    log.info("Computing normalization statistics from training split...")
    norm_stats = compute_norm_stats(train_signals)

    norm_stats_path = processed_data_dir / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    log.info("Saved norm stats → %s", norm_stats_path)

    # --- Apply normalization and save splits ---
    mean, std = norm_stats["mean"], norm_stats["std"]

    for split, data in processed.items():
        split_dir = processed_data_dir / split
        split_dir.mkdir(exist_ok=True)

        norm_signals = np.stack(
            [normalize(s, mean, std) for s in data["signals"]], axis=0
        )  # [N, 1000, 12]

        np.save(split_dir / "signals.npy",  norm_signals)
        np.save(split_dir / "labels.npy",   np.array(data["labels"], dtype=np.int64))
        np.save(split_dir / "ecg_ids.npy",  np.array(data["ecg_ids"], dtype=np.int64))

        # Reports saved as a plain text file, one per line
        with open(split_dir / "reports.json", "w", encoding="utf-8") as f:
            json.dump(data["reports"], f, ensure_ascii=False, indent=2)

        log.info(
            "Saved '%s' split: %d samples, signals shape %s",
            split, len(data["labels"]), norm_signals.shape,
        )

    # --- Class weights (inverse frequency, training split only) ---
    train_labels = np.array(processed["train"]["labels"], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=len(LABEL_MAP))
    class_weights = (class_counts.sum() / (len(LABEL_MAP) * class_counts)).tolist()

    # --- Config snapshot ---
    snapshot = {
        "likelihood_threshold": likelihood_threshold,
        "label_map": LABEL_MAP,
        "fold_splits": FOLD_SPLITS,
        "label_counts": label_counts,
        "class_weights": class_weights,
        "norm_stats_path": str(norm_stats_path),
    }
    snapshot_path = processed_data_dir / "config_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    log.info("Saved config snapshot → %s", snapshot_path)
    log.info("Preprocessing complete.")