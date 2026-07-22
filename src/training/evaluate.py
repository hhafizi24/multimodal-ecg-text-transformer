"""
Evaluation utilities for validation and test-set analysis.

Includes classification metrics, reporting, and deterministic text ablation.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

LABEL_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    text_available: bool | None = None,
) -> dict:
    """
    Evaluate a model and return loss and classification metrics.

    Set `text_available` to override text availability during fusion ablations.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            labels = batch["label"].to(device)

            availability_mask = None
            if text_available is not None:
                availability_mask = torch.full(
                    (labels.size(0),),
                    text_available,
                    dtype=torch.bool,
                    device=device,
                )
            if "signal_embedding" in batch:
                signal, input_ids, attention_mask, cached_embedding = None, None, None, None
                signal_embedding = batch["signal_embedding"].to(device)
                text_embedding   = batch["text_embedding"].to(device)
            elif "text_embedding" in batch:
                signal = batch["signal"].to(device)
                input_ids, attention_mask = None, None
                cached_embedding = batch["text_embedding"].to(device)
                signal_embedding, text_embedding = None, None
            else:
                signal         = batch["signal"].to(device)
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cached_embedding, signal_embedding, text_embedding = None, None, None

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(
                    signal=signal,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cached_embedding=cached_embedding,
                    signal_embedding=signal_embedding,
                    text_embedding=text_embedding,
                    text_available=availability_mask,
                )
            logits = logits.float()
            probs = torch.softmax(logits, dim=-1)

            if criterion is not None:
                total_loss += criterion(logits, labels).item()

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    macro_precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = {LABEL_NAMES[i]: float(per_class[i]) for i in range(len(LABEL_NAMES))}
    try:
        macro_auc = roc_auc_score(
            all_labels,
            np.array(all_probs),
            multi_class="ovr",
            average="macro",
            labels=list(range(len(LABEL_NAMES))),
        )
    except ValueError:
        macro_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "per_class_f1": per_class_f1,
        "loss": total_loss / len(loader) if criterion is not None else None,
        }
    return metrics


def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage_name: str,
    figures_dir: str,
    results_dir: str,
) -> dict:
    """
    Evaluate a model and save its metrics and confusion matrix.

    Args:
        model: Model to evaluate.
        loader: Evaluation DataLoader.
        device: Inference device.
        stage_name: Identifier used in output filenames and figure titles.
        figures_dir: Directory for confusion matrix figures.
        results_dir: Directory for metric files.

    Returns:
        Evaluation metrics.
    """
    metrics = evaluate(model, loader, device)

    # Full sklearn report for the log
    all_preds, all_labels = _collect_predictions(model, loader, device)
    report = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0
    )
    log.info("Classification report — %s:\n%s", stage_name, report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    _save_confusion_matrix(cm, stage_name, figures_dir)

    # Per-stage metrics JSON
    results_path = Path(results_dir) / f"{stage_name}_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics → %s", results_path)

    return metrics


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["signal"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
    return all_preds, all_labels


def _save_confusion_matrix(cm: np.ndarray, stage_name: str, figures_dir: str) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Normalize each row to make performance comparable across classes.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )   

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {stage_name.replace('_', ' ').title()}")
    plt.tight_layout()

    out_path = figures_dir / f"confusion_matrix_{stage_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", out_path)