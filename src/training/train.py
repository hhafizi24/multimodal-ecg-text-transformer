"""
Training loop for all three model stages.

Designed for experiments with configs instantiated directly as
Python objects. Handles MLflow logging, checkpointing, and learning-rate
scheduling in one reusable training entry point.
"""

import json
import logging
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate

log = logging.getLogger(__name__)


def load_class_weights(snapshot_path: str, device: torch.device) -> torch.Tensor | None:
    with open(snapshot_path) as f:
        snapshot = json.load(f)
    weights = torch.tensor(snapshot["class_weights"], dtype=torch.float32)
    return weights.to(device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg,
    device: torch.device,
) -> Path:
    """
    Run the full training loop for one stage.

    Logs per-epoch metrics to MLflow, saves the best checkpoint by val macro F1,
    and returns the path to the best checkpoint.

    Args:
        model:        Instantiated MultimodalECGClassifier.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        train_cfg:    TrainingConfig instance.
        device:       torch.device to train on.

    Returns:
        Path to the best saved checkpoint.
    """
    model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    if train_cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)
    elif train_cfg.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {train_cfg.scheduler!r}")

    class_weights = None
    if train_cfg.use_class_weights:
        class_weights = load_class_weights(train_cfg.class_weights_path, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"best_{train_cfg.experiment_name}.pt"

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    mlflow.set_experiment(train_cfg.experiment_name)
    with mlflow.start_run():
        # Log the full config so every run is reproducible from MLflow alone
        mlflow.log_params({
            "mode":           model.mode,
            "learning_rate":  train_cfg.learning_rate,
            "num_epochs":     train_cfg.num_epochs,
            "weight_decay":   train_cfg.weight_decay,
            "scheduler":      train_cfg.scheduler,
            "use_class_weights": train_cfg.use_class_weights,
            "batch_size":     train_loader.batch_size,
            "early_stopping_patience": train_cfg.early_stopping_patience,
        })

        for epoch in range(1, train_cfg.num_epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                signal         = batch["signal"].to(device)
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(signal, input_ids, attention_mask)
                loss   = criterion(logits, labels)
                loss.backward()

                # Gradient clipping — stabilizes training without tuning LR aggressively
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            train_loss = total_loss / len(train_loader)
            val_metrics = evaluate(model, val_loader, device, criterion)

            log.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f",
                epoch, train_cfg.num_epochs,
                train_loss, val_metrics["loss"], val_metrics["macro_f1"],
            )

            mlflow.log_metrics(
                {
                    "train_loss":   train_loss,
                    "val_loss":     val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_precision": val_metrics["macro_precision"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    **{f"val_f1_{k}": v for k, v in val_metrics["per_class_f1"].items()},
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch":      epoch,
                        "model_state_dict": model.state_dict(),
                        "val_macro_f1":     best_val_f1,
                        "train_cfg":        train_cfg.__dict__,
                    },
                    best_ckpt_path,
                )

                mlflow.log_metric("best_val_macro_f1", best_val_f1, step=epoch)
                log.info("  ↳ New best checkpoint saved (val macro F1: %.4f)", best_val_f1)

            else:
                epochs_without_improvement += 1

                if (
                    train_cfg.early_stopping_patience is not None
                    and epochs_without_improvement >= train_cfg.early_stopping_patience
                ):
                    log.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        epochs_without_improvement,
                    )
                    break

        mlflow.log_metric("best_epoch", best_epoch)            
        mlflow.log_artifact(str(best_ckpt_path))
        log.info(
                "Training complete. Best val macro F1: %.4f at epoch %d", 
                 best_val_f1,
                 best_epoch,
        )

    return best_ckpt_path