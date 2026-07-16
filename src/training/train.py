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
from src.training.focal_loss import FocalLoss
from src.training.utils import seed_everything

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
    model_cfg=None,
    run_notes: str | None = None,
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
        model_cfg:    ModelConfig instance.
        run_notes:    Optional notes describing the experiment.

    Returns:
        Path to the best saved checkpoint.
    """
    seed_everything(train_cfg.seed)
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

    def _serialize_param(value):
        if value is None:
            return "none"
        if isinstance(value, (list, tuple, set, dict)):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        return value

    if train_cfg.loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    elif train_cfg.loss_fn == "focal":
        criterion = FocalLoss(
            gamma=train_cfg.focal_gamma, 
            weight=class_weights
        )

    else:
        raise ValueError(f"Unknown loss function: {train_cfg.loss_fn}")

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = train_cfg.run_name or train_cfg.experiment_name
    best_ckpt_path = checkpoint_dir / f"best_{checkpoint_name}.pt"

    best_val_f1 = -1.0
    best_val_auc = float("nan")
    best_epoch = 0
    epochs_without_improvement = 0

    mlflow.set_experiment(train_cfg.experiment_name)
    with mlflow.start_run(run_name=train_cfg.run_name):
        # Log experiment settings so runs can be compared and reproduced from MLflow.
        aug = getattr(train_loader.dataset, "augmentation", None)

        params = {
            **(
                {k: _serialize_param(v) for k, v in model_cfg.__dict__.items()}
                if model_cfg is not None
                else {"mode": getattr(model, "mode", "unknown")}
            ),
            **{k: _serialize_param(v) for k, v in train_cfg.__dict__.items()},
            "batch_size": train_loader.batch_size,
            "augment_classes": str(sorted(aug.augment_classes)) if aug else "none",
            "aug_noise_std": aug.noise_std if aug else "none",
            "aug_amplitude_range": str(list(aug.amplitude_scale_range)) if aug else "none",
            "aug_time_shift_max": aug.time_shift_max if aug else "none",
            "aug_p": aug.p if aug else "none",
        }

        mlflow.log_params(params)

        if run_notes:
            mlflow.set_tag("run_notes", run_notes)
            mlflow.log_text(run_notes, "metadata/run_notes.txt")

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
                    "val_macro_auc": val_metrics["macro_auc"],
                    **{f"val_f1_{k}": v for k, v in val_metrics["per_class_f1"].items()},
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            # Validation macro F1 improved — reset patience and save checkpoint.
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_val_auc = val_metrics["macro_auc"]
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_macro_f1": best_val_f1,
                        "val_macro_auc": best_val_auc,
                        "train_cfg": train_cfg.__dict__,
                        "model_cfg": model_cfg.__dict__ if model_cfg is not None else None,
                        "augmentation": {
                            "augment_classes": sorted(aug.augment_classes),
                            "noise_std": aug.noise_std,
                            "amplitude_scale_range": list(aug.amplitude_scale_range),
                            "time_shift_max": aug.time_shift_max,
                            "p": aug.p,
                        } if aug else None,
                    },
                    best_ckpt_path,
                )

                mlflow.log_metric("best_val_macro_f1", best_val_f1, step=epoch)
                mlflow.log_metric("best_val_macro_auc", best_val_auc, step=epoch)
                log.info("  ↳ New best checkpoint saved (val macro F1: %.4f)", best_val_f1)

            else:
                # No improvement in validation macro F1 — increment patience counter
                # and stop training once patience is reached.
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
        mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
        log.info(
                "Training complete. Best val macro F1: %.4f | best val macro AUC: %.4f at epoch %d", 
                 best_val_f1,
                 best_val_auc,
                 best_epoch,
        )

    return best_ckpt_path