"""Hydra-Zen training entry point for signal, text, and fusion models."""

from datetime import datetime

import torch
from hydra_zen import zen

import configs.training_presets  
from configs.config import DataConfig, ModelConfig, TrainingConfig
from src.data.dataset import make_cached_fusion_dataloader, make_dataloader
from src.data.text import load_tokenizer
from src.models.model import MultimodalECGClassifier
from src.training.train import train
from src.training.checkpoint_utils import load_encoder_weights


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    fusion_cache_dir: str,
):
    if model_cfg.mode == "fusion":
        if not fusion_cache_dir:
            raise ValueError(
                "fusion_cache_dir is required for cached fusion training."
            )

        train_loader = make_cached_fusion_dataloader(
            labels_dir=f"{data_cfg.processed_data_dir}/train",
            signal_embedding_path=f"{fusion_cache_dir}/train/signal_embeddings.npy",
            text_embedding_path=f"{fusion_cache_dir}/train/text_embeddings.npy",
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            seed=train_cfg.seed,
        )
        val_loader = make_cached_fusion_dataloader(
            labels_dir=f"{data_cfg.processed_data_dir}/val",
            signal_embedding_path=f"{fusion_cache_dir}/val/signal_embeddings.npy",
            text_embedding_path=f"{fusion_cache_dir}/val/text_embeddings.npy",
            batch_size=data_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            seed=train_cfg.seed,
        )
    else:
        tokenizer = load_tokenizer(model_cfg.text_model_name) if model_cfg.mode == "text_only" else None
        train_loader = make_dataloader(
            split_dir=f"{data_cfg.processed_data_dir}/train",
            tokenizer=tokenizer,
            batch_size=data_cfg.batch_size,
            max_text_length=data_cfg.max_text_length,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            seed=train_cfg.seed,
        )
        val_loader = make_dataloader(
            split_dir=f"{data_cfg.processed_data_dir}/val",
            tokenizer=tokenizer,
            batch_size=data_cfg.batch_size,
            max_text_length=data_cfg.max_text_length,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            seed=train_cfg.seed,
        )
    return train_loader, val_loader


def run(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: str = "auto",
    fusion_cache_dir: str = "",
    signal_checkpoint: str | None = None,
    text_checkpoint: str | None = None,
):
    resolved_device = resolve_device(device)

    if not train_cfg.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_cfg.run_name = f"{model_cfg.mode}_{timestamp}"

    model = MultimodalECGClassifier(model_cfg)

    if model_cfg.mode == "fusion":
        if not signal_checkpoint or not text_checkpoint:
            raise ValueError(
                "Fusion training requires signal_checkpoint and "
                "text_checkpoint so the saved model contains both "
                "pretrained encoders."
            )

        load_encoder_weights(
            model,
            signal_checkpoint,
            "signal_encoder",
            map_location="cpu",
        )
        load_encoder_weights(
            model,
            text_checkpoint,
            "text_encoder",
            map_location="cpu",
        )

    train_loader, val_loader = build_dataloaders(
        model_cfg,
        data_cfg,
        train_cfg,
        fusion_cache_dir,
    )

    best_ckpt_path = train(
        model,
        train_loader,
        val_loader,
        train_cfg,
        resolved_device,
        model_cfg,
    )
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    zen(run).hydra_main(config_path=None, config_name="signal_only", version_base=None)