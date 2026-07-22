"""
Precompute frozen signal and text embeddings for each PTB-XL split.

Assembles a fusion-mode model from two pretrained checkpoints, then caches
both branches' output embeddings for fast downstream training.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.config import ModelConfig
from src.data.dataset import make_dataloader
from src.data.text import load_tokenizer
from src.models.model import MultimodalECGClassifier
from src.training.checkpoint_utils import load_encoder_weights

SIGNAL_FIELDS = (
    "cnn_channels", "cnn_kernel_size", "cnn_kernel_sizes", "cnn_activation",
    "cnn_pooling", "cnn_dropout", "cnn_stem",
    "multiscale_branch_channels", "multiscale_kernel_sizes",
    "dwsep_channels", "dwsep_kernel_sizes", "se_reduction_ratio",
    "transformer_hidden_dim", "transformer_num_heads",
    "transformer_num_layers", "transformer_dropout",
)


def build_fusion_config(signal_cfg: dict, text_cfg: dict) -> ModelConfig:
    """Merge a signal-encoder config and a text-encoder config into one fusion config."""
    merged = dict(text_cfg)
    merged.update({k: signal_cfg[k] for k in SIGNAL_FIELDS if k in signal_cfg})
    merged["mode"] = "fusion"
    return ModelConfig(**merged)


@torch.inference_mode()
def precompute_split(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    signal_out, text_out = [], []

    for batch in loader:
        signal = batch["signal"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        signal_out.append(model.signal_encoder(signal).cpu().numpy())
        text_out.append(model.text_encoder(input_ids, attention_mask).cpu().numpy())

    return np.concatenate(signal_out, axis=0), np.concatenate(text_out, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--signal-checkpoint", required=True)
    parser.add_argument("--text-checkpoint", required=True)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signal_ckpt = torch.load(args.signal_checkpoint, map_location="cpu", weights_only=False)
    text_ckpt = torch.load(args.text_checkpoint, map_location="cpu", weights_only=False)

    model_cfg = build_fusion_config(signal_ckpt["model_cfg"], text_ckpt["model_cfg"])
    model = MultimodalECGClassifier(model_cfg).to(device)

    load_encoder_weights(
        model, 
        args.signal_checkpoint, 
        "signal_encoder", 
        map_location="cpu"
    )
    load_encoder_weights(
        model,
        args.text_checkpoint,
        "text_encoder",
        map_location="cpu"
    )
    model.eval()

    tokenizer = load_tokenizer(model_cfg.text_model_name)
    cache_dir = Path(args.cache_dir)

    for split in ("train", "val", "test"):
        split_dir = Path(args.processed_dir) / split
        loader = make_dataloader(
            split_dir=str(split_dir),
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_text_length=args.max_text_length,
            shuffle=False,
            num_workers=2,
            augmentation=None,
            seed=42,
        )

        signal_embeddings, text_embeddings = precompute_split(model, loader, device)

        out_dir = cache_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "signal_embeddings.npy", signal_embeddings)
        np.save(out_dir / "text_embeddings.npy", text_embeddings)

        print(f"{split}: signal {signal_embeddings.shape}, text {text_embeddings.shape} -> {out_dir}")


if __name__ == "__main__":
    main()