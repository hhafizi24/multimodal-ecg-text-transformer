"""
Precompute frozen MedBERT.de CLS features for each PTB-XL split.

Run once. Saves an [N, hidden_size] float32 array per split to
data/processed/{split}/text_features.npy, row-aligned with signals.npy
and labels.npy. Enables cached-embedding training for Stage B and Stage C
for as long as the text backbone stays fully frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from configs.config import ModelConfig
from src.models.text_encoder import TextEncoder


@torch.inference_mode()
def precompute_split(
    split_dir: Path,
    text_encoder: TextEncoder,
    tokenizer,
    device: torch.device,
    max_text_length: int,
    batch_size: int = 64,
) -> np.ndarray:
    with open(split_dir / "reports.json", encoding="utf-8") as f:
        reports = json.load(f)

    features = []
    for i in range(0, len(reports), batch_size):
        encoded = tokenizer(
            reports[i : i + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        feats = text_encoder.extract_frozen_features(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )
        features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--text-model-name", default="GerMedBERT/medbert-512")
    parser.add_argument("--max-text-length", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    text_encoder = TextEncoder(ModelConfig(text_model_name=args.text_model_name)).to(device)
    text_encoder.eval()

    for split in ("train", "val", "test"):
        split_dir = Path(args.processed_dir) / split
        features = precompute_split(
            split_dir, text_encoder, tokenizer, device, args.max_text_length,
        )
        out_path = split_dir / "text_features.npy"
        np.save(out_path, features)
        print(f"{split}: saved {features.shape} → {out_path}")


if __name__ == "__main__":
    main()