"""
Entry point for the PTB-XL preprocessing pipeline.

Usage:
    python scripts/preprocess_data.py \
        --raw_data_dir data/raw \
        --processed_data_dir data/processed \
        --likelihood_threshold 50.0
"""

import argparse
from src.data.preprocess import run


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL for training.")
    parser.add_argument("--raw_data_dir",       default="data/raw",       type=str)
    parser.add_argument("--processed_data_dir", default="data/processed", type=str)
    parser.add_argument("--likelihood_threshold", default=50.0,           type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        likelihood_threshold=args.likelihood_threshold,
    )
