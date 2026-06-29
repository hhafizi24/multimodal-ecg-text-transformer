"""
Entry point for the PTB-XL preprocessing pipeline.

Usage:
    python scripts/preprocess_data.py \
        --raw_data_dir data/raw \
        --processed_data_dir data/processed \
        --likelihood_threshold 0.5
"""

import argparse
from src.data.preprocess import run


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL for training.")
    parser.add_argument("--raw_data_dir",         default="data/raw",       type=str)
    parser.add_argument("--processed_data_dir",   default="data/processed", type=str)
    parser.add_argument("--likelihood_threshold", default=0.5,              type=float)
    parser.add_argument("--apply_bandpass", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--filter_low_hz",        default=0.5,              type=float)
    parser.add_argument("--filter_high_hz",       default=40.0,             type=float)
    parser.add_argument("--filter_order",         default=4,                type=int)
    parser.add_argument("--sampling_rate",        default=100,              type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        likelihood_threshold=args.likelihood_threshold,
        apply_bandpass=args.apply_bandpass,
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
        filter_order=args.filter_order,
        sampling_rate=args.sampling_rate,
    )