"""
Fit validation-only bias per stage, then run a one-shot evaluation on the
held-out split for Stage A, B, and C.
"""

import argparse
from pathlib import Path

import torch

from configs.config import ModelConfig
from src.data.dataset import make_cached_fusion_dataloader
from src.models.model import MultimodalECGClassifier
from src.training.bias_adjustment import fit_logit_bias
from src.training.evaluate import collect_logits, evaluate_final


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["model_cfg"])
    model = MultimodalECGClassifier(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def run_stage(stage_name, checkpoint_path, val_loader, eval_loader, device, figures_dir, results_dir) -> dict:
    model = load_checkpoint_model(checkpoint_path, device)

    val_logits, val_labels = collect_logits(model, val_loader, device)
    bias = fit_logit_bias(val_logits, val_labels, num_classes=val_logits.shape[1])

    return evaluate_final(model, eval_loader, device, bias, stage_name, figures_dir, results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--fusion-cache-dir", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--stage-c-checkpoint", required=True)
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stages = [
        ("stage_a", args.stage_a_checkpoint),
        ("stage_b", args.stage_b_checkpoint),
        ("stage_c", args.stage_c_checkpoint),
    ]

    existing = [Path(args.results_dir) / f"{name}_final_metrics.json" for name, _ in stages]
    if any(p.exists() for p in existing) and not args.force:
        raise FileExistsError(
            "Final results already exist. Pass --force to intentionally overwrite."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(split: str):
        return make_cached_fusion_dataloader(
            labels_dir=f"{args.processed_dir}/{split}",
            signal_embedding_path=f"{args.fusion_cache_dir}/{split}/signal_embeddings.npy",
            text_embedding_path=f"{args.fusion_cache_dir}/{split}/text_embeddings.npy",
            batch_size=args.batch_size,
            shuffle=False,
        )

    val_loader = make_loader("val")
    eval_loader = make_loader(args.test_split)

    for stage_name, checkpoint_path in stages:
        results = run_stage(
            stage_name, checkpoint_path, val_loader, eval_loader,
            device, args.figures_dir, args.results_dir,
        )
        print(f"{stage_name}: raw F1={results['raw']['macro_f1']:.4f}, adjusted F1={results['adjusted']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()