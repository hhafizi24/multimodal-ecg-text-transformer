"""
Export the fusion model to ONNX and apply dynamic quantization.

MedBERT.de and the projection layer remain outside the ONNX graph. The exported
graph accepts raw ECG signals and pre-computed projected text embeddings,
matching the deployment inference boundary.

Usage:
    python scripts/export_onnx.py \
        --checkpoint_path models/best_fusion.pt \
        --onnx_path models/model.onnx \
        --quantized_onnx_path models/model_quantized.onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from configs.config import ExportConfig, ModelConfig
from src.models.classifier import ClassificationHead
from src.models.fusion import CrossAttentionFusion
from src.models.signal_encoder import SignalEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class ExportableECGModel(nn.Module):
    """ONNX inference graph for signal encoding, fusion, and classification."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.signal_encoder = SignalEncoder(cfg)
        self.fusion = CrossAttentionFusion(cfg)
        self.classifier = ClassificationHead(cfg)

    def forward(
        self,
        signal: torch.Tensor,
        text_embedding: torch.Tensor,
        text_available: torch.Tensor,
    ) -> torch.Tensor:
        signal_embedding = self.signal_encoder(signal)
        fused = self.fusion(signal_embedding, text_embedding, text_available=text_available)
        return self.classifier(fused)


def load_exportable_model(checkpoint_path: str | Path) -> tuple[ExportableECGModel, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**checkpoint["model_cfg"])

    if model_cfg.mode != "fusion":
        raise ValueError("ONNX export requires a fusion checkpoint.")

    model = ExportableECGModel(model_cfg)
    full_state = checkpoint["model_state_dict"]

    for name in ("signal_encoder", "fusion", "classifier"):
        prefix = f"{name}."
        state = {
            key[len(prefix):]: value
            for key, value in full_state.items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError(f"Checkpoint contains no weights for {name!r}.")
        getattr(model, name).load_state_dict(state, strict=True)

    model.eval()
    return model, model_cfg


def export(cfg: ExportConfig) -> None:
    log.info("Loading checkpoint from %s", cfg.checkpoint_path)
    exportable, model_cfg = load_exportable_model(cfg.checkpoint_path)

    Path(cfg.onnx_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.quantized_onnx_path).parent.mkdir(parents=True, exist_ok=True)

    # Representative inputs for tracing — batch size 1 matches serving
    dummy_signal = torch.randn(1, 1000, 12)
    dummy_text_emb = torch.randn(1, model_cfg.transformer_hidden_dim)
    dummy_text_available = torch.ones(1, dtype=torch.bool)

    log.info("Exporting to ONNX (opset %d)...", cfg.opset_version)
    torch.onnx.export(
        exportable,
        (dummy_signal, dummy_text_emb, dummy_text_available),
        cfg.onnx_path,
        opset_version=cfg.opset_version,
        input_names=["signal", "text_embedding", "text_available"],
        output_names=["logits"],
        dynamic_axes={
            "signal": {0: "batch"},
            "text_embedding": {0: "batch"},
            "text_available": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )
    log.info("Saved ONNX model → %s", cfg.onnx_path)

    _verify_onnx_matches_pytorch(exportable, cfg.onnx_path, model_cfg, trials_per_case=10)

    log.info("Applying dynamic quantization...")
    quantize_dynamic(
        cfg.onnx_path,
        cfg.quantized_onnx_path,
        weight_type=QuantType.QInt8,
    )
    log.info("Saved quantized ONNX model → %s", cfg.quantized_onnx_path)


def _verify_onnx_matches_pytorch(
    model: nn.Module,
    onnx_path: str,
    model_cfg: ModelConfig,
    trials_per_case: int = 10,
    atol: float = 1e-4,
) -> None:
    """
    Verify ONNX logits against PyTorch across batch sizes and text-availability masks.

    Raises if the maximum absolute difference exceeds ``atol``.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    text_available_patterns = {
        "all_present": lambda n: torch.ones(n, dtype=torch.bool),
        "all_absent": lambda n: torch.zeros(n, dtype=torch.bool),
        "mixed": lambda n: torch.arange(n) % 2 == 0,
    }

    max_diff = 0.0
    for batch_size in (1, 4):
        for pattern_name, pattern_fn in text_available_patterns.items():
            if batch_size == 1 and pattern_name == "mixed":
                continue

            for _ in range(trials_per_case):
                signal = torch.randn(batch_size, 1000, 12)
                text_emb = torch.randn(batch_size, model_cfg.transformer_hidden_dim)
                text_available = pattern_fn(batch_size)

                with torch.no_grad():
                    pt_out = model(signal, text_emb, text_available).numpy()

                ort_out = session.run(
                    None,
                    {
                        "signal": signal.numpy(),
                        "text_embedding": text_emb.numpy(),
                        "text_available": text_available.numpy(),
                    },
                )[0]

                diff = float(np.abs(pt_out - ort_out).max())
                max_diff = max(max_diff, diff)

    log.info("Max absolute difference PyTorch vs ONNX: %.2e", max_diff)
    assert max_diff < atol, (
        f"ONNX output diverges from PyTorch: max diff {max_diff:.2e} > atol {atol:.2e}"
    )
    log.info("ONNX verification passed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",      default=ExportConfig.checkpoint_path)
    parser.add_argument("--onnx_path",            default=ExportConfig.onnx_path)
    parser.add_argument("--quantized_onnx_path",  default=ExportConfig.quantized_onnx_path)
    parser.add_argument("--opset_version",        default=ExportConfig.opset_version, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_cfg = ExportConfig(
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.onnx_path,
        quantized_onnx_path=args.quantized_onnx_path,
        opset_version=args.opset_version,
    )
    export(export_cfg)
