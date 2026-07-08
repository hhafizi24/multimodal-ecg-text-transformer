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

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from configs.config import ExportConfig, ModelConfig
from src.models.model import MultimodalECGClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class ExportableECGModel(nn.Module):
    """
    Thin wrapper that isolates the ONNX-exportable portion of the fusion model:
    signal encoder + fusion module + classifier.

    The text encoder (MedBERT.de + projection) is excluded, its output is
    passed in as a pre-computed embedding.
    """

    def __init__(self, model: MultimodalECGClassifier):
        super().__init__()
        self.signal_encoder = model.signal_encoder
        self.fusion         = model.fusion
        self.classifier     = model.classifier

    def forward(self, signal: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal:   [batch, 1000, 12]
            text_emb: [batch, hidden_dim] — projected text embedding from MedBERT.de
        Returns:
            logits: [batch, num_classes]
        """
        sig_emb = self.signal_encoder(signal)
        fused   = self.fusion(sig_emb, text_emb)
        return self.classifier(fused)


def export(cfg: ExportConfig, model_cfg: ModelConfig) -> None:
    log.info("Loading checkpoint from %s", cfg.checkpoint_path)
    full_model = MultimodalECGClassifier(model_cfg)
    state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    full_model.load_state_dict(state["model_state_dict"])
    full_model.eval()

    exportable = ExportableECGModel(full_model)
    exportable.eval()

    # Representative inputs for tracing — batch size 1 matches serving
    dummy_signal   = torch.randn(1, 1000, 12)
    dummy_text_emb = torch.randn(1, model_cfg.transformer_hidden_dim)

    log.info("Exporting to ONNX (opset %d)...", cfg.opset_version)
    torch.onnx.export(
        exportable,
        (dummy_signal, dummy_text_emb),
        cfg.onnx_path,
        opset_version=cfg.opset_version,
        input_names=["signal", "text_emb"],
        output_names=["logits"],
        dynamic_axes={
            "signal":   {0: "batch"},
            "text_emb": {0: "batch"},
            "logits":   {0: "batch"},
        },
        dynamo=False,
    )
    log.info("Saved ONNX model → %s", cfg.onnx_path)

    _verify_onnx_matches_pytorch(exportable, cfg.onnx_path, model_cfg, n_samples=50)

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
    n_samples: int = 50,
    atol: float = 1e-4,
) -> None:
    """
    Run n_samples random inputs through both PyTorch and ONNX Runtime and
    assert the outputs agree within atol. Raises if any sample exceeds the
    tolerance, which would indicate a tracing or export error.
    """
    log.info("Verifying ONNX output matches PyTorch on %d samples...", n_samples)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    max_diff = 0.0
    for _ in range(n_samples):
        signal   = torch.randn(1, 1000, 12)
        text_emb = torch.randn(1, model_cfg.transformer_hidden_dim)

        with torch.no_grad():
            pt_out = model(signal, text_emb).numpy()

        ort_out = session.run(
            None,
            {"signal": signal.numpy(), "text_emb": text_emb.numpy()},
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
    export(export_cfg, ModelConfig(mode="fusion"))
