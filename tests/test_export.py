"""
ONNX export smoke test.

Validates that the ONNX-exportable portion of the model (signal encoder,
fusion module, and classifier) exports cleanly and produces correct outputs.
DistilBERT is excluded because it runs outside the ONNX graph in deployment.
"""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from configs.config import ModelConfig
from src.models.classifier import ClassificationHead
from src.models.fusion import CrossAttentionFusion
from src.models.signal_encoder import SignalEncoder

HIDDEN_DIM  = 64   # small for speed — testing exportability, not accuracy
NUM_CLASSES = 5
OPSET       = 14


def make_small_cfg() -> ModelConfig:
    # text_model_name and text_projection_dim are present in ModelConfig but
    # unused here — we never instantiate TextEncoder in these tests.
    return ModelConfig(
        mode="fusion",
        cnn_channels=[8, 16, 32],
        cnn_kernel_size=7,
        transformer_hidden_dim=HIDDEN_DIM,
        transformer_num_heads=4,
        transformer_num_layers=1,
        transformer_dropout=0.0,
        text_model_name="distilbert-base-multilingual-cased",
        text_projection_dim=HIDDEN_DIM,
        fusion_num_heads=4,
        num_classes=NUM_CLASSES,
    )


class _ExportableModel(nn.Module):
    """
    Lightweight test stand-in for ExportableECGModel.

    Assembles signal encoder + fusion + classifier directly without
    touching MultimodalECGClassifier or the text encoder.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.signal_encoder = SignalEncoder(cfg)
        self.fusion         = CrossAttentionFusion(cfg)
        self.classifier     = ClassificationHead(cfg)

    def forward(self, signal: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        sig_emb = self.signal_encoder(signal)
        fused   = self.fusion(sig_emb, text_emb)
        return self.classifier(fused)


def _export_to_tempfile(model: nn.Module, signal: torch.Tensor, text_emb: torch.Tensor, tmpdir: str) -> str:
    onnx_path = str(Path(tmpdir) / "model.onnx")
    torch.onnx.export(
        model,
        (signal, text_emb),
        onnx_path,
        opset_version=OPSET,
        input_names=["signal", "text_emb"],
        output_names=["logits"],
        dynamic_axes={
            "signal":   {0: "batch"},
            "text_emb": {0: "batch"},
            "logits":   {0: "batch"},
        },
        dynamo=False,
    )
    return onnx_path


def test_onnx_export_and_inference():
    """Export succeeds, graph passes ONNX validation, output shape is correct."""
    model = _ExportableModel(make_small_cfg())
    model.eval()

    dummy_signal   = torch.randn(1, 1000, 12)
    dummy_text_emb = torch.randn(1, HIDDEN_DIM)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export_to_tempfile(model, dummy_signal, dummy_text_emb, tmpdir)

        onnx.checker.check_model(onnx.load(onnx_path))

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.run(
            None,
            {"signal": dummy_signal.numpy(), "text_emb": dummy_text_emb.numpy()},
        )

        assert len(outputs) == 1
        assert outputs[0].shape == (1, NUM_CLASSES)


def test_onnx_output_matches_pytorch():
    """ONNX Runtime output must agree with PyTorch within numerical tolerance."""
    model = _ExportableModel(make_small_cfg())
    model.eval()

    signal   = torch.randn(1, 1000, 12)
    text_emb = torch.randn(1, HIDDEN_DIM)

    with torch.no_grad():
        pt_logits = model(signal, text_emb).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export_to_tempfile(model, signal, text_emb, tmpdir)
        session   = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_logits = session.run(
            None, {"signal": signal.numpy(), "text_emb": text_emb.numpy()}
        )[0]

    max_diff = float(np.abs(pt_logits - ort_logits).max())
    assert max_diff < 1e-4, f"PyTorch vs ONNX max diff: {max_diff:.2e}"


def test_onnx_dynamic_batch():
    """Export with batch=1, run inference with batch=4 to verify dynamic axis."""
    model = _ExportableModel(make_small_cfg())
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = _export_to_tempfile(
            model, torch.randn(1, 1000, 12), torch.randn(1, HIDDEN_DIM), tmpdir
        )
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.run(
            None,
            {
                "signal":   torch.randn(4, 1000, 12).numpy(),
                "text_emb": torch.randn(4, HIDDEN_DIM).numpy(),
            },
        )
