"""
Tests for pretrained submodule weight loading.
"""

import torch

from configs.config import ModelConfig
from src.models.model import MultimodalECGClassifier
from src.training.checkpoint_utils import load_encoder_weights


def make_cfg(mode: str) -> ModelConfig:
    return ModelConfig(
        mode=mode,
        cnn_channels=[32, 64, 128],
        transformer_hidden_dim=64,
        transformer_num_heads=4,
        transformer_num_layers=1,
        text_model_name="GerMedBERT/medbert-512",
        text_projection_dim=64,
        fusion_num_heads=4,
        num_classes=5,
    )


def test_load_encoder_weights_signal_encoder(tmp_path):
    cfg = make_cfg("signal_only")
    source_model = MultimodalECGClassifier(cfg)

    checkpoint_path = tmp_path / "fake_signal_checkpoint.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    target_model = MultimodalECGClassifier(cfg)
    load_encoder_weights(target_model, checkpoint_path, "signal_encoder")

    for p1, p2 in zip(
        source_model.signal_encoder.parameters(),
        target_model.signal_encoder.parameters(),
    ):
        assert torch.equal(p1, p2)


def test_load_encoder_weights_text_encoder(tmp_path):
    cfg = make_cfg("text_only")
    source_model = MultimodalECGClassifier(cfg)

    checkpoint_path = tmp_path / "fake_text_checkpoint.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    target_model = MultimodalECGClassifier(cfg)
    load_encoder_weights(target_model, checkpoint_path, "text_encoder")

    for p1, p2 in zip(
        source_model.text_encoder.parameters(),
        target_model.text_encoder.parameters(),
    ):
        assert torch.equal(p1, p2)