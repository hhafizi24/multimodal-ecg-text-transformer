"""
Tests for model architecture: forward passes, output shapes, parameter freezing,
component instantiation, and gradient flow.
"""

import pytest
import torch

from configs.config import ModelConfig
from src.models.model import MultimodalECGClassifier

BATCH = 4
SEQ_LEN = 128
NUM_CLASSES = 5
HIDDEN_DIM = 256


def make_cfg(mode: str) -> ModelConfig:
    return ModelConfig(
        mode=mode,
        cnn_channels=[32, 64, 128],
        cnn_kernel_size=7,
        cnn_kernel_sizes=None,
        cnn_activation="gelu",
        cnn_pooling="none",
        classifier_hidden_dim=None,
        classifier_dropout=0.0,
        transformer_hidden_dim=HIDDEN_DIM,
        transformer_num_heads=8,
        transformer_num_layers=2,  # smaller than production to keep tests fast
        transformer_dropout=0.0,
        text_model_name="distilbert-base-multilingual-cased",
        text_projection_dim=HIDDEN_DIM,
        fusion_num_heads=8,
        num_classes=NUM_CLASSES,
    )


def make_inputs():
    signal = torch.randn(BATCH, 1000, 12)
    input_ids = torch.randint(0, 1000, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    return signal, input_ids, attention_mask


@pytest.mark.parametrize("mode", ["signal_only", "text_only", "fusion"])
def test_forward_pass_output_shape(mode):
    model = MultimodalECGClassifier(make_cfg(mode))
    model.eval()

    signal, input_ids, attention_mask = make_inputs()

    with torch.no_grad():
        logits = model(signal, input_ids, attention_mask)

    assert logits.shape == (BATCH, NUM_CLASSES), (
        f"Expected ({BATCH}, {NUM_CLASSES}), got {logits.shape}"
    )


@pytest.mark.parametrize("mode", ["text_only", "fusion"])
def test_distilbert_params_frozen(mode):
    model = MultimodalECGClassifier(make_cfg(mode))

    for name, param in model.text_encoder.distilbert.named_parameters():
        assert not param.requires_grad, f"Expected frozen parameter: {name}"


def test_invalid_mode_raises_error():
    cfg = make_cfg("bad_mode")

    with pytest.raises(ValueError):
        MultimodalECGClassifier(cfg)


def test_signal_encoder_absent_in_text_only():
    model = MultimodalECGClassifier(make_cfg("text_only"))

    assert not hasattr(model, "signal_encoder")


def test_signal_encoder_with_custom_cnn_config():
    cfg = make_cfg("signal_only")
    cfg.cnn_kernel_sizes = [7, 5, 3]
    cfg.cnn_activation = "relu"
    cfg.cnn_pooling = "max"

    model = MultimodalECGClassifier(cfg)

    signal, input_ids, attention_mask = make_inputs()

    with torch.no_grad():
        logits = model(signal, input_ids, attention_mask)

    assert logits.shape == (BATCH, NUM_CLASSES)    


def test_text_encoder_absent_in_signal_only():
    model = MultimodalECGClassifier(make_cfg("signal_only"))

    assert not hasattr(model, "text_encoder")


def test_fusion_module_absent_outside_fusion_mode():
    for mode in ("signal_only", "text_only"):
        model = MultimodalECGClassifier(make_cfg(mode))

        assert not hasattr(model, "fusion"), f"Fusion present in mode='{mode}'"


def test_signal_encoder_gradients_flow():
    model = MultimodalECGClassifier(make_cfg("signal_only"))
    model.train()

    signal, input_ids, attention_mask = make_inputs()

    logits = model(signal, input_ids, attention_mask)
    loss = logits.sum()
    loss.backward()

    for name, param in model.signal_encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for: {name}"


def test_text_projection_gradients_flow():
    model = MultimodalECGClassifier(make_cfg("text_only"))
    model.train()

    signal, input_ids, attention_mask = make_inputs()

    logits = model(signal, input_ids, attention_mask)
    loss = logits.sum()
    loss.backward()

    for name, param in model.text_encoder.projection.named_parameters():
        assert param.grad is not None, f"No gradient for projection parameter: {name}"


def test_logits_are_raw_not_softmaxed():
    model = MultimodalECGClassifier(make_cfg("signal_only"))
    model.eval()

    signal, input_ids, attention_mask = make_inputs()

    with torch.no_grad():
        logits = model(signal, input_ids, attention_mask)

    row_sums = logits.sum(dim=-1)

    assert not torch.allclose(row_sums, torch.ones(BATCH), atol=1e-3)
