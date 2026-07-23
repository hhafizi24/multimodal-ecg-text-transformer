"""
Tests for cached logit collection, metric computation, and logit-bias fitting.
"""

import torch

from configs.config import ModelConfig
from src.models.model import MultimodalECGClassifier
from src.training.evaluate import collect_logits, metrics_from_logits
from src.training.bias_adjustment import fit_logit_bias

BATCH = 8
HIDDEN_DIM = 64
NUM_CLASSES = 5


def make_cfg() -> ModelConfig:
    return ModelConfig(
        mode="fusion",
        transformer_hidden_dim=HIDDEN_DIM,
        transformer_num_heads=4,
        transformer_num_layers=1,
        text_model_name="GerMedBERT/medbert-512",
        text_projection_dim=HIDDEN_DIM,
        fusion_num_heads=4,
        num_classes=NUM_CLASSES,
        freeze_encoders=True,
    )


def make_cached_batches(num_batches: int = 3) -> list[dict]:
    return [
        {
            "signal_embedding": torch.randn(BATCH, HIDDEN_DIM),
            "text_embedding": torch.randn(BATCH, HIDDEN_DIM),
            "label": torch.randint(0, NUM_CLASSES, (BATCH,)),
        }
        for _ in range(num_batches)
    ]


def test_collect_logits_shapes_and_alignment():
    model = MultimodalECGClassifier(make_cfg())
    device = torch.device("cpu")
    batches = make_cached_batches(num_batches=3)

    logits, labels = collect_logits(model, batches, device)

    assert logits.shape == (3 * BATCH, NUM_CLASSES)
    assert labels.shape == (3 * BATCH,)
    assert torch.equal(labels, torch.cat([b["label"] for b in batches]))


def test_metrics_from_logits_perfect_predictions():
    labels = torch.tensor([0, 1, 2, 3, 4])
    logits = torch.full((5, NUM_CLASSES), -10.0)
    logits[torch.arange(5), labels] = 10.0

    metrics = metrics_from_logits(logits, labels)

    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_metrics_from_logits_bias_shifts_predictions():
    labels = torch.tensor([0, 1, 2, 3, 4])
    logits = torch.eye(NUM_CLASSES) * 2.0
    logits[0] = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.0])

    raw = metrics_from_logits(logits, labels)
    bias = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0])
    adjusted = metrics_from_logits(logits, labels, bias)

    assert raw["accuracy"] == 0.8
    assert adjusted["accuracy"] == 1.0


def test_fit_logit_bias_never_decreases_f1():
    torch.manual_seed(0)
    logits = torch.randn(50, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (50,))

    raw_f1 = metrics_from_logits(logits, labels)["macro_f1"]
    bias = fit_logit_bias(logits, labels, num_classes=NUM_CLASSES)
    adjusted_f1 = metrics_from_logits(logits, labels, bias)["macro_f1"]

    assert adjusted_f1 >= raw_f1


def test_fit_logit_bias_stays_within_range():
    torch.manual_seed(1)
    logits = torch.randn(50, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (50,))

    bias = fit_logit_bias(logits, labels, num_classes=NUM_CLASSES, bias_range=(-1.0, 1.0))

    assert torch.all(bias >= -1.0) and torch.all(bias <= 1.0)