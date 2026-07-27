"""Tests for the FastAPI serving app and its ONNX-compatibility checks."""

import numpy as np
from pydantic import ValidationError
import pytest
import torch
from fastapi.testclient import TestClient

from configs.config import ModelConfig
from src.serving import serve
from src.serving.inference import InferenceResources, _validate_onnx_compatibility
from src.serving.schema import (
    CLASS_NAMES,
    EXPECTED_SAMPLES,
    LEAD_ORDER,
    PredictRequest,
)

ENV_VARS = ("ONNX_PATH", "CHECKPOINT_PATH", "NORM_STATS_PATH", "CONFIG_SNAPSHOT_PATH")


class _FakeNodeArg:
    """Minimal stand-in for onnxruntime's input/output metadata."""

    def __init__(self, name: str, shape: list):
        self.name = name
        self.shape = shape


class _FakeSession:
    """Minimal ONNX Runtime session double for compatibility tests."""

    def __init__(self, num_classes: int = 5, text_dim: int = 16, missing_input: str | None = None):
        self.num_classes = num_classes
        inputs = [
            _FakeNodeArg("signal", ["batch", EXPECTED_SAMPLES, len(LEAD_ORDER)]),
            _FakeNodeArg("text_embedding", ["batch", text_dim]),
            _FakeNodeArg("text_available", ["batch"]),
        ]
        self._inputs = [i for i in inputs if i.name != missing_input]
        self._outputs = [_FakeNodeArg("logits", ["batch", num_classes])]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, input_feed):
        batch = input_feed["signal"].shape[0]
        return [np.zeros((batch, self.num_classes), dtype=np.float32)]


class _FakeTextEncoder(torch.nn.Module):
    """Minimal text encoder double with a real projection layer."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids=None, attention_mask=None, cached_embedding=None):
        batch = input_ids.shape[0]
        return torch.zeros(batch, self.projection.out_features)


class _FakeTokenizer:
    """Minimal tokenizer double matching the call signature tokenize() uses."""

    def __call__(self, texts, padding=None, truncation=None, max_length=None, return_tensors=None):
        batch = len(texts)
        return {
            "input_ids": torch.zeros((batch, max_length), dtype=torch.long),
            "attention_mask": torch.ones((batch, max_length), dtype=torch.long),
        }


def _valid_signal() -> list[list[float]]:
    """Build a well-formed zero signal for request bodies."""
    return [[0.0] * len(LEAD_ORDER) for _ in range(EXPECTED_SAMPLES)]


@pytest.fixture
def fake_resources():
    """Build InferenceResources that need no real checkpoint or ONNX file."""
    return InferenceResources(
        session=_FakeSession(num_classes=len(CLASS_NAMES), text_dim=16),
        text_encoder=_FakeTextEncoder(hidden_dim=16),
        tokenizer=_FakeTokenizer(),
        bandpass_sos=None,
        norm_mean=np.zeros(len(LEAD_ORDER), dtype=np.float32),
        norm_std=np.ones(len(LEAD_ORDER), dtype=np.float32),
        max_text_length=16,
    )


@pytest.fixture
def client(monkeypatch, fake_resources):
    """TestClient wired to fake resources instead of real model files."""
    for name in ENV_VARS:
        monkeypatch.setenv(name, "unused")
    monkeypatch.setattr(serve, "load_resources", lambda **_: fake_resources)

    with TestClient(serve.app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_text(client):
    response = client.post("/predict", json={"signal": _valid_signal()})
    assert response.status_code == 200
    body = response.json()
    assert body["text_used"] is False
    assert set(body["probabilities"]) == set(CLASS_NAMES)


def test_predict_with_text(client):
    response = client.post(
        "/predict", json={"signal": _valid_signal(), "report_text": "Normalbefund."}
    )
    assert response.status_code == 200
    assert response.json()["text_used"] is True


def test_predict_with_whitespace_text_is_signal_only(client):
    response = client.post(
        "/predict", json={"signal": _valid_signal(), "report_text": "   "}
    )
    assert response.status_code == 200
    assert response.json()["text_used"] is False


def test_predict_rejects_wrong_sample_count(client):
    bad_signal = _valid_signal()[:-1]
    response = client.post("/predict", json={"signal": bad_signal})
    assert response.status_code == 422


def test_predict_rejects_wrong_lead_count(client):
    bad_signal = [row[:-1] for row in _valid_signal()]
    response = client.post("/predict", json={"signal": bad_signal})
    assert response.status_code == 422


def test_request_schema_rejects_non_finite_values():
    bad_signal = _valid_signal()
    bad_signal[0][0] = float("nan")

    with pytest.raises(ValidationError):
        PredictRequest(signal=bad_signal)


def test_predict_response_schema_is_stable(client):
    response = client.post("/predict", json={"signal": _valid_signal()})
    body = response.json()
    assert set(body) == {"predicted_class", "probabilities", "max_probability", "text_used"}


def test_validate_onnx_compatibility_passes_when_consistent():
    session = _FakeSession(num_classes=5, text_dim=16)
    cfg = ModelConfig(num_classes=5, text_projection_dim=16, transformer_hidden_dim=16)
    _validate_onnx_compatibility(session, cfg)


def test_validate_onnx_compatibility_rejects_text_dim_mismatch():
    session = _FakeSession(num_classes=5, text_dim=16)
    cfg = ModelConfig(num_classes=5, text_projection_dim=32, transformer_hidden_dim=32)
    with pytest.raises(ValueError):
        _validate_onnx_compatibility(session, cfg)


def test_validate_onnx_compatibility_rejects_class_count_mismatch():
    session = _FakeSession(num_classes=4, text_dim=16)
    cfg = ModelConfig(num_classes=5, text_projection_dim=16, transformer_hidden_dim=16)
    with pytest.raises(ValueError):
        _validate_onnx_compatibility(session, cfg)


def test_validate_onnx_compatibility_rejects_missing_input():
    session = _FakeSession(missing_input="text_available")
    cfg = ModelConfig(num_classes=5, text_projection_dim=16, transformer_hidden_dim=16)
    with pytest.raises(ValueError):
        _validate_onnx_compatibility(session, cfg)