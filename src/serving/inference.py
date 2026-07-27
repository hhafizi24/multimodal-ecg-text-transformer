"""Hybrid PyTorch and ONNX Runtime inference for served predictions."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from scipy.signal import sosfiltfilt

from configs.config import ModelConfig
from src.data.preprocess import build_bandpass_filter, normalize
from src.data.text import load_tokenizer, tokenize
from src.models.text_encoder import TextEncoder
from src.serving.schema import (
    CLASS_NAMES,
    EXPECTED_SAMPLES,
    LEAD_ORDER,
)


log = logging.getLogger(__name__)


@dataclass
class InferenceResources:
    """Resources loaded once at service startup."""

    session: ort.InferenceSession
    text_encoder: TextEncoder
    tokenizer: object
    bandpass_sos: np.ndarray | None
    norm_mean: np.ndarray
    norm_std: np.ndarray
    max_text_length: int


def _validate_onnx_compatibility(
    session: ort.InferenceSession,
    model_cfg: ModelConfig,
) -> None:
    """Validate the ONNX interface against the checkpoint configuration."""
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}

    required_inputs = {
        "signal",
        "text_embedding",
        "text_available",
    }
    missing_inputs = required_inputs - inputs.keys()
    if missing_inputs:
        raise ValueError(
            f"ONNX graph is missing inputs: {sorted(missing_inputs)}."
        )

    if "logits" not in outputs:
        raise ValueError("ONNX graph is missing the 'logits' output.")

    signal_shape = tuple(inputs["signal"].shape[-2:])
    expected_signal_shape = (
        EXPECTED_SAMPLES,
        len(LEAD_ORDER),
    )
    if signal_shape != expected_signal_shape:
        raise ValueError(
            f"ONNX signal shape {signal_shape} does not match "
            f"expected shape {expected_signal_shape}."
        )

    text_dim = inputs["text_embedding"].shape[-1]
    if text_dim != model_cfg.text_projection_dim:
        raise ValueError(
            f"ONNX text embedding dimension {text_dim} does not match "
            f"checkpoint text_projection_dim "
            f"{model_cfg.text_projection_dim}."
        )

    output_classes = outputs["logits"].shape[-1]
    if output_classes != model_cfg.num_classes:
        raise ValueError(
            f"ONNX output has {output_classes} classes, but the "
            f"checkpoint expects {model_cfg.num_classes}."
        )

    if model_cfg.num_classes != len(CLASS_NAMES):
        raise ValueError(
            "Checkpoint class count does not match the serving class names."
        )


def load_resources(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
    norm_stats_path: str | Path,
    config_snapshot_path: str | Path,
    max_text_length: int = 128,
) -> InferenceResources:
    """Load model and preprocessing resources for inference."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_cfg = ModelConfig(**checkpoint["model_cfg"])

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    _validate_onnx_compatibility(session, model_cfg)

    with open(norm_stats_path, encoding="utf-8") as f:
        norm_stats = json.load(f)

    norm_mean = np.asarray(
        norm_stats["mean"],
        dtype=np.float32,
    )
    norm_std = np.asarray(
        norm_stats["std"],
        dtype=np.float32,
    )

    expected_stats_shape = (len(LEAD_ORDER),)
    if (
        norm_mean.shape != expected_stats_shape
        or norm_std.shape != expected_stats_shape
    ):
        raise ValueError(
            "Normalization statistics must contain one value per ECG lead."
        )

    if np.any(norm_std < 1e-8):
        log.warning(
            "Normalization statistics contain near-zero standard "
            "deviations; normalize() will substitute 1.0."
        )

    with open(config_snapshot_path, encoding="utf-8") as f:
        preprocessing_cfg = json.load(f)

    filter_cfg = preprocessing_cfg["filter"]
    bandpass_sos = None

    if filter_cfg["apply_bandpass"]:
        bandpass_sos = build_bandpass_filter(
            filter_cfg["low_hz"],
            filter_cfg["high_hz"],
            filter_cfg["order"],
            filter_cfg["sampling_rate"],
        )

    text_encoder = TextEncoder(model_cfg)

    prefix = "text_encoder."
    text_state = {
        key[len(prefix):]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    }
    if not text_state:
        raise ValueError(
            "Checkpoint contains no weights for 'text_encoder'."
        )

    text_encoder.load_state_dict(
        text_state,
        strict=True,
    )
    text_encoder.eval()

    tokenizer = load_tokenizer(model_cfg.text_model_name)

    log.info(
        "Inference resources loaded from %s and %s.",
        checkpoint_path,
        onnx_path,
    )

    return InferenceResources(
        session=session,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        bandpass_sos=bandpass_sos,
        norm_mean=norm_mean,
        norm_std=norm_std,
        max_text_length=max_text_length,
    )


def _preprocess_signal(
    signal: np.ndarray,
    resources: InferenceResources,
) -> np.ndarray:
    """Apply the training-time filter and normalization."""
    if resources.bandpass_sos is not None:
        signal = sosfiltfilt(
            resources.bandpass_sos,
            signal,
            axis=0,
        ).astype(np.float32)

    normalized = normalize(
        signal,
        resources.norm_mean,
        resources.norm_std,
    )
    return np.asarray(
        normalized,
        dtype=np.float32,
    )


def _embed_text(
    report_text: str | None,
    resources: InferenceResources,
) -> tuple[np.ndarray, bool]:
    """Create a projected text embedding or the missing-text fallback."""
    if report_text is None:
        embedding_dim = resources.text_encoder.projection.out_features
        return np.zeros(embedding_dim, dtype=np.float32), False

    encoded = tokenize(
        [report_text],
        resources.tokenizer,
        resources.max_text_length,
    )

    with torch.inference_mode():
        embedding = resources.text_encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    return (
        embedding.squeeze(0).cpu().numpy().astype(np.float32),
        True,
    )


def predict(
    signal: list[list[float]],
    report_text: str | None,
    resources: InferenceResources,
) -> dict[str, object]:
    """Run one prediction through the serving pipeline."""
    signal_array = np.asarray(
        signal,
        dtype=np.float32,
    )
    signal_array = _preprocess_signal(
        signal_array,
        resources,
    )
    text_embedding, text_used = _embed_text(
        report_text,
        resources,
    )

    logits = resources.session.run(
        ["logits"],
        {
            "signal": signal_array[np.newaxis, ...],
            "text_embedding": text_embedding[np.newaxis, ...],
            "text_available": np.asarray(
                [text_used],
                dtype=np.bool_,
            ),
        },
    )[0][0]

    if not np.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite logits.")

    exponentiated = np.exp(logits - logits.max())
    probabilities = exponentiated / exponentiated.sum()
    predicted_index = int(probabilities.argmax())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_probability": float(probabilities.max()),
        "text_used": text_used,
    }