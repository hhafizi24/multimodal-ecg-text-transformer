"""Unit tests for preprocessing utilities."""

import numpy as np
import pandas as pd

from src.data.preprocess import (
    build_bandpass_filter,
    compute_norm_stats,
    derive_label,
    normalize,
)


def make_scp_lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diagnostic_class": ["MI", "STTC"],
        },
        index=["MI_CODE", "STTC_CODE"],
    )


def test_derive_label_returns_dominant_superclass():
    label = derive_label(
        "{'MI_CODE': 0.8, 'STTC_CODE': 0.2}",
        make_scp_lookup(),
        threshold=0.7,
    )

    assert label == "MI"


def test_derive_label_rejects_ambiguous_record():
    label = derive_label(
        "{'MI_CODE': 0.6, 'STTC_CODE': 0.4}",
        make_scp_lookup(),
        threshold=0.7,
    )

    assert label is None


def test_compute_norm_stats_uses_all_samples():
    signals = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    ]

    stats = compute_norm_stats(signals)

    np.testing.assert_allclose(stats["mean"], [4.0, 5.0])
    np.testing.assert_allclose(
        stats["std"],
        np.std(np.concatenate(signals, axis=0), axis=0),
    )


def test_normalize_applies_per_lead_statistics():
    signal = np.array(
        [[1.0, 5.0], [3.0, 9.0]],
        dtype=np.float32,
    )

    normalized = normalize(
        signal,
        mean=[1.0, 5.0],
        std=[2.0, 2.0],
    )

    np.testing.assert_allclose(
        normalized,
        [[0.0, 0.0], [1.0, 2.0]],
    )


def test_normalize_handles_zero_variance_lead():
    signal = np.array([[2.0]], dtype=np.float32)

    normalized = normalize(
        signal,
        mean=[1.0],
        std=[0.0],
    )

    np.testing.assert_allclose(normalized, [[1.0]])


def test_build_bandpass_filter_returns_sos_coefficients():
    coefficients = build_bandpass_filter(
        low_hz=0.5,
        high_hz=40.0,
        order=4,
        sampling_rate=100,
    )

    assert coefficients.ndim == 2
    assert coefficients.shape[1] == 6
    assert np.isfinite(coefficients).all()