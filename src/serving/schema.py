"""Request and response schemas for the ECG prediction API."""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Ten-second ECG window sampled at 100 Hz
EXPECTED_SAMPLES = 1000

# PTB-XL lead order used during preprocessing and training
LEAD_ORDER = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

# Class order must match LABEL_MAP in src/data/preprocess.py
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    model_config = ConfigDict(extra="forbid")

    signal: list[list[float]] = Field(
        ...,
        description=(
            "Unfiltered 12-lead ECG in millivolts with shape [1000, 12], "
            "representing 10 seconds sampled at 100 Hz. Expected lead order: "
            "I, II, III, aVR, aVL, aVF, V1-V6. Filtering and normalization "
            "are applied by the service."
        ),
    )
    report_text: str | None = Field(
        default=None,
        description=(
            "Optional German-language clinical report. Omit or provide "
            "null or blank text to run signal-only inference."
        ),
    )

    @field_validator("signal")
    @classmethod
    def validate_signal(
        cls,
        value: list[list[float]],
    ) -> list[list[float]]:
        """Validate sample count, lead count, and finiteness."""
        if len(value) != EXPECTED_SAMPLES:
            raise ValueError(
                f"Expected {EXPECTED_SAMPLES} samples, got {len(value)}."
            )

        for row in value:
            if len(row) != len(LEAD_ORDER):
                raise ValueError(
                    f"Expected {len(LEAD_ORDER)} leads per sample, "
                    f"got {len(row)}."
                )
            if not all(math.isfinite(sample) for sample in row):
                raise ValueError("Signal contains non-finite values.")

        return value

    @field_validator("report_text")
    @classmethod
    def normalize_report_text(
        cls,
        value: str | None,
    ) -> str | None:
        """Treat blank or whitespace-only text as absent."""
        if value is not None and not value.strip():
            return None
        return value


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    predicted_class: str = Field(
        description="Class with the highest raw softmax probability."
    )
    probabilities: dict[str, float] = Field(
        description=(
            "Uncalibrated softmax probabilities keyed by diagnostic class."
        )
    )
    max_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Highest raw softmax probability.",
    )
    text_used: bool = Field(
        description="Whether report text was used for the prediction."
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
