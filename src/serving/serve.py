"""FastAPI application serving ECG predictions."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.serving.inference import load_resources, predict
from src.serving.schema import HealthResponse, PredictRequest, PredictResponse


def _require_env(name: str) -> str:
    """Return a required environment variable or fail loudly if unset."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set.")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference resources once at startup."""
    app.state.resources = load_resources(
        onnx_path=_require_env("ONNX_PATH"),
        checkpoint_path=_require_env("CHECKPOINT_PATH"),
        norm_stats_path=_require_env("NORM_STATS_PATH"),
        config_snapshot_path=_require_env("CONFIG_SNAPSHOT_PATH"),
    )
    yield


app = FastAPI(title="ECG Prediction API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service liveness."""
    return HealthResponse()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest, request: Request) -> PredictResponse:
    """Run one ECG prediction, with or without report text."""
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="Model resources are not loaded.")

    try:
        result = predict(body.signal, body.report_text, resources)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(**result)