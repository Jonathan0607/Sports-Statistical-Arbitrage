"""FastAPI inference and portfolio allocation service."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.feature_contract import FEATURE_SCHEMA_VERSION
from src.serving.predictor import InferencePredictor
from src.serving.schemas import (
    AllocateRequest,
    AllocateResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    DriftReportResponse,
    ErrorResponse,
    HealthResponse,
)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts Exceptions and non-finite floats so error payloads are strictly JSON-safe."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, BaseException):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def create_app(predictor: Optional[InferencePredictor] = None) -> FastAPI:
    """Application factory creating a typed FastAPI service with injected or default predictor."""

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        if not hasattr(app_instance.state, "predictor") or app_instance.state.predictor is None:
            app_instance.state.predictor = InferencePredictor()
        yield

    application = FastAPI(
        title="StatArb Inference & Allocation Service",
        version=FEATURE_SCHEMA_VERSION,
        lifespan=lifespan,
    )
    if predictor is not None:
        application.state.predictor = predictor

    def _get_predictor() -> InferencePredictor:
        pred = getattr(application.state, "predictor", None)
        if pred is None:
            pred = InferencePredictor()
            application.state.predictor = pred
        return pred

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        sanitized_errors = _sanitize_for_json(exc.errors())
        payload = ErrorResponse(
            error="validation_error",
            detail=jsonable_encoder(sanitized_errors),
            schema_version=FEATURE_SCHEMA_VERSION,
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return _get_predictor().health()

    @application.get("/v1/drift", response_model=DriftReportResponse)
    def drift(window_days: int = 7) -> DriftReportResponse:
        return _get_predictor().drift_report(window_days=window_days)

    @application.post("/v1/predict", response_model=BatchPredictResponse)
    def predict(request_body: BatchPredictRequest) -> BatchPredictResponse:
        return _get_predictor().predict_batch(request_body)

    @application.post("/v1/allocate", response_model=AllocateResponse)
    def allocate(request_body: AllocateRequest) -> AllocateResponse:
        return _get_predictor().allocate(request_body)

    return application


app = create_app()

