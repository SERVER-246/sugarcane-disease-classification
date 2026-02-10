"""
Inference Routes
================
Multi-model prediction endpoints with validation.
Supports single image, base64, and batch prediction.
"""
from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from inference_server.engine.multi_predictor import (  # noqa: I001
    EnsemblePrediction,
    format_predictions_table,
    predict_all_models,
)
from inference_server.engine.predictor import predict_image
from inference_server.schemas import (
    EnsembleResult,
    ModelTableRow,
    PredictionResult,
    ValidationInfo,
)


logger = logging.getLogger("inference_server.routes.inference")

router = APIRouter(tags=["inference"])


def _ensemble_to_response(result: EnsemblePrediction) -> EnsembleResult:
    """Convert internal EnsemblePrediction to API response."""
    table = format_predictions_table(result)

    return EnsembleResult(
        predicted_class=result.ensemble_class,
        confidence=result.ensemble_confidence,
        voting_method=result.voting_method,
        validation=ValidationInfo(
            passed=result.validation_passed,
            message=result.validation_message,
            scores=result.validation_scores,
        ),
        total_models=result.total_models,
        agreeing_models=result.agreeing_models,
        agreement_ratio=result.agreement_ratio,
        class_votes=result.class_votes,
        class_avg_confidence=result.class_avg_confidence,
        model_predictions=[ModelTableRow(**row) for row in table],
        prediction_id=result.prediction_id,
        image_hash=result.image_hash,
        timestamp=result.timestamp,
    )


# =============================================================================
# Single-Model Endpoint (backward compatibility)
# =============================================================================

@router.post("/predict", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)) -> PredictionResult:
    """
    Classify a single image using the default backbone.

    Accepts a multipart-form upload named ``image``.
    Returns the predicted class, confidence, and per-class probabilities.

    Note: For multi-model ensemble prediction, use /predict/ensemble instead.
    """
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        result = predict_image(contents)
    except RuntimeError as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Bad image: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResult(**result)


# =============================================================================
# Multi-Model Ensemble Endpoint
# =============================================================================

@router.post("/predict/ensemble", response_model=EnsembleResult)
async def predict_ensemble(
    image: UploadFile = File(...),
    validate: bool = True,
) -> EnsembleResult:
    """
    Classify using ALL models and return ensemble result.

    Runs inference through all 15 backbone models plus student model,
    aggregates results via majority voting, and returns tabulated results.

    Args:
        image: Uploaded image file
        validate: Whether to validate image before inference (default: True)

    Returns:
        Ensemble prediction with per-model breakdown
    """
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        result = predict_all_models(contents, validate=validate)
    except RuntimeError as exc:
        logger.error("Ensemble inference error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Bad image or validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _ensemble_to_response(result)


# =============================================================================
# Base64 Endpoint
# =============================================================================

class Base64ImageRequest(BaseModel):
    """Request with base64-encoded image."""
    image_base64: str = Field(..., description="Base64-encoded image data")
    run_validation: bool = Field(default=True, description="Run image validation")


@router.post("/predict/base64", response_model=EnsembleResult)
async def predict_base64(request: Base64ImageRequest) -> EnsembleResult:
    """
    Classify a base64-encoded image using ensemble.

    Useful for clients that encode images before sending.

    Args:
        request: JSON body with base64 image and validation flag

    Returns:
        Ensemble prediction with per-model breakdown
    """
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data")

    try:
        result = predict_all_models(image_bytes, validate=request.run_validation)
    except RuntimeError as exc:
        logger.error("Ensemble inference error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Bad image or validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _ensemble_to_response(result)


# =============================================================================
# Batch Endpoint
# =============================================================================

class BatchPredictionResult(BaseModel):
    """Result for a single image in batch."""
    filename: str
    success: bool
    result: EnsembleResult | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    """Response for batch prediction."""
    total: int
    successful: int
    failed: int
    results: list[BatchPredictionResult]


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    images: list[UploadFile] = File(...),
    validate: bool = True,
) -> BatchResponse:
    """
    Classify multiple images in a single request.

    Runs ensemble inference on each image and returns aggregated results.

    Args:
        images: List of uploaded image files
        validate: Whether to validate images (default: True)

    Returns:
        Batch results with per-image predictions
    """
    results: list[BatchPredictionResult] = []
    successful = 0
    failed = 0

    for image in images:
        filename = image.filename or "unknown"
        try:
            contents = await image.read()
            if not contents:
                results.append(BatchPredictionResult(
                    filename=filename,
                    success=False,
                    error="Empty image file",
                ))
                failed += 1
                continue

            pred = predict_all_models(contents, validate=validate)
            results.append(BatchPredictionResult(
                filename=filename,
                success=True,
                result=_ensemble_to_response(pred),
            ))
            successful += 1
        except Exception as exc:
            results.append(BatchPredictionResult(
                filename=filename,
                success=False,
                error=str(exc),
            ))
            failed += 1

    return BatchResponse(
        total=len(images),
        successful=successful,
        failed=failed,
        results=results,
    )
