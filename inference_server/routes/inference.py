"""
Inference Routes
================
Basic ``/predict`` endpoint — Sprint 3A foundation.
No input validation middleware yet (deferred to Sprint 3B).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from inference_server.engine.predictor import predict_image
from inference_server.schemas import PredictionResult


logger = logging.getLogger("inference_server.routes.inference")

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)) -> PredictionResult:
    """
    Classify a single image.

    Accepts a multipart-form upload named ``image``.
    Returns the predicted class, confidence, and per-class probabilities.
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
