"""
Multi-Model Predictor
=====================
Run inference through ALL loaded models and return tabulated results.
Includes validation, ensemble voting, and confidence analysis.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from inference_server import config
from inference_server.engine.multi_loader import (
    get_all_models,
    get_model_device,
    get_student_model,
)
from inference_server.engine.validation import validate_image_bytes

logger = logging.getLogger("inference_server.multi_predictor")


@dataclass
class ModelPrediction:
    """Single model prediction result."""

    model_name: str
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]
    inference_time_ms: float


@dataclass
class EnsemblePrediction:
    """Aggregated ensemble prediction result."""

    # Ensemble result
    ensemble_class: str
    ensemble_confidence: float
    voting_method: str

    # Individual model results
    model_predictions: list[ModelPrediction]

    # Validation info
    validation_passed: bool
    validation_message: str
    validation_scores: dict[str, float]

    # Metadata
    prediction_id: str
    image_hash: str
    timestamp: str
    total_models: int
    agreeing_models: int
    agreement_ratio: float

    # Detailed analysis
    class_votes: dict[str, int] = field(default_factory=dict)
    class_avg_confidence: dict[str, float] = field(default_factory=dict)


def _build_inference_transform() -> transforms.Compose:
    """Build the deterministic (no-augmentation) transform for inference."""
    size = config.IMG_SIZE
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# Module-level singleton
_transform: transforms.Compose | None = None


def _get_transform() -> transforms.Compose:
    global _transform  # noqa: PLW0603
    if _transform is None:
        _transform = _build_inference_transform()
    return _transform


def _compute_image_hash(image_bytes: bytes) -> str:
    """Compute MD5 hash of image for deduplication."""
    return hashlib.md5(image_bytes).hexdigest()


def predict_single_model(
    model: torch.nn.Module,
    model_name: str,
    tensor: torch.Tensor,
    class_names: list[str],
) -> ModelPrediction:
    """Run inference on a single model."""
    import time

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    elapsed_ms = (time.perf_counter() - start) * 1000

    all_probs = {name: round(float(probs[i]), 6) for i, name in enumerate(class_names)}
    top_idx = int(probs.argmax())

    return ModelPrediction(
        model_name=model_name,
        predicted_class=class_names[top_idx],
        confidence=round(float(probs[top_idx]), 6),
        all_probabilities=all_probs,
        inference_time_ms=round(elapsed_ms, 2),
    )


def predict_all_models(
    image_bytes: bytes,
    validate: bool = True,
    include_student: bool = True,
) -> EnsemblePrediction:
    """
    Run inference through ALL loaded models.

    Args:
        image_bytes: Raw image file content
        validate: Whether to run image validation first
        include_student: Whether to include the student (distilled) model

    Returns:
        EnsemblePrediction with all model results and ensemble consensus
    """
    prediction_id = str(uuid.uuid4())
    image_hash = _compute_image_hash(image_bytes)
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Validation
    validation_passed = True
    validation_message = "Validation skipped"
    validation_scores = {}

    if validate:
        report = validate_image_bytes(image_bytes)
        validation_passed = report.is_valid
        validation_message = report.message
        validation_scores = {
            "quality_score": report.quality_score,
            "vegetation_score": report.vegetation_score,
            "sugarcane_score": report.sugarcane_score,
            "overall_confidence": report.confidence,
        }

        if not validation_passed:
            # Return early with validation failure
            return EnsemblePrediction(
                ensemble_class="VALIDATION_FAILED",
                ensemble_confidence=0.0,
                voting_method="none",
                model_predictions=[],
                validation_passed=False,
                validation_message=validation_message,
                validation_scores=validation_scores,
                prediction_id=prediction_id,
                image_hash=image_hash,
                timestamp=timestamp,
                total_models=0,
                agreeing_models=0,
                agreement_ratio=0.0,
                class_votes={},
                class_avg_confidence={},
            )

    # Decode image
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}") from e

    # Preprocess
    transform = _get_transform()
    tensor: torch.Tensor = transform(pil_image).unsqueeze(0)  # type: ignore[union-attr]
    device = get_model_device()
    tensor = tensor.to(device)

    class_names = config.get_class_names()
    models = get_all_models()
    predictions: list[ModelPrediction] = []

    # Run through all backbone models
    for model_name, model in models.items():
        try:
            pred = predict_single_model(model, model_name, tensor, class_names)
            predictions.append(pred)
        except Exception as e:
            logger.warning("Failed to run %s: %s", model_name, e)

    # Run student model if available
    if include_student:
        student = get_student_model()
        if student is not None:
            try:
                pred = predict_single_model(student, "StudentModel", tensor, class_names)
                predictions.append(pred)
            except Exception as e:
                logger.warning("Failed to run student model: %s", e)

    # Ensemble voting
    if not predictions:
        raise RuntimeError("No models available for inference")

    # Count votes
    votes = Counter(p.predicted_class for p in predictions)
    class_votes = dict(votes)

    # Calculate average confidence per class
    class_confidences: dict[str, list[float]] = {}
    for p in predictions:
        if p.predicted_class not in class_confidences:
            class_confidences[p.predicted_class] = []
        class_confidences[p.predicted_class].append(p.confidence)

    class_avg_confidence = {
        cls: float(round(np.mean(confs), 4)) for cls, confs in class_confidences.items()
    }

    # Majority voting
    ensemble_class = votes.most_common(1)[0][0]
    agreeing_models = votes[ensemble_class]
    agreement_ratio = agreeing_models / len(predictions)

    # Ensemble confidence = average confidence of agreeing models
    agreeing_confidences = [p.confidence for p in predictions if p.predicted_class == ensemble_class]
    ensemble_confidence = round(float(np.mean(agreeing_confidences)), 4)

    return EnsemblePrediction(
        ensemble_class=ensemble_class,
        ensemble_confidence=ensemble_confidence,
        voting_method="majority",
        model_predictions=predictions,
        validation_passed=validation_passed,
        validation_message=validation_message,
        validation_scores=validation_scores,
        prediction_id=prediction_id,
        image_hash=image_hash,
        timestamp=timestamp,
        total_models=len(predictions),
        agreeing_models=agreeing_models,
        agreement_ratio=round(agreement_ratio, 4),
        class_votes=class_votes,
        class_avg_confidence=class_avg_confidence,
    )


def format_predictions_table(result: EnsemblePrediction) -> list[dict[str, Any]]:
    """Format predictions as a table for JSON response."""
    table = []
    for p in sorted(result.model_predictions, key=lambda x: -x.confidence):
        table.append({
            "model": p.model_name,
            "prediction": p.predicted_class,
            "confidence": p.confidence,
            "inference_ms": p.inference_time_ms,
            "agrees_with_ensemble": p.predicted_class == result.ensemble_class,
        })
    return table
