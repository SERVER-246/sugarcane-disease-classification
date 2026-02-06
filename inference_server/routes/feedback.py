"""
Feedback Routes
===============
Endpoints for user correction/feedback submission.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from inference_server import config
from inference_server.schemas import (
    ClassInfo,
    ClassesListResponse,
    FeedbackRequest,
    FeedbackResponse,
)

logger = logging.getLogger("inference_server.routes.feedback")
router = APIRouter(prefix="/feedback", tags=["feedback"])

# Feedback storage directory
FEEDBACK_DIR = Path(config.BASE_DIR) / "feedback_logs"


def _ensure_feedback_dir() -> Path:
    """Ensure feedback directory exists."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    return FEEDBACK_DIR


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user correction/feedback for a prediction.

    This allows users to correct misclassifications, which can be used
    for model improvement and retraining.

    Args:
        request: Feedback with prediction_id, predicted vs corrected class

    Returns:
        Confirmation of feedback recording
    """
    feedback_id = f"fb-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"

    feedback_record = {
        "feedback_id": feedback_id,
        "prediction_id": request.prediction_id,
        "image_hash": request.image_hash,
        "predicted_class": request.predicted_class,
        "corrected_class": request.corrected_class,
        "user_notes": request.user_notes,
        "timestamp": timestamp,
    }

    # Write to feedback log file (append)
    try:
        feedback_dir = _ensure_feedback_dir()
        log_file = feedback_dir / "feedback_log.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_record) + "\n")

        logger.info("Feedback recorded: %s (pred=%s -> corrected=%s)",
                    feedback_id, request.predicted_class, request.corrected_class)

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="recorded",
            message="Feedback recorded successfully. Thank you for helping improve the model.",
        )
    except Exception as e:
        logger.error("Failed to record feedback: %s", e)
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="error",
            message=f"Failed to record feedback: {e}",
        )


@router.get("/classes", response_model=ClassesListResponse)
async def get_feedback_classes() -> ClassesListResponse:
    """
    Get list of valid classes for feedback submission.

    Returns the same class list as /classes endpoint,
    intended for populating correction dropdowns in UI.
    """
    class_names = config.get_class_names()

    class_list = [
        ClassInfo(index=i, name=name)
        for i, name in enumerate(class_names)
    ]

    return ClassesListResponse(
        classes=class_list,
        total=len(class_list),
    )


@router.get("/stats")
async def get_feedback_stats() -> dict:
    """
    Get feedback statistics.

    Returns summary of collected feedback for monitoring.
    """
    try:
        feedback_dir = _ensure_feedback_dir()
        log_file = feedback_dir / "feedback_log.jsonl"

        if not log_file.exists():
            return {
                "total_feedback": 0,
                "unique_predictions": 0,
                "class_corrections": {},
            }

        total = 0
        predictions = set()
        corrections: dict[str, dict[str, int]] = {}

        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total += 1
                    predictions.add(record.get("prediction_id", ""))

                    pred_class = record.get("predicted_class", "unknown")
                    corr_class = record.get("corrected_class", "unknown")

                    if pred_class not in corrections:
                        corrections[pred_class] = {}
                    if corr_class not in corrections[pred_class]:
                        corrections[pred_class][corr_class] = 0
                    corrections[pred_class][corr_class] += 1

        return {
            "total_feedback": total,
            "unique_predictions": len(predictions),
            "class_corrections": corrections,
        }
    except Exception as e:
        logger.error("Failed to get feedback stats: %s", e)
        return {"error": str(e)}
