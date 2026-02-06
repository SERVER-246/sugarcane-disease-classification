"""
Pydantic Schemas
================
Request / response models for the inference server API.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# =============================================================================
# Health Check Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Basic health check response."""
    status: str = Field(default="healthy", examples=["healthy"])


class ReadyResponse(BaseModel):
    """Readiness probe response — indicates model is loaded."""
    ready: bool = Field(..., examples=[True])
    model: str = Field(..., examples=["loaded"])
    backbone: str = Field(..., examples=["CustomConvNeXt"])
    num_classes: int = Field(..., examples=[13])
    device: str = Field(..., examples=["cuda"])


class LiveResponse(BaseModel):
    """Liveness probe response — indicates process is running."""
    live: bool = Field(default=True, examples=[True])


# =============================================================================
# Inference Schemas
# =============================================================================

class PredictionResult(BaseModel):
    """Single image classification result."""
    predicted_class: str = Field(..., examples=["Healthy"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.95])
    all_probabilities: dict[str, float] = Field(
        ...,
        examples=[{"Healthy": 0.95, "Brown_spot": 0.03, "Mosaic": 0.02}],
    )
    backbone: str = Field(..., examples=["CustomConvNeXt"])


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., examples=["Model not loaded"])
    detail: str | None = Field(default=None, examples=["Checkpoint file not found"])


# =============================================================================
# Multi-Model Inference Schemas
# =============================================================================

class ModelTableRow(BaseModel):
    """Single row in the model predictions table."""
    model: str = Field(..., examples=["CustomConvNeXt"])
    prediction: str = Field(..., examples=["Healthy"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.96])
    inference_ms: float = Field(..., examples=[12.5])
    agrees_with_ensemble: bool = Field(..., examples=[True])


class ValidationInfo(BaseModel):
    """Image validation details."""
    passed: bool = Field(..., examples=[True])
    message: str = Field(..., examples=["Image is valid"])
    scores: dict[str, float] = Field(
        default_factory=dict,
        examples=[{"quality_score": 85.0, "vegetation_score": 0.92}],
    )


class EnsembleResult(BaseModel):
    """Aggregated ensemble prediction from all models."""

    # Ensemble consensus
    predicted_class: str = Field(..., examples=["Healthy"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.94])
    voting_method: str = Field(default="majority", examples=["majority"])

    # Validation
    validation: ValidationInfo

    # Voting analysis
    total_models: int = Field(..., examples=[15])
    agreeing_models: int = Field(..., examples=[14])
    agreement_ratio: float = Field(..., ge=0.0, le=1.0, examples=[0.93])
    class_votes: dict[str, int] = Field(..., examples=[{"Healthy": 14, "Brown_spot": 1}])
    class_avg_confidence: dict[str, float] = Field(
        ..., examples=[{"Healthy": 0.94, "Brown_spot": 0.65}]
    )

    # Individual model results (tabulated)
    model_predictions: list[ModelTableRow]

    # Metadata
    prediction_id: str = Field(..., examples=["550e8400-e29b-41d4-a716-446655440000"])
    image_hash: str = Field(..., examples=["a3f5c2d1e4b6a7c8d9e0f1a2b3c4d5e6"])
    timestamp: str = Field(..., examples=["2025-02-06T12:34:56.789Z"])


# =============================================================================
# Models & Classes Endpoints
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a loaded model."""
    name: str = Field(..., examples=["CustomConvNeXt"])
    loaded: bool = Field(..., examples=[True])
    type: str = Field(default="backbone", examples=["backbone"])


class ModelsListResponse(BaseModel):
    """List of all available models."""
    models: list[ModelInfo]
    total: int = Field(..., examples=[16])
    device: str = Field(..., examples=["cuda"])


class ClassInfo(BaseModel):
    """Information about a disease class."""
    index: int = Field(..., examples=[0])
    name: str = Field(..., examples=["Healthy"])


class ClassesListResponse(BaseModel):
    """List of all classification classes."""
    classes: list[ClassInfo]
    total: int = Field(..., examples=[13])


# =============================================================================
# Feedback Schemas
# =============================================================================

class FeedbackRequest(BaseModel):
    """User-submitted correction/feedback."""
    prediction_id: str = Field(..., examples=["550e8400-e29b-41d4-a716-446655440000"])
    image_hash: str = Field(..., examples=["a3f5c2d1e4b6a7c8d9e0f1a2b3c4d5e6"])
    predicted_class: str = Field(..., examples=["Healthy"])
    corrected_class: str = Field(..., examples=["Brown_spot"])
    user_notes: str | None = Field(default=None, examples=["Visible brown spots on leaf edges"])


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    feedback_id: str = Field(..., examples=["fb-12345"])
    status: str = Field(default="recorded", examples=["recorded"])
    message: str = Field(..., examples=["Feedback recorded successfully"])


# =============================================================================
# Connection & Retrain Schemas
# =============================================================================

class ConnectionReportRequest(BaseModel):
    """Client connection quality report."""
    client_id: str = Field(..., examples=["mobile-app-001"])
    latency_ms: float = Field(..., examples=[150.5])
    upload_speed_mbps: float | None = Field(default=None, examples=[5.2])
    connection_type: str | None = Field(default=None, examples=["wifi"])


class ConnectionReportResponse(BaseModel):
    """Response after reporting connection quality."""
    status: str = Field(default="recorded", examples=["recorded"])


class RetrainStatusResponse(BaseModel):
    """Auto-retraining pipeline status."""
    enabled: bool = Field(..., examples=[False])
    last_retrain: str | None = Field(default=None, examples=["2025-01-15T10:00:00Z"])
    next_scheduled: str | None = Field(default=None)
    pending_feedback_count: int = Field(default=0, examples=[42])
    status: str = Field(default="idle", examples=["idle"])
