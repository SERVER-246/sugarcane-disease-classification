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
