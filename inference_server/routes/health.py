"""
Health Check Routes
===================
Provides ``/health``, ``/health/ready``, and ``/health/live`` endpoints
for container orchestrators (Docker, K8s) and monitoring.
"""
from __future__ import annotations

from fastapi import APIRouter

from inference_server import config
from inference_server.engine.loader import (
    get_model,
    get_model_backbone,
    get_model_device,
)
from inference_server.schemas import HealthResponse, LiveResponse, ReadyResponse


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health probe — always returns ``{"status": "healthy"}``."""
    return HealthResponse(status="healthy")


@router.get("/ready", response_model=ReadyResponse)
async def readiness() -> ReadyResponse:
    """
    Readiness probe.

    Returns ``ready: true`` only when the model has been loaded
    successfully.  Container orchestrators should not route traffic
    to this instance until it reports ready.
    """
    model = get_model()
    if model is None:
        return ReadyResponse(
            ready=False,
            model="not_loaded",
            backbone="none",
            num_classes=0,
            device="unknown",
        )
    return ReadyResponse(
        ready=True,
        model="loaded",
        backbone=get_model_backbone(),
        num_classes=config.NUM_CLASSES,
        device=str(get_model_device()),
    )


@router.get("/live", response_model=LiveResponse)
async def liveness() -> LiveResponse:
    """Liveness probe — confirms the process is running."""
    return LiveResponse(live=True)
