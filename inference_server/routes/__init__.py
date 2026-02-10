"""Routes package."""
from __future__ import annotations

from inference_server.routes import (  # noqa: I001
    analytics,
    feedback,
    health,
    inference,
    models,
)


__all__ = ["analytics", "feedback", "health", "inference", "models"]
