"""
FastAPI Application
===================
Entry point for the inference server.

Start with::

    uvicorn inference_server.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference_server import __version__
from inference_server.engine.loader import load_model
from inference_server.routes.health import router as health_router
from inference_server.routes.inference import router as inference_router


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("inference_server")


# ---------------------------------------------------------------------------
# Lifespan — load model on startup, cleanup on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; release on shutdown."""
    logger.info("Starting inference server v%s …", __version__)
    try:
        load_model()
        logger.info("Model loaded — server is ready.")
    except Exception:
        logger.exception("Failed to load model on startup")
        # Server will still start; /health/ready will report not ready
    yield
    logger.info("Shutting down inference server.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sugarcane Disease Classifier",
    description=(
        "Inference API for the 15-backbone sugarcane disease classification "
        "pipeline.  Provides health probes and a single-image prediction endpoint."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(inference_router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Redirect-free landing that confirms the server is alive."""
    return {"message": "Sugarcane Disease Classifier API", "version": __version__}
