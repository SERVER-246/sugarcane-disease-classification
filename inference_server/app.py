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
from fastapi.middleware.cors import CORSMiddleware

from inference_server import __version__, config
from inference_server.engine.loader import load_model
from inference_server.engine.multi_loader import load_all_models
from inference_server.engine.redrot_segmentation import load_redrot_segmenter
from inference_server.engine.v2_loader import load_v2_classifiers
from inference_server.routes.analytics import router as analytics_router
from inference_server.routes.feedback import router as feedback_router
from inference_server.routes.health import router as health_router
from inference_server.routes.inference import router as inference_router
from inference_server.routes.models import router as models_router


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
    """Load all models once at startup; release on shutdown."""
    logger.info("Starting inference server v%s …", __version__)
    try:
        # Load single model for backward compatibility
        load_model()
        logger.info("Default model loaded.")

        # Load ALL models for ensemble inference
        models = load_all_models()
        logger.info("Loaded %d backbone models for ensemble inference.", len(models))

        # Optionally load Model 2 (V2 dual-head classifiers) into the ensemble
        if config.ENABLE_V2:
            try:
                v2 = load_v2_classifiers()
                logger.info("Loaded %d V2 classifiers into the ensemble.", len(v2))
            except Exception:
                logger.exception("Failed to load V2 classifiers; continuing with V1 ensemble only.")

        # Load the Red Rot segmentation specialist (Stage 2 of Expert View)
        try:
            load_redrot_segmenter()
            logger.info("Red Rot segmentation specialist loaded.")
        except Exception:
            logger.exception(
                "Failed to load Red Rot segmenter on startup; "
                "/predict/expert will return 503 for Red Rot until resolved."
            )
    except Exception:
        logger.exception("Failed to load models on startup")
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
        "pipeline. Provides health probes, single-image prediction, "
        "multi-model ensemble inference, feedback collection, and analytics."
    ),
    version=__version__,
    lifespan=lifespan,
)

# CORS — allow the Web Expert View client to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health & inference routes
app.include_router(health_router)
app.include_router(inference_router)

# Models & classes
app.include_router(models_router)

# Feedback system
app.include_router(feedback_router)

# Analytics & connection tracking
app.include_router(analytics_router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Redirect-free landing that confirms the server is alive."""
    return {"message": "Sugarcane Disease Classifier API", "version": __version__}
