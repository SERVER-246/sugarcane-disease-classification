"""
Inference Routes
================
Multi-model prediction endpoints with validation.
Supports single image, base64, and batch prediction.
"""
from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from inference_server import config
from inference_server.engine import redrot_segmentation
from inference_server.engine.multi_predictor import (  # noqa: I001
    EnsemblePrediction,
    format_predictions_table,
    predict_all_models,
)
from inference_server.engine.predictor import predict_image
from inference_server.engine.remedial import get_remedial
from inference_server.schemas import (
    EnsembleResult,
    ExpertResult,
    ModelTableRow,
    PredictionResult,
    RedRotSegmentation,
    SymptomInfo,
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
# Expert View (two-stage) Endpoint
# =============================================================================

def _seg_to_schema(seg: redrot_segmentation.RedRotSegResult) -> RedRotSegmentation:
    return RedRotSegmentation(
        backbone=seg.backbone,
        affected_area_pct=seg.affected_area_pct,
        dominant_symptom=seg.dominant_symptom,
        symptoms=[
            SymptomInfo(
                name=s.name,
                pixel_fraction=s.pixel_fraction,
                mean_confidence=s.mean_confidence,
                color=s.color,
            )
            for s in seg.symptoms
        ],
        num_symptom_classes_detected=seg.num_symptom_classes_detected,
        overlay_png_b64=seg.overlay_png_b64,
        mask_height=seg.mask_height,
        mask_width=seg.mask_width,
    )


@router.post("/predict/expert", response_model=ExpertResult)
async def predict_expert(
    image: UploadFile = File(...),
    validate: bool | None = None,
    force_redrot: bool = False,
) -> ExpertResult:
    """
    Expert View two-stage inference.

    Stage 1 — broad 13-class ensemble classification.
    Stage 2 — if the prediction is Red Rot (or ``force_redrot=true``), run the Red
    Rot symptom-segmentation specialist and attach the overlay + symptom stats.
    Remedial-measures recommendations for the predicted disease are always included.

    Args:
        image: Uploaded image file.
        validate: Override the default sugarcane-validation gate (defaults to the
            server ``EXPERT_VALIDATE_DEFAULT`` setting when omitted).
        force_redrot: Force the Red Rot specialist to run regardless of Stage 1.
    """
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")

    do_validate = config.EXPERT_VALIDATE_DEFAULT if validate is None else validate

    try:
        ensemble = predict_all_models(contents, validate=do_validate)
    except RuntimeError as exc:
        logger.error("Expert stage-1 error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Bad image: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    classification = _ensemble_to_response(ensemble)

    if not ensemble.validation_passed:
        return ExpertResult(
            classification=classification,
            stage2_ran=False,
            redrot_segmentation=None,
            remedial={},
            note="Image failed sugarcane validation; classification and segmentation skipped.",
            overall_verdict="Not a sugarcane image",
            disagreement=False,
            low_confidence=False,
            verdict_reason=classification.validation.message,
        )

    redrot_class = config.get_redrot_class_name()
    # Pilot policy: run the Red Rot specialist on every image (unless disabled), so the
    # verdict can catch Red Rot even when the broad classifier predicts another class.
    run_stage2 = force_redrot or config.RUN_SPECIALIST_ALWAYS or (classification.predicted_class == redrot_class)

    seg_schema: RedRotSegmentation | None = None
    note: str | None = None
    if run_stage2:
        try:
            seg = redrot_segmentation.segment_redrot(contents)
        except FileNotFoundError as exc:
            logger.error("Red Rot segmenter unavailable: %s", exc)
            raise HTTPException(
                status_code=503, detail=f"Red Rot segmenter unavailable: {exc}"
            ) from exc
        except Exception as exc:  # fail loud — never silently skip Stage 2
            logger.exception("Red Rot segmentation failed")
            raise HTTPException(
                status_code=500, detail=f"Red Rot segmentation failed: {exc}"
            ) from exc
        seg_schema = _seg_to_schema(seg)

    # ---- Reconcile the two signals into an overall verdict + traffic-light status ----
    cls_name = classification.predicted_class
    low_confidence = classification.confidence < config.CLS_LOW_CONF
    seg_area = seg_schema.affected_area_pct if seg_schema else 0.0
    specialist_found = seg_schema is not None and seg_area >= config.REDROT_DISAGREE_AREA_PCT
    specialist_declares = seg_schema is not None and seg_area >= config.REDROT_SAY_PCT
    disagreement = specialist_found and cls_name != redrot_class

    # Traffic-light status (single source of truth consumed by the Farmer View).
    if specialist_declares:
        farmer_status, farmer_disease = "red", redrot_class
    elif specialist_found:
        farmer_status, farmer_disease = "amber", redrot_class  # possible Red Rot — recheck
    elif cls_name == "Healthy":
        farmer_status, farmer_disease = ("amber", None) if low_confidence else ("green", None)
    else:
        farmer_status, farmer_disease = ("amber", cls_name) if low_confidence else ("red", cls_name)

    if specialist_declares:
        overall_verdict = f"{redrot_class} — symptoms detected ({seg_area:.1f}% area)"
    elif specialist_found:
        overall_verdict = f"Possible {redrot_class} — {seg_area:.1f}% area (recheck)"
    else:
        overall_verdict = cls_name

    reasons: list[str] = []
    if disagreement:
        reasons.append(
            f"The whole-plant screen said '{cls_name}', but the Red Rot specialist found symptoms "
            f"on {seg_area:.1f}% of the image (common on close-up photos)."
        )
    if low_confidence:
        reasons.append(
            f"Whole-plant screen confidence is low ({classification.confidence * 100:.0f}%)."
        )
    verdict_reason = " ".join(reasons) or None

    return ExpertResult(
        classification=classification,
        stage2_ran=run_stage2,
        redrot_segmentation=seg_schema,
        remedial=get_remedial(farmer_disease or cls_name),
        note=note,
        overall_verdict=overall_verdict,
        disagreement=disagreement,
        low_confidence=low_confidence,
        verdict_reason=verdict_reason,
        farmer_status=farmer_status,
        farmer_disease=farmer_disease,
    )


# =============================================================================
# PDF Report Endpoint
# =============================================================================

class ReportRequest(BaseModel):
    """Expert View result (+ optional original image) to render as a PDF.

    ``result`` is accepted as a free-form object (the client echoes back the
    ExpertResult it received); the PDF builder reads fields defensively, so we do
    not re-validate the full nested schema here.
    """
    result: dict
    image_base64: str | None = Field(default=None, description="Original image, base64 (optional)")


@router.post("/report")
async def report(req: ReportRequest) -> Response:
    """Render an Expert View result as a one-page PDF."""
    from inference_server.engine.report import build_report_pdf

    try:
        pdf_bytes = build_report_pdf(req.result, req.image_base64)
    except Exception as exc:
        logger.exception("PDF report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=expert_report.pdf"},
    )


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
