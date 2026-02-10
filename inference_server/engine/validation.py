"""
Image Validation Integration
=============================
Wraps the image_validator.py for server-side validation.
Rejects non-sugarcane, corrupted, or low-quality images BEFORE inference.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image  # noqa: I001


# Add project root for image_validator import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from image_validator import ImageValidator, ValidationReport, ValidationResult  # noqa: E402, I001

logger = logging.getLogger("inference_server.validation")


# Module-level singleton
_validator: ImageValidator | None = None


def get_validator() -> ImageValidator:
    """Get or create the image validator singleton."""
    global _validator  # noqa: PLW0603
    if _validator is None:
        _validator = ImageValidator(use_deep_learning=True)
    return _validator


def validate_image_bytes(image_bytes: bytes) -> ValidationReport:
    """
    Validate image from raw bytes.

    Args:
        image_bytes: Raw image file content

    Returns:
        ValidationReport with is_valid, result, scores, and suggestions
    """
    validator = get_validator()

    # Write to temp file for validation (validator expects path)
    try:
        # First check if it's a valid image at all
        image = Image.open(BytesIO(image_bytes))
        image.verify()  # Verify it's not corrupted

        # Reopen for actual use
        image = Image.open(BytesIO(image_bytes))
        fmt = image.format or "JPEG"

        # Save to temp file
        with NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        # Run validation
        report = validator.validate(str(tmp_path))

        # Cleanup temp file
        with contextlib.suppress(Exception):
            tmp_path.unlink()

        return report

    except Exception as e:
        logger.warning("Image validation failed: %s", e)
        return ValidationReport(
            is_valid=False,
            result=ValidationResult.CORRUPTED,
            confidence=0.0,
            quality_score=0.0,
            vegetation_score=0.0,
            sugarcane_score=0.0,
            message=f"Cannot process image: {e}",
            details={"error": str(e)},
            suggestions=["Ensure the file is a valid image (JPEG, PNG, WebP)"],
        )


def quick_validate(image_bytes: bytes) -> tuple[bool, str, list[str]]:
    """
    Quick validation returning (is_valid, message, suggestions).

    For use in endpoint handlers.
    """
    report = validate_image_bytes(image_bytes)
    return report.is_valid, report.message, report.suggestions
