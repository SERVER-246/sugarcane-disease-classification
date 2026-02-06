"""
Predictor
=========
Thin wrapper that takes a PIL image, preprocesses it, runs inference,
and returns a human-readable result dict.
"""
from __future__ import annotations

import logging
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

from inference_server import config
from inference_server.engine.loader import get_model, get_model_device


logger = logging.getLogger("inference_server.predictor")


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


def predict_image(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw file content of a JPEG / PNG / WebP image.

    Returns
    -------
    dict
        ``{"predicted_class": str, "confidence": float,
           "all_probabilities": {class: prob}, "backbone": str}``

    Raises
    ------
    RuntimeError
        If the model has not been loaded yet.
    ValueError
        If the image cannot be decoded.
    """
    from inference_server.engine.loader import get_model_backbone

    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded — call load_model() first")

    # Decode
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Preprocess
    transformed = _get_transform()(image)
    tensor = torch.as_tensor(transformed).unsqueeze(0)  # (1, 3, H, W)
    device = get_model_device()
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    class_names = config.get_class_names()

    # Build probability map
    all_probs = {
        name: round(float(probs[i]), 6) for i, name in enumerate(class_names)
    }

    top_idx = int(probs.argmax())
    predicted_class = class_names[top_idx]
    confidence = round(float(probs[top_idx]), 6)

    logger.info(
        "Prediction: %s (%.2f%%) on %s",
        predicted_class,
        confidence * 100,
        get_model_backbone(),
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "backbone": get_model_backbone(),
    }
