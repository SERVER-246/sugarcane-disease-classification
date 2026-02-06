"""
Models Routes
=============
Endpoints for listing models and classes.
"""
from __future__ import annotations

from fastapi import APIRouter

from inference_server import config
from inference_server.engine.multi_loader import get_all_models, get_student_model, get_model_device
from inference_server.schemas import (
    ClassInfo,
    ClassesListResponse,
    ModelInfo,
    ModelsListResponse,
)

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    List all loaded models.

    Returns information about:
    - All 15 backbone models
    - Student (distilled) model
    - Meta-learner (if available)
    """
    models = get_all_models()
    student = get_student_model()

    model_list: list[ModelInfo] = []

    # Add backbone models
    for name in models:
        model_list.append(ModelInfo(
            name=name,
            loaded=True,
            type="backbone",
        ))

    # Add student model if available
    if student is not None:
        model_list.append(ModelInfo(
            name="StudentModel",
            loaded=True,
            type="student",
        ))

    return ModelsListResponse(
        models=model_list,
        total=len(model_list),
        device=str(get_model_device()),
    )


@router.get("/classes", response_model=ClassesListResponse)
async def list_classes() -> ClassesListResponse:
    """
    List all disease classification classes.

    Returns the 13 sugarcane disease/health classes used by the models.
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
