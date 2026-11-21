"""
Q6 tests — Expert View two-stage pipeline.

Lightweight unit tests always run. The heavy end-to-end tests (real model
weights) are marked ``slow``/``integration`` and skip automatically when the
Red Rot checkpoint is not present on the machine.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from inference_server import config
from inference_server.engine import redrot_segmentation as rr
from inference_server.engine.remedial import get_remedial


def _sample_redrot_image() -> Path | None:
    candidates = [
        config.BASE_DIR / "split_dataset" / "test" / "Red_rot",
        config.BASE_DIR / "Data" / "Red_rot",
    ]
    for d in candidates:
        if d.exists():
            for f in sorted(d.glob("*.jpg")):
                return f
    return None


_CKPT_PRESENT = config.REDROT_SEG_CKPT.exists()
_MAP_PRESENT = config.REDROT_SEG_CATEGORY_MAP.exists()
_IMG = _sample_redrot_image()


# --------------------------------------------------------------------------- #
# Unit tests (no model weights required)
# --------------------------------------------------------------------------- #
def test_remedial_placeholder_for_redrot():
    r = get_remedial("Red_rot")
    assert r["status"] == "PLACEHOLDER"
    assert r["summary"]
    assert isinstance(r["immediate_actions"], list)


def test_remedial_falls_back_to_default():
    r = get_remedial("No_Such_Disease")
    assert r["summary"]  # served from the _default entry
    assert r["disease"] == "No_Such_Disease"


def test_palette_has_rgb_per_class_and_black_background():
    palette = rr._build_palette(21)
    assert palette.shape == (21, 3)
    assert tuple(palette[0]) == (0, 0, 0)  # background is black


def test_preprocess_produces_normalized_512_tensor():
    tensor = rr._preprocess(Image.new("RGB", (123, 77), (120, 60, 40)))
    assert tuple(tensor.shape) == (1, 3, config.REDROT_SEG_IMG_SIZE, config.REDROT_SEG_IMG_SIZE)


def test_v2_disabled_by_default_returns_no_models():
    # V2 is opt-in; without DBT_ENABLE_V2 the loader yields nothing and never raises.
    from inference_server.engine import v2_loader

    if not config.ENABLE_V2:
        assert v2_loader.load_v2_classifiers() == {}


def test_v2_backbone_set_excludes_reparam_mobileone():
    # MobileOne's reparameterizable checkpoint doesn't match the fresh backbone,
    # so it must not be in the default V2 set.
    assert "CustomMobileOne" not in config.V2_BACKBONES
    assert "CustomDenseNetHybrid" in config.V2_BACKBONES


def test_report_pdf_is_valid_pdf():
    from inference_server.engine.report import build_report_pdf

    result = {
        "overall_verdict": "Red_rot — symptoms detected (12.2% area)",
        "disagreement": True,
        "low_confidence": True,
        "verdict_reason": "Whole-plant screen said Healthy but specialist found symptoms.",
        "classification": {
            "predicted_class": "Healthy", "confidence": 0.589,
            "agreeing_models": 8, "total_models": 21,
        },
        "redrot_segmentation": {
            "affected_area_pct": 12.2, "dominant_symptom": "outside_rind_discoloration",
            "overlay_png_b64": "", "symptoms": [
                {"name": "outside_rind_discoloration", "pixel_fraction": 0.10,
                 "mean_confidence": 0.69, "color": "#3cb44b"},
            ],
        },
        "remedial": {"status": "PLACEHOLDER", "summary": "Test.", "immediate_actions": ["Do X."]},
    }
    pdf = build_report_pdf(result, None)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 500


def test_hex_to_rgb_parsing():
    from inference_server.engine.report import _hex_to_rgb

    assert _hex_to_rgb("#3cb44b") == (60, 180, 75)
    assert _hex_to_rgb("bad") == (0, 0, 0)


def test_report_builder_tolerates_partial_result():
    # The /report endpoint accepts a free-form echoed result; the builder must not
    # require the full nested schema (clients may omit some ensemble fields).
    from inference_server.engine.report import build_report_pdf

    minimal = {
        "overall_verdict": "Healthy",
        "classification": {"predicted_class": "Healthy", "confidence": 0.9},
    }
    pdf = build_report_pdf(minimal, None)
    assert pdf[:4] == b"%PDF"


def test_v2_enabled_by_default():
    # Phase 3: both views should use the full Model 1+2 ensemble out of the box.
    assert config.ENABLE_V2 is True


def test_traffic_light_thresholds_are_ordered():
    # The "declare Red Rot" bar must sit above the "flag/possible" bar.
    assert config.REDROT_SAY_PCT > config.REDROT_DISAGREE_AREA_PCT
    assert config.RUN_SPECIALIST_ALWAYS is True


@pytest.mark.skipif(not _MAP_PRESENT, reason="Red Rot category map not present")
def test_category_mapping_is_21_classes():
    idx_to_name = rr._load_category_mapping()
    assert len(idx_to_name) == 21
    assert idx_to_name[0] == "background"
    # Correct mapping comes from unified_dataset.json category order.
    assert idx_to_name[12] == "Red Rot Root Infection"
    assert idx_to_name[20] == "white spot"


# --------------------------------------------------------------------------- #
# End-to-end tests (real weights; auto-skip when absent)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not (_CKPT_PRESENT and _IMG), reason="Red Rot checkpoint or sample image missing")
def test_segment_redrot_real_image():
    result = rr.segment_redrot(_IMG.read_bytes())
    assert config.REDROT_SEG_BACKBONE in result.backbone
    assert 0.0 <= result.affected_area_pct <= 100.0
    assert result.mask_height == config.REDROT_SEG_IMG_SIZE
    assert len(result.overlay_png_b64) > 0
    for s in result.symptoms:
        assert 0.0 <= s.pixel_fraction <= 1.0
        assert 0.0 <= s.mean_confidence <= 1.0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not (_CKPT_PRESENT and _IMG), reason="Red Rot checkpoint or sample image missing")
def test_expert_route_two_stage():
    from fastapi.testclient import TestClient

    from inference_server.app import app

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200

        with open(_IMG, "rb") as f:
            resp = client.post(
                "/predict/expert",
                params={"validate": "false", "force_redrot": "true"},
                files={"image": ("redrot.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["classification"]["predicted_class"]
        assert data["stage2_ran"] is True
        assert data["redrot_segmentation"]["backbone"]
        assert config.REDROT_SEG_BACKBONE in data["redrot_segmentation"]["backbone"]
        assert data["remedial"]["status"] == "PLACEHOLDER"
        # Verdict reconciliation fields are present and typed.
        assert "overall_verdict" in data and data["overall_verdict"]
        assert isinstance(data["disagreement"], bool)
        assert isinstance(data["low_confidence"], bool)
        # Traffic-light status is present and valid.
        assert data["farmer_status"] in ("green", "amber", "red")
        # Each detected symptom carries a hex color for the legend.
        for s in data["redrot_segmentation"]["symptoms"]:
            assert s["color"].startswith("#") and len(s["color"]) == 7


def _find_image(class_dir: str) -> Path | None:
    for base in (config.BASE_DIR / "split_dataset" / "test", config.BASE_DIR / "Data"):
        d = base / class_dir
        if d.exists():
            for f in sorted(d.glob("*.jpg")):
                return f
    return None


_HEALTHY_IMG = _find_image("Healthy")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not (_CKPT_PRESENT and _HEALTHY_IMG), reason="checkpoint or Healthy image missing")
def test_healthy_image_no_false_redrot_alarm():
    """The always-on specialist must not raise a RED Red Rot verdict on a healthy plant."""
    from fastapi.testclient import TestClient

    from inference_server.app import app

    with TestClient(app) as client:
        with open(_HEALTHY_IMG, "rb") as f:
            resp = client.post(
                "/predict/expert",
                params={"validate": "false"},
                files={"image": ("healthy.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # Specialist runs on every image now (pilot policy).
        assert data["stage2_ran"] is True
        # A healthy plant must not be declared Red Rot (red + Red_rot).
        if data["farmer_status"] == "red":
            assert data["farmer_disease"] != "Red_rot", (
                f"False Red Rot alarm on healthy image: area="
                f"{data['redrot_segmentation']['affected_area_pct']}%"
            )
