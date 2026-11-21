"""
Expert View PDF Report
======================
Builds a one-page PDF summary of an Expert View result (verdict, broad screen,
Red Rot specialist overlay + symptoms, remedial notes) using fpdf2.

The client posts the ExpertResult it already has (plus, optionally, the original
image to composite under the overlay), so the PDF matches exactly what was shown.
"""
from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from typing import Any

from fpdf import FPDF
from PIL import Image


def _s(text: Any) -> str:
    """Latin-1 safe text for the core PDF fonts."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _mc(pdf: FPDF, h: float, text: str) -> None:
    """Full-width multi_cell that always starts at the left margin."""
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, h, _s(text))


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _overlay_image(seg: dict[str, Any], image_base64: str | None) -> BytesIO | None:
    try:
        overlay = Image.open(BytesIO(base64.b64decode(seg["overlay_png_b64"]))).convert("RGBA").resize((512, 512))
        if image_base64:
            base = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGBA").resize((512, 512))
        else:
            base = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
        comp = Image.alpha_composite(base, overlay).convert("RGB")
        buf = BytesIO()
        comp.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


def build_report_pdf(result: dict[str, Any], image_base64: str | None = None) -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _s("Sugarcane Disease - Expert View Report"))
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 5, _s("Generated " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC") + "  -  Red Rot pilot"))
    pdf.ln(7)
    pdf.set_text_color(0, 0, 0)

    verdict = result.get("overall_verdict", "")
    pdf.set_font("Helvetica", "B", 13)
    _mc(pdf, 7, "Verdict: " + str(verdict))

    if result.get("disagreement") or result.get("low_confidence"):
        pdf.set_text_color(180, 90, 0)
        pdf.set_font("Helvetica", "", 9)
        prefix = "! " if result.get("disagreement") else ""
        _mc(pdf, 5, prefix + (result.get("verdict_reason") or ""))
        pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    cls = result.get("classification", {}) or {}
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, _s("Broad screen (whole-plant)"))
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    conf = float(cls.get("confidence", 0)) * 100
    _mc(pdf, 6,
        f"Predicted: {cls.get('predicted_class', '')}  ({conf:.1f}%)   "
        f"Agreement: {cls.get('agreeing_models', 0)}/{cls.get('total_models', 0)} models")
    pdf.ln(1)

    seg = result.get("redrot_segmentation")
    if seg:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, _s("Red Rot specialist (close-up)"))
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 10)
        _mc(pdf, 6,
            f"Affected area: {seg.get('affected_area_pct', 0)}%   "
            f"Dominant symptom: {seg.get('dominant_symptom') or '-'}")
        img = _overlay_image(seg, image_base64)
        if img is not None:
            pdf.image(img, w=65)
            pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(6, 5, "")
        pdf.cell(60, 5, _s("Symptom"))
        pdf.cell(25, 5, _s("Area"))
        pdf.cell(25, 5, _s("Conf."))
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 9)
        for sym in seg.get("symptoms", []):
            r, g, b = _hex_to_rgb(sym.get("color", "#000000"))
            pdf.set_fill_color(r, g, b)
            pdf.cell(5, 5, "", fill=True)
            pdf.cell(1, 5, "")
            pdf.cell(60, 5, _s(sym.get("name", "")))
            pdf.cell(25, 5, _s(f"{float(sym.get('pixel_fraction', 0)) * 100:.1f}%"))
            pdf.cell(25, 5, _s(f"{float(sym.get('mean_confidence', 0)) * 100:.1f}%"))
            pdf.ln(5)
        pdf.ln(1)

    rem = result.get("remedial") or {}
    if rem.get("summary"):
        pdf.set_font("Helvetica", "B", 11)
        tag = "  [PLACEHOLDER]" if rem.get("status") == "PLACEHOLDER" else ""
        pdf.cell(0, 6, _s("Remedial measures" + tag))
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 9)
        _mc(pdf, 5, rem.get("summary", ""))
        for action in (rem.get("immediate_actions") or []):
            _mc(pdf, 5, "- " + action)

    return bytes(pdf.output())
