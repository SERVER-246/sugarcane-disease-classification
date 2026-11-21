import { useRef, useState } from "react";
import { predictExpert } from "./api";
import { t } from "./i18n";
import "./FarmerView.css";

function prettyDisease(name) {
  return (name || "").replace(/_/g, " ");
}

export default function FarmerView({ locale }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [needsConfirm, setNeedsConfirm] = useState(false);
  const cameraInput = useRef(null);
  const galleryInput = useRef(null);

  function onPick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setNeedsConfirm(false);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function analyze(validate = true) {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setNeedsConfirm(false);
    try {
      const data = await predictExpert(file, { validate, forceRedrot: false });
      const failedValidation =
        validate &&
        (data.overall_verdict === "Not a sugarcane image" || data.classification.validation?.passed === false);
      if (failedValidation) {
        // Gentle fallback: don't hard-block — let the farmer analyze anyway.
        setNeedsConfirm(true);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (e) {
      setError(e.message.includes("Failed to fetch") ? t("server_unreachable", {}, locale) : e.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError(null);
    setNeedsConfirm(false);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
  }

  // Server-computed traffic-light status is the single source of truth.
  const status = result?.farmer_status || null;
  const seg = result?.redrot_segmentation;
  const showOverlay = status === "red" && result?.farmer_disease === "Red_rot" && seg && seg.affected_area_pct > 0;
  const action = result?.remedial?.immediate_actions?.[0] || result?.remedial?.summary || null;

  function headline() {
    if (status === "green") return t("status_healthy", {}, locale);
    if (status === "red") return t("status_disease", { disease: prettyDisease(result.farmer_disease) }, locale);
    // amber
    if (result.farmer_disease === "Red_rot") return t("status_possible_redrot", {}, locale);
    return t("status_uncertain", {}, locale);
  }

  return (
    <div className="farmer">
      <p className="farmer-tagline">{t("farmer_tagline", {}, locale)}</p>

      {!result && (
        <div className="farmer-capture">
          {previewUrl && <img className="farmer-preview" src={previewUrl} alt="preview" />}
          <button className="big-btn capture" onClick={() => cameraInput.current?.click()}>
            {t("capture", {}, locale)}
          </button>
          <button className="big-btn ghost" onClick={() => galleryInput.current?.click()}>
            {t("gallery", {}, locale)}
          </button>
          <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={onPick} />
          <input ref={galleryInput} type="file" accept="image/*" hidden onChange={onPick} />
          {file && !needsConfirm && (
            <button className="big-btn primary" disabled={loading} onClick={() => analyze(true)}>
              {loading ? t("analyzing", {}, locale) : t("analyze", {}, locale)}
            </button>
          )}
          {needsConfirm && (
            <div className="farmer-confirm">
              <p>⚠ {t("not_sugarcane", {}, locale)}</p>
              <button className="big-btn primary" disabled={loading} onClick={() => analyze(false)}>
                {loading ? t("analyzing", {}, locale) : t("analyze_anyway", {}, locale)}
              </button>
            </div>
          )}
          {error && <div className="farmer-error">⚠ {error}</div>}
        </div>
      )}

      {result && (
        <div className="farmer-result">
          <div className={`status-card ${status}`}>
            <div className="status-icon">
              {status === "green" ? "✓" : status === "red" ? "!" : "?"}
            </div>
            <div className="status-text">{headline()}</div>
            {status === "green" && (
              <div className="status-conf">
                {t("confidence", {}, locale)}: {t("likely", {}, locale)}
              </div>
            )}
          </div>

          {showOverlay ? (
            <div className="farmer-visual">
              <div className="farmer-overlay-stack" style={{ aspectRatio: (seg.mask_width / seg.mask_height) || 1 }}>
                <img className="layer" src={previewUrl} alt="crop" />
                <img className="layer" src={`data:image/png;base64,${seg.overlay_png_b64}`} alt="signs" />
              </div>
              <p className="farmer-caption">
                {t("redrot_symptoms", {}, locale)} · {t("affected_area", {}, locale)} {seg.affected_area_pct}%
              </p>
            </div>
          ) : (
            previewUrl && <img className="farmer-preview" src={previewUrl} alt="crop" />
          )}

          {status === "amber" && (
            <div className="farmer-guidance">
              <h3>{t("guidance_title", {}, locale)}</h3>
              <p>{t("guidance_body", {}, locale)}</p>
              <p className="consult">{t("consult_expert", {}, locale)}</p>
            </div>
          )}

          {status !== "amber" && action && (
            <div className="farmer-action">
              <h3>{t("recommended_action", {}, locale)}</h3>
              <p>{action}</p>
            </div>
          )}

          <button className="big-btn ghost" onClick={reset}>{t("retake", {}, locale)}</button>
        </div>
      )}
    </div>
  );
}
