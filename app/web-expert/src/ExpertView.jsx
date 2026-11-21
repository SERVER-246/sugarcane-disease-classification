import { useEffect, useRef, useState } from "react";
import {
  checkReady,
  downloadReport,
  fetchClasses,
  fileToBase64,
  getApiBase,
  predictExpert,
  setApiBase,
  submitFeedback,
} from "./api";
import "./App.css";

const pct = (x) => `${(x * 100).toFixed(1)}%`;

function Confidence({ value }) {
  return (
    <div className="conf-bar" title={pct(value)}>
      <div className="conf-fill" style={{ width: pct(value) }} />
    </div>
  );
}

function VerdictCard({ result }) {
  const tone = result.disagreement ? "warn" : result.low_confidence ? "soft" : "ok";
  return (
    <section className={`card verdict ${tone}`}>
      <div className="verdict-top">
        <span className="verdict-label">Expert verdict</span>
        <span className="verdict-value">{result.overall_verdict || "—"}</span>
      </div>
      {result.disagreement && (
        <div className="banner warn">⚠ Models disagree — expert review suggested</div>
      )}
      {result.low_confidence && !result.disagreement && (
        <div className="banner soft">ⓘ Low screening confidence</div>
      )}
      {result.verdict_reason && <p className="muted small">{result.verdict_reason}</p>}
    </section>
  );
}

function CaptureGuidance({ result }) {
  if (!result.disagreement && !result.low_confidence) return null;
  return (
    <section className="card guidance">
      <h3>📸 Capture guidance</h3>
      <p>
        The whole-plant screen was unsure. For a more reliable result, take{" "}
        <strong>one full-plant photo</strong> for screening and a{" "}
        <strong>close-up of the affected stalk/leaf section</strong> so the Red Rot specialist can
        confirm symptoms.
      </p>
    </section>
  );
}

function ClassificationCard({ c }) {
  const [open, setOpen] = useState(false);
  const votes = c.class_votes || {};
  const maxVote = Math.max(1, ...Object.values(votes));
  return (
    <section className="card">
      <h3>Broad screen <span className="sub">(whole-plant)</span></h3>
      <div className="class-head">
        <span className="disease">{c.predicted_class}</span>
        <Confidence value={c.confidence} />
        <span className="muted">{pct(c.confidence)}</span>
      </div>
      <p className="muted small">
        {c.agreeing_models}/{c.total_models} models agree ({pct(c.agreement_ratio)}) · {c.voting_method} vote
      </p>

      <div className="votes">
        {Object.entries(votes)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, n]) => (
            <div className="vote-row" key={cls}>
              <span className="vote-label">{cls}</span>
              <div className="vote-track">
                <div
                  className={`vote-fill ${cls === c.predicted_class ? "lead" : ""}`}
                  style={{ width: `${(n / maxVote) * 100}%` }}
                />
              </div>
              <span className="vote-n">{n}</span>
            </div>
          ))}
      </div>

      <button className="link" onClick={() => setOpen((v) => !v)}>
        {open ? "Hide" : "Show"} per-model detail
      </button>
      {open && (
        <table className="model-table">
          <thead>
            <tr><th>Model</th><th>Prediction</th><th>Conf.</th></tr>
          </thead>
          <tbody>
            {c.model_predictions.map((m) => (
              <tr key={m.model} className={m.agrees_with_ensemble ? "agree" : ""}>
                <td>{m.model}</td>
                <td>{m.prediction}</td>
                <td>{pct(m.confidence)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function SegmentationCard({ seg, previewUrl }) {
  const [opacity, setOpacity] = useState(0.85);
  const [showOverlay, setShowOverlay] = useState(true);
  const [sideBySide, setSideBySide] = useState(false);
  const [zoom, setZoom] = useState(1);
  if (!seg) return null;
  const overlaySrc = `data:image/png;base64,${seg.overlay_png_b64}`;
  const ar = seg.mask_width && seg.mask_height ? seg.mask_width / seg.mask_height : 1;

  return (
    <section className="card">
      <h3>
        Red Rot specialist <span className="sub">(close-up)</span>
        <span className="badge">{seg.backbone}</span>
      </h3>

      <div className="seg-controls">
        <label className="chk"><input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} /> Overlay</label>
        <label className="range">Opacity
          <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
        </label>
        <label className="range">Zoom
          <input type="range" min="1" max="3" step="0.1" value={zoom} onChange={(e) => setZoom(+e.target.value)} />
        </label>
        <label className="chk"><input type="checkbox" checked={sideBySide} onChange={(e) => setSideBySide(e.target.checked)} /> Side-by-side</label>
      </div>

      <div className="seg-grid">
        {sideBySide && (
          <div className="overlay-stack" style={{ aspectRatio: ar }}>
            <img className="layer" src={previewUrl} alt="original" />
            <span className="corner">original</span>
          </div>
        )}
        <div className="overlay-stack" style={{ aspectRatio: ar, overflow: "hidden" }}>
          <img className="layer" src={previewUrl} alt="input" style={{ transform: `scale(${zoom})` }} />
          {showOverlay && (
            <img className="layer" src={overlaySrc} alt="symptom overlay" style={{ opacity, transform: `scale(${zoom})` }} />
          )}
          <span className="corner">overlay</span>
        </div>

        <div className="seg-stats">
          <p className="big">Affected area: <strong>{seg.affected_area_pct}%</strong></p>
          <p>Dominant: <strong>{seg.dominant_symptom || "—"}</strong></p>
          <p className="muted small">{seg.num_symptom_classes_detected} symptom class(es) detected</p>
          <div className="legend">
            {seg.symptoms.map((s) => (
              <div className="legend-item" key={s.name}>
                <span className="swatch" style={{ background: s.color }} />
                <span className="legend-name">{s.name}</span>
                <span className="legend-meta">{pct(s.pixel_fraction)} · {pct(s.mean_confidence)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function RemedialCard({ remedial }) {
  if (!remedial || !remedial.summary) return null;
  const placeholder = remedial.status === "PLACEHOLDER";
  return (
    <section className="card">
      <h3>Remedial measures {placeholder && <span className="badge warn">PLACEHOLDER</span>}</h3>
      <p>{remedial.summary}</p>
      {remedial.immediate_actions?.length > 0 && (
        <>
          <h4>Immediate actions</h4>
          <ul>{remedial.immediate_actions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </>
      )}
      {remedial.management?.length > 0 && (
        <>
          <h4>Management</h4>
          <ul>{remedial.management.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </>
      )}
    </section>
  );
}

function ExpertCorrection({ result, classes, onSubmitted }) {
  const [open, setOpen] = useState(result.disagreement);
  const [corrected, setCorrected] = useState(result.classification.predicted_class);
  const [notes, setNotes] = useState("");
  const [status, setStatus] = useState(null);

  async function send() {
    setStatus("sending");
    try {
      await submitFeedback({
        predictionId: result.classification.prediction_id,
        imageHash: result.classification.image_hash,
        predicted: result.classification.predicted_class,
        corrected,
        notes,
      });
      setStatus("done");
      onSubmitted?.();
    } catch (e) {
      setStatus(`error: ${e.message}`);
    }
  }

  return (
    <section className="card correction">
      <h3>🧑‍🔬 Expert correction {result.disagreement && <span className="badge warn">review suggested</span>}</h3>
      {!open && <button onClick={() => setOpen(true)}>Correct this result</button>}
      {open && (
        <div className="correction-form">
          <label>Correct disease
            <select value={corrected} onChange={(e) => setCorrected(e.target.value)}>
              {(classes.length ? classes : [result.classification.predicted_class]).map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <label>Notes
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Optional expert notes…" />
          </label>
          <button className="primary" disabled={status === "sending"} onClick={send}>
            {status === "sending" ? "Submitting…" : "Submit correction"}
          </button>
          {status === "done" && <span className="status ok">✓ Recorded for retraining</span>}
          {typeof status === "string" && status.startsWith("error") && <span className="status down">{status}</span>}
        </div>
      )}
    </section>
  );
}

export default function ExpertView() {
  const [apiBase, setApiBaseState] = useState(getApiBase());
  const [ready, setReady] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [validate, setValidate] = useState(true);
  const [forceRedrot, setForceRedrot] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [classes, setClasses] = useState([]);
  const fileInput = useRef(null);

  useEffect(() => {
    checkReady().then(setReady).catch(() => setReady(null));
    fetchClasses().then((d) => setClasses((d.classes || []).map((c) => c.name))).catch(() => {});
  }, [apiBase]);

  function onPick(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  function saveApiBase() {
    setApiBase(apiBase);
    setApiBaseState(getApiBase());
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictExpert(file, { validate, forceRedrot });
      setResult(data);
      setHistory((h) => [
        { name: file.name, verdict: data.overall_verdict, ts: new Date().toLocaleTimeString() },
        ...h,
      ].slice(0, 10));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function onDownloadReport() {
    try {
      const imgB64 = file ? await fileToBase64(file) : null;
      const blob = await downloadReport(result, imgB64);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "expert_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🌿 Sugarcane Disease — Expert View</h1>
        <p className="muted">Red Rot pilot · broad screen → Red Rot specialist</p>
      </header>

      <section className="card settings">
        <label>API base URL
          <input value={apiBase} onChange={(e) => setApiBaseState(e.target.value)} />
        </label>
        <button onClick={saveApiBase}>Save</button>
        <span className={`status ${ready ? "ok" : "down"}`}>
          {ready ? `ready · ${ready.device}` : "server unreachable"}
        </span>
      </section>

      <section className="card">
        <div className="controls">
          <button onClick={() => fileInput.current?.click()}>Choose image</button>
          <input ref={fileInput} type="file" accept="image/*" hidden onChange={onPick} />
          <label className="chk"><input type="checkbox" checked={validate} onChange={(e) => setValidate(e.target.checked)} /> Sugarcane validation gate</label>
          <label className="chk"><input type="checkbox" checked={forceRedrot} onChange={(e) => setForceRedrot(e.target.checked)} /> Force Red Rot specialist</label>
          <button className="primary" disabled={!file || loading} onClick={analyze}>
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        {previewUrl && !result && <img className="preview" src={previewUrl} alt="preview" />}
      </section>

      {error && <section className="card error">⚠ {error}</section>}

      {result && (
        <>
          <VerdictCard result={result} />
          <CaptureGuidance result={result} />
          <div className="actions">
            <button onClick={onDownloadReport}>⬇ Download PDF report</button>
          </div>
          <ClassificationCard c={result.classification} />
          <SegmentationCard seg={result.redrot_segmentation} previewUrl={previewUrl} />
          <RemedialCard remedial={result.remedial} />
          <ExpertCorrection result={result} classes={classes} />
        </>
      )}

      {history.length > 0 && (
        <section className="card">
          <h3>History</h3>
          <ul className="history">
            {history.map((h, i) => (
              <li key={i}><span>{h.ts}</span> {h.name} → <strong>{h.verdict}</strong></li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

