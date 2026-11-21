// API client for the Sugarcane Disease Expert View.
// The base URL is user-configurable and persisted, so switching from local
// testing to the Q7/Q8 persistent web tunnel is a one-field change.

const DEFAULT_BASE = "http://127.0.0.1:8011";

export function getApiBase() {
  return localStorage.getItem("dbt_api_base") || DEFAULT_BASE;
}

export function setApiBase(value) {
  localStorage.setItem("dbt_api_base", value.replace(/\/+$/, ""));
}

async function parseError(res) {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body.detail || JSON.stringify(body);
  } catch {
    /* non-JSON error body */
  }
  return new Error(`HTTP ${res.status}: ${detail}`);
}

export async function checkReady() {
  const res = await fetch(`${getApiBase()}/health/ready`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function predictExpert(file, { validate = true, forceRedrot = false } = {}) {
  const params = new URLSearchParams();
  params.set("validate", String(validate));
  if (forceRedrot) params.set("force_redrot", "true");

  const form = new FormData();
  form.append("image", file);

  const res = await fetch(`${getApiBase()}/predict/expert?${params.toString()}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function downloadReport(result, imageBase64) {
  const res = await fetch(`${getApiBase()}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ result, image_base64: imageBase64 || null }),
  });
  if (!res.ok) throw await parseError(res);
  return res.blob();
}

export async function submitFeedback({ predictionId, imageHash, predicted, corrected, notes }) {
  const res = await fetch(`${getApiBase()}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prediction_id: predictionId,
      image_hash: imageHash,
      predicted_class: predicted,
      corrected_class: corrected,
      user_notes: notes || null,
    }),
  });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function fetchClasses() {
  const res = await fetch(`${getApiBase()}/classes`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}
