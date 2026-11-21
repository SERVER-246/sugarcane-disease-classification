// Lightweight i18n for the Farmer View (and shared chrome).
// English is fully populated. Hindi (hi) is intentionally left empty for now and
// falls back to English, so real Hindi translations can be dropped in later by
// filling the `hi` table — no structural change required.

const STRINGS = {
  en: {
    app_title: "🌿 Sugarcane Disease",
    expert_view: "Expert",
    farmer_view: "Farmer",
    choose_view: "Welcome — choose your view",
    choose_view_desc: "You can switch anytime from the top bar.",
    im_expert: "I'm an expert",
    im_farmer: "I'm a farmer",
    language: "Language",

    farmer_tagline: "Point, capture, and check your crop",
    capture: "📷 Capture photo",
    gallery: "Choose from gallery",
    analyzing: "Checking…",
    analyze: "Check crop",
    retake: "Check another photo",

    status_healthy: "Healthy",
    status_disease: "{disease} detected",
    status_uncertain: "Uncertain — please recheck",
    status_possible_redrot: "Possible Red Rot — please recheck",

    confidence: "Confidence",
    likely: "Likely",
    uncertain_word: "Uncertain",

    recommended_action: "Recommended action",
    consult_expert: "Please consult an agricultural expert to confirm.",

    guidance_title: "📸 Take a clearer photo",
    guidance_body:
      "Take one full-plant photo for screening and a close-up of the affected stalk or leaf, then check again.",

    redrot_symptoms: "Red Rot signs",
    affected_area: "Affected area",

    not_sugarcane: "We couldn't confirm this is a sugarcane image.",
    analyze_anyway: "Analyze anyway",
    server_unreachable: "Cannot reach the server. Please check your connection.",
    server_ready: "Connected",
  },
  hi: {
    // Hindi translations pending expert review — fall back to English for now.
  },
};

export const LOCALES = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिंदी" },
];

export function getLocale() {
  return localStorage.getItem("dbt_locale") || "en";
}

export function setLocale(code) {
  localStorage.setItem("dbt_locale", code);
}

export function t(key, vars = {}, locale = getLocale()) {
  const table = STRINGS[locale] || {};
  let s = table[key] ?? STRINGS.en[key] ?? key;
  for (const [k, v] of Object.entries(vars)) {
    s = s.replace(`{${k}}`, String(v));
  }
  return s;
}
