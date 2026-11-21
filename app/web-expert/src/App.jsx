import { useEffect, useState } from "react";
import { checkReady } from "./api";
import ExpertView from "./ExpertView";
import FarmerView from "./FarmerView";
import { getLocale, LOCALES, setLocale, t } from "./i18n";
import "./App.css";
import "./Shell.css";

const VIEW_KEY = "dbt_view";

export default function App() {
  const [view, setView] = useState(() => localStorage.getItem(VIEW_KEY)); // null until chosen
  const [locale, setLocaleState] = useState(getLocale());
  const [ready, setReady] = useState(null);

  useEffect(() => {
    checkReady().then(setReady).catch(() => setReady(null));
  }, []);

  function chooseView(v) {
    setView(v);
    localStorage.setItem(VIEW_KEY, v);
  }

  function switchLocale(code) {
    setLocale(code);
    setLocaleState(code);
  }

  // First-run: ask which view to use.
  if (!view) {
    return (
      <div className="shell">
        <div className="first-run">
          <h1>{t("app_title", {}, locale)}</h1>
          <h2>{t("choose_view", {}, locale)}</h2>
          <p className="muted">{t("choose_view_desc", {}, locale)}</p>
          <div className="first-run-actions">
            <button className="choice farmer" onClick={() => chooseView("farmer")}>
              🌾 {t("im_farmer", {}, locale)}
            </button>
            <button className="choice expert" onClick={() => chooseView("expert")}>
              🔬 {t("im_expert", {}, locale)}
            </button>
          </div>
          <LanguageToggle locale={locale} onChange={switchLocale} />
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <div className="topbar">
        <span className="brand">{t("app_title", {}, locale)}</span>
        <div className="toggle">
          <button className={view === "farmer" ? "on" : ""} onClick={() => chooseView("farmer")}>
            🌾 {t("farmer_view", {}, locale)}
          </button>
          <button className={view === "expert" ? "on" : ""} onClick={() => chooseView("expert")}>
            🔬 {t("expert_view", {}, locale)}
          </button>
        </div>
        <LanguageToggle locale={locale} onChange={switchLocale} compact />
      </div>

      {view === "farmer" ? <FarmerView locale={locale} ready={ready} /> : <ExpertView />}
    </div>
  );
}

function LanguageToggle({ locale, onChange, compact }) {
  return (
    <div className={`lang ${compact ? "compact" : ""}`}>
      {!compact && <span className="lang-label">{t("language", {}, locale)}:</span>}
      {LOCALES.map((l) => (
        <button key={l.code} className={locale === l.code ? "on" : ""} onClick={() => onChange(l.code)}>
          {l.label}
        </button>
      ))}
    </div>
  );
}
