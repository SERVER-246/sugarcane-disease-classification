# Expert View Fixes + Q7/Q8 — Planning & Decisions

**Status:** 🟡 Planning in progress — Round 1 complete, Round 2 open
**Rule:** No implementation until all aspects are explicitly agreed and the user approves.
**Sequencing (locked):** Plan Q7+Q8 now; **build Expert View fixes (A/B/C) first**, then Q7.

---

## Workstreams

- **A. Verdict reconciliation** — behaviour when broad classifier and Red Rot specialist disagree.
- **B. Overlay alignment fix** — aspect-ratio mismatch (not EXIF).
- **C. Fuller, interactive Expert View display.**
- **D. Q7** — Farmer View + dataset collection/cleaning/preprocessing.
- **E. Q8** — dataset cleaning + overlapping-symptom identification + final Expert View integration & testing.

---

## Grounded findings (evidence)

- **Overlay misalignment** is an **aspect-ratio** issue, not rotation. Flagged images have no EXIF
  rotation (portrait 2656×3984). The model squashes to 512², but clients display the base with
  crop (`object-fit: cover` / `ContentScale.Crop`) while the square overlay fills the box → mismatch.
  Fix = size the display to the image aspect ratio and render base + overlay with `fill` (un-squash).
- **"Healthy" while symptoms found** is a **train-distribution mismatch**: the broad classifier was
  trained on whole-plant images (close-up lesion → "Healthy", low confidence), while the segmenter was
  trained on close-up annotations (finds the lesion). The two stages expect different photo types.

---

## Round 1 — Locked decisions

| # | Topic | Decision |
|---|---|---|
| A1 | Disagreement handling | **Flag disagreement**: keep both signals, add a clear "models disagree" banner |
| A2 | Headings | **Reframe** so "Classification" isn't read as the final verdict (e.g. Broad screen / Red Rot specialist) |
| B1 | Overlay fix | **Aspect-ratio box + un-squash overlay** (`fill`), no retrain — apply to Web + Android |
| C1 | Display additions | Symptom **legend**, overlay **opacity/toggle**, **per-model vote chart**, **zoom/pan**, **side-by-side** original vs overlay, **report export**, **capture-guidance** tips |
| C2 | Severity | Not added now (consistent with earlier severity deferral) |
| D1 | Sequencing | Plan Q7+Q8 now; build Expert View fixes first |
| D2 | Dataset scope (mine) | **Tooling AND clean/reorganize existing datasets** |
| D3 | Farmer View platforms | **Android + Web** |
| D4 | Farmer View scope | ⬜ **OPEN — discuss (Round 2)** |
| — | Planning doc | Maintained (this file) |

---

## Round 2 — Locked decisions

| # | Topic | Decision |
|---|---|---|
| D4 | Farmer View disease scope | **All 13 classes** (normal classification) with **Red Rot as specialist detail**; extensible as new diseases are added |
| D5 | Farmer View result | **Verdict + action + simple visual** (overlay thumbnail) |
| C3 | Report export | **PDF** (human-readable) |
| C4 | Capture guidance | **Dynamic**; when the classifier is unsure, prompt for close-up photos of the plant / specific sections, guiding the user |
| DS1 | Dataset cleaning | **Deferred** — no datasets to be cleaned now |

---

## Round 3 — Locked decisions

| # | Topic | Decision |
|---|---|---|
| A3 | Disagreement rule | Flag when classifier top ≠ Red_rot **AND** specialist affected-area ≥ **2%**; low-confidence note when classifier top < **60%**. Thresholds **configurable**. |
| A4 | Headings | **"Broad screen (whole-plant)"** / **"Red Rot specialist (close-up)"** + overall verdict + disagreement banner |
| A5 | Continuous learning | Disagreements are **expert-handled**: expert confirms/corrects the verdict → feeds a **feedback → retraining** cycle for ongoing improvement (wire capture to the existing feedback API). |
| D6 | Farmer connectivity | **Online for pilot; on-device TFLite as an explicit later milestone** |
| D7 | Farmer language | **English + one regional language** (which = Round 4) |
| C5 | Capture guidance | Dynamic, as described (whole-plant for screening, close-up for symptom confirmation) — **confirmed** |
| E1 | Overlapping-symptom analysis | **Deferred** with the dataset work |

---

## Round 4 — Locked decisions

| # | Topic | Decision |
|---|---|---|
| D7 | Regional language | **English + Hindi** (i18n table for more later) |
| A6 | Continuous learning | **Build expert correction/feedback capture now** (wired to the existing feedback API); **plan** the retraining pipeline as a later milestone |

---

## Consolidated Build Plan

### Phase 1 — Expert View fixes & enhancements  (build first, on approval)

**Backend** ([inference_server/](../inference_server))
- `config.py`: add `REDROT_DISAGREE_AREA_PCT` (default 2.0) and `CLS_LOW_CONF` (default 0.60), env-overridable.
- `engine/redrot_segmentation.py`: include a per-symptom **color (hex)** (from the overlay palette) in each symptom, so the legend matches the overlay exactly.
- `schemas.py`: `ExpertResult` += `overall_verdict`, `disagreement`, `low_confidence`, `verdict_reason`; `SymptomInfo` += `color`.
- `routes/inference.py`: compute verdict/disagreement/low-confidence in `/predict/expert`; add a **`/report`** endpoint returning a one-page **PDF** (lightweight dep, e.g. `fpdf2` — to confirm at build time). Existing `/feedback` route reused for expert corrections.

**Web** ([app/web-expert/](../app/web-expert)) and **Android** ([app/android-expert/](../app/android-expert)) — same feature set:
- Reframed headings: **"Broad screen (whole-plant)"** / **"Red Rot specialist (close-up)"** + **overall verdict** + **disagreement banner** + **low-confidence note**.
- **Overlay fix**: aspect-ratio box + `fill` (un-squash) so the overlay aligns with the photo.
- **Symptom legend** (colors from API), overlay **opacity/toggle**, **per-model vote chart**, **zoom/pan**, **side-by-side** original vs overlay, **PDF report** download, **dynamic capture guidance**.
- **Expert correction form** → submits to `/feedback` (prediction_id, corrected_class, notes), prompted especially on disagreement (continuous-learning capture).

**Tests:** backend unit (verdict/disagreement/low-confidence, symptom color, `/report` PDF); rebuild web + Android; re-demo (browser + emulator).

### Phase 2 — Farmer View (Q7)  (after Phase 1)
- New Farmer UI (**Android + Web**): all **13 classes** + Red Rot specialist detail; **verdict + action + simple visual**; **English + Hindi** (i18n table); **online** (swappable API URL); **dynamic capture guidance**. Extensible as new diseases are added.

### Deferred (planned, explicit later milestones)
- On-device **TFLite** export (Farmer offline use).
- **Retraining pipeline** consuming accumulated expert feedback (continuous-learning cycle).
- **Dataset cleaning** + **overlapping-symptom analysis** (Q8).
- **Severity index** (agronomy-calibrated).
- Real **remedial-measures** content (Q1 database).

---

## Approval gate

**Phase 1 begins only on explicit user approval.** Phase 2 (Farmer View) begins after Phase 1 is
accepted. Deferred items are each their own later, separately-approved milestone.

---

## Phase 1 — COMPLETED & VALIDATED (2026-11-20)

All Phase 1 items are built and verified on **both Web and Android** against the live server.

| Item | Status |
|---|---|
| Config thresholds (`REDROT_DISAGREE_AREA_PCT`=2.0, `CLS_LOW_CONF`=0.60, env-overridable) | ✅ |
| Verdict reconciliation in `/predict/expert` (`overall_verdict`, `disagreement`, `low_confidence`, `verdict_reason`) | ✅ |
| Per-symptom hex **color** in API + schema (legend matches overlay) | ✅ |
| **Overlay alignment fix** — aspect-ratio box + fill (un-squash); overlay now lines up with the photo | ✅ Web + Android |
| Reframed headings (Broad screen / Red Rot specialist) + verdict card + disagreement banner + low-conf note | ✅ Web + Android |
| Symptom **legend**, overlay **opacity/toggle**, **vote chart**, **zoom** (web) / controls, **side-by-side** (web) | ✅ |
| **Dynamic capture guidance** (whole-plant + close-up) | ✅ Web + Android |
| **PDF report** — server `/report` (fpdf2), tolerant dict input; downloaded 630 KB PDF on Android, 6.5 KB on web | ✅ |
| **Expert correction** → `/feedback` (continuous-learning capture), pre-opened on disagreement | ✅ Web + Android (recorded real prediction_id/image_hash) |
| Tests | ✅ 10 unit + 2 end-to-end pass; ruff clean |
| CI | `web-build` + `android-build` jobs already in `ci.yml` cover the rebuilt clients |

**New dependency:** `fpdf2` (added to `requirements.txt`, `requirements-ci.txt`).

### Next: Phase 2 — Farmer View (awaiting Phase 1 acceptance)
All 13 classes + Red Rot specialist; verdict + action + simple visual; **English + Hindi** (i18n);
online pilot (on-device later); dynamic capture guidance. Android + Web.

---

## Phase 2 — Farmer View: Locked decisions & Build Plan

### Locked decisions

| # | Topic | Decision |
|---|---|---|
| F1 | Structure | **Same app + Expert⇄Farmer toggle** (one codebase, shared API client) |
| F2 | Entry default | **Ask on first run**, then remember last-used view (persisted) |
| F3 | Capture | **Camera-first** (big capture button); gallery secondary |
| F4 | Result style | **Traffic-light** status (green Healthy / red disease / amber uncertain) + disease name |
| F5 | Confidence | **Plain words** (Likely / Uncertain), no percentage |
| F6 | Visual | **Overlay thumbnail** when Red Rot detected; plain photo otherwise |
| F7 | Disagreement/low-conf | **Capture guidance + "consult an expert"** (no correction form — expert-only) |
| F8 | Hindi | **Not translated now**; build **i18n-ready** with a working language toggle, English populated, Hindi falling back to English so translations drop in later with no restructuring |
| F9 | Language toggle | **Present** (English ⇄ हिंदी), persisted |
| F10 | Validation gate | **On** (reject non-sugarcane before analysis) |
| F11 | Report/share | **None** — keep the farmer flow minimal |

### Build plan (no backend change — Farmer View reuses `/predict/expert`)

**Web** ([app/web-expert/](../app/web-expert))
- `src/i18n.js`: locale string table (`en` full, `hi` fallback→en), `t(key)` + persisted locale.
- Extract current UI into `ExpertView`; add `FarmerView`; `App` hosts the Expert⇄Farmer toggle + first-run prompt + language toggle.
- `FarmerView`: camera-first capture, traffic-light result, plain-words confidence, Red Rot overlay thumbnail, one-line action (from remedial), capture-guidance + consult-expert on disagreement, validation gate on.

**Android** ([app/android-expert/](../app/android-expert))
- `ui/Strings.kt`: locale map (`en` full, `hi` fallback→en) + `t(key)`.
- Extract `ExpertScreen` (unchanged); add `FarmerScreen`; `MainActivity` hosts the mode toggle + first-run dialog + language toggle, persisted in `Settings`.
- `FarmerScreen`: camera-first, traffic-light card, plain-words confidence, Red Rot overlay thumbnail, action line, guidance + consult on disagreement, validation gate on.

**Traffic-light / confidence logic (client-side, shared definition)**
- **Amber "Uncertain"** if `disagreement` or `low_confidence` → show capture guidance + consult expert.
- else **Green "Healthy"** if overall verdict is Healthy.
- else **Red "<Disease> detected"**.
- Confidence word: **Likely** when not low-confidence/disagreement, else **Uncertain**.

**Validation:** build both clients; live-demo in browser + emulator (farmer flow, toggle, language toggle, traffic-light, overlay thumbnail, guidance). CI `web-build`/`android-build` already cover the rebuilds.

### Phase 2 approval gate
Build begins only on explicit user approval of this plan.

---

## Phase 2 — COMPLETED & VALIDATED (2026-11-21)

Farmer View built and verified on **both Web and Android** against the live server.

| Item | Status |
|---|---|
| Expert⇄Farmer toggle (same app, shared API client) | ✅ Web + Android |
| First-run view chooser, then remember last used | ✅ Web (localStorage) + Android (SharedPreferences) |
| Language toggle (English ⇄ हिंदी), persisted; i18n table (`i18n.js` / `Strings.kt`), Hindi falls back to English | ✅ |
| Camera-first capture (+ gallery) | ✅ |
| Traffic-light result (🟢 Healthy / 🔴 disease / 🟠 uncertain) + disease name | ✅ (green + amber demoed) |
| Plain-words confidence (Likely / Uncertain) | ✅ |
| Red Rot overlay thumbnail when detected; plain photo otherwise | ✅ |
| One-line action from remedial DB | ✅ |
| Disagreement/low-conf → capture guidance + consult-expert (no correction form) | ✅ |
| Sugarcane validation gate on | ✅ |
| No PDF in farmer flow | ✅ |
| No backend change (reuses `/predict/expert`); backend tests still 10/10 + 2 e2e | ✅ |

**Validation:** live browser demo (first-run, farmer green + amber, language + view toggles) and
API-36 emulator demo (first-run, farmer amber result, Expert toggle). Both clients build; backend lint
clean, tests pass. CI `web-build`/`android-build` cover the rebuilds.

**Hindi note:** the toggle is live and the structure is i18n-ready; actual Hindi strings are deferred —
filling the `hi` table (web `i18n.js`, Android `Strings.kt`) adds Hindi with no restructuring.

---

## Phase 3 — Post-review fixes (approved 2026-11-21)

Addressing three review findings. Locked decisions:

| # | Topic | Decision |
|---|---|---|
| G1 | Farmer≠Expert on same image | Root cause: **sugarcane validation gate** (needs ≥40% green) rejects diseased close-ups; Farmer forced it on with no escape. Fix: **relax gate** (lower vegetation threshold, config-driven) **+ gentle fallback** ("couldn't confirm — Analyze anyway") in Farmer View. |
| G2 | "Say Red Rot when the specialist finds it" | **Always run the Red Rot specialist** during the pilot (so Farmer catches classifier misses); **server computes one traffic-light status**; specialist **overrides** a wrong "Healthy". |
| G3 | Traffic-light thresholds | **RED "Red Rot detected"** when specialist affected-area ≥ **5%** (`REDROT_SAY_PCT`); **AMBER "possible Red Rot — recheck"** band for 2–5%; GREEN when classifier Healthy + no specialist signal. All configurable; tuned against healthy images. |
| G4 | V2 (Model 2) | **Enabled by default** so both views use the full Model 1+2 ensemble (was opt-in). |
| G5 | Status source of truth | **Server-side** (`/predict/expert`), consumed by both clients. |
| G6 | Expert View | Unchanged UX — still shows both signals + disagreement banner; now always has the specialist result. |

**Then:** screenshots of every step (Web + Android) into `F:\Report_Images`; scoped Q5–Q8 commit
(secrets gitignored, file list shown) pushed to `origin/main` on final OK.

