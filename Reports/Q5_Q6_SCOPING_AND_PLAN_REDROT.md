# Q5–Q6 Scoping & Execution Plan — Red Rot First

**Status:** FROZEN FOR REVIEW — no code to be written until explicitly approved
**Scope:** DBT Year 2, Quarter 5 and Quarter 6 deliverables
**Strategy:** Red Rot is the first application use-case; other diseases are added later behind the same architecture.

---

## 1. Official Milestone Basis

From the official DBT milestone document (`G:\DBT-Milestones-ISRI.pdf`):

| Quarter | Official Deliverable |
|---|---|
| Q5 | GUI development for Mobile application (Expert View) |
| Q6 | Integration of Sugarcane disease Classification Models 1 & 2 to the Mobile app (Expert View) & testing |

---

## 2. Locked Decisions

| # | Aspect | Decision |
|---|---|---|
| 1 | Inference architecture | **Server-based** now (reuse `inference_server/` FastAPI); on-device (TFLite) added later |
| 2 | "Models 1 & 2" | **Broad 13-class classifiers** — V1 and V2 from Year 1 |
| 3 | Red Rot specialist | **Segmentation model** from `F:\Migration_RedRot_20260727` (per-symptom, 21 classes) |
| 4 | Future diseases | Add one specialist segmentation model per disease behind the same broad classifier |
| 5 | Expert View design | **Two-stage**: broad classifier → Red Rot segmentation overlay when Red Rot is detected |
| 6 | Models 1 & 2 presentation | **Ensembled** into one result; expert can expand per-model detail |
| 7 | Tech stack | **Native Android** (Kotlin + Jetpack Compose) |
| 8 | Platforms | **Android + Web** (both are Expert View clients calling the same API) |
| 9 | Web role | **Web = Expert View** (browser version of the same app) |
| 10 | Code location | **This repo** (new `app/` area) |
| 11 | Q6 testing | unit + integration + accuracy parity + field, **plus full CI/CD** |
| 12 | Red Rot deployment model | **CustomResNetMish (Snapshot 3 retrain)** primary — best individual on the deployed retrain weights (22.80% mIoU, 154 MB); ensemble optional server-side mode |
| 13 | Web framework | **React** |
| 14 | Remedial-measures content | **Scaffold placeholder** Red Rot recommendations file; fill in later |
| 15 | Model 1 & 2 order | **V1 (Model 1) first**, add V2 (Model 2) right after |
| 16 | Sugarcane validation gate | **On by default** in Expert View |
| 17 | Execution approach | **Local-first**: build full backend + Android design, then define & test API contracts locally; base URL kept swappable for a persistent web tunnel in Q7/Q8 |
| 18 | Severity definition | **Deferred** — show raw % affected area + detected symptoms + confidence now; formal severity index (with agronomy-calibrated bands) later |

---

## 3. Target Architecture

```
          ┌─────────────────────────┐        ┌─────────────────────────┐
          │   Android Expert View   │        │     Web Expert View     │
          │  (Kotlin + Compose)     │        │      (browser)          │
          └────────────┬────────────┘        └────────────┬────────────┘
                       │  HTTPS (same REST API)            │
                       └───────────────┬───────────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │   FastAPI inference server    │   (inference_server/)
                        │                               │
                        │  Stage 1: Broad classifier    │   Models 1 & 2 (13-class), ensembled
                        │     └─ disease + confidence    │
                        │                               │
                        │  Stage 2: Red Rot specialist  │   InceptionV4 seg (21-class)
                        │     └─ symptom overlay,         │   runs when Stage 1 = Red Rot
                        │        % area, severity         │   (or Red Rot mode selected)
                        │                               │
                        │  + remedial recommendations    │
                        └──────────────────────────────┘
```

**Why this fits "Red Rot first, expand later":** adding a new disease = register its specialist
segmentation model behind the same broad classifier and the same API. No client redesign.

---

## 4. Model Inventory & Evidence (Red Rot)

Source: `F:\Migration_RedRot_20260727\Paper_redrot`

**Segmentation task:** 21 classes (background + 20 Red Rot symptom categories).

**Authoritative basis = Snapshot 3 retrain** (full 3,595 images, 2,876 train / 719 val). These are the
weights in `seg_checkpoints_retrain/` that we actually deploy. Per-backbone retrain mIoU (top of ranking):

| Rank | Backbone | Retrain mIoU | Retrain size |
|---|---|---|---|
| **1** | **CustomResNetMish** | **22.80%** | 154 MB |
| 2 | CustomDenseNetHybrid | 19.29% | 29 MB |
| 3 | CustomDeiTStyle | 19.16% | 386 MB |
| 4 | CustomSwinTransformer | 18.44% | 352 MB |
| 5 | CustomViTHybrid | 17.73% | 548 MB |
| … | … | … | … |
| 11 | CustomInceptionV4 | 16.18% | 30 MB |

The retrain `pipeline_summary_retrain.json` explicitly records `best_individual = CustomResNetMish
(0.22795 mIoU)`. ResNetMish also exceeds the old seed-based weighted ensemble (21.94%).

**Context (superseded for deployment):** on the *seed* Snapshot 1 (~509 images) InceptionV4 looked best
(20.47%), and the seed K-fold put DenseNetHybrid on top — but neither reflects the deployed retrain
weights. Deployment follows Snapshot 3.

**Why Snapshot 3 over Snapshot 1** (506-image holdout): mean max confidence 0.316 → 0.377; images above
0.5 confidence 11.3% → 23.5%; unique classes covered 4 → 5.

**Per-symptom reliability (honest):** ResNetMish retrain is strong on dominant symptoms (nodal
discoloration 0.46 IoU, mid-rib infection 0.53, red-rot root infection, nodal sporulation 0.34) but
near-zero on rare symptoms (bud death, rind splitting, nodal break-off) due to class imbalance. High
pixel accuracy is dominated by background. **Expert View must display confidence and avoid over-claiming
on rare symptoms.**

**Size note:** 154 MB is a non-issue for the server-based Q5/Q6 architecture; quantize/distill for the
later on-device path.

---

## 5. Q5 Scope — Expert View GUI

**Goal:** Build the Expert View app shell (Android + Web) with the full UX, wired to a stubbed/real API.

In scope:
- Image capture (CameraX) + gallery selection (Android); upload (Web)
- Optional "is this sugarcane?" validation gate (`image_validator.py` logic server-side)
- Results screen: disease + confidence, expandable per-model detail, severity indicator
- Red Rot symptom overlay view (segmentation mask rendering) — UI container ready
- Recommended remedial measures panel
- Local history (Android: Room; Web: session/local storage)
- Settings: confidence threshold, model/version info

Out of scope for Q5 (belongs to Q7 Farmer View or later):
- Simplified Farmer View
- Admin/log-verification/correction console (Q11)
- On-device offline inference (TFLite)

**Q5 exit criteria:** app runs on Android + Web; user can capture/upload an image, call the API, and
see a structured Expert View result; UI stable and ready for real model integration.

---

## 6. Q6 Scope — Integration + Testing

**Goal:** Wire the real models end-to-end and validate.

In scope:
- Serve **Models 1 & 2** (broad 13-class classifiers) via FastAPI, ensembled into one result
- Register **CustomInceptionV4 Red Rot specialist** (Snapshot 3 retrain) as Stage 2; return symptom
  mask, % affected area, per-symptom confidence, severity
- Optional server-side "max accuracy" ensemble mode (deferable)
- Wire remedial-measures content into the response
- Testing bar:
  - **Unit** — inference, validation, mask post-processing
  - **Integration** — capture/upload → API → result on both clients
  - **Accuracy parity** — app result matches the reference pipeline on a fixed sample set
  - **Field** — real field images, including non-sugarcane rejection
  - **CI/CD** — extend `.github/workflows/` (ci.yml, test-matrix.yml, docker-build.yml) to cover the
    app + server endpoints (lint, tests, build)

**Q6 exit criteria:** Expert View runs real inference (broad classification + Red Rot overlay) on both
clients; all four test categories pass; CI/CD green.

---

## 7. Dependencies, Risks & Inputs Needed

| Item | Status / Need |
|---|---|
| Red Rot retrain weights | ✅ Present (`seg_checkpoints_retrain/`, all 15) |
| Broad classifier serving | Model 1 (V1) already loads in `inference_server/`; **V1 first**, Model 2 (V2 dual-head) wired right after |
| Remedial-measures database | ✅ Decision: **scaffold placeholder** Red Rot recommendations file; real content filled in later |
| Web framework | ✅ **React** |
| Severity definition | ✅ **Deferred** for Q5/Q6 — show raw % symptomatic area + detected symptoms + confidence; formal agronomy-calibrated index added later |
| Sugarcane validator default | ✅ **On by default** |
| Persistent web tunnel | Q7/Q8 forward dependency — reserve a dedicated config/module; keep API base URL swappable so local → tunnel is a one-line change |
| Expert auth/users | Assumed later (Q11); not in Q5/Q6 unless required |
| Segmentation accuracy | Modest mIoU — surface confidence; set expectations in UI |

---

## 8. Proposed Work Breakdown (NOT started — awaiting approval)

Execution approach: **local-first**. Build the full backend + Android design first, then define and
test the API contracts locally; keep the base URL swappable so the Q7/Q8 persistent web tunnel drops in
with a one-line config change.

1. Resolve the remaining open input (severity definition, Section 7).
2. Backend (server-based): two-stage pipeline endpoint (broad **V1** ensemble → Red Rot specialist), response schema, remedial placeholder content; reserve a config slot for the future web tunnel base URL.
3. Red Rot specialist integration: load **CustomResNetMish** Snapshot 3 retrain, mask post-processing, severity, overlay encoding.
4. Android Expert View: screens, camera/gallery, API client, result + overlay rendering, history, settings.
5. Web Expert View: **React** client against the same API.
6. API contracts: finalize and test locally end-to-end across both clients.
7. Add **V2 (Model 2)** to the broad-classifier ensemble.
8. Tests: unit + integration + parity + field harness.
9. CI/CD: extend workflows to app + server.
10. Documentation + Q5/Q6 completion report.

---

## 9. Approval Gate

**No implementation begins until the user explicitly approves this plan and names the first step to start.**
Open inputs in Section 7 should be resolved before or at the start of the corresponding task.
