# Q5–Q6 Completion Report — Red Rot Expert View (Pilot)

**Status:** ✅ Delivered and validated (backend + Web + Android, Models 1 & 2 integrated)
**Scope:** DBT Year 2, Q5 (Expert View GUI) + Q6 (model integration & testing)
**Strategy:** Red Rot-first; server-based inference now, on-device later.

---

## 0. Post-review fixes (this iteration)

Three issues raised in review were resolved and verified:

1. **Symptom labels were seriously wrong** — root cause was a **category-mapping mismatch**, not the
   model. The retrain masks index symptoms by `unified_dataset.json` `category_id` order (id 12 = *Red
   Rot Root Infection*, id 20 = *white spot*), but the server used the seed `category_mapping.json` (a
   different order), so every channel was mislabeled. Fixed by bundling the correct mapping
   ([data/redrot_category_mapping.json](../inference_server/data/redrot_category_mapping.json)).
   Cross-checked against the expert COCO annotations: predicted dominant symptoms now match ground
   truth on common classes (e.g. white spot, outside_rind_discoloration).
2. **Single-model segmentation → ensemble.** The Red Rot specialist is now a **3-backbone soft-voting
   ensemble** (ResNetMish + DenseNetHybrid + RegNet), like the classification ensemble.
3. **UI redesign.** Both clients moved off the plain white look to a colorful, modern palette
   (gradient header, disease chip, green/amber confidence bars, framed symptom overlay).

Honest caveat: the true *Red Rot Root Infection* symptom (id 12) is rare in the data (~88 annotations)
and is under-segmented — a class-imbalance limitation surfaced in the UI via per-symptom confidence.

---

## 1. What was delivered

### Q5 — Expert View GUI (Android + Web)
- **Android native app** ([app/android-expert/](../app/android-expert)) — Kotlin + Jetpack Compose:
  image capture (camera), gallery, and bundled sample; classification view with per-model detail;
  Red Rot symptom overlay; remedial panel; history; configurable API base URL (defaults to
  `http://10.0.2.2:8011` for the emulator). Colorful Material3 theme. **Builds (`assembleDebug`) and
  runs in the API‑36 emulator (overlay aligns with real lesions).**
- **React Web app** ([app/web-expert/](../app/web-expert)) — Vite + React: upload, classification with
  per-model detail, Red Rot overlay, symptom table, remedial panel, swappable API base URL. Redesigned
  colorful palette. **Builds and runs; verified live in-browser.**

### Q6 — Model integration & testing
- **Two-stage backend** ([inference_server/](../inference_server)):
  - **Stage 1** — broad 13-class ensemble (**Model 1 / V1**: 15 backbones + student). **Model 2 / V2**
    dual-head classifiers are integrated as an **opt-in** addition (`DBT_ENABLE_V2=1`): the light CNN set
    (DenseNetHybrid, InceptionV4, EfficientNetV4, CSPDarkNet, GhostNetV2) folds its `cls_logits` into the
    ensemble vote ([v2_loader.py](../inference_server/engine/v2_loader.py)). Verified: ensemble grows
    from 16 → 21 voting models.
  - **Stage 2** — Red Rot specialist **3-backbone soft-voting ensemble** (ResNetMish + DenseNetHybrid +
    RegNet, Snapshot 3 retrain) in
    [redrot_segmentation.py](../inference_server/engine/redrot_segmentation.py): real weights, 21-class
    symptom mask with the **corrected** category mapping, % affected area, per-symptom confidence, RGBA
    overlay PNG.
  - **Route** `POST /predict/expert`: Stage 1 → Stage 2 when Red Rot (or `force_redrot`) → remedial.
  - **Remedial** placeholder + loader ([remedial.py](../inference_server/engine/remedial.py),
    [remedial_measures.json](../inference_server/data/remedial_measures.json)).
  - CORS for the web client; Red Rot ensemble + optional V2 loaded at startup.
- **Tests** ([tests/test_expert_pipeline.py](../tests/test_expert_pipeline.py)) — 7 unit + 2 end-to-end
  (gated on real weights via `--runslow`). **All 9 pass.**
- **CI/CD** — extended [.github/workflows/ci.yml](../.github/workflows/ci.yml) with `web-build`
  (npm build) and `android-build` (`assembleDebug`) jobs, both required for `ci-success`.

---

## 2. Validation evidence

| Check | Result |
|---|---|
| Red Rot label fix (GT cross-check) | Predicted dominant symptom matches expert annotations on common classes (white spot, outside_rind_discoloration) |
| Segmentation ensemble | 3-backbone soft vote; overlay aligns with visible lesions on annotated images |
| Backend `/predict/expert` on Red Rot image | `Red_rot` 0.90; specialist overlay + corrected symptom names returned |
| Backend branch logic (Healthy image) | Stage 2 correctly skipped with explanatory note |
| Model 2 (V2) integration | `DBT_ENABLE_V2=1` → ensemble 16 → 21 voting models (5 V2 added), all voting |
| Web Expert View (browser) | Redesigned UI; live analyze run rendered classification + overlay + remedial |
| Android Expert View (API-36 emulator) | Redesigned UI; analyzed annotated sample → overlay aligned with lesions |
| Q6 tests | 7 unit + 2 end-to-end — **all 9 pass** |
| Ruff lint (inference_server/, tests/) | clean |

---

## 3. Key technical decisions recorded

- **Red Rot deployment model:** on the Snapshot 3 retrain weights the best single backbone is
  **CustomResNetMish (22.8% mIoU)** (not InceptionV4, which only led on the seed snapshot). Now served as
  a 3-backbone soft-voting ensemble.
- **Hook-spec drift fixed:** retrain weights hook `layer2.0.conv1` (128 ch); the server sizes the decoder
  from the checkpoint itself (drift-proof), independent of the repo's V2 adapter.
- **Category mapping:** symptom channel→name labels come from `unified_dataset.json` `category_id` order,
  not the seed `category_mapping.json`. The correct map is bundled with the server.
- **V2 reparameterization:** MobileOne's V2 checkpoint is in multi-branch form that the fresh backbone
  doesn't match, so it is excluded from the default V2 set (fail-loud guard remains if re-added).

---

## 4. How to run

```powershell
# 1. Backend (loads V1 ensemble + Red Rot ensemble; set DBT_ENABLE_V2=1 to add Model 2)
cd F:\DBT-Base-DIr
$env:DBT_ENABLE_V2 = "1"   # optional: fold V2 classifiers into the ensemble
python -m uvicorn inference_server.app:app --host 127.0.0.1 --port 8011

# 2. Web Expert View
cd app\web-expert ; npm install ; npm run dev        # http://localhost:5173

# 3. Android Expert View (emulator reaches host via 10.0.2.2)
cd app\android-expert ; .\gradlew.bat assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk

# Tests
pytest tests/test_expert_pipeline.py -m "not slow"   # fast
pytest tests/test_expert_pipeline.py --runslow -m slow  # end-to-end
```

Red Rot weights are read from `DBT_REDROT_SEG_DIR` (default
`F:\Migration_RedRot_20260727\Paper_redrot\seg_checkpoints_retrain\`); the symptom category mapping is
bundled at `inference_server/data/redrot_category_mapping.json`.

---

## 5. Remaining / next increments

| Item | Notes |
|---|---|
| **Real remedial content** | Replace the PLACEHOLDER `remedial_measures.json` with agronomy-reviewed Q1 database content (no code change). |
| **Rare-symptom accuracy** | *Red Rot Root Infection* (id 12) is rare (~88 annotations) and under-segmented; needs class-rebalancing / more annotation to improve. |
| **Severity index** | Deferred for Q5/Q6 (raw % area + symptoms + confidence shown). Add agronomy-calibrated bands later. |
| **Persistent web tunnel (Q7/Q8)** | API base URL is already swappable in both clients; drop in the tunnel URL with no code change. |
| **On-device inference** | Server-based now; TFLite/CoreML export of the Red Rot backbone for later on-device use. |
| **V2 transformers / MobileOne** | Default V2 set is light CNNs; heavy transformers and reparameterized MobileOne can be added later (env-configurable). |

---

## 6. File map

| Area | Location |
|---|---|
| Two-stage route | [inference_server/routes/inference.py](../inference_server/routes/inference.py) |
| Red Rot specialist | [inference_server/engine/redrot_segmentation.py](../inference_server/engine/redrot_segmentation.py) |
| Remedial | [inference_server/engine/remedial.py](../inference_server/engine/remedial.py), [data/remedial_measures.json](../inference_server/data/remedial_measures.json) |
| Schemas / config | [inference_server/schemas.py](../inference_server/schemas.py), [inference_server/config.py](../inference_server/config.py) |
| Web app | [app/web-expert/](../app/web-expert) |
| Android app | [app/android-expert/](../app/android-expert) |
| Tests | [tests/test_expert_pipeline.py](../tests/test_expert_pipeline.py) |
| CI | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| Plan | [Reports/Q5_Q6_SCOPING_AND_PLAN_REDROT.md](Q5_Q6_SCOPING_AND_PLAN_REDROT.md) |
