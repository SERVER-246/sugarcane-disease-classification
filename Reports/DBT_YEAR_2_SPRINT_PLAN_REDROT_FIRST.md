# DBT Year 2 Sprint Plan: Red Rot First, Then Expansion to Other Diseases

## Executive Summary

This report sets the Year 2 work plan for the DBT-funded sugarcane disease classification project. The official milestone document for the three-year project identifies Q1-Q4 as the Year 1 foundation phase and Q5-Q8 as the Year 2 application and deployment phase. In line with the project’s maturity and the current research progress, the Year 2 strategy will focus first on Red Rot as the primary production use-case, then expand to other diseases once the deployment architecture is validated.

The Red Rot pathway is the most advanced and operationally ready domain in the current pipeline. It already has a strong research base, annotation work, and model-development progress. Therefore, Year 2 should prioritize:

- building the mobile expert-view application,
- integrating the disease classification models into the app,
- developing the farmer-view inference interface,
- validating the app under real field conditions,
- and finalizing the Red Rot deployment workflow before expanding the same architecture to other diseases.

This approach is realistic, milestone-aligned, and reduces technical risk while still meeting the DBT promise for Year 2 outcomes.

---

## 1. Strategic Direction

### Core Decision

- The first production application phase will be designed around Red Rot.
- After the Red Rot workflow is validated, the same mobile/edge/cloud architecture will be expanded to other diseases as model development continues.
- This strategy keeps the product focused, testable, and aligned with the project’s official quarter-wise milestones.

### Why Red Rot First

- Red Rot is already the most mature disease pathway in the research and annotation pipeline.
- The project has already invested in annotation workflows, segmentation work, and model development for this disease.
- A Red Rot-first deployment gives the team a working, credible application baseline before broadening to more disease classes.

---

## 2. Official Year 2 Milestone Mapping

The official DBT milestone file sets the project schedule as:

- Q1–Q4: Year 1 foundation and model development
- Q5–Q8: Year 2 application and deployment milestones

The official Year 2 promises are:

| Quarter | Official Milestone | Interpretation |
|---|---|---|
| Q5 | GUI development for Mobile application (Expert View) | Build the expert-mode mobile interface |
| Q6 | Integration of Sugarcane disease Classification Models 1 & 2 to the Mobile app (Expert View) & testing | Deploy and validate the first models in the app |
| Q7 | GUI development of Mobile application - Farmer view (Getting inference) | Create the farmer-facing inference interface |
| Q8 | Dataset cleaning, overlapping symptoms, preprocessing, and app integration/testing | Finalize Red Rot quality and complete integration/testing |

---

## 3. Year 2 Scope and Product Logic

The Year 2 plan should be structured around a phased product deployment model:

### Phase A: Red Rot as the first production disease
- Expert-view mobile application
- Red Rot inference workflow
- Farmer-view interface
- Red Rot model validations
- Deployment readiness checks

### Phase B: Expansion to other diseases
- Add other disease classes as their models mature
- Reuse the same mobile and backend architecture
- Extend the app without redesigning the platform

### Phase C: Full multi-disease deployment
- Broader disease coverage
- Cloud analytics, model versioning, and retraining pipeline
- Field-scale deployment across disease categories

This is the most practical implementation path and is consistent with the official milestone logic.

---

## 4. Year 2 Sprint Structure (Q5–Q8)

### Sprint 1: Q5 — Red Rot Expert-View Mobile App Foundation

Objective:
- Build the initial expert-view mobile application for Red Rot diagnosis.

Key deliverables:
- Expert View interface layout and navigation
- Image capture and upload flow
- Result presentation screen
- Basic app state handling and logs
- Foundation for backend inference connection

Exit criteria:
- App launches and is stable on target devices
- User can open the Red Rot workflow
- Image input is accepted in the app
- App is ready for model integration

---

### Sprint 2: Q6 — Red Rot Model Integration and Testing

Objective:
- Integrate the first Red Rot classification and/or segmentation model into the app.

Key deliverables:
- Backend model inference connection
- Prediction pipeline integration
- App-level error handling
- Expert-view testing and real sample validation
- Feedback/diagnosis review mechanism

Exit criteria:
- Expert app can run inference on Red Rot samples
- Outputs are stable and interpretable
- Testing feedback is captured
- Model integration is validated before moving to farmer-view work

---

### Sprint 3: Q7 — Farmer View and Data Quality Refinement

Objective:
- Build the farmer-facing interface and improve decision-quality for real field use.

Key deliverables:
- Farmer View interface for quick image-based inference
- Simplified output for non-expert users
- Dataset cleaning and preprocessing refinement
- Overlapping symptom handling work
- Deployment-quality input preparation workflow

Exit criteria:
- Farmer-view app is usable for field-level diagnosis
- Red Rot data pipeline quality is improved for real-world images
- The system is ready for final validation and handoff

---

### Sprint 4: Q8 — Final Red Rot Deployment Readiness and Integration

Objective:
- Finalize and validate the full Red Rot application workflow.

Key deliverables:
- Full end-to-end Red Rot app validation
- Integration testing across app, backend, and model layers
- Deployment package for Red Rot-first phase
- Documentation of app usage and operational flow
- Roadmap for expansion to other diseases

Exit criteria:
- Red Rot application works end-to-end
- Performance and user experience are validated
- System is ready for wider deployment and future disease expansion

---

## 5. Q5–Q8 Deliverables Table

| Quarter | Deliverable | Outcome | Priority |
|---|---|---|---|
| Q5 | Mobile application GUI — Expert View | Application shell and diagnostic workflow for experts | High |
| Q6 | Red Rot model integration into mobile app | App performs disease inference using trained model(s) | High |
| Q7 | Farmer View mobile app | Simpler field-use interface for non-expert decisions | High |
| Q8 | Dataset refinement + overlapping symptom handling + final app integration | Deployment-quality Red Rot pipeline and final validation | High |

---

## 6. Recommended Report Summary

> The Year 2 project plan will focus on translating the validated Red Rot research pipeline into a deployable mobile application. The first deployment will target Red Rot as the primary use-case, with expert-view and farmer-view interfaces, model integration, dataset refinement, and field validation. Once the Red Rot application is validated, the same architecture will be expanded to additional diseases. This phased approach is aligned with the official DBT quarterly milestones and is the most realistic and low-risk path for the project.

---

## 7. Immediate Next Steps

1. Define the expert-view app screens and interaction flow.
2. Finalize backend architecture and inference connection.
3. Confirm the Red Rot model set to be deployed first.
4. Lock Q5-Q8 sprint goals and ownership.
5. Begin implementation with the Red Rot expert-view application.

---

## 8. Final Position

The correct Year 2 plan is not to attempt all diseases at once. The correct plan is to build a working Red Rot-first application, validate it, and then extend the same platform to other diseases as the rest of the disease pipelines mature. This is the cleanest way to meet the DBT milestone promises while keeping the project technically manageable and deployment-focused.
