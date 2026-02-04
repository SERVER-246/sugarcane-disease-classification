# DISEASE CLASSIFICATION PIPELINE — 5-SPRINT PRODUCTION PLAN

**Created:** 2026-02-04  
**Authority:** User-approved master directive  
**Status:** 📋 **AWAITING USER APPROVAL** — No code changes until explicitly approved  
**Reference Model:** PEST Detection Pipeline (Production-grade, CI/CD active, 96.25% accuracy)

---

## MASTER DIRECTIVE COMPLIANCE SUMMARY

| Precondition | Status |
|--------------|--------|
| No code modifications without explicit instruction | ✅ Enforced |
| No directory restructuring without explicit instruction | ✅ Enforced |
| No silenced warnings/errors/failures | ✅ Enforced |
| Every issue: Identified → Logged → Fixed → Verified 3x | ✅ Enforced |
| No assumptions — ask user if unclear | ✅ Enforced |
| Backward compatibility non-negotiable | ✅ Enforced |

---

## USER-PROVIDED CONFIGURATION

| Parameter | Value |
|-----------|-------|
| **Primary Python Version** | 3.10 (explicit) |
| **CI/CD Python Matrix** | 3.9, 3.10, 3.11, 3.12 |
| **Target OS** | Windows + Linux (both) |
| **Primary Inference Target** | NVIDIA GPU (CUDA 11.8 / 12.x) |
| **Fallback Inference Target** | CPU (fully tested) |
| **CI Platform** | GitHub Actions |
| **Dependency Policy** | Multi-format always; no single-dependency reliance; version whitelist/blacklist maintained |
| **Export Formats Required** | ONNX + TorchScript + smoke tests (minimum); TensorRT, CoreML, OpenVINO as secondary |
| **Data Privacy** | Images treated as sensitive; S3 with encryption at rest; user consent on feedback API |
| **Docker** | Required; CPU + GPU images; multi-stage builds; publish to registry |
| **Deployment Model** | PEST-equivalent: FastAPI server, multi-repo git, auto-retraining, Phase 3 features |

---

## TABLE OF CONTENTS

1. [Sprint 1 — Repository Integrity & Safety Baseline](#sprint-1--repository-integrity--safety-baseline)
2. [Sprint 2 — CI/CD Without Behavior Change](#sprint-2--cicd-without-behavior-change)
3. [Sprint 3A — Inference Server Foundation](#sprint-3a--inference-server-foundation)
4. [Sprint 3B — Inference Server Hardening & Testing](#sprint-3b--inference-server-hardening--testing)
5. [Sprint 4 — Deployment Discipline & Model Governance](#sprint-4--deployment-discipline--model-governance)
6. [Sprint 5 — Continuous Validation & Production Safeguards](#sprint-5--continuous-validation--production-safeguards)
6. [Appendix A — Dependency Whitelist/Blacklist](#appendix-a--dependency-whitelistblacklist)
7. [Appendix B — Export Format Matrix](#appendix-b--export-format-matrix)
8. [Appendix C — Blocker Reporting Template](#appendix-c--blocker-reporting-template)

---

## SPRINT 1 — REPOSITORY INTEGRITY & SAFETY BASELINE

### Objective

Make the codebase safe to modify without breaking anything. Establish a verified baseline where all scripts run, all imports resolve, and all static analysis passes.

### Prerequisites

- [ ] User confirms Python 3.10 environment is active
- [ ] User confirms `F:\DBT-Base-DIr` is the correct workspace root
- [ ] User confirms git repository is initialized and clean

### Exact Files Touched

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | CREATE | Unified tooling config (ruff, pytest, mypy, coverage) |
| `pyrightconfig.json` | CREATE | Type checking configuration |
| `.pre-commit-config.yaml` | CREATE | Pre-commit hooks for lint/format |
| `requirements-dev.txt` | CREATE | Development dependencies (ruff, mypy, pytest, pre-commit) |
| `DEPENDENCY_MANIFEST.md` | CREATE | Version whitelist/blacklist documentation |

### Exact Commands to Run

#### 1.1 Environment Verification

```powershell
# Verify Python version
python --version
# Expected: Python 3.10.x

# Verify pip is available
python -m pip --version

# Verify workspace path
Get-Location
# Expected: F:\DBT-Base-DIr
```

#### 1.2 Install Development Dependencies

```powershell
# Install dev dependencies
python -m pip install ruff mypy pytest pytest-cov pre-commit pyright --upgrade

# Verify installations
ruff --version
mypy --version
pytest --version
pyright --version
```

#### 1.3 Static Analysis — Ruff (Linting)

```powershell
# Run ruff check (no auto-fix yet — report only)
ruff check . --output-format=full > sprint1_ruff_report.txt 2>&1

# Count issues
Select-String -Path sprint1_ruff_report.txt -Pattern "error|warning" | Measure-Object
```

**Exit Criteria for 1.3:**
- Report generated
- All errors documented
- No errors silenced or ignored

#### 1.4 Static Analysis — Compile Check

```powershell
# Verify all Python files compile
python -m compileall . -q

# Expected: No output (all files compile successfully)
# If errors: Document exact file and line
```

**Exit Criteria for 1.4:**
- Zero compilation errors
- All `.py` files parse correctly

#### 1.5 Static Analysis — Import Graph Validation

```powershell
# Test all imports resolve
python -c "import sys; sys.path.insert(0, '.'); import Base_backbones; print('✅ Base_backbones imports successfully')"

# Test other key modules
python -c "import run_pipeline; print('✅ run_pipeline imports successfully')"
python -c "import disease_classifier_gui; print('✅ disease_classifier_gui imports successfully')"
python -c "import image_validator; print('✅ image_validator imports successfully')"
python -c "import reproduce_pipeline; print('✅ reproduce_pipeline imports successfully')"
python -c "import setup_verify; print('✅ setup_verify imports successfully')"
python -c "import test_dependencies; print('✅ test_dependencies imports successfully')"
```

**Exit Criteria for 1.5:**
- All imports succeed
- No `ModuleNotFoundError` or `ImportError`

#### 1.6 Static Analysis — Type Checking (Pyright)

```powershell
# Run pyright in report mode
pyright . --outputjson > sprint1_pyright_report.json 2>&1

# Summary
pyright . 2>&1 | Select-String -Pattern "error|warning" | Measure-Object
```

**Exit Criteria for 1.6:**
- Report generated
- All type errors documented (not silenced)

#### 1.7 Dependency Integrity Check

```powershell
# Freeze current environment
python -m pip freeze > sprint1_pip_freeze.txt

# Check for outdated packages
python -m pip list --outdated > sprint1_outdated_packages.txt

# Verify requirements.txt installs cleanly
python -m pip install -r requirements.txt --dry-run
```

**Exit Criteria for 1.7:**
- `requirements.txt` installs without conflicts
- All dependencies documented with versions

#### 1.8 Dead Code Detection (Report Only — NO DELETION)

```powershell
# Install vulture for dead code detection
python -m pip install vulture

# Run vulture (report only)
vulture . --min-confidence 80 > sprint1_dead_code_report.txt 2>&1

# Review report
Get-Content sprint1_dead_code_report.txt | Measure-Object -Line
```

**Exit Criteria for 1.8:**
- Dead code report generated
- NO code deleted in this sprint
- Report preserved for Sprint 2+ review

#### 1.9 Script Execution Verification

```powershell
# Test that key scripts run without immediate crash
# Using --help or minimal invocation

python setup_verify.py 2>&1 | Tee-Object -FilePath sprint1_setup_verify_output.txt
python test_dependencies.py 2>&1 | Tee-Object -FilePath sprint1_test_deps_output.txt

# Debug mode smoke test (model creation only)
$env:DBT_DEBUG_MODE = "true"
$env:DBT_DEBUG_FUNCTION = "model_creation"
$env:DBT_DEBUG_BACKBONE = "CustomConvNeXt"
python Base_backbones.py 2>&1 | Tee-Object -FilePath sprint1_backbone_smoke_test.txt
Remove-Item Env:\DBT_DEBUG_MODE
Remove-Item Env:\DBT_DEBUG_FUNCTION
Remove-Item Env:\DBT_DEBUG_BACKBONE
```

**Exit Criteria for 1.9:**
- All scripts execute without immediate crash
- Output captured for review

### Validation Steps

| Check | Command | Expected Result |
|-------|---------|-----------------|
| Ruff passes | `ruff check .` | 0 errors (after fixes) |
| Compile passes | `python -m compileall . -q` | No output (success) |
| Imports resolve | All import commands above | All print success messages |
| Pyright documented | `pyright .` | Errors documented, not silenced |
| Scripts run | Execution commands above | No immediate crashes |

### Failure Conditions

If ANY of the following occur, **STOP and report using Appendix C template**:

1. Python version is not 3.10.x
2. Any required package fails to install
3. Any `.py` file fails to compile
4. Any critical import fails (Base_backbones, run_pipeline)
5. Environment variable `DBT_BASE_DIR` is missing and required
6. Dataset path `F:\DBT-Base-DIr\Data\` does not exist or is empty

### Rollback Strategy

Sprint 1 creates only new files and reports. Rollback:

```powershell
# Remove created files (if needed)
Remove-Item -Path "pyproject.toml" -ErrorAction SilentlyContinue
Remove-Item -Path "pyrightconfig.json" -ErrorAction SilentlyContinue
Remove-Item -Path ".pre-commit-config.yaml" -ErrorAction SilentlyContinue
Remove-Item -Path "requirements-dev.txt" -ErrorAction SilentlyContinue
Remove-Item -Path "DEPENDENCY_MANIFEST.md" -ErrorAction SilentlyContinue
Remove-Item -Path "sprint1_*.txt" -ErrorAction SilentlyContinue
Remove-Item -Path "sprint1_*.json" -ErrorAction SilentlyContinue
```

### Exit Criteria

- [ ] All static analysis reports generated
- [ ] All compilation errors fixed (not silenced)
- [ ] All import errors fixed (not silenced)
- [ ] All scripts verified to run locally
- [ ] `pyproject.toml` created with ruff/pytest/mypy config
- [ ] `pyrightconfig.json` created
- [ ] `requirements-dev.txt` created
- [ ] `DEPENDENCY_MANIFEST.md` created with whitelist/blacklist
- [ ] **User approval obtained to proceed to Sprint 2**

---

## SPRINT 2 — CI/CD WITHOUT BEHAVIOR CHANGE

### Objective

Automate validation via GitHub Actions without changing any runtime behavior. CI must fail on any warning, error, or test failure.

### Prerequisites

- [ ] Sprint 1 completed and approved
- [ ] GitHub repository initialized and accessible
- [ ] GitHub Actions enabled for repository

### Exact Files Touched

| File | Action | Purpose |
|------|--------|---------|
| `.github/workflows/ci.yml` | CREATE | Main CI pipeline (lint, typecheck, test) |
| `.github/workflows/test-matrix.yml` | CREATE | Multi-Python version test matrix |
| `.github/workflows/docker-build.yml` | CREATE | Docker image build verification |
| `tests/conftest.py` | CREATE | Pytest fixtures and configuration |
| `tests/test_imports.py` | CREATE | Import validation tests |
| `tests/test_model_creation.py` | CREATE | Model instantiation tests |
| `tests/test_forward_pass.py` | CREATE | Forward pass smoke tests |
| `tests/test_export_formats.py` | CREATE | Export format validation |
| `Dockerfile.cpu` | CREATE | CPU inference Docker image |
| `Dockerfile.gpu` | CREATE | GPU inference Docker image |
| `docker-compose.yml` | CREATE | Multi-container orchestration |
| `.dockerignore` | CREATE | Docker build exclusions |

### Exact Commands to Run

#### 2.1 Create GitHub Actions Directory

```powershell
# Create .github/workflows directory
New-Item -ItemType Directory -Path ".github/workflows" -Force
```

#### 2.2 Create Test Directory Structure

```powershell
# Create tests directory
New-Item -ItemType Directory -Path "tests" -Force
```

#### 2.3 Verify CI Configuration Locally

```powershell
# Install act (local GitHub Actions runner) — optional but recommended
# https://github.com/nektos/act

# Run ruff (same as CI will run)
ruff check . --exit-non-zero-on-fix

# Run pytest with collection only (verify tests discovered)
pytest --collect-only

# Run pytest with coverage
pytest --cov=. --cov-report=term-missing --cov-report=xml
```

#### 2.4 Docker Build Verification

```powershell
# Build CPU image
docker build -f Dockerfile.cpu -t disease-classifier:cpu-latest .

# Build GPU image
docker build -f Dockerfile.gpu -t disease-classifier:gpu-latest .

# Verify images created
docker images | Select-String "disease-classifier"
```

#### 2.5 Fresh Clone Test

```powershell
# Clone to temporary directory and verify CI passes
$tempDir = "C:\Temp\disease-classifier-test-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
git clone . $tempDir
Push-Location $tempDir
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
ruff check .
pytest --collect-only
Pop-Location
Remove-Item -Recurse -Force $tempDir
```

### CI Pipeline Specification

#### `.github/workflows/ci.yml` Structure

```yaml
name: CI Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install ruff
      - run: ruff check . --exit-non-zero-on-fix
      - run: ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install pyright
      - run: pip install -r requirements.txt
      - run: pyright . --warnings

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v4
        if: matrix.os == 'ubuntu-latest'

  build-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python -c "from Base_backbones import create_custom_backbone; print('✅ Import successful')"
```

#### `.github/workflows/test-matrix.yml` Structure

```yaml
name: Test Matrix (Nightly)
on:
  schedule:
    - cron: "0 2 * * *"  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  test-python-versions:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest -v --tb=short
```

### Validation Steps

| Check | Command | Expected Result |
|-------|---------|-----------------|
| CI config valid | `act --list` (if installed) | Jobs listed without errors |
| Lint passes | `ruff check .` | Exit code 0 |
| Tests discovered | `pytest --collect-only` | Tests listed |
| Docker CPU builds | `docker build -f Dockerfile.cpu .` | Build succeeds |
| Docker GPU builds | `docker build -f Dockerfile.gpu .` | Build succeeds |
| Fresh clone works | Clone test above | All commands pass |

### Failure Conditions

If ANY of the following occur, **STOP and report using Appendix C template**:

1. GitHub Actions syntax errors
2. Docker build fails
3. Any test fails on fresh clone
4. CI passes locally but fails on GitHub (environment mismatch)
5. Python version matrix fails for any required version (3.9-3.12)

### Rollback Strategy

```powershell
# Remove CI files
Remove-Item -Recurse -Force ".github" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "tests" -ErrorAction SilentlyContinue
Remove-Item -Path "Dockerfile.cpu" -ErrorAction SilentlyContinue
Remove-Item -Path "Dockerfile.gpu" -ErrorAction SilentlyContinue
Remove-Item -Path "docker-compose.yml" -ErrorAction SilentlyContinue
Remove-Item -Path ".dockerignore" -ErrorAction SilentlyContinue
```

### Exit Criteria

- [ ] `.github/workflows/ci.yml` created and validated
- [ ] `.github/workflows/test-matrix.yml` created and validated
- [ ] `.github/workflows/docker-build.yml` created and validated
- [ ] Test suite created with minimum coverage
- [ ] Docker images build successfully (CPU and GPU)
- [ ] CI fails on any warning, error, or test failure (verified)
- [ ] Fresh clone → CI passes (verified)
- [ ] CI failure reproduces locally (verified)
- [ ] **User approval obtained to proceed to Sprint 3A**

---

## SPRINT 3A — INFERENCE SERVER FOUNDATION

### Objective

Create the foundational inference server structure with basic endpoints, configuration, and model loading. This sprint focuses on scaffolding only — no hardening, no load testing.

### Non-Goals (Sprint 3A)

- ❌ Input validation middleware (deferred to Sprint 3B)
- ❌ Structured logging middleware (deferred to Sprint 3B)
- ❌ Load testing (deferred to Sprint 3B)
- ❌ Determinism verification (deferred to Sprint 3B)
- ❌ Any changes to existing model code

### Prerequisites

- [ ] Sprint 2 completed and approved
- [ ] CI pipeline passing
- [ ] Docker images building successfully

### Exact Files Touched

| File | Action | Purpose |
|------|--------|---------|
| `inference_server/__init__.py` | CREATE | Server package initialization |
| `inference_server/app.py` | CREATE | FastAPI application entry point |
| `inference_server/config.py` | CREATE | Server configuration management |
| `inference_server/schemas.py` | CREATE | Pydantic request/response schemas |
| `inference_server/routes/__init__.py` | CREATE | Routes package initialization |
| `inference_server/routes/health.py` | CREATE | Health check endpoints |
| `inference_server/routes/inference.py` | CREATE | Basic inference endpoint (no validation yet) |
| `inference_server/engine/__init__.py` | CREATE | Engine package initialization |
| `inference_server/engine/loader.py` | CREATE | Model loading utility |
| `inference_server/engine/predictor.py` | CREATE | Basic prediction wrapper |

### Exact Commands to Run

#### 3A.1 Create Server Directory Structure

```powershell
New-Item -ItemType Directory -Path "inference_server" -Force
New-Item -ItemType Directory -Path "inference_server/routes" -Force
New-Item -ItemType Directory -Path "inference_server/engine" -Force
```

#### 3A.2 Install Server Dependencies

```powershell
python -m pip install fastapi uvicorn pydantic python-multipart --upgrade
```

#### 3A.3 Start Server Locally

```powershell
# Start server in development mode
uvicorn inference_server.app:app --host 0.0.0.0 --port 8000 --reload
```

#### 3A.4 Health Check Verification

```powershell
# Test health endpoints
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
Invoke-RestMethod -Uri "http://localhost:8000/health/ready" -Method GET
Invoke-RestMethod -Uri "http://localhost:8000/health/live" -Method GET
```

#### 3A.5 Basic Inference Test

```powershell
# Test basic inference endpoint (no validation in Sprint 3A)
# Use any existing test image from the dataset
$testImage = Get-ChildItem "Data\Healthy\*.jpg" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($testImage) {
    Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Form @{image = Get-Item $testImage.FullName}
} else {
    Write-Host "⚠️ No test image found in Data\Healthy\ - create one for testing"
}
```

### Server Specification (Sprint 3A — Foundation Only)

#### Health Check Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /health` | Basic health | `{"status": "healthy"}` |
| `GET /health/ready` | Readiness (model loaded) | `{"ready": true, "model": "loaded"}` |
| `GET /health/live` | Liveness (process running) | `{"live": true}` |

#### Inference Endpoint (Basic — No Validation)

| Endpoint | Purpose | Request | Response |
|----------|---------|---------|----------|
| `POST /predict` | Single image classification | Multipart form with image | `{"class": "...", "confidence": 0.95, "all_probabilities": {...}}` |

### Failure Conditions (Sprint 3A)

If ANY of the following occur, **STOP and report using Appendix C template**:

1. Server fails to start
2. Health endpoints return errors
3. Model fails to load
4. Basic inference endpoint does not return a prediction
5. Any existing code is modified

### Validation Steps (Sprint 3A)

| Check | Command | Expected Result |
|-------|---------|-----------------|
| Server starts | `uvicorn inference_server.app:app` | No errors |
| Health endpoint | `curl localhost:8000/health` | 200 OK |
| Ready endpoint | `curl localhost:8000/health/ready` | 200 OK with model loaded |
| Live endpoint | `curl localhost:8000/health/live` | 200 OK |
| Basic inference | `POST /predict` with valid image | Prediction returned |

### Rollback Strategy (Sprint 3A)

```powershell
# Remove server files (does not affect existing code)
Remove-Item -Recurse -Force "inference_server" -ErrorAction SilentlyContinue
```

### Exit Criteria

- [ ] FastAPI server created and starts without errors
- [ ] Health check endpoints return expected responses
- [ ] Basic inference endpoint accepts image and returns prediction
- [ ] Model loads successfully on server startup
- [ ] Server runs in Docker container (CPU and GPU)
- [ ] No existing code modified
- [ ] **User approval obtained to proceed to Sprint 3B**

---

## SPRINT 3B — INFERENCE SERVER HARDENING & TESTING

### Objective

Harden the inference server created in Sprint 3A with input validation, structured logging, determinism verification, and load testing. No changes to inference logic.

### Non-Goals (Sprint 3B)

- ❌ Changes to model architecture
- ❌ Changes to prediction logic
- ❌ New model training or fine-tuning
- ❌ Model export modifications

### Prerequisites

- [ ] Sprint 3A completed and approved
- [ ] Inference server running and responding
- [ ] Basic inference endpoint functional

### Exact Files Touched

| File | Action | Purpose |
|------|--------|--------|
| `inference_server/middleware/__init__.py` | CREATE | Middleware package initialization |
| `inference_server/middleware/logging.py` | CREATE | Structured logging middleware |
| `inference_server/middleware/validation.py` | CREATE | Input validation middleware |
| `inference_server/middleware/error_handler.py` | CREATE | Centralized error handling |
| `tests/test_health_endpoints.py` | CREATE | Health check tests |
| `tests/test_inference_endpoints.py` | CREATE | Inference endpoint tests |
| `tests/test_input_validation.py` | CREATE | Input validation tests |
| `tests/test_determinism.py` | CREATE | Deterministic output tests |
| `tests/locustfile.py` | CREATE | Load test configuration |

### Exact Commands to Run

#### 3B.1 Create Middleware Directory

```powershell
New-Item -ItemType Directory -Path "inference_server/middleware" -Force
```

#### 3B.2 Install Additional Dependencies

```powershell
python -m pip install structlog locust --upgrade
```

#### 3B.3 Run Validation Tests

```powershell
# Run all server tests
pytest tests/test_health_endpoints.py tests/test_inference_endpoints.py tests/test_input_validation.py -v
```

#### 3B.4 Determinism Verification

```powershell
# Run determinism test
python -c @"
import torch
import hashlib
torch.manual_seed(42)

# Load model
from inference_server.engine.loader import load_model
model = load_model()

# Create deterministic input
dummy_input = torch.randn(1, 3, 224, 224)

# Run 3 times and compare
results = []
for i in range(3):
    with torch.no_grad():
        output = model(dummy_input)
    results.append(hashlib.md5(output.numpy().tobytes()).hexdigest())

assert len(set(results)) == 1, f'Non-deterministic outputs: {results}'
print('✅ Determinism verified: all 3 runs produced identical output')
"@
```

#### 3B.5 Load Test

```powershell
# Run load test (10 users, 60 seconds)
locust -f tests/locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000
```

### Input Validation Rules

| Field | Validation | Error Response |
|-------|------------|----------------|
| Image file | Required, max 10MB | 400: "Image required" / "Image too large" |
| Image format | JPEG, PNG, WebP | 400: "Unsupported format" |
| Image dimensions | Min 32x32, Max 4096x4096 | 400: "Invalid dimensions" |

### Structured Logging Format

```json
{
  "timestamp": "2026-02-04T10:30:00Z",
  "level": "INFO",
  "event": "inference_request",
  "request_id": "uuid",
  "duration_ms": 45,
  "model_version": "v1.0.0",
  "input_size": [224, 224],
  "prediction": "Healthy",
  "confidence": 0.95
}
```

### Load Test Thresholds

| Metric | Threshold | Action if Exceeded |
|--------|-----------|--------------------|
| p50 latency | < 100ms | Log warning |
| p99 latency | < 500ms | Fail test |
| Error rate | < 1% | Fail test |
| Throughput | > 10 req/s (CPU), > 50 req/s (GPU) | Log warning |

### Validation Steps

| Check | Command | Expected Result |
|-------|---------|------------------|
| Validation rejects bad input | Send invalid image | 400 Bad Request |
| Validation accepts good input | Send valid image | 200 OK |
| Logging works | Check server logs | Structured JSON logs |
| Deterministic output | Determinism test above | All runs identical |
| Load test passes | Locust test | Under thresholds |

### Failure Conditions

If ANY of the following occur, **STOP and report using Appendix C template**:

1. Validation logic changes inference behavior
2. Determinism test fails
3. Load test shows > 1% error rate
4. Load test p99 latency > 500ms
5. Any existing inference logic is modified
6. Logging adds > 10ms overhead per request

### Rollback Strategy

```powershell
# Remove hardening additions (keeps Sprint 3A foundation)
Remove-Item -Recurse -Force "inference_server/middleware" -ErrorAction SilentlyContinue
Remove-Item -Path "tests/test_health_endpoints.py" -ErrorAction SilentlyContinue
Remove-Item -Path "tests/test_inference_endpoints.py" -ErrorAction SilentlyContinue
Remove-Item -Path "tests/test_input_validation.py" -ErrorAction SilentlyContinue
Remove-Item -Path "tests/test_determinism.py" -ErrorAction SilentlyContinue
Remove-Item -Path "tests/locustfile.py" -ErrorAction SilentlyContinue
```

### Exit Criteria

- [ ] Input validation middleware implemented and tested
- [ ] Structured logging middleware implemented
- [ ] Deterministic outputs verified (3 identical runs)
- [ ] Load test passes all thresholds
- [ ] No inference logic changed
- [ ] All tests pass in CI
- [ ] **User approval obtained to proceed to Sprint 4**

---

## SPRINT 4 — DEPLOYMENT DISCIPLINE & MODEL GOVERNANCE

### Objective

Control what is deployed and why. Implement model versioning, integrity checks, and rollback mechanisms.

### Prerequisites

- [ ] Sprint 3B completed and approved
- [ ] Inference server functional with validation and logging
- [ ] All tests passing

### ⛔ NON-GOALS (STRICTLY FORBIDDEN)

The following actions are **PROHIBITED** in Sprint 4 unless explicitly approved by the user in writing:

| Action | Status | Enforcement |
|--------|--------|-------------|
| Modify model weights | ❌ FORBIDDEN | No `.pth` files may be altered |
| Retrain any model | ❌ FORBIDDEN | No training scripts may be executed |
| Re-export models | ❌ FORBIDDEN | No new ONNX/TorchScript/TensorRT generation |
| Quantize models | ❌ FORBIDDEN | No precision reduction (FP16, INT8, etc.) |
| Automated rollback | ❌ FORBIDDEN | User approval required for any rollback |
| Automated restart | ❌ FORBIDDEN | User approval required for any restart |
| Automated scaling | ❌ FORBIDDEN | User approval required for scaling actions |
| Silent mitigation | ❌ FORBIDDEN | All remediation requires user approval |

**Exception Process:** Any deviation from the above requires:
1. Written user approval in this chat
2. Documented justification with risk assessment
3. Explicit rollback plan before execution
4. Verification that rollback plan works

### 🛡️ STRICT RISK CONTAINMENT

#### Blast Radius Limits

| Risk Category | Containment Measure |
|---------------|---------------------|
| Model corruption | All operations are **READ-ONLY** on model files |
| Manifest corruption | Backup created **BEFORE** any manifest write |
| Rollback failure | Manifest snapshots retained (last 5 versions) |
| Hash collision | SHA-256 used (cryptographically collision-resistant) |
| Registry inconsistency | Atomic writes with temp file + rename pattern |
| Concurrent access | File locking with timeout and retry |

#### Immutable Artifacts (READ-ONLY)

The following files are **IMMUTABLE** during Sprint 4 — no writes, no deletes:

```
checkpoints/*.pth              # All training checkpoints
checkpoints/**/*.pth           # Nested checkpoints (ensembles)
deployment_models/*.onnx       # All ONNX exports
deployment_models/*.pt         # All TorchScript exports  
deployment_models/*.engine     # All TensorRT engines
Base_backbones.py              # Core training script
run_pipeline.py                # Pipeline orchestrator
```

#### Mandatory Backup Before Manifest Changes

```powershell
# REQUIRED before ANY manifest modification
$manifestPath = "MODEL_MANIFEST.json"
if (Test-Path $manifestPath) {
    $backupPath = "MODEL_MANIFEST.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    Copy-Item $manifestPath $backupPath -ErrorAction Stop
    Write-Host "✅ Backup created: $backupPath"
} else {
    Write-Host "ℹ️ No existing manifest to backup (first creation)"
}
```

#### Prohibited Operations Checklist

Before executing ANY Sprint 4 script, verify the script does NOT contain:

- [ ] `torch.save()` — Would modify model files
- [ ] `model.train()` — Would enable training mode
- [ ] `optimizer.step()` — Would update weights
- [ ] `torch.onnx.export()` — Would create new exports
- [ ] `torch.jit.save()` — Would create new TorchScript
- [ ] `shutil.copy()` to `checkpoints/` — Would modify checkpoint dir
- [ ] `os.remove()` on model files — Would delete models
- [ ] Any write operation to `checkpoints/` directory
- [ ] Any write operation to `deployment_models/` directory

#### Verification Commands

```powershell
# Verify no prohibited operations in Sprint 4 scripts
$prohibitedPatterns = @(
    'torch\.save',
    'model\.train\(',
    'optimizer\.step',
    'torch\.onnx\.export',
    'torch\.jit\.save',
    'shutil\.copy.*checkpoints',
    'os\.remove.*\.pth',
    'os\.remove.*\.onnx'
)

Get-ChildItem -Path "deployment","scripts" -Filter "*.py" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    foreach ($pattern in $prohibitedPatterns) {
        if ($content -match $pattern) {
            Write-Host "❌ PROHIBITED: $($_.Name) contains '$pattern'" -ForegroundColor Red
            exit 1
        }
    }
}
Write-Host "✅ Prohibited operations check complete — no violations found"
```

### Exact Files Touched

| File | Action | Purpose |
|------|--------|---------|
| `MODEL_MANIFEST.json` | CREATE | Central model registry (references only, no model data) |
| `deployment/model_registry.py` | CREATE | Model registration utilities (read-only operations) |
| `deployment/integrity.py` | CREATE | Hash-based integrity verification |
| `deployment/selector.py` | CREATE | Active model selection logic |
| `deployment/rollback.py` | CREATE | Rollback mechanism |
| `deployment/metadata_schema.py` | CREATE | Model metadata Pydantic schema |
| `scripts/register_model.py` | CREATE | CLI for model registration |
| `scripts/verify_model.py` | CREATE | CLI for integrity verification |
| `scripts/rollback_model.py` | CREATE | CLI for model rollback |
| `tests/test_model_registry.py` | CREATE | Registry tests |
| `tests/test_integrity.py` | CREATE | Integrity verification tests |
| `tests/test_rollback.py` | CREATE | Rollback mechanism tests |

### Model Manifest Specification

```json
{
  "manifest_version": "1.0.0",
  "active_model": "CustomConvNeXt_v1.0.0",
  "models": {
    "CustomConvNeXt_v1.0.0": {
      "name": "CustomConvNeXt",
      "version": "1.0.0",
      "checkpoint_path": "checkpoints/CustomConvNeXt_final.pth",
      "formats": {
        "pytorch": {
          "path": "checkpoints/CustomConvNeXt_final.pth",
          "sha256": "abc123...",
          "size_bytes": 12345678
        },
        "onnx": {
          "path": "deployment_models/CustomConvNeXt.onnx",
          "sha256": "def456...",
          "size_bytes": 11234567
        },
        "torchscript": {
          "path": "deployment_models/CustomConvNeXt.pt",
          "sha256": "ghi789...",
          "size_bytes": 12000000
        }
      },
      "metadata": {
        "training_data_ref": "Data/split_dataset_v1",
        "commit_hash": "abc123def",
        "training_date": "2026-02-01T10:00:00Z",
        "hyperparameters": {
          "epochs_head": 40,
          "epochs_finetune": 25,
          "batch_size": 32,
          "learning_rate_head": 0.001,
          "learning_rate_backbone": 0.000001
        },
        "metrics": {
          "validation_accuracy": 0.9625,
          "validation_loss": 0.1234,
          "test_accuracy": 0.9580
        },
        "num_classes": 10,
        "input_shape": [1, 3, 224, 224],
        "normalization": {
          "mean": [0.485, 0.456, 0.406],
          "std": [0.229, 0.224, 0.225]
        }
      },
      "status": "active",
      "registered_at": "2026-02-04T10:00:00Z",
      "registered_by": "user"
    }
  },
  "rollback_history": []
}
```

### Exact Commands to Run

#### 4.1 Create Deployment Directory

```powershell
New-Item -ItemType Directory -Path "deployment" -Force
New-Item -ItemType Directory -Path "scripts" -Force
```

#### 4.2 Generate Initial Manifest

```powershell
python scripts/register_model.py --checkpoint checkpoints/CustomConvNeXt_final.pth --name CustomConvNeXt --version 1.0.0
```

#### 4.3 Verify Model Integrity

```powershell
python scripts/verify_model.py --model CustomConvNeXt_v1.0.0

# Expected output:
# ✅ PyTorch checkpoint: VALID (sha256 matches)
# ✅ ONNX export: VALID (sha256 matches)
# ✅ TorchScript export: VALID (sha256 matches)
```

#### 4.4 Test Rollback Mechanism

```powershell
# Register a second model version
python scripts/register_model.py --checkpoint checkpoints/CustomConvNeXt_final.pth --name CustomConvNeXt --version 1.0.1

# Set as active
python scripts/register_model.py --set-active CustomConvNeXt_v1.0.1

# Rollback to previous
python scripts/rollback_model.py --to CustomConvNeXt_v1.0.0

# Verify rollback
python -c "import json; m = json.load(open('MODEL_MANIFEST.json')); print(f'Active: {m[\"active_model\"]}')"
# Expected: Active: CustomConvNeXt_v1.0.0
```

### Multi-Format Export Verification

For each model in the manifest, ALL formats must be verified:

| Format | Verification Command | Expected |
|--------|---------------------|----------|
| PyTorch | `python -c "import torch; torch.load('path.pth')"` | No errors |
| ONNX | `python -c "import onnx; onnx.checker.check_model('path.onnx')"` | No errors |
| TorchScript | `python -c "import torch; torch.jit.load('path.pt')"` | No errors |
| TensorRT | `trtexec --onnx=path.onnx --saveEngine=path.engine` | Engine created |

### Validation Steps

| Check | Command | Expected Result |
|-------|---------|-----------------|
| Manifest valid JSON | `python -c "import json; json.load(open('MODEL_MANIFEST.json'))"` | No errors |
| All hashes match | `python scripts/verify_model.py --all` | All VALID |
| Active model loads | `python scripts/verify_model.py --load-test` | Model loads and runs |
| Rollback works | Rollback test above | Previous version restored |
| All formats present | Check manifest | PyTorch + ONNX + TorchScript minimum |

### Failure Conditions

If ANY of the following occur, **STOP and report using Appendix C template**:

1. Model checkpoint missing or corrupted
2. Hash mismatch detected
3. Required export format missing
4. Rollback fails to restore previous state
5. Manifest schema validation fails
6. Any model format fails to load

### Rollback Strategy

```powershell
# Remove deployment files
Remove-Item -Recurse -Force "deployment" -ErrorAction SilentlyContinue
Remove-Item -Path "MODEL_MANIFEST.json" -ErrorAction SilentlyContinue
Remove-Item -Path "scripts/register_model.py" -ErrorAction SilentlyContinue
Remove-Item -Path "scripts/verify_model.py" -ErrorAction SilentlyContinue
Remove-Item -Path "scripts/rollback_model.py" -ErrorAction SilentlyContinue
```

### Exit Criteria

- [ ] `MODEL_MANIFEST.json` created with all models
- [ ] Hash-based integrity checks implemented
- [ ] All models have PyTorch + ONNX + TorchScript formats (minimum)
- [ ] Active model selector functional
- [ ] Rollback mechanism tested and working
- [ ] All metadata explicit (training data, commit, hyperparams, metrics)
- [ ] **User approval obtained to proceed to Sprint 5**

---

## SPRINT 5 — CONTINUOUS VALIDATION & PRODUCTION SAFEGUARDS

### Objective

Ensure system does not silently degrade. Implement runtime monitoring, error rate tracking, and drift detection. **NO auto-retraining yet** — detection and reporting only.

### Prerequisites

- [ ] Sprint 4 completed and approved
- [ ] Model manifest in place
- [ ] Inference server functional with logging

### Exact Files Touched

| File | Action | Purpose |
|------|--------|---------|
| `monitoring/__init__.py` | CREATE | Monitoring package |
| `monitoring/metrics.py` | CREATE | Prometheus metrics definitions |
| `monitoring/collector.py` | CREATE | Metrics collection logic |
| `monitoring/drift.py` | CREATE | Basic drift detection |
| `monitoring/alerts.py` | CREATE | Alerting thresholds and triggers |
| `monitoring/dashboard.py` | CREATE | Metrics dashboard configuration |
| `monitoring/config.yaml` | CREATE | Monitoring configuration |
| `tests/test_metrics.py` | CREATE | Metrics collection tests |
| `tests/test_drift_detection.py` | CREATE | Drift detection tests |
| `tests/test_alerting.py` | CREATE | Alert trigger tests |
| `docker-compose.monitoring.yml` | CREATE | Prometheus + Grafana stack |

### Metrics Specification

#### Inference Metrics (Prometheus)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `inference_requests_total` | Counter | `model`, `status` | Total inference requests |
| `inference_latency_seconds` | Histogram | `model` | Request latency distribution |
| `inference_errors_total` | Counter | `model`, `error_type` | Total errors by type |
| `model_confidence_score` | Histogram | `model`, `class` | Prediction confidence distribution |
| `input_image_size_bytes` | Histogram | | Input image sizes |

#### Drift Detection Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_distribution` | Histogram | Class prediction frequency |
| `confidence_mean` | Gauge | Rolling mean confidence |
| `confidence_stddev` | Gauge | Rolling confidence std deviation |
| `low_confidence_rate` | Gauge | % of predictions < 0.7 confidence |

#### Alert Thresholds

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `HighErrorRate` | error_rate > 1% over 5 min | WARNING | Log + notify |
| `CriticalErrorRate` | error_rate > 5% over 5 min | CRITICAL | Log + notify + page |
| `HighLatency` | p99 latency > 1000ms | WARNING | Log + notify |
| `LowConfidenceSpike` | low_confidence_rate > 20% | WARNING | Log + notify |
| `ClassDistributionDrift` | chi-square p < 0.01 | INFO | Log only |

### Exact Commands to Run

#### 5.1 Create Monitoring Directory

```powershell
New-Item -ItemType Directory -Path "monitoring" -Force
```

#### 5.2 Install Monitoring Dependencies

```powershell
python -m pip install prometheus-client prometheus-fastapi-instrumentator --upgrade
```

#### 5.3 Start Monitoring Stack

```powershell
# Start Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# Verify Prometheus is scraping
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/targets" -Method GET
```

#### 5.4 Verify Metrics Endpoint

```powershell
# Check metrics endpoint
Invoke-RestMethod -Uri "http://localhost:8000/metrics" -Method GET
```

#### 5.5 Simulate Alert Trigger

```powershell
# Generate test errors to trigger alert
python -c @"
import requests
for i in range(100):
    try:
        requests.post('http://localhost:8000/predict', files={'image': b'invalid'})
    except:
        pass
print('Test errors generated')
"@

# Check alert state
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/alerts" -Method GET
```

### Drift Detection Specification

```python
# Basic drift detection (no auto-retraining)
class DriftDetector:
    def __init__(self, baseline_distribution: dict, threshold: float = 0.01):
        self.baseline = baseline_distribution
        self.threshold = threshold
        self.window = []
        self.window_size = 1000
    
    def check(self, predictions: list) -> dict:
        # Chi-square test against baseline
        # Returns: {"drift_detected": bool, "p_value": float, "recommendation": str}
        pass
    
    def report(self) -> dict:
        # Generate drift report (NO auto-action)
        pass
```

### Validation Steps

| Check | Command | Expected Result |
|-------|---------|-----------------|
| Metrics endpoint | `curl localhost:8000/metrics` | Prometheus format metrics |
| Prometheus scraping | Check Prometheus targets | All targets UP |
| Grafana accessible | `curl localhost:3000` | Grafana login page |
| Alert fires on errors | Simulate errors | Alert in firing state |
| Drift detection runs | Check logs | Drift report generated |

### Failure Conditions

If ANY of the following occur, **STOP and report using Appendix C template**:

1. Prometheus fails to scrape metrics
2. Metrics endpoint returns errors
3. Alert configuration invalid
4. Drift detection produces false positives on stable data
5. Monitoring adds > 10ms latency to inference

### Rollback Strategy

```powershell
# Stop monitoring stack
docker-compose -f docker-compose.monitoring.yml down

# Remove monitoring files
Remove-Item -Recurse -Force "monitoring" -ErrorAction SilentlyContinue
Remove-Item -Path "docker-compose.monitoring.yml" -ErrorAction SilentlyContinue
```

### Exit Criteria

- [ ] Prometheus metrics exposed and scraped
- [ ] Grafana dashboard configured
- [ ] Error rate tracking functional
- [ ] Latency tracking functional
- [ ] Basic drift detection implemented (reporting only)
- [ ] Alert thresholds defined and tested
- [ ] No auto-retraining implemented (detection only)
- [ ] No silent model replacement
- [ ] All monitoring verified in Docker environment
- [ ] **Production deployment approved by user**

---

## APPENDIX A — DEPENDENCY WHITELIST/BLACKLIST

### Whitelist (Approved Dependencies)

| Package | Min Version | Max Version | Purpose | Notes |
|---------|-------------|-------------|---------|-------|
| torch | 2.0.0 | 2.5.x | Core ML framework | CUDA 11.8/12.x support required |
| torchvision | 0.15.0 | 0.20.x | Image transforms | Must match torch version |
| onnx | 1.14.0 | latest | Model export | Required for multi-format |
| onnxruntime | 1.15.0 | latest | ONNX inference | CPU fallback |
| onnxruntime-gpu | 1.15.0 | latest | ONNX inference | GPU primary |
| fastapi | 0.100.0 | latest | API server | Production server |
| uvicorn | 0.22.0 | latest | ASGI server | With uvloop on Linux |
| pydantic | 2.0.0 | latest | Validation | V2 only |
| prometheus-client | 0.17.0 | latest | Metrics | Monitoring |
| structlog | 23.0.0 | latest | Logging | Structured logs |
| pillow | 9.0.0 | latest | Image processing | Security updates |
| numpy | 1.24.0 | 1.26.x | Numerics | Compatibility |
| ruff | 0.1.0 | latest | Linting | Dev only |
| pytest | 7.0.0 | latest | Testing | Dev only |
| mypy | 1.0.0 | latest | Type checking | Dev only |

### Blacklist (Forbidden Dependencies)

| Package | Reason | Alternative |
|---------|--------|-------------|
| tensorflow (standalone) | Platform-unstable for exports | Use torch + onnx |
| tflite-runtime (exclusive) | Closed-source limitations | Use onnxruntime as fallback |
| opencv-python-headless | Version conflicts | Use pillow + torchvision |
| pickle (raw) | Security risk | Use torch.save with safetensors |
| requests (sync only) | Blocking in async context | Use httpx for async |

### Version Lock Policy

1. **Lock major.minor** for production dependencies
2. **Allow patch updates** via `>=x.y.0,<x.y+1.0`
3. **Pin exactly** for CUDA/cuDNN versions
4. **Document breaking changes** in CHANGELOG.md

---

## APPENDIX B — EXPORT FORMAT MATRIX

### Required Formats (All Models)

| Format | Extension | Purpose | Verification |
|--------|-----------|---------|--------------|
| PyTorch State Dict | `.pth` | Training/fine-tuning | `torch.load()` |
| TorchScript | `.pt` | Production inference | `torch.jit.load()` |
| ONNX | `.onnx` | Cross-platform | `onnx.checker.check_model()` |

### Optional Formats (When Available)

| Format | Extension | Purpose | Verification |
|--------|-----------|---------|--------------|
| TensorRT | `.engine` | NVIDIA GPU optimization | `trtexec --loadEngine` |
| OpenVINO | `.xml` + `.bin` | Intel optimization | `openvino.Core().read_model()` |
| CoreML | `.mlmodel` | Apple devices | `coremltools.models.MLModel()` |

### Smoke Test Requirements

Each exported model MUST pass:

1. **Load test**: Model loads without errors
2. **Shape test**: Input/output shapes match specification
3. **Dtype test**: Input/output dtypes correct
4. **Inference test**: Dummy input produces valid output
5. **Determinism test**: Same input → same output (3 runs)

---

## APPENDIX C — BLOCKER REPORTING TEMPLATE

When any blocker is encountered, report using this exact format:

```markdown
## BLOCKER REPORT

**Sprint:** [1-5]
**Task:** [Task number and name]
**Severity:** [CRITICAL / HIGH / MEDIUM]

### Description
[Clear description of the blocker]

### Files Involved
- `path/to/file1.py`
- `path/to/file2.py`

### Exact Error Message
```
[Paste exact error output]
```

### Exact Reason Execution Cannot Continue
[Explain why this blocks progress]

### What Is Required From User
1. [First action needed]
2. [Second action needed]

### Attempted Resolutions
- [What was tried and failed]

### Suggested Path Forward
[Recommendation if any]
```

---

## APPROVAL GATE

### Sprint Transition Requirements

| From | To | Requirements |
|------|----|--------------|
| Plan | Sprint 1 | User explicitly approves this plan |
| Sprint 1 | Sprint 2 | All Sprint 1 exit criteria met + user approval |
| Sprint 2 | Sprint 3A | All Sprint 2 exit criteria met + user approval |
| Sprint 3A | Sprint 3B | All Sprint 3A exit criteria met + user approval |
| Sprint 3B | Sprint 4 | All Sprint 3B exit criteria met + user approval |
| Sprint 4 | Sprint 5 | All Sprint 4 exit criteria met + user approval |
| Sprint 5 | Production | All Sprint 5 exit criteria met + user approval |

---

## DOCUMENT METADATA

| Field | Value |
|-------|-------|
| Created | 2026-02-04 |
| Author | Copilot (per user directive) |
| Status | AWAITING USER APPROVAL |
| Supersedes | DISEASE_PIPELINE_NEXT_STEPS_PLAN.md (deleted) |
| Reference | PROJECT_OVERSEER_REPORT_PEST.md |

---

**⚠️ NO CODE CHANGES ARE PERMITTED UNTIL USER EXPLICITLY APPROVES THIS PLAN ⚠️**
