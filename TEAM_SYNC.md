# SureInspect: Team Status & Collision Prevention Guide
> **Repository:** https://github.com/Gyantwik/sureinspect.git  
> **Project:** SureInspect (OpenCV AI Competition 2026)  
> **Last Updated:** October 17, 2026  

---

## 1. Live Team Status Board

| Role | Assigned Developer | Branch | Current Status |
| :--- | :--- | :--- | :--- |
| **Person 1 (OpenCV & ML Lead)** | `@Biswabasini90` | `feat/core-opencv` | **Ready to start** (scaffolding & data contracts ready) |
| **Person 2 (Backend & AWS Lead)** | `@Gyantwik` (Owner) | `feat/backend-contract` | **COMPLETED: Day 1 Contract & Mock Engine** (PR open) |
| **Person 3 (Frontend & Docs Lead)** | Assigned Contributor | `feat/frontend-scaffold` | **Ready to start** (OpenAPI spec & mock API ready) |

---

## 2. Completed Milestones (by Person 2 @Gyantwik)

### 2.1 Basic GitHub & Monorepo Scaffolding
* [x] **GitHub Repository & Main Branch:** Created at `https://github.com/Gyantwik/sureinspect.git` and live on `origin/main`.
* [x] **Complete Directory Setup:** All monorepo folders generated with tracking files:
  * `/core` & `/core/tests`: OpenCV perception, tools, and agent logic.
  * `/backend` & `/backend/tests`: FastAPI application, endpoints, and mock engine.
  * `/train`: Dataset preparation, split freezing, and ONNX export scripts.
  * `/eval`, `/eval/results`, & `/eval/tests`: Evaluation metrics, tables, and Wilson statistics.
  * `/config`: Dataset configuration (`categories.yaml` pre-configured for VisA `pcb1`).
  * `/contracts/fixtures`: Shared OpenAPI specifications and mock payloads.
  * `/data/templates` & `/data/samples`: Golden references and UI demo samples.
  * `/deploy`: Multi-stage Dockerfile and AWS EC2 Graviton deployment scripts.
  * `/docs` & `/docs/failure_cases`: Architecture diagrams, technical reports, and failure analysis (PDF research papers moved here).
  * `/.github/workflows` & `/.github/scripts`: Automated CI pipeline with boundary checking.
* [x] **Configuration & Dependencies:** Added `.gitignore`, `.gitattributes`, `.dockerignore`, `.env.example`, `requirements*.txt`.
* [x] **CI & Ownership Rules:** P1 (`@Biswabasini90`), P2 (`@Gyantwik`), and P3 mapped in `CODEOWNERS` and CI boundary checker.

### 2.2 Day 1 API Contract, Mock Engine & OpenAPI Specification
* [x] **API Schemas (`backend/schemas.py`):** Pydantic v2 models covering all decision enums (`PASS`, `FAIL`, `HUMAN`, `RETAKE`), 10 decision paths, quality gates, measurements, traces, and feedback.
* [x] **Mock Engine (`backend/mock_service.py`):** Generates realistic traces and overlays for `pcb1` covering all 4 decision outcomes.
* [x] **FastAPI App Factory (`backend/app_factory.py`):** Provides cryptographic bearer sessions (`POST /api/session`), mock inspection endpoints, sample listings, and feedback logging.
* [x] **Exported OpenAPI Contract (`contracts/openapi.json`):** 44.7 KB tracked specification ready for Person 3's TypeScript type generator.
* [x] **Test Verification (`backend/tests/test_api.py`):** 18/18 tests passing across `core/tests`, `backend/tests`, and `eval/tests`.

---

## 3. Strict File Ownership Matrix (Zero Collision Rules)

To ensure developers never collide or produce merge conflicts, **each person edits ONLY their assigned directories**:

| Directory / File | Allowed Owner | Description |
| :--- | :--- | :--- |
| `core/` (except quality_gate/degrade) | **Person 1** (`@Biswabasini90`) | OpenCV DNN perception, calibration, tools T1-T5, agent policy |
| `train/` | **Person 1** (`@Biswabasini90`) | VisA dataset download, 80/20 split freeze, ONNX export, memory bank |
| `config/` | **Person 1** (`@Biswabasini90`) | Dataset and category configuration (`categories.yaml`) |
| `data/templates/` | **Person 1** (`@Biswabasini90`) | Curated golden defect-free reference templates |
| `docs/failure_cases/` | **Person 1** (`@Biswabasini90`) | Failure mode images and explanations |
| `backend/` | **Person 2** (`@Gyantwik`) | FastAPI endpoints, Pydantic schemas, session auth, mock service |
| `eval/` | **Person 2** (`@Gyantwik`) | Evaluation protocols (E1-E4), Wilson statistics, result tables |
| `deploy/` | **Person 2** (`@Gyantwik`) | Dockerfile, AWS EC2 Graviton deployment, Caddy HTTPS setup |
| `data/samples/` | **Person 2** (`@Gyantwik`) | Curated test samples for web UI demo and manifest |
| `core/quality_gate.py` & `degrade.py` | **Person 2** (`@Gyantwik`) | Image quality gate and deterministic blur/glare simulations |
| `frontend/` | **Person 3** | React + Vite + Tailwind CSS 3-panel operator console |
| `docs/` (general) | **Person 3** | 7-section competition technical report, architecture diagram |
| Root files & `.github/` | **Shared** | Requires review before modifying |

---

## 4. Instructions for Person 1 (@Biswabasini90)

Welcome! The repository setup and data contracts have already been completed by `@Gyantwik`.

### Step 1: Clone the Repo
Open Git Bash and run:
```bash
git clone https://github.com/Gyantwik/sureinspect.git
cd sureinspect
git switch -c feat/core-opencv
```

### Step 2: Set Up Python Virtual Environment
```bash
py -3.11 -m venv .venv-train
source .venv-train/Scripts/activate
python -m pip install -r requirements-train.txt
```

### Step 3: Your Day 1 Tasks
1. Run `train/prepare_dataset.py`: Download the VisA dataset and generate frozen, hashed train/calib splits for `pcb1` with fixed seed 42.
2. Run `train/export_backbone_onnx.py`: Export the WideResNet50 feature extractor to ONNX and verify it loads in OpenCV 5 DNN (`cv.dnn.readNetFromONNX`).
3. Build the coreset memory bank in `train/build_memory_bank.py`.
4. Implement OpenCV tools T1-T5 in `core/tools.py`.
5. Remember: **Never commit `.onnx` or `.npy` files to Git!** Store them locally in `artifacts/`.

---

## 5. Instructions for Person 3

Welcome! The repository setup and backend OpenAPI contracts are already live.

### Step 1: Clone the Repo
```bash
git clone https://github.com/Gyantwik/sureinspect.git
cd sureinspect
git switch -c feat/frontend-scaffold
```

### Step 2: Initialize Vite in `/frontend`
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite lucide-react
npm install -D openapi-typescript vitest
```

### Step 3: Your Day 1 Tasks
1. Configure Tailwind CSS with the Vite plugin.
2. In `vite.config.ts`, set up the proxy: `"/api": "http://127.0.0.1:8000"`.
3. Auto-generate TypeScript types directly from the tracked contract:
   ```bash
   npx openapi-typescript ../contracts/openapi.json -o src/services/api-types.ts
   ```
4. Build the 3-panel layout shell (Control Panel, Trace Timeline, Decision Card) with full type-safety.
