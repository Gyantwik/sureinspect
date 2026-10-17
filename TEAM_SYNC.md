# SureInspect: Team Status & Collision Prevention Guide
> **Repository:** https://github.com/Gyantwik/sureinspect.git  
> **Project:** SureInspect (OpenCV AI Competition 2026)  
> **Last Updated:** October 17, 2026  

---

## 1. Live Team Status Board

| Role | Assigned Developer | Branch | Current Status |
| :--- | :--- | :--- | :--- |
| **Person 1 (OpenCV & ML Lead)** | `@Biswabasini90` | `feat/core-opencv` | **Ready to start** (scaffolding is already done) |
| **Person 2 (Backend & AWS Lead)** | `@Gyantwik` (Owner) | `feat/backend-contract` | **ACTIVE** (building API schemas, mock service & contracts) |
| **Person 3 (Frontend & Docs Lead)** | Assigned Contributor | `feat/frontend-scaffold` | **Ready to start** (Vite frontend shell) |

---

## 2. What Has Already Been Done by Person 2 (@Gyantwik)
* Initialized the GitHub repository and established the `main` branch.
* Built the unified monorepo directory tree (`core/`, `backend/`, `train/`, `eval/`, `config/`, `data/`, `deploy/`, `docs/`, `.github/`).
* Created all configuration, environment, and requirements files:
  * `.gitignore`, `.gitattributes`, `.dockerignore`, `.env.example`
  * `requirements.txt`, `requirements-dev.txt`, `requirements-train.txt`, `requirements-eval.txt`
  * `config/categories.yaml` (configured for Amazon VisA `pcb1`)
* Created CI automation in `.github/workflows/ci.yml` with automated boundary checking (`boundary_check.sh`) and anti-large-file checking (`large_file_check.sh`).
* Verified initial bootstrap tests and pushed to GitHub `main`.

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

Welcome! The repository setup has already been completed by `@Gyantwik`. You do not need to initialize anything.

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
4. Remember: **Never commit `.onnx` or `.npy` files to Git!** Store them locally in `artifacts/`.

---

## 5. Instructions for Person 3

Welcome! The repository setup is already done.

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
3. Build the 3-panel layout shell (Control Panel, Trace Timeline, Decision Card) using mock data first (`VITE_USE_MOCK=true`).
4. Person 2 will provide `contracts/openapi.json` within four hours so you can auto-generate TypeScript types using `openapi-typescript`.

---

## 6. The Golden Git Rules for All 3 Developers
1. **Never push directly to `main`.** Always work on your personal feature branch (`feat/...`).
2. **Never edit files outside your assigned ownership.** CI will automatically reject PRs with out-of-boundary changes.
3. **Pull `main` frequently:**
   ```bash
   git fetch origin
   git merge origin/main
   ```
4. **Communicate via PRs:** When a task is ready, push your branch and open a Pull Request for code review.
