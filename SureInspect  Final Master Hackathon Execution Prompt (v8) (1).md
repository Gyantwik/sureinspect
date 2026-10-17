# SureInspect: Final Master Hackathon Execution Prompt (v8.1)

**Status:** authoritative implementation specification. This file replaces v3 through v8 and all review addenda. Give each developer this Markdown file and their role prompt below. Do not copy commands out of a PDF.

**Important:** “final” means the planning document is consolidated, not that the application is built or its performance is certified. The team must pass the execution gates in this file. Do not add features before the first real vertical slice works.

## Start here

- **Product:** a one-page industrial inspection application using OpenCV 5 for anomaly scoring, visual evidence collection and explainable decisions; AWS for deployment and audit storage.
- **Default dataset:** Amazon VisA, primary category `pcb1`. `cashew` is optional after the primary category passes deployment and evaluation gates. VisA is listed under CC BY 4.0 by its publisher. ([AWS Open Data Registry](https://registry.opendata.aws/visa/))
- **People:** Person 1 owns OpenCV/ML; Person 2 owns backend/integration/AWS/evaluation; Person 3 owns frontend/report/video.
- **Submission target:** evening of October 26 IST. The rules page lists October 26, 2026, 11:45 p.m. PDT, equivalent to October 27, 12:15 p.m. IST. Check the actual submission form and any eligibility or registration obligations immediately; do not rely on the final minutes. ([Competition rules](https://opencv26.devpost.com/rules))
- **Schedule:** Day 1 means the team's actual start, not “today at noon.” Target the contract handoff within four working hours of bootstrap. If starting late, shorten scope rather than moving the submission deadline.
- **Costs:** no paid inference API or LLM is required. AWS resources are not assumed free: verify account eligibility, remaining credits, instance, storage, networking and monitoring charges in the console before creating them. Budget alerts are notifications, not a hard spending cap. ([AWS budget guidance](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html))

### Final maintenance fixes

v8.1 retains v8's architecture and roles. It makes the placeholder check fail safely, documents the conservative third-ROI rule, adds evaluation-statistics tests to CI, schedules a Day 7 immutable report-bundle deployment, uses JPEG for display previews while preserving lossless inputs/masks, and starts DNS/HTTPS verification with the Day 3 hello-world deployment.

### What changed in the final consolidation

The production scorer stays frozen. Human labels are logged, and optional session adaptation runs only as an experimental shadow result; it never changes the official decision or inherits the production calibration claim. Public deployment uses HTTPS and server-issued session credentials. Preprocessing, asset delivery, artifact rebuilding, session ownership, concurrency, CI rename checks and evaluation rules are now explicit.

## Role prompts to copy

### Person 1: OpenCV and ML lead

> Read the entire SureInspect Final v8.1 specification. You are Person 1, responsible for OpenCV/ML. Run the tools check and repository bootstrap, replacing the usernames before pushing. Then work through the Person 1 tasks. First merge preprocessing and the shared engine interface so Person 2 is unblocked. Implement only your owned paths. Use short task branches, reviewed PRs and the checks below. Never invent results, runtimes or guarantees. A backbone, input-size or preprocessing change invalidates every downstream scoring artifact. Final decisions always use the frozen production snapshot. Ask the team before changing shared contracts. Show commands and test results before every handoff.

### Person 2: backend, integration, AWS and evaluation lead

> Read the entire SureInspect Final v8.1 specification. You are Person 2. First accept the collaborator invitation, run the P2 clone/setup commands, and open the contract/mock PR within four working hours of bootstrap. Own API schemas, server-issued sessions, ownership checks, bounded inference, uploads, assets, report templates, quality gate/degradations, AWS and evaluation. Never silently fall back to mock. Do not mutate the production bank from a web request. Feedback feature extraction uses the same bounded engine executor as inspection. Propose shared changes in reviewed PRs. Verify costs before creating resources, and do not upgrade plans or subscribe to paid products without approval.

### Person 3: frontend, report and video lead

> Read the entire SureInspect Final v8.1 specification. You are Person 3. First accept the invitation and run the P3 clone/setup commands. Until the contract PR merges, build only Vite/Tailwind and the layout shell. Then generate TypeScript types from the tracked OpenAPI contract and use the shared fixtures. Obtain a session from the backend before inspecting. Show REAL/MOCK provenance, official frozen decisions, missing measurements and experimental shadow results distinctly. Use the API's assets for comparison; do not reconstruct them from a checkerboard overlay. Use measured evaluation output for report/video claims. Stay within your folders and use reviewed PRs.

## Product scope and invariants

### Required first release

1. One supported category and one frozen, versioned OpenCV DNN anomaly scorer.
2. Input quality gate and deterministic blur/darken/glare simulations.
3. Evidence-dependent reference alignment, ROI comparison and pixel measurements.
4. PASS, FAIL, HUMAN and RETAKE with reasons and real trace images.
5. HTTPS deployment, session-isolated uploads/traces and protected human label recording.
6. Held-out baseline-versus-agent evaluation with coverage, escapes and uncertainty.
7. Reproducible setup, architecture diagram, report and demo recording.

### Optional only after the required slice works

- A second category.
- Session shadow adaptation and its held-out evaluation.
- Counterfactual risk display, calibration panel, localisation metrics, CSV export and monitoring dashboard.

Do not add VLM narration, learned matchers, COOL/Marketplace subscriptions, physical millimetre claims, retraining, automatic global bank updates or a second dataset integration.

### Decision and statistical invariants

- Score `s` is a raw nearest-neighbour feature distance, not a probability.
- The scorer, preprocessing, bank and image-score recipe are fixed before production calibration. Choose the backbone/input on development data, not calibration or final-test outcomes.
- For a preselected alpha, the conformal upper threshold bounds the normal-image upper-tail event marginally under exchangeability. Restricting automatic FAIL to a subset of that event preserves that marginal false-FAIL bound; it does not certify each part, fixed-calibration realised error, defect recall or arbitrary user uploads. ([Conformal tutorial](https://arxiv.org/html/2107.07511))
- Use `s > t_fail`, not `>=`, for a FAIL candidate. Use `s <= t_pass` for FAST_PASS.
- Lighting-shift inputs never auto-PASS or auto-FAIL.
- “No lighting warning” does not prove exchangeability. The API reports calibration status and assumptions, not a Boolean “guarantee valid.”
- Every final FAIL additionally needs valid alignment and high residual evidence.
- Never feed CLAHE images, sharpened images or zoomed crops back to the anomaly scorer.
- Grounded text names measurements, not predicted defect subtypes. No invented “scratch” label.
- Human labels never alter official production decisions in this release.
- Make comparisons on full-precision values. Round only display/serialized measurements; do not round thresholds before decision-making.

## Ownership and repository

```text
sureinspect/
  .github/
    CODEOWNERS
    workflows/ci.yml
    scripts/boundary_check.sh
    scripts/large_file_check.sh
  .gitignore  .gitattributes  .dockerignore  .env.example
  README.md  requirements.txt  requirements-dev.txt
  requirements-train.txt  requirements-eval.txt
  contracts/
    openapi.json
    fixtures/
  config/categories.yaml
  train/                         # P1
    prepare_dataset.py
    export_backbone_onnx.py
    verify_onnx_opencv.py
    check_detector.py
    build_memory_bank.py
    fit_thresholds.py
    select_templates.py
    upload_artifacts.py
  core/                          # P1 except listed P2 files
    preprocess.py  perception.py  calibration.py
    tools.py  agent.py  artifacts.py  interfaces.py
    session_shadow.py            # optional
    degrade.py                   # P2
    quality_gate.py              # P2
    tests/
  backend/                       # P2
    main.py  app_factory.py  schemas.py  sessions.py  executor.py
    assets.py  storage.py  report.py  mock_service.py
    export_openapi.py  tests/
  eval/                          # P2
    stats.py  tests/test_stats.py  run_baseline.py  run_agent.py
    run_calibration_splits.py  run_corruptions.py
    run_feedback.py  run_localisation.py
    check_determinism.py  make_tables.py  results/
  deploy/                        # P2
    Dockerfile  ec2_setup.sh  Caddyfile
  data/
    templates/                   # P1
    samples/manifest.json        # P2
    samples/LICENSE.md           # P2, covers both sample and template attribution
  frontend/                      # P3
  docs/                          # P3 except two paths below
    failure_cases/               # P1
    aws_section.md               # P2
```

| Owner | Allowed paths |
|---|---|
| P1 | `core/`, `train/`, `config/`, `data/templates/`, `docs/failure_cases/` |
| P2 | `backend/`, `deploy/`, `eval/`, `data/samples/`, `docs/aws_section.md`; `core/degrade.py`, `core/quality_gate.py` and their two named test files |
| P3 | `frontend/`, `docs/` except `docs/failure_cases/` and `docs/aws_section.md` |
| Shared | Root README, requirements/lock files, ignore/attributes/environment examples, `.github/`; require reviewed `shared-ok` label |
| Contract | `contracts/`, `backend/schemas.py`, `core/interfaces.py`; require `contract-ok` label after all three explicitly approve |

CODEOWNERS requests reviewers; it does not enforce all-three approval. Multiple listed owners do not mean all must approve. ([GitHub CODEOWNERS documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners))

P1 may fix the P2-owned core files as engine lead, but coordinates with P2 first. No dataset archives, model files, credentials, user uploads or individual files over 5 MiB go into Git. Small attributed VisA samples/templates may be committed.

## Tools and bootstrap

### Everyone: tools and identity

Use Git Bash on Windows. Install Git for Windows, GitHub CLI, Python 3.11 x64 and Node 24. P2 also needs a container build path, either Docker Desktop or the target server.

```bash
git --version
gh --version
py -3.11 --version
node --version
npm --version
git config --global user.name "YOUR REAL NAME"
git config --global user.email "YOUR VERIFIED OR NOREPLY EMAIL"
git config --global init.defaultBranch main
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Each person uses their own account. Do not share passwords, keys or tokens.

### P1: create the scaffold locally

This block is only for a new project directory. It creates files but does not publish anything.

```bash
mkdir sureinspect
cd sureinspect
git init -b main
mkdir -p core/tests backend/tests train eval/results eval/tests config contracts/fixtures
mkdir -p data/templates data/samples deploy docs/failure_cases
mkdir -p .github/workflows .github/scripts
touch core/__init__.py backend/__init__.py
touch core/tests/__init__.py backend/tests/__init__.py
touch train/__init__.py eval/__init__.py eval/tests/__init__.py
touch contracts/fixtures/.gitkeep eval/results/.gitkeep
touch data/templates/.gitkeep data/samples/.gitkeep
touch deploy/.gitkeep docs/failure_cases/.gitkeep

cat <<'EOF' > .gitignore
__pycache__/
*.py[cod]
.venv/
.venv-train/
.env
.env.*
!.env.example
*.onnx
*.onnx.data
*.npy
*.npz
*.pt
*.pth
*.tar
*.tar.gz
*.zip
artifacts/
datasets/
data/visa/
data/VisA*/
node_modules/
frontend/dist/
dist/
build/
*.log
.pytest_cache/
.ruff_cache/
.DS_Store
EOF

cat <<'EOF' > .gitattributes
* text=auto eol=lf
*.png binary
*.jpg binary
*.jpeg binary
EOF

cat <<'EOF' > .dockerignore
.git
.venv
.venv-train
**/__pycache__
**/node_modules
frontend/dist
artifacts
datasets
data/visa
data/VisA*
eval/results
docs
.env
.env.*
*.log
EOF

cat <<'EOF' > .env.example
USE_MOCK=false
ARTIFACT_DIR=./artifacts
MODEL_VERSION=UNSET
S3_BUCKET=sureinspect-artifacts-P1_LOGIN
AWS_REGION=ap-south-1
OPENCV_IO_MAX_IMAGE_PIXELS=25000000
ENABLE_SHADOW_ADAPTATION=false
EOF

cat <<'EOF' > requirements.txt
opencv-python-headless==5.0.0.93
numpy>=1.26,<3
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic>=2.7
pyyaml>=6
boto3>=1.34
slowapi>=0.1.9
psutil>=5.9
EOF

cat <<'EOF' > requirements-dev.txt
pytest>=8
httpx>=0.27
ruff>=0.5
EOF

cat <<'EOF' > requirements-train.txt
-r requirements.txt
torch
torchvision
onnx
onnxscript
scikit-learn
tqdm
pandas
pillow
EOF

cat <<'EOF' > requirements-eval.txt
-r requirements.txt
scipy
pandas
matplotlib
scikit-learn
EOF

cat <<'EOF' > config/categories.yaml
dataset: visa
image_max_side: 1024
categories:
  - id: pcb1
    display_name: PCB 1
    enabled: true
    backbone: wide_resnet50_2
    backbone_input: 256
    bank_images: 200
    calib_images: 300
    alpha_pass: 0.20
    alpha_fail_default: 0.05
    alpha_fail_options: [0.01, 0.02, 0.05, 0.10]
    templates: 5
  - id: cashew
    display_name: Cashew
    enabled: false
    backbone: wide_resnet50_2
    backbone_input: 256
    bank_images: 150
    calib_images: 150
    alpha_pass: 0.20
    alpha_fail_default: 0.05
    alpha_fail_options: [0.02, 0.05, 0.10]
    templates: 5
EOF

cat <<'EOF' > README.md
# SureInspect
Industrial visual inspection with OpenCV 5 and AWS.
See the Final v8.1 specification for ownership and implementation gates.
No accuracy or deployment claims have been measured yet.
EOF

cat <<'EOF' > core/tests/test_bootstrap.py
from pathlib import Path
import cv2
import yaml

def test_opencv_major():
    assert cv2.__version__.startswith("5.")

def test_enabled_primary_category():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config/categories.yaml").read_text())
    assert any(x["id"] == "pcb1" and x["enabled"] for x in config["categories"])
EOF

cat <<'EOF' > eval/stats.py
"""Two-sided binomial intervals; n=0 is unavailable, not a fabricated zero rate."""
import math
from scipy.stats import beta, norm


def _validate(k, n, confidence):
    if not isinstance(k, int) or not isinstance(n, int):
        raise ValueError("k and n must be integers")
    if n < 0 or k < 0 or k > n or not 0 < confidence < 1:
        raise ValueError("Invalid count or confidence")


def wilson(k, n, confidence=0.95):
    _validate(k, n, confidence)
    if n == 0:
        return None
    z = float(norm.ppf((1 + confidence) / 2))
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson(k, n, confidence=0.95):
    _validate(k, n, confidence)
    if n == 0:
        return None
    tail = (1 - confidence) / 2
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lower, upper
EOF

cat <<'EOF' > eval/tests/test_stats.py
import pytest
from eval.stats import wilson, clopper_pearson


def test_wilson_known_count():
    assert wilson(3, 100) == pytest.approx((0.010, 0.085), abs=0.0005)


def test_exact_known_count():
    assert clopper_pearson(3, 100) == pytest.approx((0.006, 0.085), abs=0.0005)


@pytest.mark.parametrize("interval", [wilson, clopper_pearson])
def test_empty_denominator(interval):
    assert interval(0, 0) is None


@pytest.mark.parametrize("interval", [wilson, clopper_pearson])
def test_invalid_count(interval):
    with pytest.raises(ValueError):
        interval(2, 1)


def test_exact_zero_events_two_sided():
    assert clopper_pearson(0, 20) == pytest.approx((0.0, 1 - 0.025 ** (1 / 20)))


def test_exact_all_events_two_sided():
    assert clopper_pearson(20, 20) == pytest.approx((0.025 ** (1 / 20), 1.0))
EOF

cat <<'EOF' > .github/CODEOWNERS
/core/                 @P1_LOGIN @P2_LOGIN
/train/                @P1_LOGIN @P2_LOGIN
/config/               @P1_LOGIN @P2_LOGIN
/data/templates/       @P1_LOGIN @P3_LOGIN
/backend/              @P2_LOGIN @P1_LOGIN
/deploy/               @P2_LOGIN @P1_LOGIN
/eval/                 @P2_LOGIN @P1_LOGIN
/data/samples/         @P2_LOGIN @P3_LOGIN
/frontend/             @P3_LOGIN @P2_LOGIN
/docs/                 @P3_LOGIN @P1_LOGIN
/docs/failure_cases/   @P1_LOGIN @P3_LOGIN
/docs/aws_section.md   @P2_LOGIN @P3_LOGIN
/.github/              @P1_LOGIN @P2_LOGIN @P3_LOGIN
/contracts/            @P1_LOGIN @P2_LOGIN @P3_LOGIN
/backend/schemas.py    @P1_LOGIN @P2_LOGIN @P3_LOGIN
/core/interfaces.py    @P1_LOGIN @P2_LOGIN @P3_LOGIN
EOF

cat <<'EOF' > .github/scripts/boundary_check.sh
#!/usr/bin/env bash
set -euo pipefail
P1="P1_LOGIN"; P2="P2_LOGIN"; P3="P3_LOGIN"
AUTHOR="${AUTHOR:?}"; BASE="${BASE:?}"; LABELS=",${LABELS:-},"
case "${AUTHOR,,}" in
  "${P1,,}") OWN='^(core|train|config|data/templates|docs/failure_cases)/' ;;
  "${P2,,}") OWN='^(backend|deploy|eval|data/samples)/|^docs/aws_section\.md$|^core/(degrade|quality_gate)\.py$|^core/tests/test_(degrade|quality_gate)\.py$' ;;
  "${P3,,}") OWN='^(frontend|docs)/' ;;
  *) echo "Unknown author: $AUTHOR"; exit 1 ;;
esac
SHARED='^(README\.md|requirements[^/]*\.txt|\.gitignore|\.gitattributes|\.dockerignore|\.env\.example)$|^\.github/'
CONTRACT='^contracts/|^backend/schemas\.py$|^core/interfaces\.py$'
BAD=0
while IFS= read -r -d '' f; do
  if [[ "$f" =~ $CONTRACT ]]; then
    [[ "$LABELS" == *",contract-ok,"* ]] || { echo "Needs contract-ok: $f"; BAD=1; }
  elif [[ "$f" =~ $SHARED ]]; then
    [[ "$LABELS" == *",shared-ok,"* ]] || { echo "Needs shared-ok: $f"; BAD=1; }
  elif [[ "${AUTHOR,,}" == "${P3,,}" && ( "$f" =~ ^docs/failure_cases/ || "$f" == "docs/aws_section.md" ) ]]; then
    echo "Another owner's path: $f"; BAD=1
  elif ! [[ "$f" =~ $OWN ]]; then
    echo "Outside $AUTHOR ownership: $f"; BAD=1
  fi
done < <(git diff --no-renames --name-only -z "$BASE"...HEAD)
exit "$BAD"
EOF

cat <<'EOF' > .github/scripts/large_file_check.sh
#!/usr/bin/env bash
# Full tracked HEAD tree, not only added/modified paths: renames cannot bypass it.
set -euo pipefail
MAX=5242880; BAD=0
while IFS= read -r -d '' f; do
  case "${f,,}" in
    *.onnx|*.onnx.data|*.npy|*.npz|*.pt|*.pth|*.tar|*.tar.gz|*.tgz|*.zip)
      echo "Forbidden model/archive: $f"; BAD=1 ;;
    .env|.env.*|*/.env|*/.env.*)
      case "$f" in .env.example|*/.env.example) ;; *) echo "Forbidden env file: $f"; BAD=1 ;; esac ;;
  esac
  SIZE=$(git cat-file -s "HEAD:$f")
  if (( SIZE > MAX )); then
    echo "Over 5 MiB: $f ($SIZE bytes)"; BAD=1
  fi
done < <(git ls-tree -r --name-only -z HEAD)
exit "$BAD"
EOF

cat <<'EOF' > .github/workflows/ci.yml
name: ci
on:
  pull_request:
    types: [opened, synchronize, reopened, labeled, unlabeled]
permissions:
  contents: read
jobs:
  boundary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with: {fetch-depth: 0}
      - name: Trusted base-branch checks
        env:
          AUTHOR: ${{ github.event.pull_request.user.login }}
          BASE: ${{ github.event.pull_request.base.sha }}
          LABELS: ${{ join(github.event.pull_request.labels.*.name, ',') }}
        run: |
          git show "$BASE:.github/scripts/boundary_check.sh" > "$RUNNER_TEMP/ownership.sh"
          git show "$BASE:.github/scripts/large_file_check.sh" > "$RUNNER_TEMP/large.sh"
          bash "$RUNNER_TEMP/ownership.sh"
          bash "$RUNNER_TEMP/large.sh"
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: {python-version: '3.11'}
      - run: python -m pip install -r requirements.txt -r requirements-dev.txt
      - run: python -m pip check
      - name: Tests; zero collected tests is a failure
        env:
          USE_MOCK: "true"
          OPENCV_IO_MAX_IMAGE_PIXELS: "25000000"
        run: python -m pytest core/tests backend/tests -q
      - name: Tracked OpenAPI and schema drift
        env:
          USE_MOCK: "true"
          OPENCV_IO_MAX_IMAGE_PIXELS: "25000000"
        run: |
          if [ -f backend/export_openapi.py ]; then
            git ls-files --error-unmatch contracts/openapi.json
            python -m backend.export_openapi
            git diff --exit-code -- contracts/openapi.json
          elif [ -f backend/main.py ]; then
            echo "API exists without its required exporter."
            exit 1
          else
            echo "Bootstrap only: backend application not created yet."
          fi
  evaluation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: {python-version: '3.11'}
      - run: python -m pip install -r requirements-eval.txt -r requirements-dev.txt
      - run: python -m pip check
      - name: Evaluation statistics tests
        env:
          USE_MOCK: "true"
        run: python -m pytest eval/tests -q
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v5
        with: {node-version: '24'}
      - name: Build and test frontend
        run: |
          if [ ! -d frontend ]; then
            test ! -f backend/main.py
            echo "Bootstrap only: neither frontend nor API exists yet."
            exit 0
          fi
          test -f frontend/package-lock.json
          cd frontend
          npm ci
          npm run build
          npm run test -- --run
          if [ -f ../backend/main.py ]; then
            npm run types:check
          fi
EOF
```

**Stop before committing.** Set real usernames and verify that the authenticated GitHub account is P1:

```bash
P1=YourP1Login; P2=YourP2Login; P3=YourP3Login
sed -i "s/P1_LOGIN/$P1/g; s/P2_LOGIN/$P2/g; s/P3_LOGIN/$P3/g" .github/CODEOWNERS .github/scripts/boundary_check.sh
sed -i "s/P1_LOGIN/$(printf '%s' "$P1" | tr '[:upper:]' '[:lower:]')/g" .env.example
if grep -rnE 'P[123]_LOGIN' .github .env.example; then
  echo "STOP: replace the remaining placeholders before continuing."
  exit 1
else
  status=$?
  if [ "$status" -eq 1 ]; then
    echo "All placeholders replaced."
  else
    echo "STOP: placeholder search failed."
    exit "$status"
  fi
fi
gh api user --jq .login
```

The placeholder check uses `grep`, and stops on remaining placeholders or a search failure; it does not silently accept a missing search tool. No placeholder should remain in the generated files. S3 bucket names must be lowercase and globally unique; if a suffix is needed, update `.env.example` and use that exact bucket consistently.

Install dependencies before checking OpenCV:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check
python -m pytest core/tests backend/tests -q
git add -A
git diff --cached --stat
git diff --cached
git commit -m "chore: bootstrap contracts, ownership checks and CI"
bash .github/scripts/large_file_check.sh
```

These requirements are bootstrap inputs, not tested cross-platform locks. After resolution, record tested runtime locks for Windows and ARM64 separately, plus the exact training torch/torchvision pair. Shared-file edits require review.

### Publish the repo and invite collaborators

Use a public repo only after the team approves making the code and selected samples public. If private is preferred, verify plan support for the branch protection you intend to use; GitHub's protected-branch availability depends on repository visibility and plan. ([GitHub documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches))

The following is a team-run command, not something this document has executed:

```bash
# Same terminal where P1, P2 and P3 were defined.
gh repo create "$P1/sureinspect" --public --source=. --remote=origin --push
gh api --method PUT "repos/$P1/sureinspect/collaborators/$P2" -f permission=push
gh api --method PUT "repos/$P1/sureinspect/collaborators/$P3" -f permission=push
gh label create shared-ok --color FBCA04 --description "Shared changes reviewed"
gh label create contract-ok --color B60205 --description "All three approved the contract"
```

P2/P3 accept invitations. In repository settings, allow squash merges and delete merged branches. Protect `main`: one non-author approval, required `boundary`, `python`, `evaluation`, `frontend` checks after their first run, and no force pushes. CODEOWNERS approval enforcement may remain off; all-three contract approval is a team rule plus a reviewed label.

**First GitHub test:** make throwaway PRs that demonstrate a forbidden path, a renamed `.onnx`, a missing lockfile after frontend creation and a missing tracked OpenAPI file all fail. Do not merge failing experiments. Stable base-branch scripts prevent a PR from bypassing its ownership/file checks by editing those scripts, but workflow changes still need human scrutiny.

### P2: clone and setup

```bash
gh repo clone P1_LOGIN/sureinspect
cd sureinspect
git switch -c feat/backend-contract
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Replace `P1_LOGIN` in the clone command. For the first API PR, follow the four-PR sequence below: once `backend/main.py` exists, CI requires a real frontend build, tests and tracked type check.

### P3: clone and scaffold

```bash
gh repo clone P1_LOGIN/sureinspect
cd sureinspect
git switch -c feat/frontend-scaffold
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite lucide-react
npm install -D openapi-typescript vitest
```

Before opening the scaffold PR:
- Configure the Tailwind Vite plugin and CSS import according to the installed major version. ([Tailwind Vite guide](https://tailwindcss.com/docs/installation/using-vite))
- Add `/api` proxy to `http://127.0.0.1:8000`.
- Add a real Vitest smoke test; define `"test": "vitest"` in package scripts.
- Define `types:generate` to generate `src/services/api-types.ts` from `../contracts/openapi.json`. Define `types:check` to first require that file to be tracked with `git ls-files --error-unmatch src/services/api-types.ts`, then regenerate and run `git diff --exit-code -- src/services/api-types.ts`.
- Commit `package-lock.json`. Generated types must also be tracked; the API handoff PR must not merge before the generated file exists.
- Before the API exists, type generation can wait; build/test cannot.

```bash
npm run build
npm run test -- --run
```

First handoff sequence, without cross-owner edits or CI deadlock:
1. P3 merges the scaffold, real smoke test and package lock.
2. P2 merges schemas, mock route definitions in `backend/app_factory.py`, exporter and tracked `contracts/openapi.json`, with `contract-ok`. The app factory builds the documented routes in mock mode without artifacts. Do not add `backend/main.py` yet. CI checks the exporter but does not yet require generated frontend types.
3. P3 merges the generated, tracked `api-types.ts` and tests the type-check script.
4. P2 adds `backend/main.py` as the runtime entry point importing the app factory. All API, OpenAPI, frontend and type checks are now mandatory.

After bootstrap, material API changes require the same contract-first coordination. If a contract change breaks the existing frontend, split backward-compatible schema changes from consumer migration rather than bypassing CI.

### Everyone: daily Git and review loop

```bash
# Start with a clean working tree.
git switch main
git pull --ff-only origin main
git switch -c feat/YOUR-SMALL-TASK

# Edit, run the task's tests, then stage specific owned paths.
git status --short
git add PATH1 PATH2
git diff --cached
git commit -m "feat: describe this tested change"
git fetch origin --prune
git merge origin/main
# Re-run tests after merging.
bash .github/scripts/large_file_check.sh
git push -u origin HEAD
gh pr create --base main --fill --reviewer REVIEWER_LOGIN
```

Reviewer:

```bash
gh pr checkout PR_NUMBER
gh pr diff PR_NUMBER
# Run the relevant tests locally.
gh pr checks PR_NUMBER --watch
gh pr review PR_NUMBER --approve --body "Reviewed and reproduced the listed tests."
gh pr merge PR_NUMBER --squash --delete-branch
```

After merge, return to `main`, pull with `--ff-only`, and start a fresh task branch. A squash-merged local branch may need `git branch -D OLD_BRANCH`, but only after confirming it is merged and contains no unpushed work. Never use force-push on `main` or destructive reset to silence conflicts.

For conflicts: `git status`, resolve with the relevant owner, `git add RESOLVED_PATHS`, `git merge --continue`, then retest. Use `git merge --abort` only to abandon an unfinished merge.

For shared changes, a reviewer adds `shared-ok`; for contract changes, all three explicitly approve the latest contract diff before a reviewer adds `contract-ok`. Remove labels and request fresh review if the approved content changes materially.

## Shared engine interfaces

P1 and P2 freeze `core/interfaces.py` with the API contract:

```python
# Type definitions are illustrative; implement dataclasses/TypedDicts or Pydantic adapters.
def standardize(image_bgr): ...
def to_blob(image_std, size): ...
def fit_gate_thresholds(images_std): ...
def check_quality(image_std, thresholds): ...
def inspect_standardized(image_std, category, flags, alpha_fail, snapshot): ...
def decide(score, lighting_shift, evidence, t_pass, t_fail, budget_state): ...
```

`inspect_standardized()` accepts an already standardized BGR uint8 image. It never silently resizes it a second time. Backend retains original dimensions, supplies standardized dimensions and assembles metadata. CLI/training paths call the same standardizer before this interface.

Quality threshold dictionary:

```text
blur_thr: float
dark_max: float
bright_max: float
glare_thr: float
l_mean_ref: float
l_mean_sd: float, minimum 2.0 L units
l_std_ref: float, guarded against zero
n_fit: integer
gate_version: string
```

`fit_gate_thresholds()` takes standardized development normals. `check_quality()` returns the API's quality object. P2 may merge a stub early with `gate_version="stub"`; no live-ready artifact may contain stub thresholds.

## API contract

P2 owns Pydantic schemas and a deterministic OpenAPI exporter. P1/P3 approve semantics before independent integration. Generate types from the tracked schema; do not hand-write parallel TypeScript API types.

### Sessions and ownership

- `POST /api/session`: creates an unguessable server-generated token using a cryptographic secret generator; returns `{session_token, expires_at}`. Keep it in tab `sessionStorage`, never URLs or logs.
- Protected requests use `Authorization: Bearer <session_token>`.
- Expiry: 30-minute idle TTL, bounded active sessions; return 401 `SESSION_EXPIRED` when needed.
- Bind every inspection, asset, feedback event and trace to its owning session.
- Validate ownership for all protected reads/writes. Unknown or non-owned inspection/asset IDs return the same 404.
- Session reset clears experimental shadow state, not other visitors' history.
- Public HTTPS is a release requirement. `crypto.randomUUID()` is no longer a session dependency; it requires a secure context and cannot be assumed on public HTTP. ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/Crypto/randomUUID))

### Endpoints

| Endpoint | Behaviour |
|---|---|
| `GET /api/health` | Readiness, `engine`, model version, enabled/loaded categories. No secrets. Missing artifacts produce degraded readiness, not mock success. |
| `GET /api/categories` | Enabled category IDs, display names, valid alpha options, default alpha, calibration counts and sample counts. |
| `GET /api/samples?category=...` | Sample ID, category, ground-truth dataset label, attribution, sample image URL, optional demo role. |
| `GET /api/samples/{id}/image` | Curated sample bytes; allowlisted lookup, no arbitrary filesystem paths. |
| `POST /api/inspect` | Protected multipart: exactly one `file` or `sample_id`, category, optional allowed `alpha_fail`, three simulation booleans. |
| `GET /api/trace/{id}` | Protected, complete stored response JSON for the owning session. |
| `GET /api/assets/{id}` | Protected, session-owned image bytes with bounded retention. |
| `GET /api/calibration/{category}` | Explicitly `scope="frozen_production"`, production thresholds/counts, `report_bundle_id` (nullable), `evaluation_status="pending"|"available"` and approved evaluation summaries. |
| `POST /api/feedback` | Protected: inspection ID, `NORMAL` or `DEFECT`, idempotency key. Category comes from the stored inspection. |
| `POST /api/session/reset` | Protected: resets only shadow state; returns current frozen snapshot identity. |

Frontend asset fetches use the authorization header, then create/revoke object URLs; a bare `<img src>` cannot attach the bearer header.

### Inspection fields

The following is the required schema shape, not fabricated model output:

```text
schema_version: "1.0"
inspection_id, request_id, timestamp: strings
engine: "live" | "mock"
category: string
snapshot: {
  model_version, bank_version, calibration_version, policy_version: strings
}
image: {
  original_size: [width, height],
  standardized_size: [width, height],
  scale_x, scale_y: floats
}
decision: "PASS" | "FAIL" | "HUMAN" | "RETAKE"
decision_path: enum below
suggested_decision: "PASS" | "FAIL" | null
anomaly_score: float | null
thresholds: {
  t_pass: float | null, t_fail: float | null,
  alpha_pass, alpha_fail: floats, n_calib: integer
}
calibration: {
  status: "frozen_assumption_dependent" | "not_applicable_lighting_shift" | "not_evaluated",
  claim: "normal_marginal_upper_tail" | null,
  assumptions: string
}
quality_gate: {
  status: "PASSED" | "FAILED",
  action: "REQUEST_RETAKE" | null, reason: string,
  blur_variance, blur_threshold, dark_fraction, bright_fraction: floats,
  glare_fraction, glare_threshold, l_mean: floats,
  exposure_ok, lighting_shift_detected: booleans
}
alignment: {
  template_id, method: strings,
  quality, q_min, rotation_deg: floats
} | null
measurements: {
  roi_bbox: [x, y, width, height],
  roi_source: "threshold" | "peak_fallback",
  rois_checked, rois_total: integers,
  residual_mean, r_low, r_high: floats,
  area_pct: float | null,
  area_denominator: "image" | "part_mask",
  elongation: float | null,
  shape_class: "elongated" | "compact" | "irregular" | null
} | null
assets: {
  input_standardized: protected asset URL,
  reference_aligned, anomaly_heatmap, contour_mask: protected asset URL | null
}
execution_trace: list of {
  step: integer, tool: enum, reason, status: strings,
  counts_toward_budget: boolean,
  duration_ms: float, overlay_url: protected asset URL | null
}
steps_used, step_budget: integers
timing_ms: {quality, perception, tools, total: floats}
grounded_report: string
counterfactuals: list of {
  alpha_fail, t_fail: floats,
  decision, decision_path: enum | null,
  needs_rerun: boolean
}
experimental_shadow: null | {
  status: "experimental_no_calibration_claim",
  shadow_version: string,
  score, shadow_t_pass, shadow_t_fail: floats,
  approximate: boolean,
  note: string
}
```

- All boxes are standardized-image coordinates. Map x/width by `scale_x`, y/height by `scale_y` to the original. Rounding can make the two scale factors slightly different.
- Core returns numerical results/assets; backend adds original metadata, IDs, timestamp, session ownership and protected URLs. P2's report renderer uses actual fields.
- Every overlay is full standardized size; T3 shows zoom as an inset. Underlying compare assets remain separate.
- RETAKE: score, decision thresholds, alignment and measurements are null; input asset and quality trace still exist.
- FAST_PASS: alignment/measurements/reference/contour are null; perception heatmap can exist.
- T3 skipped or no valid contour: shape/area/elongation are null, not zero.
- Residual-only results still report their bbox and residual. Area defaults to image-area percentage; use part-mask area only if an image-derived mask has passed validation.
- Calibration status describes applicability assumptions, not proof that an arbitrary upload is exchangeable. No `guarantee_valid` Boolean.

### Decisions

| Path | Decision | Rule |
|---|---|---|
| `QUALITY_REJECTED` | RETAKE | Input quality fails; no anomaly inference. |
| `FAST_PASS` | PASS | No lighting shift and `s <= t_pass`. |
| `AGENT_CLEARED` | PASS | No shift, `s <= t_fail`, valid alignment, all proposed ROIs checked with low residual. |
| `AGENT_CONFIRMED` | FAIL | No shift, `s > t_fail`, valid alignment, high residual evidence. |
| `SUSPECTED_DEFECT` | HUMAN | High residual but below/equal upper threshold or lighting shifted; suggested FAIL. |
| `EVIDENCE_CONFLICT` | HUMAN | Score above upper threshold but checked residuals low. |
| `LIGHTING_SHIFT_REVIEW` | HUMAN | Low residuals but lighting shifted; suggested PASS only. |
| `ALIGN_FAILED` | HUMAN | Failed/implausible registration or insufficient valid overlap. |
| `INCONCLUSIVE` | HUMAN | Mid-band residual, or any remaining unchecked third proposed ROI. |
| `BUDGET_EXHAUSTED` | HUMAN | Mandatory next evidence tool cannot run within the budget. |

Tools: `SIMULATED_DEGRADATION`, `QUALITY_GATE`, `PATCHCORE_PERCEPTION`, `T1_ALIGN_REFERENCE`, `T2_ROI_COMPARE`, `T3_ROI_ZOOM_MEASURE`, `T4_LIGHT_NORMALIZE`, `T5_NEXT_ROI_COMPARE`.

Every non-2xx response uses:

```json
{"error":{"code":"VALIDATION_ERROR","message":"Human-readable message","request_id":"server-generated-id"}}
```

Map 400 bad request/category/alpha; 401 invalid/expired session; 404 unavailable/non-owned inspection or asset; 409 session shadow limit or idempotency conflict; 413 byte/pixel limit; 415 invalid image; 422 typed input validation; 429 rate limit; 503 queue full/artifacts unavailable; 500 unexpected failure. Override framework validation/HTTP/rate-limit exception handlers to preserve this shape.

Fixtures: all ten decision paths, skipped-T3 and no-contour variants, session creation/expiry, errors, categories/samples, frozen calibration, log-only feedback, optional shadow feedback and reset. `engine="mock"` is mandatory for inspection/health mock data; do not add that field to unrelated arrays just to satisfy a global fixture rule.

## Person 1 execution tasks

### Dataset preparation and split freeze

Create the training venv before running the publisher preparation script:

```bash
# From repo root in Git Bash.
py -3.11 -m venv .venv-train
source .venv-train/Scripts/activate
python -m pip install -r requirements-train.txt

mkdir -p "$HOME/datasets"
cd "$HOME/datasets"
curl -fL -o VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
mkdir -p VisA
tar -xf VisA_20220922.tar -C VisA
git clone https://github.com/amazon-science/spot-diff.git
cd spot-diff
python utils/prepare_data.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file split_csv/1cls.csv
```

Check extraction nesting; `--data-folder` must contain the category folders. The publisher script creates `VisA_pytorch/1cls/<category>/train/good`, `test/good`, `test/bad` and `ground_truth/bad`. ([Publisher preparation script](https://raw.githubusercontent.com/amazon-science/spot-diff/main/utils/prepare_data.py))

`train/prepare_dataset.py` writes hashed, disjoint lists using seed 42:
- Bank normals and calibration normals from training normals, counts from config.
- Development normals: remaining training normals, further split for tuning/checking the quality gate if enough data.
- Development anomalies: 30% of the publisher's test anomalies, labelled as a nonstandard development split.
- Final test: remaining 70% anomalies and publisher test normals, untouched until final evaluation.
- Separate small demo pool where practical. If final-test images are later selected for presentation, freeze metrics first and never tune from them.

Record image/mask hashes, counts and near-duplicate groups. No group crosses splits. Check counts before allocating; fail with an actionable message if insufficient. Report expected interval widths for the actual test sizes. Ground-truth masks are evaluation-only, never inference-time object masks.

### Preprocessing and export

Merge preprocessing and `core/interfaces.py` first.

- `standardize`: retain aspect ratio, longest side at most 1024, `INTER_AREA` when shrinking, no upscaling. Return image plus exact dimensions/scales.
- Masks use nearest-neighbour resize.
- Backbone tensor: standardized BGR → RGB, resize to configured square input using one fixed interpolation policy, float32 divided by 255, subtract `[0.485, 0.456, 0.406]`, divide channelwise by `[0.229, 0.224, 0.225]`, contiguous NCHW.
- The channel mean/std match the documented weight normalization, but this project intentionally uses a fixed whole-image square resize, not the weight preset's center crop. Use this identical project policy in reference and runtime. ([Torchvision weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html))
- Do normalization explicitly with NumPy, or embed it inside ONNX, never both. Classic `blobFromImage`'s single scale factor cannot alone implement three different std values. ([OpenCV DNN docs](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html))
- Export `wide_resnet50_2` with explicit pretrained weight enum recorded in the manifest; outputs `layer2_out` and `layer3_out`.
- Start with 256 input. Export one fixed-shape graph per size. Set `external_data=False`; test modern export with `onnxscript`, or a tested legacy exporter if necessary. Do not assume exporter compatibility without the forward test. ([PyTorch export docs](https://docs.pytorch.org/docs/stable/onnx.html))
- First compare normalized tensors, then OpenCV DNN vs PyTorch features with the same input. Target max absolute feature difference below `1e-3`; document actual tolerances and failures.
- If export is still broken by end of Day 2, switch to ResNet18, recording the different channel dimensions. A graph that imports but produces wrong features is not a pass.

### Detector, resolution gate and calibration

- Upsample layer3 to the actual layer2 spatial size; concatenate, then stride-1 3×3 average pool with an explicit padding policy.
- At 256 input, WRN layer2/layer3 yield a 32×32 combined grid and 1,536 channels; ResNet18 uses 384 combined channels. Derive dimensions, do not hard-code WRN shapes.
- Build a coreset from bank normals, initially 5% to limit CPU setup cost. Record the projection seed and selection procedure. Cap absolute bank size and report it.
- Save fp16 on disk if desired, but cast the final saved bank to fp32 once for scoring. Calibration uses this saved/reloaded bank, not a higher-precision bank that is later quantized.
- Use squared norms and blocked matrix multiplication; block both queries and bank if necessary, avoiding query×bank×descriptor broadcasting. Clip numerical negatives before square root.
- Store image score as max raw patch distance. Build the displayed heatmap by upsampling and fixed Gaussian blur.
- Day 2 detection gate: image AUROC on development normals/anomalies, never final test or calibration. Team target 0.85, not an expected benchmark.
- If weak, test 320 then 384 input with rebuilt bank. Choose the smallest adequate configuration. Tiling is excluded from this release.
- Only after selecting the scorer, score the untouched calibration normals. Let `k=ceil((n+1)*(1-alpha))`; threshold is sorted score `k-1`. If `k>n`, omit that alpha option instead of clipping.
- Fit `t_pass` at alpha 0.20 and allowed `t_fail` options from category config. Store sorted scores and counts.
- Fit heatmap `patch_thr` on the same upsampled/blurred representation used at runtime, using development normals and image-derived masks or the whole image.
- Find up to three ROI boxes by peak heatmap value. If none pass, use a clipped 3×3 feature-patch box around the peak; ROI1 always exists. Map stride independently with standardized width/grid-width and height/grid-height.

Acceptance: one real image gives a score/heatmap; tensor and feature parity tests pass; bank RAM and development AUROC are measured; split leakage tests and tie tests pass.

### Templates, alignment and evidence

- Select up to five bank-normal templates by deterministic embedding diversity; keep original standardized PNGs and embeddings.
- Start ORB with BF Hamming ratio filtering and RANSAC `estimateAffinePartial2D`.
- Fallback: deterministic 10-degree rotation search on reduced images, then ECC refinement on the best candidate. Try up to two nearest templates.
- Store one transform convention: homogeneous matrix from reference-standardized coordinates to test-standardized coordinates.
- Convert reduced transforms with `M_full = inverse(S_test) @ M_small @ S_reference`; compose coarse rotation and ECC refinement explicitly. Convert inverse-map outputs to this convention before using them.
- Validate finite transform, plausible scale/rotation, matches/inliers when applicable, overlap and quality. Warp an all-ones validity mask and exclude padding from residuals.
- Use image-derived part masks only if validated for the category; otherwise compare within valid overlap and report area as image-area percentage.
- Fit alignment cutoff `q_min` using development normals, initially their 1st percentile. Report actual held-out alignment failure rather than promising a 1% deployment rate.
- T2 compares aligned grayscale images after fixed blur using `absdiff/255` in the ROI and valid mask. No contrib SSIM dependency.
- Fit `r_low` and `r_high` on development data only, starting at normal residual 90th/99th percentiles; assess anomaly capture on development anomalies and record the chosen trade-off.
- Fit separate CLAHE-comparison residual cutoffs. Final-test outcomes never trigger threshold retuning.
- T3: padded crop, difference contour and rotated rectangle. Handle empty mask/contour and zero-width rectangles as unavailable measurements, not fabricated zeros or infinities.
- Shape: elongated at ratio ≥3; compact below 1.5; irregular otherwise.
- Unit-test a known 30-degree rotation plus translation, resolution scaling, transform direction, padding exclusion and low-overlap failure.

### Agent state machine

Official decisions always use the frozen scorer. Tool budget: four evidence tools and a configured soft tool-time deadline, initially 2,000 ms. Quality/perception are outside the tool count but included in end-to-end latency.

```text
Apply requested deterministic degradations to standardized input.
Quality failed -> QUALITY_REJECTED / RETAKE.
Score original/degraded input with the frozen scorer.
No lighting shift and s <= t_pass -> FAST_PASS / PASS.

Lighting shifted:
  T4 prepares normalized comparison copies of test and candidate templates.
  T1 still registers original grayscale inputs, then applies its transform
  to the corresponding normalized reference. Scoring remains unchanged.
Otherwise:
  T1 registers originals.

T1 invalid -> ALIGN_FAILED / HUMAN.
T2 compares ROI1, using the correct raw/CLAHE residual cutoffs.

High residual:
  If budget/time allows, T3 measures.
  If mandatory T3 cannot run because time expired -> BUDGET_EXHAUSTED.
  No shift and s > t_fail -> AGENT_CONFIRMED / FAIL.
  Otherwise -> SUSPECTED_DEFECT / HUMAN, suggested FAIL.

Mid-band residual -> INCONCLUSIVE / HUMAN.

Low residual:
  ROI2 exists:
    No next-tool budget/time -> BUDGET_EXHAUSTED / HUMAN.
    T5 compares ROI2.
    High -> same confirmation logic above.
    Mid-band -> INCONCLUSIVE / HUMAN.
    Low -> CLEAR.
  No ROI2 -> CLEAR.

CLEAR:
  Any unchecked third proposed ROI -> INCONCLUSIVE / HUMAN.
  s > t_fail -> EVIDENCE_CONFLICT / HUMAN.
  Lighting shifted -> LIGHTING_SHIFT_REVIEW / HUMAN, suggested PASS.
  Otherwise -> AGENT_CLEARED / PASS.
```

All non-fallback proposed ROIs are above `patch_thr` by construction. “Third ROI” means a third contour-derived box returned by perception, not a new undefined suspicion test. With three boxes, low residuals in the first two never produce AGENT_CLEARED: the unchecked third forces HUMAN. This is an intentional conservative trade-off; report its contribution to review burden. Do not introduce a “t_pass-equivalent patch threshold,” because image-score and heatmap thresholds are different quantities.

With a lighting shift, T4→T1→T2→T5 uses all four steps. If ROI2 is high, skip T3 explicitly and return SUSPECTED_DEFECT with residual-only measurements. Without a shift, T1→T2→T5→T3 fits four steps. Tool-count and elapsed-time exhaustion are separate conditions.

The time budget is soft: check before each expensive operation and configure internal alignment iteration limits. Do not pretend this can preempt an arbitrary running native call. End-to-end time is always reported separately.

`decide()` is a pure rule evaluator over score, evidence completeness, lighting and budget. Counterfactuals reuse the same evidence with another preselected alpha's upper threshold; incomplete evidence yields `needs_rerun=true`. No hypothetical claim may imply an unrun tool succeeded.

Tests cover all ten paths, score/residual ties, empty contour, third ROI, skipped T3, no decisions under lighting shift, budget boundaries and counterfactuals.

### Artifact lifecycle

Every scorer change invalidates downstream bank, calibration, heatmap, embedding and evidence-threshold artifacts. Rebuild them before publishing.

Publish immutable snapshots:
- Day 3 early: perception-only snapshot, `readiness="perception_only"`. It cannot serve final agent decisions.
- Before Day 4 real deployment: complete snapshot with `readiness="agent_ready"`, no stub thresholds and all required checks.
- Manifest includes versions, explicit weight identity, preprocessing, input size, feature layers/shapes, score recipe, bank precision/size, split hashes, seeds, tool policy, thresholds and per-file sha256.
- Generate both machine-readable `manifest.json` and standard `SHA256SUMS`. Only the latter is passed to `sha256sum -c`.
- Upload under `s3://<bucket>/artifacts/<snapshot-id>/`, never overwrite a published snapshot.
- P2 downloads and verifies on another machine before activation. Readiness fails closed for partial/mismatched artifacts.

Optional second category repeats all gates independently; enabling it is not just changing a dropdown.

## Person 2 execution tasks

### Contract/mock and guarded engine integration

Within four working hours, publish schemas, fixtures and deterministic OpenAPI export. Coordinate the initial UI/types handoff with P3 as described in bootstrap.

- `USE_MOCK=true`: explicit fixture service with MOCK labels.
- Live mode with missing artifacts: readiness degraded, inspect 503.
- No catch-all fallback from real errors to mock results.
- JSON report templates use actual measured fields. Separate templates cover RETAKE, FAST_PASS, HUMAN and skipped/no-contour measurements.

### Uploads, queue and access control

- Compressed byte limit 10 MiB, enforced while streaming/reading, not after unbounded body allocation. Configure reverse-proxy body limits too.
- Check JPEG/PNG signature cheaply; validate decoded output, channels and dimensions.
- `OPENCV_IO_MAX_IMAGE_PIXELS=25000000` must be set before any OpenCV import in the process, including tests.
- Admit at most one running decode/inference job and four waiting jobs. Additional work returns 503. Only bounded raw bytes enter the queue.
- Use a bounded async admission mechanism; do not wait on a synchronous semaphore in the event loop. Execute decode, standardization, DNN work and encoding in the worker.
- Feedback re-extraction shares the same worker/model lock; do not run concurrent forwards on the shared `cv.dnn.Net`.
- Unknown internal `cv2.error` is not automatically 413. Pixel-limit/header violation is 413, malformed decode is 415, unexpected processing failure is 500.
- Required subprocess test: a tiny PNG declaring 20000×20000 fails before large allocation. Also test malformed small PNG, valid 25 MP boundary, compressed-byte limit, queue saturation and session ownership. Set the pixel-limit environment before the test subprocess imports OpenCV.
- Rate-limit session creation, inspection and feedback. Proxy-header trust must match the actual reverse-proxy network, not all incoming addresses.
- Browser CORS is not authorization. The bearer token and ownership checks are mandatory even for same-origin requests.

### Degradations and gate

P2 owns `core/degrade.py`, `core/quality_gate.py` and their named tests.

- Degrade after standardization, once only. Blur: fixed motion kernel; darken: fixed gamma; glare: feathered white blob seeded from image hash. Record applied effects in the trace.
- Fit blur Laplacian-variance floor from development normals.
- Fit dark/bright fraction limits and HSV glare-fraction limit from development normals. Record concrete margins, not undefined “plus a margin.”
- Lighting detector: development LAB L mean/std statistics, L-mean SD floored at 2.0, guarded std denominator. Initial shift test: mean outside three reference SDs or std ratio outside [0.7,1.3].
- Assess joint gate rejection rate, not each individual threshold alone. Target fewer than 2% clean development rejects and reliable rejection of strong simulations; these are measured acceptance targets, not promises.
- Assess lighting false-trigger rate on a development hold-out. Tune only there; final test fire rate is reported unchanged.
- P1's threshold-fitting wrapper calls P2's frozen interface on the complete development split.

### Storage and feedback

- Cache the exact post-degradation standardized pixels used for inference as lossless PNG. Retain original dimension metadata separately.
- Keep binary/contour masks lossless too. Use JPEG at an explicitly configured display quality for photographic trace overlays, aligned-reference previews and UI heatmap previews. Never extract features or compute measurements from the lossy previews; numerical comparisons use the original arrays.
- Cache metadata includes owner session, category, snapshot, flags, selected alpha, label/idempotency state and expiry.
- Bound cache by bytes, count and TTL, initially 200 MiB, 100 inspections and 30 minutes. Evict assets/metadata coherently and give an expiry response.
- These are independent upper limits, not a promise that 100 inspections fit. Measure actual encoded bytes per inspection and evictions on the target; the byte or TTL limit may evict records first. Keep the cap even if previews compress well.
- Trace reads require session ownership. Audit logs contain no bearer credentials.
- NORMAL and DEFECT always create an audit event. Reject labels on RETAKE/expired records, resolve category from stored metadata, and make repeated idempotency keys safe.
- Default response: `{status:"LOGGED", scope:"audit_only", production_changed:false}`.
- Never change the frozen production bank or official thresholds.

### Optional experimental shadow adaptation

Only implement after the required release gates pass. Keep it disabled by default.

- A NORMAL label may add up to 512 patch descriptors to that session's shadow overlay, at most three corrections, 20 active sessions, with TTL.
- Share immutable calibration descriptor arrays across sessions; only small distance/overlay state is session-specific.
- Shadow re-extraction runs in the same bounded worker and updates a locked/session-versioned snapshot.
- Keep final score, final thresholds, final decision and production calibration status unchanged.
- Return a separate `experimental_shadow` object with `status="experimental_no_calibration_claim"`. UI says “experimental, this session only; official result unchanged.”
- Recalibration using top-K saved calibration descriptors may compute an upper bound, not an unconditional exact score. If updated top-K max drops below the original Kth distance, retain that Kth distance as a conservative upper bound.
- Lossless fp32 calibration descriptors are required for an exact arithmetic claim; fp16 descriptors require an explicit tolerance/approximation label. Quantization error is not automatically conservative.
- Thresholds are nonincreasing for append-only raw 1-NN updates with fixed calibration membership and recipe; test this. Measure update runtime rather than claiming milliseconds.
- Same-image score reduction demonstrates memorization, not generalization. Assess separate held-out images; this distinction is discussed in the memory-bank-correction study. ([Feedback study](https://arxiv.org/html/2608.17775v1))
- Reset clears shadow state only. Serve no adapted PASS/FAIL as the official production result.

### AWS, HTTPS and runtime

Before launch, verify costs and credits, enable billing alerts, record selected region and teardown instructions. No Marketplace or paid-plan upgrade is needed.

- Private S3: `artifacts/`, `traces/`, `feedback/`, optionally approved report summaries.
- EC2 role: read immutable artifacts; write audit prefixes and monitoring logs. Do not grant runtime permission to overwrite production artifacts.
- P1 upload credentials use a separate least-privilege identity and secure local AWS configuration, never Git.
- Select an ARM64 instance only after verifying the wheel, RAM and latency on it. A small 2-vCPU/2-GiB target is a starting hypothesis, not proof it fits or is free.
- Day 3: deploy a minimal container, verify S3 access from the execution context, health endpoint and ARM imports. Configure the DNS hostname and Caddy certificate now; verify HTTPS on the hello-world endpoint so DNS/certificate issues surface before the Day 4 real app. If container role access fails, inspect instance-metadata/network configuration; do not solve it with broad hard-coded keys.
- Docker stage 1: tested Node 24, `npm ci`, frontend build. Stage 2: tested Python 3.11 runtime, runtime dependencies only, backend/core/config/curated data and built frontend.
- Day 4 uses the immutable model snapshot with `evaluation_status="pending"` and `report_bundle_id=null`; do not manufacture evaluation values. Runtime reads approved report JSON separately from the excluded development `eval/results/` directory.
- Day 7, after evaluation approval: publish an immutable report bundle at `reports/<report-bundle-id>/` with summary JSON, its checksums, evaluation-protocol version, split hashes and the exact model/calibration/policy snapshot IDs. Verify/download it and update the deployment's report-bundle configuration.
- This report-only update must not rebuild or overwrite the frozen model snapshot, bank, scores or thresholds. `/api/calibration` now returns `evaluation_status="available"` and the new `report_bundle_id`; the frontend refreshes its panel. If a bundle references the wrong scorer/splits or fails checksum verification, reject it rather than displaying mismatched results.
- Training/evaluation dependencies stay off the runtime server; only approved summary artifacts are deployed.
- Start one uvicorn worker. Set the pixel limit before import and bounded CPU/BLAS thread settings before NumPy import. Measure peak RSS including cached banks and requests.
- Mount artifact snapshot read-only. Configure restart policy, readiness and startup verification.
- Public app: Caddy plus a DNS name and HTTPS before judge-facing deployment. A free hostname may suffice; verify certificate issuance. CloudFront is unnecessary for this design.
- Bind backend only to localhost when Caddy is on the host. Set `--proxy-headers --forwarded-allow-ips=<actual-trusted-proxy-address>`, confirming the network address rather than assuming `172.17.0.1`.
- If topology changes, retest real client IP, forged forwarded headers, rate limits and origin security. Never blindly trust `X-Forwarded-For`.
- Keep artifacts, logs, cache and secrets outside the frontend/public static mount. Mount static frontend routes last, after `/api`.
- Restart test: reboot and verify the same snapshot, readiness, session-expiry behaviour and one real inspection.

Storage lifecycle: audit object expiration eligibility at 14 days, local cache/log cleanup on a schedule, and noncurrent-version expiration if S3 versioning is enabled. S3 expiration is asynchronous, so the UI must not promise exact deletion of every copy at day 14. ([AWS lifecycle documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/lifecycle-expire-general-considerations.html))

Minimum observability: JSON logs with request ID, version, mode, decision/path, queue wait, stage timings, errors and RSS; health check and one useful alert. Dashboard is optional. Do not log uploaded pixels or tokens into general logs.

## Evaluation protocol

Freeze scorer, splits, policy and alpha values before final-test outcomes. Publish raw counts and denominators alongside rates. Report actual laptop/ARM differences rather than assuming identical floating-point scores.

### E1: baseline versus official frozen agent

- Same untouched final images for both.
- Baseline: PASS at/below `t_fail`, FAIL above; document whether the common quality gate is on. Also report a gate-matched comparison to isolate agent benefit.
- Coverage `(PASS+FAIL)/all`; HUMAN/RETAKE separately.
- Defect escapes `defective PASS/all defective`, split FAST_PASS/AGENT_CLEARED.
- Automatic defect recall `defective FAIL/all defective`.
- Normal false-FAIL and review burden, lighting-shift rate and fallback/alignment rates.
- Selective accuracy only beside coverage; do not count HUMAN as an automatic correct decision.
- Counts and Wilson 95% intervals for rates; handle zero denominators as unavailable.
- Use the shared, CI-tested `eval.stats.wilson()` and `clopper_pearson()` helpers. They return two-sided intervals by default and `None` when `n=0`. Do not substitute a one-sided upper bound without explicitly labelling it.
- p50/p95 total latency, stage timing, tool counts, cold start and peak RSS on the target.

### E2: calibration

- Canonical: frozen bank and scorer, train-derived calibration normals → untouched test normals, per preselected alpha. Report upper-tail score exceedance and final false-FAIL separately.
- Wilson and exact binomial intervals with raw counts; note whether an interval is one- or two-sided.
- Twenty reshuffled normal calibration/evaluation partitions are a descriptive sensitivity analysis, not proof that the original deployment distribution is exchangeable.
- Never pool repeated appearances as independent evidence. Report per-split results and descriptive range.
- A small test set cannot empirically certify the target false-alarm bound; the statistical assumption and observed interval are different things.

### E3: robustness

Evaluate deterministic strong simulations plus milder fixed corruptions with the gate on/off, keeping corrupted variants grouped with their original image. Report rejects, escapes, false-FAIL and review burden. Do not count variants as independent new physical samples for confidence claims.

### E4: optional shadow feedback

- Only if shadow adaptation exists.
- Twenty seeded, stratified 70/30 partitions of the final-evaluation pool, with corrections drawn solely from the 70% review partition. Never include development anomalies as final test.
- Up to 1, 2 and 3 actual normal corrections; note when fewer false alarms exist.
- Compare shadow score decisions against frozen baseline on each 30% partition, while retaining the official production result separately.
- Report paired counts, correction counts, held-out false alarms/escapes and descriptive variability. Overlapping seeds do not create independent samples or establish statistical significance.
- Same-image rescore is a separate memorization demonstration.

### E5: optional localisation

- Ground-truth masks are used only here.
- Top-1/top-3 ROI hit rate, using a predeclared overlap criterion.
- Contour IoU and area error only for measured cases, with measurement coverage and count of anomalies never receiving T3.
- Map contours back to standardized coordinates; exclude warp padding. Never quietly drop missed cases from an overall localisation claim.

### Determinism and reproducibility

- Same artifact/input/flags/session baseline produces repeatable scores and deterministic policy decisions under a controlled clock.
- Remove IDs, timestamp, protected asset identifiers, top-level timings and nested trace durations when comparing semantic JSON. Compare underlying image content hashes separately.
- Live wall-clock budget outcomes may differ under load; test/report this separately instead of promising byte-identical real-time responses.
- Record seeds, file hashes, dependencies and machine architecture. Clean-clone rebuild/redeploy must be reproduced by a teammate.

## Person 3 execution tasks

### Application

- API client creates a backend session, attaches its bearer token to protected calls, handles expiry and never stores secrets in `VITE_*`.
- Fetch category/sample options from the API, not hard-coded lists.
- Controls: upload/sample selection, category, simulation flags, preselected alpha, inspect/loading and clear actionable errors.
- Trace: reason, tool, status, timing and budget; asset fetch failures show expired/unavailable rather than a fake image.
- Compare: API input/reference/heatmap assets, authorization-aware fetch, object URL cleanup and opacity slider.
- Decision: text plus colour, score, thresholds, calibration assumptions, measurements or “not measured,” grounded report and snapshot identity.
- Any shadow score is visually separate: “experimental; no calibration claim; official decision unchanged.”
- REAL/MOCK ribbon always visible. A live API failure remains an error.
- Optional history/CSV and trace download stay session-scoped. Neutralize spreadsheet-formula-leading user text in CSV exports.
- Storage notice: “Uploads are processed for inspection and audit. Local cache is short-lived; audit records become eligible for automatic deletion after 14 days. Do not upload confidential images.”
- Keyboard access, readable contrast, mobile layout and empty/error/loading states are required.

### Demo selection and report

- Fix demo sample IDs after the frozen evaluation, preferably from a separate demo pool.
- Demo roles: FAST_PASS, AGENT_CONFIRMED, RETAKE glare, HUMAN conflict/shift; optional alpha flip and experimental shadow.
- Confirm on the deployed ARM host with headroom from each relevant threshold: score vs score thresholds, residual vs residual thresholds, alignment quality vs its cutoff. Units are not interchangeable.
- A flip sample is optional. Try 0.05↔0.02 or 0.10↔0.05, but neither gap is guaranteed wider or populated. If no stable sample exists, omit the live flip. Never synthesize a result.
- Selected demo outcomes are presentation examples, not an unbiased accuracy estimate.
- Report sections: problem/users; architecture; OpenCV implementation; AWS/security/costs from P2; evaluation; failure cases; responsible use and limitations.
- Use measured tables only. Include true URLs for external claims and attribution for samples/templates.

Video target: roughly five minutes, subject to the submission form:
1. Problem and architecture.
2. Real FAST_PASS, confirmed defect and evidence-driven HUMAN case.
3. Simulated glare RETAKE with the measured reason.
4. Optional alpha counterfactual and log-only/experimental feedback, honestly labelled.
5. Coverage, escapes, intervals, failure case and AWS operation.

Draft by Day 8; final and backup screen recording by Day 9. Do not present a mock or replay as live inference.

## Schedule and acceptance gates

| Stage | P1 | P2 | P3 | Acceptance |
|---|---|---|---|---|
| Day 1 | Bootstrap, preprocessing/interface, dataset/splits, export start | Contract/mock and session schema | Scaffold, smoke test, contract types | Reviewed contract and local tests; relative handoff within four working hours |
| Day 2 | Feature parity, bank, development AUROC, scorer choice | Queue/uploads, gate/degrade, storage/session ownership | All result/error states | Real local score/heatmap; fallback decision made |
| Day 3 | Perception snapshot early; templates/evidence/fitting | ARM hello-world, S3 verification, DNS + HTTPS certificate, report adapter | Live API/assets/compare | Complete immutable agent snapshot by handoff; hello-world HTTPS verified |
| Day 4 | State machine and numerical tests | Real HTTPS deployment, security/restart/load checks | Real UI flow | Public real inspection and protected trace |
| Day 5 | Fix failures; optional second category only if on track | Audit feedback, observability; optional shadow | Feedback labels, trace download | Frozen baseline remains unaffected |
| Day 6 | Support evaluation/failure exports | Freeze and run E1–E3 | Architecture/report shell | No test-driven retuning |
| Day 7 | Numerical/report audit | Finish tables; optional E4/E5; publish and deploy immutable report bundle | Report and stable demo pool; refresh calibration panel | Counts, coverage, escapes, latency and limitations visible in the live report panel |
| Day 8 | Clean-clone support | Fresh deployment reproduction | Video draft | Feature freeze |
| Day 9 | Final verification | Keep server healthy | Final video + backup | Release tag and judge-access checks |
| Day 10 | Team submission representative | Monitor endpoint | Verify artifacts | Submit with deadline margin |

If behind, cut optional shadow adaptation, `cashew`, dashboards, E4/E5 and cosmetic extras first. Never cut the real OpenCV pipeline, honest states, ownership protection, HTTPS, basic evaluation, reproducible setup or actual submission checks.

## Submission and operating checklist

- Confirm eligibility, registration, team membership, current form fields and deadlines directly in the competition site.
- Provide judge-accessible code/instructions, architecture, measured report, real demo and backup video according to the actual form. Do not assume a particular licence, video limit or deployment format without checking it.
- Preserve third-party notices and VisA attribution; choose an application-code licence deliberately.
- No credentials, full dataset, model binaries or user uploads in Git.
- Ensure an unfamiliar teammate can run a real inspection from the README.
- Archive snapshot IDs, checksum files, evaluation counts and submitted URLs.
- Keep the endpoint available for the required judging period shown in the portal; document teardown and stop resources afterwards.
- No claim of production safety, guaranteed defect-free PASS, generalization from same-image correction, certified physical dimensions or a promised competition win.

**Final build-ready gate:** feature parity, frozen artifacts, one real UI/API inspection, working HTTPS sessions and ownership tests, actual GitHub negative CI checks, and held-out paired evaluation must all be demonstrated. This document defines the work; it does not claim those gates have already passed.

## Document validation performed

The updated v8.1 scaffold passed 36 local specification checks in a disposable Linux repository. These covered shell execution/syntax, CI YAML and its new evaluation job, placeholder success/remaining-placeholder/search-error cases, ownership/labels, paths with spaces, forbidden extensions and renames, large files, environment files and base-branch script protection. The validation also executed all 10 embedded pytest tests successfully: two bootstrap tests and eight evaluation-statistics test cases.

These results validate the local scaffold behaviour, not the finished product. Windows Git Bash, actual GitHub Actions/branch protection, exported model parity, AWS deployment, authentication, detector performance and measured server latency still require the team's execution evidence. No remote repository or AWS resource was created by preparing this file.
