# MASTER BUILD PROMPT: SureInspect Industrial Operator Frontend (Person 3 Lead)

You are the Senior Frontend Architect & Staff React/TypeScript Engineer on the **SureInspect** project (OpenCV AI Competition 2026).
Your task is to implement the complete, production-ready frontend web application inside a single directory: `/frontend`.

Once completed, your code will be zipped, unpacked into the repository root at `/frontend`, and validated against our strict GitHub Actions CI pipeline running **Node.js 24**.

---

## 1. CRITICAL CI/CD & REPOSITORY CONSTRAINTS
Our GitHub Actions workflow (`.github/workflows/ci.yml`) executes these exact commands on Node 24:
```bash
test -f frontend/package-lock.json
cd frontend
npm ci
npm run build
npm run test -- --run
npm run types:check
```

### Absolute Rules:
1. **Strict Directory Boundary:** All frontend code must reside inside `/frontend/`. No edits to backend, core, or training folders.
2. **Mandatory Scripts in `frontend/package.json`:**
   - `"build": "tsc -b && vite build"`
   - `"test": "vitest"`
   - `"types:check": "tsc --noEmit"`
   - `"generate:types": "openapi-typescript ../contracts/openapi.json -o src/services/api-types.ts"`
3. **Valid Lockfile:** You must instruct how to generate or include the exact `package-lock.json` so `npm ci` never fails.
4. **Zero Compilation/Type Errors:** TypeScript must compile cleanly (`tsc --noEmit` returns exit code 0). No implicit `any` types.
5. **Passing Tests:** Vitest unit tests in `src/tests/` must pass with `npm run test -- --run`.

---

## 2. BACKEND API SPECIFICATION & INTEGRATION RULES
The backend is a FastAPI service running at `http://127.0.0.1:8000`. The Vite development server must proxy all `/api` requests to `http://127.0.0.1:8000`.

### 2.1 Authentication & Session Handling
- **Session Generation:** On first load, check `sessionStorage` for `session_token`. If missing or expired, call `POST /api/session`.
  - Response: `{ "session_token": "<token>", "expires_at": "<ISO-8601>" }`
  - Store token **ONLY in `sessionStorage`** (never in `localStorage`, URL parameters, or terminal logs).
- **Idle TTL:** 30-minute idle TTL. All protected requests include: `Authorization: Bearer <session_token>`.
- **Session Expiry (401):** When any endpoint returns `401` with `code: "SESSION_EXPIRED"`, automatically refresh the session and notify the operator via a toast banner.
- **Session Reset:** `POST /api/session/reset` resets the session's experimental shadow state.

### 2.2 ⚠️ CRITICAL BROWSER REQUIREMENT: Protected Asset Fetching
- Assets (`input_standardized`, `reference_aligned`, `anomaly_heatmap`, `contour_mask`, and trace `overlay_url`) are protected routes (`GET /api/assets/{id}`) requiring the Bearer token.
- **NEVER use bare `<img src="/api/assets/..." />` tags!** HTML `<img>` tags cannot send HTTP Bearer headers and will fail with `401 Unauthorized`.
- **Mandatory Blob Pattern:** Implement an authenticated asset loader hook (`useAssetUrl`):
  1. `fetch(url, { headers: { Authorization: `Bearer ${token}` } })`
  2. `const blob = await response.blob();`
  3. `const objectUrl = URL.createObjectURL(blob);`
  4. Always invoke `URL.revokeObjectURL(objectUrl)` when the component unmounts or the image changes to prevent browser memory leaks.
  5. If fetching fails (e.g., 404 expired asset), display an "Asset Unavailable or Expired" placeholder card.

### 2.3 API Endpoints Reference

| Endpoint | Method | Auth | Payload / Parameters | Response |
| :--- | :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | Public | None | `{ status: "healthy"\|"degraded", engine: "live"\|"mock", model_version: string, enabled_categories: string[] }` |
| `/api/session` | `POST` | Public | None | `{ session_token: string, expires_at: string }` |
| `/api/categories` | `GET` | Public | None | `{ categories: CategoryInfo[] }` |
| `/api/samples` | `GET` | Public | `?category=pcb1` | `{ samples: SampleItem[] }` (`id`, `category`, `display_name`, `ground_truth`, `attribution`, `image_url`, `role`) |
| `/api/samples/{id}/image`| `GET` | Public | Path `id` | Raw image stream (for sample selector thumbnails) |
| `/api/calibration/{category}` | `GET` | Public | Path `category` | `{ category, t_pass, t_fail, alpha_pass, alpha_fail, n_calib, evaluation_status, evaluation_summary }` |
| `/api/session/reset` | `POST` | Bearer | None | `{ status: "RESET", message: string, snapshot_id: string }` |
| `/api/inspect` | `POST` | Bearer | `multipart/form-data`: `file` (File) OR `sample_id` (string), `category` (string), `alpha_fail` (number), `simulate_blur` (bool), `simulate_darken` (bool), `simulate_glare` (bool), optional `path` (string) | `InspectionResponse` (Master payload) |
| `/api/trace/{id}` | `GET` | Bearer | Path `id` | `InspectionResponse` (Full stored record) |
| `/api/assets/{id}` | `GET` | Bearer | Path `id` | Protected image stream (PNG/JPEG) |
| `/api/feedback` | `POST` | Bearer | JSON `{ inspection_id, user_label: "NORMAL"\|"DEFECT", idempotency_key }` | `{ status: "LOGGED", scope: "audit_only", production_changed: false }` |

### 2.4 Error Envelope
All error responses adhere strictly to:
```json
{
  "error": {
    "code": "BAD_REQUEST" | "UNAUTHORIZED" | "SESSION_EXPIRED" | "NOT_FOUND" | "CONFLICT" | "PAYLOAD_TOO_LARGE" | "UNSUPPORTED_MEDIA_TYPE" | "VALIDATION_ERROR" | "RATE_LIMIT_EXCEEDED" | "INTERNAL_SERVER_ERROR" | "SERVICE_UNAVAILABLE",
    "message": "Human readable error description",
    "request_id": "req_xxxx"
  }
}
```

---

## 3. UI ARCHITECTURE: 3-PANEL INDUSTRIAL OPERATOR CONSOLE

Design a modern, high-contrast dark industrial theme (`slate-900`/`zinc-900`, cyan/emerald/amber/rose accents, Lucide icons, monospace stats).

### 3.1 Header & Compliance Ribbon
1. **Engine Ribbon Badge:**
   - If `engine === "mock"`: Amber badge `MOCK ENGINE (Synthetic Evaluation)`.
   - If `engine === "live"`: Emerald badge `LIVE OPENCV INFERENCE`.
2. **Snapshot Identity:** Displays active model version (e.g., `wrn50_v1_frozen`), calibration version, and bank version.
3. **Session Status:** Displays session token status, expiry countdown, and "Reset Session" button.
4. **Mandatory Storage Notice:**
   > *"Uploads are processed for inspection and audit. Local cache is short-lived; audit records become eligible for automatic deletion after 14 days. Do not upload confidential images."*

### 3.2 Panel 1: Control & Ingestion (Left Panel, ~25% width)
- **Category Selector:** Populated dynamically from `/api/categories` (default: `pcb1`).
- **Input Source Tabs:**
  - **Curated Samples Tab:** Visual grid of demo items from `/api/samples`. Displays thumbnail, label (`good`/`bad`), and demo role badge (`FAST_PASS`, `AGENT_CONFIRMED`, `RETAKE`, `HUMAN`).
  - **Upload Tab:** Drag-and-drop zone with client-side file size (< 10 MB) and format (PNG/JPEG) validation.
- **Risk Budget ($\alpha_{\text{fail}}$):** Segmented buttons for allowed FPR values: `0.01`, `0.02`, `0.05`, `0.10`.
- **Degradation Stress-Test Toggles:** Checkboxes for `simulate_blur`, `simulate_darken`, `simulate_glare`.
- **Action:** Primary "Inspect Surface" button with loading spinner and live elapsed timer.

### 3.3 Panel 2: Visual Evidence & Interactive Studio (Center Panel, ~45% width)
- **Multi-Layer Canvas:**
  - Layer 0: Standardized Input (`input_standardized`).
  - Layer 1: Aligned Golden Reference (`reference_aligned`).
  - Layer 2: Anomaly Heatmap (`anomaly_heatmap`) with continuous Opacity Slider (0% - 100%).
  - Layer 3: Defect Contour Mask & Bounding Box (`contour_mask`).
- **Interactive Tool Step Timeline Viewer:**
  - Clicking any step in the execution trace (e.g. `T1_ALIGN_REFERENCE`, `T2_ROI_COMPARE`, `T3_ROI_ZOOM_MEASURE`, `T4_LIGHT_NORMALIZE`) swaps the canvas to render that tool's specific `overlay_url`.
- **Alignment Metrics Card:** Displays Template ID, Method, Quality Score $Q$ vs $Q_{\min}$, and Rotation Angle $\theta^\circ$.

### 3.4 Panel 3: Decision Card, Trace & Audit (Right Panel, ~30% width)
- **Primary Decision Card:**
  - `PASS`: Emerald Green badge.
  - `FAIL`: Crimson Red badge.
  - `HUMAN`: Amber Warning badge (Shows `suggested_decision` if present).
  - `RETAKE`: Orange/Purple badge (Quality Gate Rejection with actionable reason: blur variance, glare fraction).
  - Decision Path Tag (e.g., `FAST_PASS`, `AGENT_CLEARED`, `AGENT_CONFIRMED`, `SUSPECTED_DEFECT`, `EVIDENCE_CONFLICT`, `LIGHTING_SHIFT_REVIEW`, `ALIGN_FAILED`, `INCONCLUSIVE`, `BUDGET_EXHAUSTED`, `QUALITY_REJECTED`).
- **Anomaly Score Gauge:** Horizontal bar showing score $s$ relative to $t_{\text{pass}}$ and $t_{\text{fail}}$.
- **Budget Meter & Timing:** "Steps Used: X / 4 (Tool Budget)". Latency breakdown: Quality Gate ms, Perception ms, Tools ms, Total ms.
- **Chronological Trace Steps:** Ordered list of steps with tool name, reason, duration, and status.
- **Grounded Verification Report:** Formatted synthesis text explaining the verdict.
- **Measurements Table:** ROI bounding box, residual mean, area %, elongation, shape classification (`elongated`, `compact`, `irregular`).
- **Counterfactual Risk Analysis Table:** Alternate outcomes under $\alpha_{\text{fail}} \in \{0.01, 0.02, 0.05, 0.10\}$.
- **Experimental Shadow Score Card (when present):** Distinct card with notice: *"Experimental Shadow State (Session Only). Official decision unchanged."*
- **Human-in-the-Loop (HITL) Feedback:**
  - Dual action buttons: "Verify as Normal" vs "Confirm Defect".
  - Submits unique UUID `idempotency_key` to `POST /api/feedback`. Displays confirmation: *"Audit event logged. Production weights immutable."*
  - Automatically disabled on `RETAKE` or expired records.
- **Trace Export:**
  - "Download Trace JSON" & "Export CSV" buttons.
  - **CSV Formula Sanitization:** Any text starting with `=`, `+`, `-`, or `@` must be prepended with a single quote `'` to prevent spreadsheet formula injection.

---

## 4. CODEBASE DIRECTORY STRUCTURE
Implement the code across these exact paths in `/frontend`:

```
frontend/
├── package.json
├── package-lock.json (or instructions to generate)
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── services/
│   │   ├── api-types.ts         # Complete TypeScript schemas matching OpenAPI
│   │   ├── api-client.ts        # Fetch client attaching Bearer & handling 401
│   │   ├── session.ts           # Token storage in sessionStorage
│   │   └── asset-loader.ts      # Authenticated blob loader + URL.revokeObjectURL
│   ├── hooks/
│   │   ├── useSession.ts        # Auto session acquisition & refresh
│   │   ├── useInspection.ts     # Inspection mutation, loading, & error state
│   │   ├── useCategories.ts     # Categories & calibration queries
│   │   └── useAssetUrl.ts       # Hook wrapping asset-loader
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx       # Logo, Engine ribbon, Snapshot ID, Session status
│   │   │   ├── NoticeBanner.tsx # Mandatory storage & compliance notice
│   │   │   └── Shell.tsx        # 3-panel responsive layout
│   │   ├── controls/
│   │   │   ├── CategorySelect.tsx
│   │   │   ├── SamplePicker.tsx
│   │   │   ├── UploadDropzone.tsx
│   │   │   ├── RiskSelector.tsx
│   │   │   └── DegradationToggles.tsx
│   │   ├── viewer/
│   │   │   ├── EvidenceStudio.tsx
│   │   │   ├── LayerControls.tsx
│   │   │   └── AlignmentMetrics.tsx
│   │   ├── decision/
│   │   │   ├── DecisionCard.tsx
│   │   │   ├── ScoreMeter.tsx
│   │   │   ├── GroundedReport.tsx
│   │   │   ├── MeasurementsTable.tsx
│   │   │   ├── CounterfactualView.tsx
│   │   │   └── ShadowScoreCard.tsx
│   │   ├── trace/
│   │   │   ├── TraceTimeline.tsx
│   │   │   ├── BudgetMeter.tsx
│   │   │   └── TimingSummary.tsx
│   │   ├── feedback/
│   │   │   └── HITLFeedback.tsx
│   │   └── common/
│   │       ├── ErrorAlert.tsx
│   │       ├── Badge.tsx
│   │       └── Spinner.tsx
│   └── tests/
│       ├── api-client.test.ts   # Vitest unit test for auth headers & session refresh
│       └── DecisionCard.test.tsx# Vitest unit test for PASS/FAIL/HUMAN/RETAKE rendering
```

---

## 5. COMPLETE TYPESCRIPT SCHEMAS (`src/services/api-types.ts`)
Ensure your `api-types.ts` contains these exact schemas matching backend Pydantic models:

```typescript
export type DecisionEnum = "PASS" | "FAIL" | "HUMAN" | "RETAKE";

export type DecisionPathEnum = 
  | "QUALITY_REJECTED"
  | "FAST_PASS"
  | "AGENT_CLEARED"
  | "AGENT_CONFIRMED"
  | "SUSPECTED_DEFECT"
  | "EVIDENCE_CONFLICT"
  | "LIGHTING_SHIFT_REVIEW"
  | "ALIGN_FAILED"
  | "INCONCLUSIVE"
  | "BUDGET_EXHAUSTED";

export type ToolEnum =
  | "SIMULATED_DEGRADATION"
  | "QUALITY_GATE"
  | "PATCHCORE_PERCEPTION"
  | "T1_ALIGN_REFERENCE"
  | "T2_ROI_COMPARE"
  | "T3_ROI_ZOOM_MEASURE"
  | "T4_LIGHT_NORMALIZE"
  | "T5_NEXT_ROI_COMPARE";

export interface SnapshotInfo {
  model_version: string;
  bank_version: string;
  calibration_version: string;
  policy_version: string;
}

export interface ImageMetadata {
  original_size: [number, number];
  standardized_size: [number, number];
  scale_x: number;
  scale_y: number;
}

export interface ThresholdsInfo {
  t_pass: number | null;
  t_fail: number | null;
  alpha_pass: number;
  alpha_fail: number;
  n_calib: number;
}

export interface CalibrationInfo {
  status: "frozen_assumption_dependent" | "not_applicable_lighting_shift" | "not_evaluated";
  claim?: "normal_marginal_upper_tail" | null;
  assumptions: string;
}

export interface QualityGateResult {
  status: "PASSED" | "FAILED";
  action?: "REQUEST_RETAKE" | null;
  reason: string;
  blur_variance: number;
  blur_threshold: number;
  dark_fraction: number;
  bright_fraction: number;
  glare_fraction: number;
  glare_threshold: number;
  l_mean: number;
  exposure_ok: boolean;
  lighting_shift_detected: boolean;
}

export interface AlignmentInfo {
  template_id: string;
  method: string;
  quality: number;
  q_min: number;
  rotation_deg: number;
}

export interface MeasurementStats {
  roi_bbox: [number, number, number, number];
  roi_source: "threshold" | "peak_fallback";
  rois_checked: number;
  rois_total: number;
  residual_mean: number;
  r_low: number;
  r_high: number;
  area_pct?: number | null;
  area_denominator: "image" | "part_mask";
  elongation?: number | null;
  shape_class?: "elongated" | "compact" | "irregular" | null;
}

export interface AssetsInfo {
  input_standardized: string;
  reference_aligned: string | null;
  anomaly_heatmap: string | null;
  contour_mask: string | null;
}

export interface TraceStep {
  step: number;
  tool: ToolEnum;
  reason: string;
  status: string;
  counts_toward_budget: boolean;
  duration_ms: number;
  overlay_url: string | null;
}

export interface TimingInfo {
  quality: number;
  perception: number;
  tools: number;
  total: number;
}

export interface CounterfactualItem {
  alpha_fail: number;
  t_fail: number;
  decision: DecisionEnum | null;
  decision_path: DecisionPathEnum | null;
  needs_rerun: boolean;
}

export interface ExperimentalShadowInfo {
  status: "experimental_no_calibration_claim";
  shadow_version: string;
  score: number;
  shadow_t_pass: number;
  shadow_t_fail: number;
  approximate: boolean;
  note: string;
}

export interface InspectionResponse {
  schema_version: "1.0";
  inspection_id: string;
  request_id: string;
  timestamp: string;
  engine: "live" | "mock";
  category: string;
  snapshot: SnapshotInfo;
  image: ImageMetadata;
  decision: DecisionEnum;
  decision_path: DecisionPathEnum;
  suggested_decision?: "PASS" | "FAIL" | null;
  anomaly_score?: number | null;
  thresholds: ThresholdsInfo;
  calibration: CalibrationInfo;
  quality_gate: QualityGateResult;
  alignment?: AlignmentInfo | null;
  measurements?: MeasurementStats | null;
  assets: AssetsInfo;
  execution_trace: TraceStep[];
  steps_used: number;
  step_budget: number;
  timing_ms: TimingInfo;
  grounded_report: string;
  counterfactuals: CounterfactualItem[];
  experimental_shadow?: ExperimentalShadowInfo | null;
}

export interface CategoryInfo {
  id: string;
  display_name: string;
  enabled: boolean;
  alpha_pass: number;
  alpha_fail_default: number;
  alpha_fail_options: number[];
  n_calib: number;
  sample_count: number;
}

export interface SampleItem {
  id: string;
  category: string;
  display_name: string;
  ground_truth: "good" | "bad";
  attribution: string;
  image_url: string;
  role?: string | null;
}

export interface FeedbackRequest {
  inspection_id: string;
  user_label: "NORMAL" | "DEFECT";
  idempotency_key: string;
}

export interface FeedbackResponse {
  status: "LOGGED" | "UPDATED" | "DUPLICATE";
  scope: "audit_only" | "session_shadow";
  production_changed: boolean;
  message?: string | null;
}
```

---

## 6. DELIVERABLE INSTRUCTIONS FOR CLAUDE

Provide the complete file implementations with **zero placeholders, no `// TODO` stubs, and no omitted lines**.

### Output Structure:
1. Provide the complete `package.json`, `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`, and `index.html`.
2. Provide all utility services: `api-client.ts`, `session.ts`, `asset-loader.ts`.
3. Provide custom hooks: `useSession.ts`, `useInspection.ts`, `useCategories.ts`, `useAssetUrl.ts`.
4. Provide all UI components across `components/layout/`, `components/controls/`, `components/viewer/`, `components/decision/`, `components/trace/`, and `components/feedback/`.
5. Provide passing Vitest test suites in `src/tests/` to guarantee that running `npm run test -- --run` passes 100%.
6. Provide exact shell instructions to package this output into `frontend.zip` so it can be unzipped cleanly into `c:\Users\Khush\OneDrive\Desktop\OpenCV\frontend`.
