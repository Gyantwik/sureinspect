# SureInspect Judge-Ready Execution Plan

## Objective

Prepare SureInspect for the OpenCV AI Competition 2026 without fabricating results, tuning on final-test data, or describing proposed infrastructure as implemented.

## Current Frozen Starting Point

- Audit start: 2026-10-22 IST
- Audit-start commit: `536618b43ee52d1369011f868a722f5a4fcdb214`
- Current base commit: `d749f32d70ecf53cbc7fdd5f07c91f644a52efad`
- Current pipeline fingerprint: `a241045168624da082502a1b36d78e6588902afad264f3cbe166ef6e3cbc8094`
- Working tree: intentionally modified for the current hardening pass; no PR or push performed
- Python: 3.11.9
- OpenCV: 5.0.0
- NumPy: 2.4.6
- FastAPI: 0.143.3
- Pydantic: 2.14.1
- Node.js: 24.4.0
- npm: 11.4.2
- Verified Python test run: 180 passed, 5 skipped, with 2 warnings
- Current snapshot: `pcb1-wide_resnet50_2-256-edc526f4da11`
- Current measured-result commit: `d749f32d70ecf53cbc7fdd5f07c91f644a52efad`
- The measured result matches current inference content through its pipeline fingerprint, including uncommitted changes.

## Non-Negotiable Evaluation Rules

1. Development data may be used for diagnosis, thresholds, policy changes, and controlled experiments.
2. The current `final_test_split` is a regression set because its own label says it was previously seen.
3. No result from that split may be called an untouched holdout.
4. No Markdown file is a source of truth for benchmark metrics.
5. `PASS`, `FAIL`, `HUMAN`, and `RETAKE` must remain separate in reports.
6. Human escalation may be called interception only when the assumed human-review behavior is stated.
7. At most five controlled detection experiments may be run.
8. The final independent test may be run only after code, model, bank, calibration, templates, policy, and evaluation scripts are frozen.

## Phase Status

| Phase | Purpose | Status | Exit condition |
|---|---|---|---|
| P0.1 | Establish trusted baseline and provenance | Complete | Canonical schema, provenance record, discrepancy inventory, and verified commands exist |
| P0.2 | Reconcile conflicting metrics | Complete | All tracked headline metrics are generated from one canonical result |
| P0.3 | Complete reproducibility and test reliability | Complete locally | Full suite, frontend, API drift, checksums, and browser smoke pass |
| P0.4 | Publish agent-ready snapshot metadata | Complete | Snapshot is versioned, checksum-valid, and accurately marked |
| P0.5 | Prove live OpenCV 5 execution | Complete locally | Health, runtime proof, and reproducible inspection prove OpenCV 5 DNN execution |
| P1 | Development-only controlled improvements | Complete for acquisition hardening | Declared regression assertions pass without model/threshold tuning |
| P2 | Freeze and independent final evaluation | External data pending | Current candidate is fingerprinted; genuinely independent data is still required |
| P3 | Agentic Vision proof | Complete locally | Real traces show OpenCV evidence changing later actions |
| P4 | Operator UX and policy consistency | Complete locally | UI, retention, feedback, accessibility, and failure states agree with implementation |
| P5 | Additional category or dataset | Optional | Added only with complete calibration and evaluation |
| P6 | AWS eligibility | Not started | Meaningful live workload and judge-accessible proof exist |
| P7 | Authoritative report generation | Complete locally | Reports map to canonical evidence and distinguish pending external proof |
| P8 | Five-minute video and live demonstration | Not started | Video and endpoint or screen-share are judge-accessible |
| P9 | Submission verification | Final archive rebuild pending | Local judge command passes; archive must be rebuilt after current changes |

## P0 Work Queue

### P0.1 Trusted baseline

- Record repository, runtime, result, snapshot, and split hashes.
- Define the canonical benchmark schema.
- Map every report to the artifact and commit that generated it.
- Verify per-image predictions are sufficient to regenerate aggregate metrics.
- Treat the regenerated measured v2 result as canonical regression evidence, not as an independent final result.

### P0.2 Metric reconciliation

- Inventory every conflict involving escapes, review, rejection, retake, coverage, latency, test counts, and split labels.
- Build a generator that reads the canonical result and produces the publication tables.
- Remove manual metric duplication from submission documents.
- Correct stale claims only after the source and generating command are verified.

### P0.3 Reproducibility

- Record that the admission-queue test stalls only in the restricted Windows sandbox because `asyncio` cannot create its loopback socket pair; it passes outside that sandbox.
- Repeat the full Python suite before the final freeze. Latest verified run: 180 passed, 5 skipped.
- Repeat frontend tests, type checking, and production build before the final freeze. Latest verified runs pass.
- Repeat OpenAPI drift, snapshot checksums, split integrity, archive contents, API health, and one deterministic inspection before the final freeze. OpenAPI, checksum, archive, and local inference checks currently pass.
- Consolidate these checks into one judge command.

### P0.4 Snapshot integrity

- Reconcile `readiness: perception_only`, `tool_policy: null`, and `evaluation_status: pending` with the live agent application.
- Define the agent-ready manifest fields and validator.
- Do not mutate the existing immutable snapshot in place; publish a new versioned snapshot when ready.

### P0.5 OpenCV proof

- Capture OpenCV version, DNN model load, output-layer shapes, engine mode, and live snapshot in reproducible output.
- Demonstrate that the API uses the live engine rather than mock fallback.

## P0 Acceptance Criteria

P0 is complete only when:

- one documented current evidence candidate exists;
- every conflicting claim has an owner and disposition;
- the complete verification command finishes without hanging;
- the snapshot metadata accurately describes the submitted pipeline;
- live OpenCV 5 execution is reproducibly demonstrated;
- no detection threshold or policy was changed before these conditions were satisfied.

## Work Log

### 2026-10-22 — P0 initialization

- Confirmed clean working tree at commit `536618b43ee52d1369011f868a722f5a4fcdb214`.
- Recorded runtime versions.
- Recorded snapshot and result hashes in `docs/BENCHMARK_PROVENANCE.md`.
- Verified the complete Python suite: 171 passed, 5 skipped, 2 warnings in 56.24 seconds.
- Confirmed the admission-queue test passes outside the restricted Windows sandbox; the earlier stall was caused by sandbox denial of `asyncio` loopback socket-pair creation.
- Verified all four frontend tests, TypeScript type checking, and the production build.
- Verified `pip check`, OpenAPI regeneration without drift, snapshot checksums, archive structure, and local OpenCV 5 inference.
- Corrected stale manual test counts; the current verified result is 171 passed and 5 skipped.
- Hardened the archive verifier so it checks the explicitly requested ZIP and does not describe local checks as proof of deployment or eligibility.
- Added `python -m eval.verify_project` as the single local judge command for dependencies, tests, measured-result validation, snapshot checksums, OpenAPI drift, and frontend verification.
- Regenerated all 493 predictions from an empty cache against commit `536618b43ee52d1369011f868a722f5a4fcdb214`.
- Validated the regenerated artifact: test defect escapes 4/70, test normal HUMAN 18/100, development defect escapes 0/26, and development normal HUMAN 35/297.
- Updated the generated benchmark report to derive its headline and per-escape table from commit-matched evidence rather than stale hard-coded prose.
- No detection code, thresholds, policy, model artifact, or benchmark output was changed.

### 2026-10-23 — Agentic and acquisition hardening

- Added bounded 0.60x–1.60x OpenCV alignment recovery while retaining overlap, quality, rotation, and safety gates.
- Added a conservative autonomy rule for strong acquisition-scale shifts so resampling cannot silently erase a small defect and still receive automatic clearance.
- Added measured scale, overlap, and translation evidence to the API and operator UI.
- Added a structured recommendation-only next action with explicit human-confirmation and external-actuation fields.
- Added content-addressed cache and candidate compatibility across inference code, snapshot artifacts, and templates.
- Regenerated all 493 real-image predictions from an empty cache; canonical outcomes were reproduced.
- Added and passed a 15-case acquisition benchmark; the 0.729x known-good screenshot is recovered while transformed known defects never pass.
- Verified 180 Python tests, 4 frontend tests, TypeScript, production build, all consolidated judge gates, and a live browser inspection with no console errors.
