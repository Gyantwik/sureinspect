# SureInspect Judge-Ready Execution Plan

## Objective

Prepare SureInspect for the OpenCV AI Competition 2026 without fabricating results, tuning on final-test data, or describing proposed infrastructure as implemented.

## Current Frozen Starting Point

- Audit start: 2026-10-22 IST
- Working-tree commit: `536618b43ee52d1369011f868a722f5a4fcdb214`
- Commit subject: `docs: generate comprehensive MODEL_AUDIT_REPORT with artifact audit and escape analysis`
- Working tree at audit start: clean
- Python: 3.11.9
- OpenCV: 5.0.0
- NumPy: 2.4.6
- FastAPI: 0.143.3
- Pydantic: 2.14.1
- Node.js: 24.4.0
- npm: 11.4.2
- Verified Python test run: 171 passed, 5 skipped, with 2 warnings
- Current snapshot: `pcb1-wide_resnet50_2-256-edc526f4da11`
- Current measured-result commit: `536618b43ee52d1369011f868a722f5a4fcdb214`
- The measured result matches the audit-start commit. Subsequent uncommitted work is limited to documentation and evaluation safeguards.

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
| P0.2 | Reconcile conflicting metrics | In progress | All tracked headline metrics are generated from one canonical result |
| P0.3 | Complete reproducibility and test reliability | In progress | Full suite finishes repeatedly; frontend, API drift, checksums, archive, and smoke test pass |
| P0.4 | Publish agent-ready snapshot metadata | Not started | Snapshot is complete, versioned, checksum-valid, and accurately marked |
| P0.5 | Prove live OpenCV 5 execution | Not started | Health, logs, and a reproducible inspection prove OpenCV 5 DNN execution |
| P1 | Development-only controlled improvements | Blocked by P0 | Up to five registered experiments are accepted or rejected against fixed criteria |
| P2 | Freeze and independent final evaluation | Blocked by P1 | Immutable candidate and independent final evidence exist |
| P3 | Agentic Vision proof | Not started | Real traces show OpenCV evidence changing later actions |
| P4 | Operator UX and policy consistency | Not started | UI, retention, feedback, accessibility, and failure states agree with implementation |
| P5 | Additional category or dataset | Optional | Added only with complete calibration and evaluation |
| P6 | AWS eligibility | Not started | Meaningful live workload and judge-accessible proof exist |
| P7 | Authoritative report generation | Blocked by canonical results | All reports agree with canonical evidence |
| P8 | Five-minute video and live demonstration | Not started | Video and endpoint or screen-share are judge-accessible |
| P9 | Submission verification | Not started | Clean archive passes every submission gate |

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
- Repeat the full Python suite before the final freeze. Latest verified run: 171 passed, 5 skipped.
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
