# SureInspect Benchmark Provenance

## Purpose

This document records which artifacts exist, where reported metrics came from, and which claims are verified. It is not itself a benchmark result.

## Repository State at Audit Start

| Field | Value |
|---|---|
| Audit date | 2026-10-22 IST |
| Working-tree commit | `536618b43ee52d1369011f868a722f5a4fcdb214` |
| Working tree | Clean |
| Measured v2 embedded commit | `536618b43ee52d1369011f868a722f5a4fcdb214` |
| Commit match | Yes, against the audit-start commit; subsequent uncommitted changes are documentation and evaluation safeguards |

The measured v2 result was regenerated from scratch against the audit-start commit after rejecting the older cache. It describes the current detection implementation; subsequent uncommitted changes do not alter detection policy, model weights, thresholds, or runtime inference.

## Runtime Inventory

| Component | Version |
|---|---|
| Python | 3.11.9 |
| OpenCV | 5.0.0 |
| NumPy | 2.4.6 |
| FastAPI | 0.143.3 |
| Pydantic | 2.14.1 |
| Node.js | 24.4.0 |
| npm | 11.4.2 |

## Snapshot Inventory

Snapshot directory: `artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11`

| File | Bytes | SHA-256 |
|---|---:|---|
| `bank.manifest.json` | 1,062 | `940166d3bdad837f2edd3dfef172d422b0fae260f2b62d3dfb8b0346271afe1b` |
| `bank.npz` | 50,631,067 | `5d02a7f53f8116280b072ce556e7edd5436a2653eb67e0b416f1afe5b20c98e7` |
| `calibration.json` | 10,250 | `f6bddbe0f2be695f83654d4fb2118aa4a8d0ca748a49653ae324f8d388310b4b` |
| `manifest.json` | 3,943 | `55b2f93a8c2ecc7b46c90f229525b79d613ea239a12f8f749f693d295cf14ee0` |
| `model.manifest.json` | 734 | `d0fe69daabf11422f06a89155814b7caef65c28da10ac1ca1ac6e9a486787553` |
| `model.onnx` | 99,390,584 | `48642eec4d02294b8e230c04289aadf296fd1b827bafe9a4fc9f9115875a3939` |
| `quality_gate.json` | 325 | `b484aef445e1bf2a988ad1bb24e57570f28468a78c5b166138719d4d1888af94` |
| `SHA256SUMS` | 648 | `ffe22c103e52051bb2cc70912cc14037caeb8540dfd2b5eb1f96d9428161a5aa` |
| `splits.json` | 256,109 | `1bacaf66f05b8965837e2f4f72b82e07ee4ebf3d91ccda353926858427e6ce78` |

## Evaluation Result Inventory

| File | Bytes | SHA-256 | Current disposition |
|---|---:|---|---|
| `eval/results/measured_e1_results_v2.json` | 13,228 | `6744108b974c0d20aa4f018e33546d996440191cf0c61e361584f800ce7dc1d7` | Current canonical regression evidence; generated from the audit-start commit |
| `eval/results/measured_e1_results.json` | 7,159 | `be170b94106503c1b654f43134868b88e27bd726e4d2fea3a15062dbce571367` | Superseded candidate pending provenance confirmation |
| `eval/results/benchmark_summary.json` | 31,088 | `4cd76890ec532263da216379e96f4b6dd54fe952108ed5af43cd457f7abd56b6` | Generated summary linked to the canonical regression result |
| `eval/results/cached_e1_predictions.json` | 236,037 | `72ce1bd9c8fe1e760bd62fba73f912b1edc99ffee5f782bf07595b50a2b114ca` | Per-image cache bound to the measured implementation commit |
| `eval/results/BENCHMARK_REPORT.md` | 15,430 | `36fd6dde24637fbf227b00c80f7b3061b8d13833b592e7413b3c236ee96313bd` | Generated human-readable report, never canonical |

## Current Measured v2 Evidence Candidate

### Development split

- Normals: 297
- Anomalies: 26
- Normal fast PASS: 212
- Normal agent-cleared PASS: 49
- Normal FAIL: 0
- Normal HUMAN: 35
- Normal RETAKE: 1
- Defect fast PASS: 0
- Defect agent-cleared PASS: 0
- Defect FAIL: 3
- Defect HUMAN: 21
- Defect RETAKE: 2
- Defect escapes: 0/26, 0.00%
- Normal automated FAIL: 0/297
- Selective accuracy: 264/264, 100.00%

### Previously seen final-test regression split

- Normals: 100
- Anomalies: 70
- Label in source artifact: `re-run after fixes; split previously seen; not a clean hold-out`
- Normal fast PASS: 61
- Normal agent-cleared PASS: 21
- Normal FAIL: 0
- Normal HUMAN: 18
- Normal RETAKE: 0
- Defect fast PASS: 1
- Defect agent-cleared PASS: 3
- Defect FAIL: 10
- Defect HUMAN: 55
- Defect RETAKE: 1
- Defect escapes: 4/70, 5.71%
- Exact 95% interval for defect escapes: 1.58% to 13.99%
- Normal automated FAIL: 0/100
- Normal HUMAN burden: 18/100, 18.00%
- Total HUMAN burden: 73/170, 42.94%
- Selective accuracy: 92/96, 95.83%

These values describe the audit-start commit and a previously seen regression split. They are canonical regression evidence, but they are not an independent final evaluation.

## Confirmed Claim Conflicts

| Topic | Current measured v2 candidate | Conflicting claim | Location | Required disposition |
|---|---|---|---|---|
| Test normal HUMAN | 18/100 | 17/100 | `FINAL_SUBMISSION_REPORT.md` | Corrected to 18/100 |
| Development normal HUMAN | 35/297, plus one RETAKE | 24/297 | `FINAL_SUBMISSION_REPORT.md` | Corrected to 35/297 |
| Python test result | 171 passed, 5 skipped | 156 tests | `README.md` | Corrected to the verified result on 2026-10-22 |
| Snapshot readiness | `perception_only` | Full agent-ready behavior described | snapshot manifest and reports | Publish a new correctly versioned snapshot rather than rewriting history |
| Snapshot evaluation | `pending` | Final hardened/evaluated wording | snapshot manifest and reports | Publish a new evaluated snapshot after verified evaluation |

## Canonicalization Decision

`eval/results/measured_e1_results_v2.json` is the canonical evidence for the current regression evaluation.

It is machine-readable, validated for count/rate arithmetic, generated from a clean cache against the audit-start commit, and now drives the benchmark report. It is deliberately limited because:

- its final-test split was previously seen;
- the complete current test suite must remain green after the candidate is frozen;
- current snapshot metadata is not agent-ready.

## Verification Record

Verified on 2026-10-22 against the audit-start working tree plus documentation and verifier changes:

- Python: 171 passed, 5 skipped, 2 warnings in 56.24 seconds.
- The admission-queue cancellation test passes outside the restricted Windows sandbox; the sandbox cannot create the `asyncio` loopback socket pair used by the test runtime.
- Frontend: 4 tests passed.
- TypeScript: type checking passed.
- Frontend production build: passed.
- Python dependencies: `pip check` reported no broken requirements.
- OpenAPI: regeneration produced the same file hash.
- Snapshot: all recorded checksums matched.
- Submission archive: required structure, internal checksums, and local OpenCV 5 inference passed.

These local checks do not prove AWS deployment, public endpoint availability, video eligibility, or competition eligibility.

## Required Next Provenance Actions

1. Verify cached per-image predictions against dataset hashes and the embedded commit.
2. Identify the exact generator for every aggregate field.
3. Define a versioned canonical result schema.
4. Add schema and arithmetic validation tests.
5. Generate publication tables from the canonical artifact.
6. Rerun current-commit regression evidence only after P0 test reliability is established.
7. Keep regression evidence separate from a future independent final test.
