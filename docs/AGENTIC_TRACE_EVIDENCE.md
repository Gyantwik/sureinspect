# Agentic Trace Evidence

These demonstrations are reproducible from the frozen local candidate archive with:

```powershell
.venv\Scripts\python.exe -m eval.verify_submission_zip sureinspect_submission_candidate.zip
```

## Quality rejection halts ordinary inference

The extracted-package test runs `pcb1_good_01` with the deterministic blur degradation. The result is `RETAKE / QUALITY_REJECTED`. The trace records the simulated degradation and quality-gate measurements; no DNN score or evidence-tool decision is used. The verifier fails if the blurred case is anything other than RETAKE.

## Fast path

The packaged normal regression fixture `0011.JPG` returns `PASS / FAST_PASS`, score `1.8109`, after the quality gate and OpenCV DNN scorer. No unnecessary evidence tools execute.

## Evidence-assisted escalation

The packaged ambiguous defect fixture `025.JPG` returns `HUMAN / SUSPECTED_DEFECT`. Its trace records ORB/RANSAC alignment (quality `0.8628`, minimum `0.74`), ROI structural residual (`0.0826`), geometry, three budgeted steps, reasons, and durations. The score alone does not become an automatic verdict; the evidence path changes the action to trained review.

## Budget-safe escalation

`core/tests/test_agent.py` exercises the pure frozen policy with exhausted and timed-out budget states and requires `HUMAN / BUDGET_EXHAUSTED`. The production policy has a maximum of four evidence steps and a two-second evidence-loop ceiling. Unchecked ROIs, timeout, or exhausted budget cannot return PASS.

The trace schema includes tool input context, output status, reason, duration, whether a step counts toward budget, total steps used, and the configured step budget. Downloaded JSON is session-bound and audit-only.

## Complete policy contract

`eval/results/agentic_policy_matrix.json` executes explicit fixtures through the pure frozen policy for all ten advertised paths. `docs/AGENTIC_POLICY_MATRIX.md` provides the judge-readable decision, path, and grounded reason for each fixture. These policy fixtures complement the live image traces and do not pretend to be image-level performance evidence.
