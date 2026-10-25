# SureInspect Submission Checklist

## Eligibility Gates

- [ ] OpenCV 5 performs substantive image analysis in the judged workload.
- [ ] A meaningful component actually runs on AWS.
- [ ] AWS execution evidence is preserved.
- [ ] A judge-accessible endpoint or arranged live screen-share exists.
- [ ] A public or unlisted video is no longer than five minutes.
- [ ] Every team member satisfies competition eligibility requirements.
- [ ] Dataset, model, image, and media rights are documented.

## Evidence Integrity

- [x] One canonical machine-readable regression result is identified.
- [x] Canonical regression result records the implementation commit, dataset, snapshot ID, split hash, and model/bank/calibration/quality-gate hashes.
- [x] Per-image predictions regenerate all aggregate metrics through the measured runner and arithmetic validator.
- [x] README headline metrics match the canonical regression result.
- [x] Final report headline tables match the canonical regression result; historical experiment notes are explicitly non-canonical.
- [x] Benchmark report metrics are generated from the canonical regression result.
- [x] Demo script headline metrics match the canonical regression result.
- [x] Devpost text metrics match the canonical result.
- [x] PASS, FAIL, HUMAN, and RETAKE are reported separately.
- [x] Counts and denominators accompany percentages.
- [x] Confidence intervals are included.
- [x] Human-interception assumptions are explicit.
- [x] Previously seen data is never called untouched or held out.

## Reproducibility

- [ ] Clean-environment dependency installation succeeds.
- [x] Python dependency check succeeds.
- [x] Complete Python test suite finishes without hanging outside the restricted Windows sandbox: 180 passed, 5 skipped.
- [x] Frontend tests pass: 4 passed.
- [x] TypeScript type checking passes.
- [x] Frontend production build passes.
- [x] OpenAPI drift check passes.
- [x] Snapshot checksum verification passes.
- [x] Dataset split structure, counts, unique paths, and near-duplicate group isolation pass; full source-file re-hash requires the intentionally pruned source dataset.
- [x] Deterministic sample inspection passes locally for the packaged normal and defect fixtures.
- [x] `sureinspect_submission_candidate.zip` passes structure, checksum, secure extraction, and extracted packaged-runtime verification (rebuild after the current evidence changes before submission).
- [x] One judge verification command runs the local reproducibility and evidence checks: `python -m eval.verify_project`.

## Snapshot and Live Engine

- [x] Final snapshot is versioned and immutable.
- [x] Snapshot readiness accurately states `agent_ready`.
- [x] Policy version is recorded.
- [x] Evidence thresholds are recorded.
- [x] Tool budget and timeout are recorded.
- [x] Evaluation status is accurate.
- [x] Live health output selects the agent-ready snapshot.
- [x] Mock, live, and unknown modes are unmistakably labeled.
- [x] OpenCV version is captured.
- [x] OpenCV DNN execution is demonstrated.

## Evaluation

- [x] Development baseline is frozen; no post-freeze model/policy experiment was selected.
- [ ] No more than five controlled experiments are run.
- [ ] Each experiment changes one primary factor.
- [ ] Each experiment has predeclared acceptance criteria.
- [ ] Negative experiments are retained.
- [x] Ablations cover the score-only and full-agent paths.
- [x] Latency includes mean, median, p95, p99, and maximum.
- [x] Coverage and selective accuracy are defined.
- [x] Final candidate is frozen before independent testing.
- [ ] Independent final test is run once.
- [x] Every recorded defect escape has generated visual evidence; important false-alarm evidence is represented by the zero automated normal FAIL count.

## Agentic Vision Evidence

- [x] Workflow diagram matches the real code.
- [x] At least one trace shows OpenCV evidence changing the next tool call.
- [x] At least one trace shows quality rejection halting ordinary inference.
- [x] At least one trace shows budget or uncertainty causing human escalation.
- [x] Trace records inputs, outputs, reason, duration, and budget state.
- [x] All ten advertised decision paths have deterministic policy fixtures; principal live paths additionally have image demonstrations.

## UX and Responsible Operation

- [x] PASS, FAIL, HUMAN, and RETAKE meanings are clear.
- [x] Score and thresholds are understandable.
- [x] ROI, reference, heatmap, difference, and contour evidence are visible where available.
- [x] Loading, upload, session, and asset failures have recovery paths; queue status is API-driven.
- [x] Keyboard and accessible-label checks pass at code/test level; manual assistive-technology QA remains recommended.
- [x] Verdicts do not rely on color alone.
- [x] Feedback behavior is accurately labeled.
- [x] Retention wording matches actual 30-minute idle in-memory storage behavior.
- [x] Security limits and human oversight are documented.
- [x] Production-readiness claims are appropriately limited.

## AWS

- [ ] Live OpenCV workload executes on AWS.
- [ ] Service, region, compute configuration, and deployment commands are documented.
- [ ] HTTPS or an arranged live demonstration is available.
- [ ] Logs prove a real inspection executed on AWS.
- [ ] IAM follows least privilege.
- [ ] Credentials and private keys are absent from the repository and archive.
- [ ] Health checks and rollback steps are documented.
- [ ] Cost controls are configured and evidenced.
- [x] Proposed services are not presented as implemented.

## Documentation and Video

- [ ] Technical report contains every required competition section.
- [ ] Architecture diagram renders correctly.
- [x] Implemented, verified, inferred, and proposed elements are distinguished.
- [x] Failure cases and limitations are prominent.
- [ ] README setup commands work from a clean environment.
- [ ] Video shows the team, application, architecture, agent trace, results, failures, AWS evidence, and responsible use.
- [ ] Video link is judge-accessible.

## Archive Hygiene

- [ ] No credentials or private keys are present.
- [x] No virtual environment is included.
- [x] No dependency cache is included.
- [x] No unlicensed dataset is redistributed.
- [x] No stale benchmark file is included.
- [x] No contradictory canonical metric remains.
- [x] Required local artifacts are included.
- [x] Archive size is measured at 141.06 MiB; confirm the target platform's upload limit before submission.
- [ ] Archive contents match the documented commit.

## Current Blocking Findings

- [x] Resolve test defect escapes: fresh current-code rerun gives 4/70.
- [x] Resolve test normal HUMAN: fresh current-code rerun gives 18/100.
- [x] Resolve development defect escapes: fresh current-code rerun gives 0/26.
- [x] Replace stale manual test counts with the verified result: 180 passed, 5 skipped.
- [x] Resolve measured-result compatibility: fresh rerun embeds pipeline fingerprint `a24104516862...`, including uncommitted inference files and immutable artifacts.
- [x] Diagnose the earlier admission-queue test stall: the restricted Windows sandbox blocks `asyncio` loopback socket-pair creation; the test passes outside it.
- [x] Replace perception-only/pending snapshot status with a newly published, verified agent-ready snapshot.
- [x] Explicitly retain regression-only claims and document the independent-test protocol.
- [ ] Implement and prove meaningful AWS execution.
