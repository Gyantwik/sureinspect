# SureInspect Controlled Experiment Log

## Experiment Policy

- Maximum experiments: five
- Allowed tuning data: development data only
- Current final-test split: regression evidence only; prohibited for experiment selection
- Baseline must be frozen before Experiment 1
- One primary factor may change per experiment
- Negative results must be retained
- The accepted candidate must remain reproducible from a versioned configuration

## Default Acceptance Criteria

An experiment may be kept only when all applicable conditions hold:

1. It improves a predefined development failure mode or produces important negative evidence.
2. Automated FAIL on verified development normals remains zero unless a different ceiling was declared before execution.
3. Normal-board intervention does not rise by more than three absolute percentage points from the verified baseline unless a stronger trade-off was declared before execution.
4. p95 latency does not rise by more than 20% unless justified before execution.
5. Incomplete verification, timeout, budget exhaustion, and unchecked ROIs cannot return PASS.
6. Quality-rejected inputs cannot enter ordinary scoring logic.
7. Results include per-image changes, counts, denominators, and uncertainty.
8. The experiment does not use the previously seen final-test split for tuning or selection.

## Baseline Registration

Status: frozen for this submission cycle.

The canonical regression evidence is `eval/results/measured_e1_results_v2.json`, rerun from an empty cache at commit `d749f32d70ecf53cbc7fdd5f07c91f644a52efad` with pipeline fingerprint `a241045168624da082502a1b36d78e6588902afad264f3cbe166ef6e3cbc8094`. The fingerprint includes current inference files, snapshot artifacts, and templates, including uncommitted inference changes. This hardening pass changed acquisition recovery and conservative autonomy behavior, not model weights or conformal thresholds; its dedicated 15-case acquisition benchmark is recorded separately.

The baseline registration must contain:

- Git commit
- Snapshot ID and hashes
- Policy version
- Dataset and split hashes
- Development counts
- Configuration
- Runtime versions
- Latency summary
- Per-image prediction file
- Command used

## Experiment Register

| ID | Hypothesis | Primary change | Predeclared acceptance criteria | Status | Decision |
|---|---|---|---|---|---|
| E1 | Adaptive ROI tightening | None | Not run: candidate frozen | Deferred | No change |
| E2 | Localized high-percentile residual | None | Not run: candidate frozen | Deferred | No change |
| E3 | Multi-scale ROI comparison | None | Not run: candidate frozen | Deferred | No change |
| E4 | Component-aware masks | None | Not run: candidate frozen | Deferred | No change |
| E5 | Calibrated evidence fusion | None | Not run: candidate frozen | Deferred | No change |

## Required Experiment Record

For each experiment, add a complete section containing:

- ID and date
- Git commit
- Hypothesis
- Targeted images and failure mechanism
- Primary factor changed
- Exact configuration difference
- Development-only data declaration
- Predeclared acceptance criteria
- Command
- Runtime and hardware
- Baseline metrics
- Experimental metrics
- Per-image decision changes
- Normal intervention delta
- Defect escape delta
- Automated FAIL delta
- RETAKE delta
- Coverage delta
- Selective-accuracy delta
- Mean and p95 latency delta
- Safety-invariant results
- Failure analysis
- Decision: keep, reject, or refine
- Decision rationale
- Artifact hashes

## Candidate Hypotheses

These are candidates, not authorized experiments until registered against the verified baseline:

1. Adaptive ROI tightening may reduce background dilution around micro-defects.
2. Localized high-percentile residual aggregation may recover defects hidden by ROI-wide means.
3. Multi-scale ROI comparison may improve robustness across missing components, melt defects, and scratches.
4. Component-aware foreground masks may suppress stable PCB background.
5. Interpretable calibrated fusion may combine score, alignment, localized residual, contour evidence, and ROI completeness more safely than a single cutoff.

## Final-Test Guard

No experiment section may include final-test performance as an acceptance criterion. Final-test results may be added only after the candidate is frozen, and they may not be used to revise the accepted experiment decision.
