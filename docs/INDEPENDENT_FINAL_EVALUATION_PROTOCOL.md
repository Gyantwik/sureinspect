# Independent Final Evaluation Protocol

The currently reported 100-normal/70-defect test result is a **previously seen regression set**, not an untouched holdout. It must never be described as independent or final evidence.

## Freeze before data access

1. Record the candidate Git commit, snapshot checksums, policy version, dependency lock files, and verification output.
2. Do not change thresholds, policy, preprocessing, templates, or evidence tools after opening the independent set.
3. Have a person who did not tune the model provide a new, rights-cleared image manifest with labels concealed until predictions are sealed.

## One permitted run

Run the frozen candidate once. Preserve per-image score, PASS/FAIL/HUMAN/RETAKE decision, decision path, tool trace, timing, and artifact hashes. Reveal labels only after the prediction file is hashed.

Report all four outcomes separately with counts and denominators. Report defect escapes, normal automated failures, review/retake rates, coverage, selective accuracy, Wilson or exact binomial confidence intervals, and mean/median/p95/p99/max latency. Preserve every escape and important false alarm for visual review without using them to revise the submitted candidate.

Until this protocol is completed, all quantitative claims remain regression evidence.

## Current external-data status

`docs/INDEPENDENT_PCB1_DATA_AUDIT.md` records the public-data search. A small rights-cleared external HC-SR04 normal-image development set is available, with zero exact or near-duplicate overlap against the frozen VisA PCB1 manifest. Because it has now informed preprocessing and policy development, contains no independently labelled defective boards, and was not label-blinded, it is robustness evidence only and does not complete this protocol.
