# SureInspect — Devpost Submission Draft

> Submission-safe draft. Replace the AWS and video placeholders only with verified public evidence.

## Inspiration and problem

SureInspect is a human-centered visual inspection assistant for PCB manufacturing. It aims to automate only confident cases, expose the evidence behind each decision, reject unusable captures, and escalate uncertainty instead of hiding it.

## What it does

OpenCV 5 performs image standardization, DNN feature inference, anomaly heatmap generation, ORB/ECC alignment, photometric normalization, structural residual comparison, contour extraction, and geometric measurement. The frozen policy returns four distinct outcomes: PASS, FAIL, HUMAN, or RETAKE. Every inspection includes a bounded, downloadable execution trace.

## Verified regression results

On the 297-normal/26-defect development regression set: 212 FAST_PASS, 49 AGENT_CLEARED, 0 automated FAIL, 35 HUMAN, and 1 RETAKE for normals; 3 FAIL, 21 HUMAN, 2 RETAKE, and 0 PASS for defects. This is 0/26 defect escapes and 264/264 selective accuracy on automated decisions.

On the previously seen 100-normal/70-defect test regression set: normal outcomes were 61 FAST_PASS, 21 AGENT_CLEARED, 0 FAIL, 18 HUMAN, and 0 RETAKE. Defect outcomes were 4 PASS, 10 FAIL, 55 HUMAN, and 1 RETAKE. Defect escapes were 4/70 (exact 95% CI 1.58%–13.99%); FAIL-or-HUMAN interception was 65/70; selective accuracy was 92/96. This set is explicitly not an independent holdout.

## Responsible operation

SureInspect is a prototype decision-support system, not a certified quality-control replacement. HUMAN means a trained reviewer must decide; RETAKE means ordinary inference was stopped because capture quality was unacceptable. Feedback is audit-only and does not alter the live policy. Local assets expire after 30 idle minutes.

## Built with

Python, FastAPI, React, TypeScript, OpenCV 5, ONNX, NumPy, Docker, and Caddy.

## Required evidence before submission

- AWS workload and region: **PENDING VERIFIED DEPLOYMENT**
- Public HTTPS endpoint or arranged screen-share: **PENDING**
- AWS execution log/screenshot: **PENDING**
- Public or unlisted video URL (≤5 minutes): **PENDING TEAM RECORDING**
- Team eligibility and final media-rights confirmation: **PENDING OWNER SIGN-OFF**
