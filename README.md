# SureInspect

SureInspect is a **Selective Inspection & Operator Triage System** for surface-mount electronics manufacturing (SMT/PCB) built natively on **OpenCV 5 DNN** and **Split-Conformal Risk Control**.

Each decision includes a bounded OpenCV evidence trace and a structured, recommendation-only next action. Strong capture-scale shifts trigger conservative clearance rules so geometry can be recovered without silently treating resampled screenshots as calibrated sensor frames.

> **Key Operational Finding:** On the previously seen test regression split, SureInspect eliminated automated normal-board scrap (0/100), routed 18/100 normal boards to human review, and passed 4/70 defects. The cautious binary baseline rejected 27/100 normal boards and also passed 4/70 defects. Interception assumes human reviewers catch HUMAN-routed defects.

- **OpenCV AI Competition 2026 Entry**
- **Repository:** https://github.com/Gyantwik/sureinspect.git
- **Core Technology:** OpenCV 5 DNN (`cv2.dnn.readNetFromONNX` WideResNet-50) + Conformal Decision Banding + Multi-Tool Physical Verification (ORB/ECC alignment, multi-ROI photometric residuals, rotated-bounding box geometry).
- **Evaluation Guide:** See `eval/AI_REVIEW_GUIDE.md` for a comprehensive architectural and evaluation walk-through.
- **Empirical Benchmark Results:** See `eval/results/BENCHMARK_REPORT.md` for measured results across 493 authentic Amazon VisA `pcb1` circuit boards (Experiments E1–E4).

### Operational Trade-Off Comparison (Previously Seen Test Regression Split: 100 Normals, 70 Anomalies)

| Inspection Strategy | Good-Board Review / Scrap Rate | Defect Escapes (Bad $\to$ PASS) | Defect Interception (FAIL + HUMAN) | Operational Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Cautious Baseline ($t_{\text{pass}}=2.04$)** | 27.00% (27/100 scrap) | 5.71% (4/70) [1.58%, 13.99%] | 94.29% (66/70) | Conservative single cutoff; halts 27% of good boards as false alarms. |
| **Permissive Baseline ($t_{\text{fail}}=2.29$)** | 9.00% (9/100 scrap) | 27.14% (19/70) [17.80%, 38.97%] | 72.86% (51/70) | Higher throughput; more defect escapes. |
| **SureInspect v1 (Pre-Remediation)** | 22.00% (14 scrap + 8 review) | 7.14% (5/70) [3.09%, 15.69%] | 91.43% (64/70) | Inconsistent tool routing; high automated scrap. |
| **SureInspect v2 (Current Measured Pipeline)** | **18.00%** (0 scrap + 18 review) | **5.71%** (4/70) [1.58%, 13.99%] | **92.86%** (65/70) | Eliminates automated normal-board scrap while routing uncertainty to review. |

> **Production-workload implication:** When normal boards dominate a production stream, operator workload is driven mainly by normal routing. In these measured splits, SureInspect routes 18.00% (18/100) of test normals and 11.78% (35/297) of development normals to human review. Automated normal-board FAIL is zero on both splits.

### Key Verified Results (Amazon VisA `pcb1` Measured Inference)
- **100% False Alarm Scrap Elimination:** Automated false-alarm scrap (clean boards misclassified as `FAIL`) drops to **0.00%** on both the development (0/297) and test (0/100) splits. Ambiguous boards are safely escalated to operator triage (`HUMAN`) rather than scrapped.
- **Operator Review Trade-Off:** On the seen test regression split, the current policy routes **18.00%** (18/100) of normal boards to human review while keeping automated normal-board FAIL at 0/100. The development normal-review rate is **11.78%** (35/297).
- **Defect Interception Efficacy:** On the test split (*re-run after fixes; split previously seen; not a clean hold-out*), SureInspect routes **92.86%** (65/70) of defects to FAIL or HUMAN, passes 4/70 (5.71%), and routes one defect to RETAKE. The 95% exact interval for escapes is [1.58%, 13.99%]. On development, no defect is passed (0/26), 24/26 route to FAIL or HUMAN, and 2/26 route to RETAKE.
- **Conformal Risk Bounding:** Normal marginal coverage ($1 - \alpha \ge 90\%$ at $\alpha=0.10$) is mathematically guaranteed on calibration data; borderline ambiguity in $(t_{\text{pass}}, t_{\text{fail}}]$ is escalated to human review rather than blindly passed.
- **Commodity CPU Inference:** Full local execution on CPU without requiring cloud GPUs: **~397 ms** for Fast-Pass boards and **~538 ms** for multi-tool inspection boards on commodity hardware (`11th Gen Intel(R) Core(TM) i5-11260H @ 2.60GHz`).
- **Admission Quality Gate:** Shipped admission filter rejects 100% of out-of-focus blur, gamma darkening, and specular glare corruptions prior to neural evaluation. Quality Gate v1.0 was used in E1 to ensure snapshot reproducibility, while Quality Gate v1.1 achieves a false-rejection rate of only **1.01%** (3/297) on out-of-sample development normals. Dev defect images `043.JPG` and `050.JPG` were rejected due to sitting near marginal optical threshold boundaries alongside normal boards.

### Quick Start
```bash
# 1. Activate environment
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

# 2. Run test suite (currently 180 passed, 5 skipped)
pytest -q

# 3. Run benchmarks (E1-E4)
python -m eval.run_benchmarks

# 4. Start backend server
python -m backend.main --host 127.0.0.1 --port 8000
```

### Judge Verification

Run the local reproducibility, evidence, checksum, API-contract, frontend, and complete Python gates with one command:

```bash
python -m eval.verify_project
```

To include a freshly built submission archive, add `--archive path/to/submission.zip`. This local command does not prove AWS deployment, endpoint availability, video access, eligibility, or an independent holdout evaluation.

The judge gate also executes and records a real OpenCV 5 ONNX forward pass, ORB feature extraction, Laplacian focus measurement, Otsu thresholding, and contour extraction in `eval/results/opencv5_runtime_proof.json`. Candidate/result compatibility is recorded separately in `eval/results/candidate_compatibility.json`, and all ten frozen policy paths are regenerated into `eval/results/agentic_policy_matrix.json`.

Known defect escapes are documented visually in `docs/failure_cases/DEFECT_ESCAPE_EVIDENCE.md`; they are retained as limitations rather than hidden from the evaluation.
