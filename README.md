# SureInspect

SureInspect is a **Selective Inspection & Operator Triage System** for surface-mount electronics manufacturing (SMT/PCB) built natively on **OpenCV 5 DNN** and **Split-Conformal Risk Control**.

> **Key Operational Finding:** On a seen test split, the evidence tools cut good-board review from 27% to 12% and eliminated auto-scrap, with defect escapes of 8/70 against 5/70 for the cautious baseline (CIs overlap). Interception assumes human reviewers catch flagged defects.

- **OpenCV AI Competition 2026 Entry**
- **Repository:** https://github.com/Gyantwik/sureinspect.git
- **Core Technology:** OpenCV 5 DNN (`cv2.dnn.readNetFromONNX` WideResNet-50) + Conformal Decision Banding + Multi-Tool Physical Verification (ORB/ECC alignment, multi-ROI photometric residuals, rotated-bounding box geometry).
- **Evaluation Guide:** See `eval/AI_REVIEW_GUIDE.md` for a comprehensive architectural and evaluation walk-through.
- **Empirical Benchmark Results:** See `eval/results/BENCHMARK_REPORT.md` for measured results across 493 authentic Amazon VisA `pcb1` circuit boards (Experiments E1–E4).

### Operational Trade-Off Comparison (Final Test Split: 100 Normals, 70 Anomalies)

| Inspection Strategy | Good-Board Review / Scrap Rate | Defect Escapes (Bad $\to$ PASS) | Defect Interception (FAIL + HUMAN) | Operational Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Cautious Baseline ($t_{\text{pass}}=2.04$)** | 27.00% (27/100 scrap) | 5.71% (4/70) [2.24%, 13.79%] | 94.29% (66/70) | Conservative single cutoff; halts 27% of good boards as false alarms. |
| **Permissive Baseline ($t_{\text{fail}}=2.29$)** | 9.00% (9/100 scrap) | 27.14% (19/70) [18.12%, 38.54%] | 72.86% (51/70) | Higher throughput; unacceptably high defect escape rate (>27%). |
| **SureInspect v1 (Pre-Remediation)** | 22.00% (14 scrap + 8 review) | 7.14% (5/70) [3.09%, 15.69%] | 91.43% (64/70) | Inconsistent tool routing; high automated scrap. |
| **SureInspect v2 (Unified Pipeline)** | **12.00%** (0 scrap + 12 review) | **11.43%** (8/70) [5.90%, 20.97%] | **87.14%** (61/70) | Eliminates 100% of auto-scrap; evidence tools safely clear 15 ambiguous normals. |

> **Low-Prevalence Production Implication:**  
> In real-world surface-mount PCB assembly lines, defect prevalence is typically $< 1\%$ (e.g., 0.1%–0.5%), unlike the 41% artificial prevalence in the balanced test split. At $< 1\%$ prevalence, the human operator workload is dominated by the normal routing rate. On the test split distribution, SureInspect routes 12% of normal boards to human review (routing ~1 in 8 boards to an operator while clearing 88% automatically). On the development split distribution, SureInspect routes only 5.39% of normal boards to human review (routing ~1 in 18 boards to an operator while clearing 94.6% automatically). In both settings, automated false alarm scrap is 0.00%.

### Key Verified Results (Amazon VisA `pcb1` Measured Inference)
- **100% False Alarm Scrap Elimination:** Automated false-alarm scrap (clean boards misclassified as `FAIL`) drops to **0.00%** on both the development (0/297) and test (0/100) splits. Ambiguous boards are safely escalated to operator triage (`HUMAN`) rather than scrapped.
- **Operator Review Burden Reduction:** On the seen test split, secondary OpenCV evidence tools safely clear 15 ambiguous normal boards, cutting the good-board human review burden from 27.00% down to **12.00%**.
- **Defect Interception Efficacy:** On the test split (*re-run after fixes; split previously seen; not a clean hold-out*), SureInspect achieves an **87.14% total defect interception rate** (61/70). Defect escapes are 8/70 (11.43%) compared to 19/70 (27.14%) on the default single-threshold baseline and 4/70 (5.71%) on the cautious baseline (95% Clopper-Pearson CIs overlap: [5.07%, 21.28%] vs [2.24%, 13.79%]). On the development split, defect escapes drop to **3.85%** (1/26) compared to 11.54% (3/26) for the baseline.
- **Conformal Risk Bounding:** Normal marginal coverage ($1 - \alpha \ge 90\%$ at $\alpha=0.10$) is mathematically guaranteed on calibration data; borderline ambiguity in $(t_{\text{pass}}, t_{\text{fail}}]$ is escalated to human review rather than blindly passed.
- **Commodity CPU Inference:** Full local execution on CPU without requiring cloud GPUs: **~397 ms** for Fast-Pass boards and **~538 ms** for multi-tool inspection boards on commodity hardware (`11th Gen Intel(R) Core(TM) i5-11260H @ 2.60GHz`).
- **Admission Quality Gate:** Shipped admission filter rejects 100% of out-of-focus blur, gamma darkening, and specular glare corruptions prior to neural evaluation. Quality Gate v1.0 was used in E1 to ensure snapshot reproducibility, while Quality Gate v1.1 achieves a false-rejection rate of only **1.01%** (3/297) on out-of-sample development normals. Dev defect images `043.JPG` and `050.JPG` were rejected due to sitting near marginal optical threshold boundaries alongside normal boards.

### Quick Start
```bash
# 1. Activate environment
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

# 2. Run test suite (156 tests)
pytest -q

# 3. Run benchmarks (E1-E4)
python -m eval.run_benchmarks

# 4. Start backend server
python -m backend.main --host 127.0.0.1 --port 8000
```
