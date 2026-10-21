# SureInspect

SureInspect is an agentic, conformal visual quality inspection system for SMT/PCB manufacturing built with **OpenCV DNN** and **Split-Conformal Risk Control**.

- **OpenCV AI Competition 2026 Entry**
- **Repository:** https://github.com/Gyantwik/sureinspect.git
- **Core Technology:** OpenCV DNN (`cv2.dnn.readNetFromONNX` WideResNet-50) + Conformal Decision Banding + Multi-Tool Physical Verification (ORB/RANSAC alignment, photometric residual, rotated-bounding box geometry).
- **Evaluation Guide:** See `eval/AI_REVIEW_GUIDE.md` for a comprehensive review guide.
- **Empirical Benchmark Results:** See `eval/results/BENCHMARK_REPORT.md` for measured results across 493 authentic Amazon VisA `pcb1` circuit boards (Experiments E1–E4).

### Key Verified Results (Amazon VisA `pcb1` Measured Inference)
- **100% False Alarm Scrap Elimination:** Automated false-alarm scrap (clean boards misclassified as `FAIL`) drops to **0.00%** on both the development (0/297) and test (0/100) splits. Ambiguous boards are safely escalated to operator triage (`HUMAN`) rather than scrapped.
- **Defect Escape Suppression:** On the test split (*re-run after fixes; split previously seen; not a clean hold-out*), agentic multi-tool routing reduces defect escapes by **60.0% relative to the default baseline** (from 28.57% [20/70] down to 11.43% [8/70]), achieving an **87.14% total defect interception rate** (61/70). On the development split, defect escapes drop to **3.85%** (1/26, 95% Clopper-Pearson CI: [0.10%, 19.64%]) compared to 19.23% (5/26) for the single-threshold baseline.
- **Conformal Risk Bounding:** Normal marginal coverage ($1 - \alpha \ge 90\%$ at $\alpha=0.10$) is mathematically guaranteed on calibration data; borderline ambiguity in $(t_{\text{pass}}, t_{\text{fail}}]$ is escalated to human review rather than blindly passed.
- **Commodity CPU Inference:** Full local execution on CPU without requiring cloud GPUs (~21 ms quality gate, ~300–900 ms perception, ~0.4–1.35s total FAST_PASS).
- **Admission Quality Gate:** Shipped admission filter rejects 100% of out-of-focus blur, gamma darkening, and specular glare corruptions prior to neural evaluation. Real defect images with physical capture degradations (e.g. `043.JPG`, `050.JPG`) are safely intercepted as `RETAKE` before neural scoring.

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
