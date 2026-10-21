# SureInspect: Comprehensive AI Review & Evaluation Guide
> **OpenCV AI Competition 2026 Entry**  
> **Repository:** https://github.com/Gyantwik/sureinspect.git  
> **Team:** Biswabasini90 (ML Lead), Gyantwik (Backend/AWS Lead), Khushbu656 (Frontend Lead)

---

## 1. Executive Summary & Problem Solved

In high-volume surface-mount technology (SMT) and printed circuit board (PCB) assembly, visual quality control faces a fundamental dilemma:
1. **Traditional Automated Optical Inspection (AOI)** uses brittle rule-based matching that triggers excessive false-alarm scrap rates (5%–15%), wasting thousands of dollars of good hardware.
2. **Standard Deep Learning Anomaly Detection** (e.g. Vanilla PatchCore or PaDiM) uses an uncalibrated single static threshold. To catch subtle defects, manufacturers lower the threshold, triggering false alarms; to avoid false alarms, they raise the threshold, allowing escaped defects to reach end customers.
3. **Black-box Deep Models** lack physical dimensional grounding and fail silently when lenses smudge or lighting fluctuates.

### The SureInspect Solution
SureInspect is a **statistically rigorous, agentic visual inspection system** built natively on **OpenCV 5 DNN** and **Split-Conformal Prediction**:
* **Provable Risk Bounds:** Replaces heuristic cutoffs with a two-sided conformal decision band:
  $$s \le t_{\text{pass}} \implies \text{FAST\_PASS}$$
  $$s > t_{\text{fail}} \implies \text{AGENT\_CONFIRMED}$$
  $$s \in (t_{\text{pass}}, t_{\text{fail}}] \implies \text{EVIDENCE\_TOOLS} \rightarrow \text{ESCALATION}$$
* **100% False Alarm Scrap Elimination:** Eliminates automated false alarm scrap (`Normal` $\to$ `FAIL`) to **0.00%** on both dev and test splits by escalating borderline ambiguity to human review or multi-tool clearance rather than blindly rejecting.
* **Defect Escape Suppression:** On the final test split (*re-run after fixes; split previously seen; not a clean hold-out*), agentic multi-tool routing reduces defect escapes by **60.0% relative to the baseline** (from 28.57% down to 11.43%), intercepting 87.14% of all defects (and reducing escapes to 3.85% [1/26] on the development split).
* **Dual OpenCV DNN + Vector Architecture:** Deep layer-2 and layer-3 embedding extraction in C++ via `cv2.dnn.readNetFromONNX`, followed by vectorized Euclidean nearest-neighbor memory bank search in NumPy.
* **Commodity CPU Inference:** Runs entirely on standard Intel/AMD or AWS Graviton CPUs without requiring GPU hardware (~21 ms quality gate, ~300–900 ms perception, ~0.4–1.35s total FAST_PASS).
* **Industrial Admission Quality Gate:** Rejects optical blur, specular glare, and lighting shifts in ~21 ms before neural inference using the shipped calibration parameters. Real captures with camera degradations (e.g. `043.JPG`, `050.JPG`) are safely intercepted as `RETAKE` prior to neural evaluation.

---

## 2. Monorepo Architecture

```
sureinspect/
├── core/                  # Core OpenCV Perception, Calibration & Agent State Machine
│   ├── agent.py           # Deterministic 10-path decision policy & AgentRunner
│   ├── calibration.py     # Split-conformal quantile calibration & p-value computation
│   ├── degrade.py         # Deterministic physical corruptions (motion blur, glare, darken)
│   ├── interfaces.py      # Abstract protocol definitions & contracts
│   ├── perception.py      # OpenCV 5 DNN PatchCore inference & coreset search
│   ├── preprocess.py      # Aspect-preserving standardization (1024px) & NCHW blob formatting
│   ├── quality_gate.py    # Laplacian variance, HSV specular glare, CIELAB lighting shift
│   ├── tools.py           # Evidence tools T1-T5 (ORB alignment, ROI residual, geometric measurement)
│   └── tests/             # 97 unit tests covering all OpenCV components
├── backend/               # Production FastAPI Engine & Admission Controls
│   ├── admission.py       # Streaming upload limiter, magic byte validator, 25 MP bomb guard
│   ├── app_factory.py     # FastAPI application factory, session auth, static SPA mount
│   ├── executor.py        # Pluggable dual-mode executor (Dynamic Live vs Mock engine)
│   ├── live_runner.py     # Live OpenCV 5 DNN PatchCore inference & tool orchestrator
│   ├── main.py            # ASGI CLI entrypoint with thread caps and proxy headers
│   ├── mock_service.py    # Realistic Day 1 mock engine simulating all 10 decision paths
│   ├── schemas.py         # Pydantic v2 schemas for all API requests, responses, and enums
│   ├── storage.py         # Bounded LRU/TTL in-memory cache (200 MiB limit, asset streaming)
│   └── tests/             # 37 integration & unit tests
├── frontend/              # Production React 18 + Vite + Tailwind CSS Console
│   ├── src/components/    # ControlPanel, Visual Evidence Compare, DecisionCard, TraceTimeline
│   ├── src/services/      # api.ts client & openapi-typescript generated contract types
│   └── tests/             # Vitest frontend component tests
├── eval/                  # Empirical Benchmark Suite (Experiments E1-E4)
│   ├── run_benchmarks.py  # Master evaluation orchestrator
│   ├── reference_calibration.json # Committed reference calibration data for headless CI
│   ├── results/           # BENCHMARK_REPORT.md & benchmark_summary.json
│   └── tests/             # 19 benchmark and statistical validation tests
├── data/                  # Industrial Templates & Test Samples
│   ├── templates/pcb1/    # Golden reference templates from Amazon VisA
│   └── samples/           # Curated UI demo samples (FAST_PASS, AGENT_CONFIRMED, etc.)
├── contracts/             # Frozen OpenAPI specifications & fixtures
├── deploy/                # Multi-stage Dockerfile, Caddy TLS reverse proxy, AWS EC2 Graviton scripts
└── docs/                  # Architecture diagrams, failure modes, AWS cost report
```

---

## 3. How to Test and Verify the Project

### A. Run Automated Test Suite (153 Tests)
The entire monorepo is tested via `pytest`. All tests pass locally and in CI:
```bash
# In Python environment with requirements installed:
pytest -q
```
Expected output:
```text
153 passed, 5 skipped in ~15s (100% pass rate)
```

### B. Run the Benchmark Suite (Experiments E1–E4)
Execute the master benchmark runner:
```bash
python -m eval.run_benchmarks
```
This generates:
* `eval/results/benchmark_summary.json` (Machine-readable empirical metrics)
* `eval/results/BENCHMARK_REPORT.md` (Publication-grade tables with Wilson & Clopper-Pearson 95% confidence intervals)

### C. Run the Backend Server
Start the Uvicorn server:
```bash
python -m backend.main --host 127.0.0.1 --port 8000
```
Check health:
```bash
curl http://127.0.0.1:8000/api/health
```

### D. Run Real Industrial Image Tests
Execute the automated test script against live Amazon VisA PCB images:
```bash
python scratch/run_real_test.py
```
This tests:
1. Pristine clean board $\rightarrow$ Verified `PASS` (`FAST_PASS`)
2. Bent pin anomaly $\rightarrow$ Verified `FAIL` (`AGENT_CONFIRMED`) with geometric defect measurements
3. Ambiguous border board $\rightarrow$ Verified `HUMAN` (`INCONCLUSIVE`) routing to operator
4. Trace scratch $\rightarrow$ Cleared by tool $T_2$ (`AGENT_CLEARED`)
5. Specular glare optical corruption $\rightarrow$ Rejected in $22\text{ ms}$ by Admission Quality Gate (`RETAKE`)

---

## 4. Key Quantitative Results (Empirically Measured Across 493 Authentic VisA Images)

| Experiment | Metric | Baseline (Single Threshold $t_{\text{fail}}$) | SureInspect v2 (Unified Agent) | Advantage / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **E1 (Dev Split)** | Normal False Alarm (Scrap) | **4.38%** (13/297 false scrap) | **0.00%** (0/297 false scrap) | **100% scrap reduction** (95% CI: [0.00%, 1.23%]) |
| **E1 (Dev Split)** | Total Normal False Alarm | 4.38% (13/297) | **5.39%** (16/297 FAIL+HUMAN) | Controlled human triage on ambiguous boards |
| **E1 (Dev Split)** | Defect Escape Rate | **19.23%** (5/26 missed defects) | **3.85%** (1/26 escapes) | **-80.0% relative reduction in escapes** (95% CI: [0.10%, 19.64%]) |
| **E1 (Dev Split)** | Defect Interception Rate | 80.77% (21/26) | **88.46%** (23/26 FAIL+HUMAN) | Intercepts 23 of 26 defects (and 2 camera RETAKEs) |
| **E1 (Dev Split)** | Automated Coverage | 100.0% (forced binary) | **88.24%** (285/323) | 88.2% automation; 35 boards escalated to human review |
| **E1 (Dev Split)** | Admission Gate RETAKE | 0.00% (No gate) | **0.93%** (3/323) | Intercepts 1 normal blur + 2 camera-degraded defects |
| **E1 (Final Test\*)** | Normal False Alarm (Scrap) | **9.00%** (9/100 false scrap) | **0.00%** (0/100 false scrap) | **100% scrap reduction** (95% CI: [0.00%, 3.62%]) |
| **E1 (Final Test\*)** | Total Normal False Alarm | 9.00% (9/100) | **12.00%** (12/100 FAIL+HUMAN) | 12 borderline normal boards escalated to review |
| **E1 (Final Test\*)** | Defect Escape Rate | **28.57%** (20/70 missed defects) | **11.43%** (8/70 escapes) | **-60.0% relative reduction in escapes** (95% CI: [5.07%, 21.28%]) |
| **E1 (Final Test\*)** | Defect Interception Rate | 71.43% (50/70) | **87.14%** (61/70 FAIL+HUMAN) | Intercepts 61 of 70 defects (and 1 camera RETAKE) |
| **E1 (Final Test\*)** | Automated Coverage | 100.0% (forced binary) | **62.94%** (107/170) | 62.9% automation; 62 boards escalated to human review |
| **E1 (Final Test\*)** | Admission Gate RETAKE | 0.00% (No gate) | **0.59%** (1/170) | Intercepts 1 camera-degraded capture |
| **E2** | $\alpha=0.10$ Empirical Coverage | N/A | **94.00%** (6/100 exceedances) | Conformal bound holds ($1 - \alpha \ge 90\%$) |
| **E3** | Motion Blur Rejection | 0.0% (Runs inference blindly) | **100.0%** Rejected by Quality Gate | Shipped gate eliminates focus noise |
| **E3** | Specular Glare Rejection | 0.0% (Triggers false positive) | **100.0%** Rejected by Quality Gate | Shipped gate eliminates glare artifacts |
| **E4** | Admission Gate Latency | N/A | **21.0 ms (CPU)** | Fast pre-inference filter |
| **E4** | PatchCore Scorer Latency | Requires GPU | **~290–900 ms (CPU)** | OpenCV DNN + NumPy coreset on commodity CPU |
| **E4** | End-to-End FAST_PASS | N/A | **~395 ms – 1.35 s (CPU)** | Full inspection without cloud GPUs |

*\*Note on Final Test Split: re-run after fixes; split previously seen; not a clean hold-out. Thresholds were frozen and never fitted on this split.*

---

## 5. Weights & Snapshot Note
* Binary neural network weights (`model.onnx`, 99 MB) and memory bank (`bank.npz`, 50 MB) are omitted from this compact review archive to stay within typical AI upload size limits (< 15 MB).
* The codebase includes full reference calibration metadata (`eval/reference_calibration.json`) and a production mock engine (`backend/mock_service.py`), allowing **all unit tests, benchmark pipelines, and API endpoints to execute seamlessly out-of-the-box**.
* If live ONNX inference is desired, run `python train/export_backbone_onnx.py` and `python train/build_memory_bank.py` to regenerate the weights.
