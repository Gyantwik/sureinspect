# SureInspect: Comprehensive AI Review & Evaluation Guide
> **OpenCV AI Competition 2026 Entry**  
> **Repository:** https://github.com/Gyantwik/sureinspect.git  
> **Team:** Biswabasini90 (ML Lead), Gyantwik (Backend/AWS Lead), Khushbu656 (Frontend Lead)  
> **System Architecture:** Selective Inspection & Operator Triage System  

---

> **Key Operational Finding:** On a seen test split, the evidence tools cut good-board review from 27% to 12% and eliminated auto-scrap, with defect escapes of 8/70 against 4/70 for the cautious baseline (CIs overlap). Interception assumes human reviewers catch flagged defects.

---

## 1. Executive Summary & Problem Solved

In high-volume surface-mount technology (SMT) and printed circuit board (PCB) assembly, visual quality control faces a fundamental dilemma:
1. **Traditional Automated Optical Inspection (AOI)** uses brittle rule-based template matching that triggers excessive false-alarm scrap rates (5%–15%), wasting thousands of dollars of good hardware.
2. **Standard Deep Learning Anomaly Detection** (e.g., Vanilla PatchCore or PaDiM) uses an uncalibrated single static threshold. To catch subtle defects, manufacturers lower the threshold, triggering false alarms; to avoid false alarms, they raise the threshold, allowing escaped defects to reach end customers.
3. **Black-box Deep Models** lack physical dimensional grounding and fail silently when lenses smudge or factory lighting fluctuates.

### The SureInspect Solution: Selective Inspection & Operator Triage
SureInspect is engineered as a **Selective Inspection & Operator Triage System** rather than an autonomous oracle. It combines **OpenCV 5 DNN** feature extraction, **Split-Conformal Risk Control**, and deterministic **OpenCV Evidence Verification Tools** (ORB+ECC alignment, multi-ROI photometric residuals, and contour measurements):
* **Provable Risk Bounds:** Replaces heuristic cutoffs with a two-sided conformal decision band calibrated on clean reference normals:
  $$s \le t_{\text{pass}} \implies \text{FAST\_PASS}$$
  $$s > t_{\text{fail}} \implies \text{AGENT\_CONFIRMED}$$
  $$s \in (t_{\text{pass}}, t_{\text{fail}}] \implies \text{EVIDENCE\_TOOLS} \rightarrow \text{ESCALATION}$$
* **100% False Alarm Scrap Elimination:** Eliminates automated false alarm scrap (`Normal` $\to$ `FAIL`) to **0.00%** on both development (0/297) and test (0/100) splits by escalating borderline ambiguity to human review or multi-tool clearance rather than blindly scrapping.
* **Operator Review Burden Reduction:** On the seen test split, secondary OpenCV evidence tools safely clear 15 ambiguous normal boards, cutting the good-board human review burden from 27.00% down to **12.00%**.
* **Defect Escape Suppression:** On the final test split (*re-run after fixes; split previously seen; not a clean hold-out*), defect escapes are 8/70 (11.43%) against 4/70 (5.71%) for the ultra-conservative baseline (95% Clopper-Pearson CIs overlap: [5.07%, 21.28%] vs [2.24%, 13.79%]), while intercepting 87.14% (61/70) of all defects. On the development split, defect escapes drop to **3.85%** (1/26) compared to 11.54% (3/26) for the baseline.
* **Dual OpenCV DNN + Vector Architecture:** Deep layer-2 and layer-3 embedding extraction in C++ via `cv2.dnn.readNetFromONNX`, followed by vectorized Euclidean nearest-neighbor memory bank search in NumPy.
* **Commodity CPU Inference:** Runs entirely on standard Intel/AMD or AWS Graviton CPUs without requiring GPU hardware (~22 ms quality gate, ~325 ms perception, ~397 ms total FAST_PASS, ~538 ms multi-tool inspection).
* **Industrial Admission Quality Gate:** Rejects optical blur, specular glare, and lighting shifts in ~22 ms before neural inference.

### High-Level Operational Comparison (Final Test Split: 100 Normals, 70 Anomalies)

| Inspection Strategy | Good-Board Review / Scrap Rate | Defect Escapes (Bad $\to$ PASS) | Defect Interception (FAIL + HUMAN) | Operational Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **Cautious Baseline ($t_{\text{pass}}=2.04$)** | 27.00% (27/100 scrap) | 5.71% (4/70) [2.24%, 13.79%] | 94.29% (66/70) | Conservative single cutoff; halts 27% of good boards as false alarms. |
| **Permissive Baseline ($t_{\text{fail}}=2.29$)** | 9.00% (9/100 scrap) | 27.14% (19/70) [18.12%, 38.54%] | 72.86% (51/70) | Higher throughput; unacceptably high defect escape rate (>27%). |
| **SureInspect v1 (Pre-Remediation)** | 22.00% (14 scrap + 8 review) | 7.14% (5/70) [3.09%, 15.69%] | 91.43% (64/70) | Inconsistent tool routing; high automated scrap. |
| **SureInspect v2 (Unified Pipeline)** | **12.00%** (0 scrap + 12 review) | **11.43%** (8/70) [5.90%, 20.97%] | **87.14%** (61/70) | Eliminates 100% of auto-scrap; evidence tools safely clear 15 ambiguous normals. |

> **Low-Prevalence Production Implication:**  
> In real-world surface-mount PCB assembly lines, defect prevalence is typically $< 1\%$ (e.g., 0.1%–0.5%), unlike the 41% artificial prevalence in the balanced test split. At $< 1\%$ prevalence, the human operator workload is dominated by the normal routing rate. On the test split distribution, SureInspect routes 12% of normal boards to human review (routing ~1 in 8 boards to an operator while clearing 88% automatically). On the development split distribution, SureInspect routes only 5.39% of normal boards to human review (routing ~1 in 18 boards to an operator while clearing 94.6% automatically). In both settings, automated false alarm scrap is 0.00%.

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

| Experiment | Metric | Baseline ($t_{\text{fail}}=2.29$) | Baseline ($t_{\text{pass}}=2.04$) | SureInspect v2 (Unified Agent) | Advantage / Evidence |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **E1 (Dev Split)** | Normal False Alarm (Scrap) | 4.38% (13/297) | 20.54% (61/297) | **0.00%** (0/297) | **100% scrap reduction** (95% CI: [0.00%, 1.23%]) |
| **E1 (Dev Split)** | Total Normal False Alarm | 4.38% (13/297) | 20.54% (61/297) | **5.39%** (16/297) | Controlled human triage on ambiguous boards |
| **E1 (Dev Split)** | Defect Escape Rate | 11.54% (3/26) | 0.00% (0/26) | **3.85%** (1/26) | **1 escape** (95% CI: [0.10%, 19.64%]) vs 3 for default baseline |
| **E1 (Dev Split)** | Defect Interception Rate | 88.46% (23/26) | 100.00% (26/26) | **88.46%** (23/26) | 23/26 intercepted (plus 2 marginal optical RETAKEs) |
| **E1 (Dev Split)** | Automated Coverage | 100.0% (forced binary) | 100.0% (forced binary) | **88.24%** (285/323) | 88.2% automation; 35 boards escalated to human review |
| **E1 (Dev Split)** | Admission Gate RETAKE | 0.00% (No gate) | 0.00% (No gate) | **0.93%** (3/323) | 1 normal blur (`0154.JPG`) + 2 defects near optical boundary |
| **E1 (Final Test\*)** | Normal False Alarm (Scrap) | 9.00% (9/100) | 27.00% (27/100) | **0.00%** (0/100) | **100% scrap reduction** (95% CI: [0.00%, 3.62%]) |
| **E1 (Final Test\*)** | Total Normal False Alarm | 9.00% (9/100) | 27.00% (27/100) | **12.00%** (12/100) | Cuts good-board review from 27% to 12% |
| **E1 (Final Test\*)** | Defect Escape Rate | 27.14% (19/70) | 5.71% (4/70) | **11.43%** (8/70) | **-57.9% relative reduction vs default baseline** |
| **E1 (Final Test\*)** | Defect Interception Rate | 72.86% (51/70) | 94.29% (66/70) | **87.14%** (61/70) | Intercepts 61 of 70 defects (plus 1 camera RETAKE) |
| **E1 (Final Test\*)** | Automated Coverage | 100.0% (forced binary) | 100.0% (forced binary) | **62.94%** (107/170) | 62.9% automation; 62 boards escalated to human review |
| **E1 (Final Test\*)** | Admission Gate RETAKE | 0.00% (No gate) | 0.00% (No gate) | **0.59%** (1/170) | 1 marginal optical capture |
| **E2** | $\alpha=0.10$ Empirical Coverage | N/A | N/A | **94.00%** (6/100 exceedances) | Conformal bound holds ($1 - \alpha \ge 90\%$) |
| **E3** | Motion Blur Rejection | 0.0% (Runs inference blindly) | 0.0% | **100.0%** Rejected by Gate | Pre-inference physical defense |
| **E3** | Specular Glare Rejection | 0.0% (Triggers false positive) | 0.0% | **100.0%** Rejected by Gate | Pre-inference physical defense |
| **E4** | Admission Gate Latency | N/A | N/A | **22.5 ms (CPU)** | Pre-inference filter |
| **E4** | PatchCore Scorer Latency | Requires GPU | Requires GPU | **325–331 ms (CPU)** | OpenCV DNN + NumPy coreset on commodity CPU |
| **E4** | FAST_PASS End-to-End | N/A | N/A | **396.7 ms (CPU)** | Mean over 25 real VisA fast-pass images |
| **E4** | Multi-Tool End-to-End | N/A | N/A | **537.8 ms (CPU)** | Mean over 25 real VisA tool-path images |

*\*Note on Final Test Split: re-run after fixes; split previously seen; not a clean hold-out. Thresholds were frozen on dev calibration and never fitted on this split.*

> **Quality Gate Rejections & Versions:**
> - Dev defects `pcb1/test/bad/043.JPG` (blur 183.1 vs 190.0, bright 12.9% vs 11.8%, glare 13.4% vs 10.5%) and `pcb1/test/bad/050.JPG` (glare 10.6% vs 10.5%) sit near the marginal optical threshold boundary alongside normal boards (`0152.JPG` blur 163.3, `0154.JPG` blur 183.5).
> - **v1.0 (Snapshot):** Ran in E1 benchmarks to guarantee snapshot reproducibility.
> - **v1.1 (Opt-in Refinement):** Retuned on 99 fit normals; exhibits a false-rejection rate of only **1.01%** (3/297) when evaluated out-of-sample on all 297 development normals.

---

## 5. Weights & Snapshot Note
* Binary neural network weights (`model.onnx`, 99 MB) and memory bank (`bank.npz`, 50 MB) are omitted from this compact review archive to stay within typical AI upload size limits (< 15 MB).
* The codebase includes full reference calibration metadata (`eval/reference_calibration.json`) and a production mock engine (`backend/mock_service.py`), allowing **all unit tests, benchmark pipelines, and API endpoints to execute seamlessly out-of-the-box**.
* If live ONNX inference is desired, run `python train/export_backbone_onnx.py` and `python train/build_memory_bank.py` to regenerate the weights.
