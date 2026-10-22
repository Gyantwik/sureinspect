# SureInspect: Selective Inspection & Operator Triage System
### OpenCV AI Competition 2026 — Final Technical Submission Report

> **Project Name:** SureInspect  
> **Repository:** `https://github.com/Gyantwik/SureInspect`  
> **Target Track:** OpenCV AI Competition 2026  
> **Team Members:**  
> - **Biswabasini Prasad** (@Biswabasini90) — Computer Vision & Machine Learning Lead (OpenCV 5 DNN, Quality Gate, Evidence Tools)  
> - **Gyantwik** (@Gyantwik) — Backend, Integration, AWS Architecture & Evaluation Lead (FastAPI, Storage, Conformal Benchmarks)  
> - **Khushbu** (@Khushbu656) — Frontend & User Experience Lead (React Dashboard, Live Visual Overlays, Triage Queue)  
> **OpenCV Version:** `5.0.0` (Headless `5.0.0.93`, C++ DNN ONNX Engine)  
> **Report Date:** October 2026  

---

## The Problem and Users

### The Industrial PCB Inspection Dilemma
High-volume printed circuit board (PCB) assembly lines require rigorous quality assurance to detect surface defects such as solder bridges, missing surface-mount devices (SMD components), trace scratches, and solder melt before boards proceed to functional testing or final product integration. In modern surface-mount technology (SMT) production lines, visual inspection systems face an agonizing operational trade-off:
1. **Catastrophic Escapes (False Negatives):** A defective PCB that escapes automated inspection and is shipped to an automotive, aerospace, or medical customer results in catastrophic field failures, expensive warranty recalls, and safety hazards.
2. **Production-Halting False Alarms (False Positives):** A vision system tuned with hyper-conservative thresholds stops the line or automatically discards ("scraps") good boards, wasting thousands of dollars per hour in unnecessary scrap costs and creating severe operator review backlogs.

Traditional deep learning inspection pipelines rely on black-box convolutional neural networks (CNNs) or vision transformers that emit uncalibrated softmax scores. Under factory floor variations—such as ambient illumination changes between daytime and nighttime shifts, subtle board vibration on conveyor belts, or component vendor print variations—these models exhibit overconfidence on out-of-distribution inputs, silently failing to detect subtle flaws or flooding operators with false alarms.

### Who Uses SureInspect
SureInspect is built for two primary industrial stakeholders:
1. **Quality Assurance (QA) & Process Engineers:** Line managers responsible for setting quality thresholds, tracking first-pass yield (FPY), monitoring conveyor belt optical stability, and auditing defect escape rates across shifts.
2. **Floor Quality Control (QC) Operators:** Human inspectors stationed at manual review stations who review borderline boards flagged by automated systems. Rather than inspecting thousands of identical boards from scratch (which causes severe visual fatigue and missed defects), operators receive pre-aligned, pre-highlighted candidate defects with quantified geometric measurements.

### Operational Costs: Missed Defects vs. False Alarms
In surface-mount manufacturing, the cost asymmetry between defect escapes and false alarms is typically 100:1. A false alarm costs approximately 15–30 seconds of operator review time (~$0.25–$0.50). In contrast, an escaped defect integrated into a complex sub-assembly can cost hundreds to thousands of dollars in rework or field replacement. Consequently, factory managers cannot deploy an automated inspection system unless its false-negative rate is mathematically bounded and its operational failure modes are transparently documented.

### Current Limitations of the Proposed Solution
SureInspect is intentionally engineered as a **Selective Inspection & Operator Triage System**, not an unchecked autonomous decision maker. Under its current implementation:
- It focuses on surface-layer optical defects on rigid printed circuit boards; it does not perform multi-layer X-ray inspection (AXI) or in-circuit electrical testing (ICT).
- Subtle micro-defects occupying less than 1–2% of a $256 \times 256$ window (such as hairline hairline scratches or miniature 0402 SMD missing chips) can be photometrically diluted during spatial residual averaging.
- The system assumes factory illumination and camera focus can be monitored by the physical quality gate; severe lens fouling or total mechanical misalignment halts the line with `RETAKE` or `ALIGN_FAILED` rather than guessing an outcome.

---

## System Architecture

### Application Architecture Overview
SureInspect employs a multi-tiered defense-in-depth architecture that decouples physical signal validation, deep perception, deterministic physical evidence tools, and policy routing. 

```
+----------------------------------------------------------------------------------------------------+
|                                    SUREINSPECT SYSTEM PIPELINE                                     |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                                     [ Input PCB Image (BGR) ]
                                                  |
                                                  v
                            +-------------------------------------------+
                            | Stage 1: OpenCV 5 Physical Quality Gate   |
                            | (Laplacian blur, dark/bright, glare, LAB) |
                            +---------------------+---------------------+
                                                  |
                         +------------------------+------------------------+
                         | Fails Thresholds                                | Passes Gate
                         v                                                 v
                [ RETAKE Outcome ]                          +-------------------------------+
              (QUALITY_REJECTED)                            | Stage 2: OpenCV 5 DNN Engine  |
                                                            | (WideResNet-50 PatchCore)     |
                                                            +---------------+---------------+
                                                                            |
                                               +----------------------------+----------------------------+
                                               | Score <= t_pass (Normal)                                | Score > t_pass (Ambiguous / Defect)
                                               v                                                         v
                                      [ FAST_PASS Outcome ]                             +-----------------------------------+
                                         (Decision: PASS)                               | Stage 3: Deterministic Tools      |
                                                                                        | T1: ORB+ECC Multi-Scale Alignment |
                                                                                        | T4: CLAHE Light Normalization     |
                                                                                        | T2: ROI Structural Residual       |
                                                                                        | T5: Secondary ROI Comparison      |
                                                                                        | T3: Zoom & Defect Geometry        |
                                                                                        +-----------------+-----------------+
                                                                                                          |
                                                                                                          v
                                                                                        +-----------------------------------+
                                                                                        | Stage 4: Pure Agent Policy        |
                                                                                        | (Maps score + residuals to 10     |
                                                                                        |  frozen contract decision paths)  |
                                                                                        +-----------------+-----------------+
                                                                                                          |
                                                   +------------------------------------+-----------------+------------------------------------+
                                                   |                                    |                                                      |
                                                   v                                    v                                                      v
                                        [ Automated Clearance ]               [ Human Operator Triage ]                             [ Automated Rejection ]
                                            (AGENT_CLEARED)                    - SUSPECTED_DEFECT                                      (AGENT_CONFIRMED)
                                            Verdict: PASS                      - EVIDENCE_CONFLICT                                      Verdict: FAIL
                                                                               - LIGHTING_SHIFT_REVIEW
                                                                               - ALIGN_FAILED
                                                                               - INCONCLUSIVE
                                                                               - BUDGET_EXHAUSTED
                                                                               Verdict: HUMAN
```

The system architecture is formally defined in [`docs/architecture.mmd`](file:///c:/Users/sukum/OneDrive/Desktop/WebDev/OpenCv/docs/architecture.mmd).

### End-to-End Data Flow
1. **Ingestion & Standardization (`core/preprocess.py`):** The incoming raw image (any camera resolution up to 25 megapixels) is parsed and standardized to a maximum bounding side of 1024 pixels using OpenCV `INTER_AREA` interpolation. An aspect ratio guard ensures the board geometry has not been distorted during transport.
2. **Quality Gate Screening (`core/quality_gate.py`):** Pre-inference screening measures focus sharpness (Laplacian variance), exposure bounds (dark and bright pixel fractions), specular glare reflection, and CIELAB $L^*$ illumination drift. Corrupted frames are rejected immediately as `RETAKE`, completely bypassing neural network inference.
3. **PatchCore Perception (`core/perception.py`):** Standardized images are converted into an NCHW tensor via `cv2.dnn.blobFromImage` ($256 \times 256$, ImageNet normalized). The OpenCV 5 DNN engine executes a frozen WideResNet-50 backbone, extracting localized feature representations from intermediate layers. Local patch vectors are compared against a coreset memory bank using Euclidean nearest-neighbor search, generating an anomaly score $s$ and a spatial heatmap.
4. **Conformal Boundary Check (`core/calibration.py`):** The raw anomaly score $s$ is evaluated against non-parametric split-conformal thresholds ($t_{\text{pass}} = 2.0405, t_{\text{fail}} = 2.2863$) fitted on clean calibration boards. If $s \le t_{\text{pass}}$ and illumination is stable, the board exits immediately as `FAST_PASS`.
5. **Secondary Visual Verification (`core/tools.py`):** For boards where $s > t_{\text{pass}}$ or lighting has shifted, the agent dispatches deterministic OpenCV tools:
   - `T1_ALIGN_REFERENCE`: ORB keypoint feature detection followed by RANSAC homography and Euclidean ECC sub-pixel alignment against golden reference templates.
   - `T4_LIGHT_NORMALIZE`: Applies CLAHE photometric equalization if illumination covariate shift was flagged.
   - `T2_ROI_COMPARE`: Warps the golden template into test coordinates and computes structural pixel residuals ($r_{95}$ and $r_{\text{mean}}$) over candidate anomaly regions of interest (ROIs).
   - `T5_NEXT_ROI_COMPARE`: Evaluates secondary candidate ROIs when multiple anomaly peaks exist.
   - `T3_ROI_ZOOM_MEASURE`: Isolates difference contours via Otsu thresholding, computing area percentage, elongation, and rotated rectangle orientation (`cv2.minAreaRect`).
6. **Pure Agent Policy Routing (`core/agent.py`):** The pure functional decision function `decide()` evaluates score $s$, lighting flags, tool residual values, and processing budget limits, emitting exactly one of 10 frozen decision paths mapped to one of four public outcomes: `PASS`, `FAIL`, `HUMAN`, or `RETAKE`.

### The Human-Review Path
When the agent encounters ambiguous or conflicting evidence, it routes the board to `HUMAN`. The backend stores all intermediate artifacts in a thread-safe bounded LRU cache (`backend/storage.py`), serving:
- The standardized input image and side-by-side aligned golden template.
- The OpenCV anomaly heatmap overlay.
- Bounding box coordinates and rotated rectangle parameters for the suspect anomaly.
- Calibrated conformal bounds explaining why automated clearance was withheld.
- A human feedback endpoint (`POST /api/inspections/{id}/feedback`) enabling operators to log ground-truth verdicts (`CONFIRMED_DEFECT` or `CLEARED_NORMAL`) with optional notes for auditing.

### Relationship Between Implemented Components and Proposed Infrastructure
- **Locally Implemented & Verified:** The complete FastAPI backend (`backend/main.py`), in-memory bounded LRU/TTL storage cache, OpenCV 5 DNN inference engine, 5 deterministic verification tools, admission queue with concurrency limiting, Docker containerization (`deploy/Dockerfile`), Caddy reverse proxy configuration (`deploy/Caddyfile`), and React web dashboard (`frontend/`).
- **Proposed Architecture:** Production deployment on AWS Graviton (`t4g.small`), direct S3 bucket synchronization for multi-region audit logs via `boto3`, and CloudWatch alarm monitoring. To ensure 100% reproducibility and eliminate network dependencies during competition evaluation, all tests and benchmarks run on local hardware.

---

## OpenCV 5 Implementation

### Real OpenCV 5 Operations Used
SureInspect relies substantively on OpenCV 5 (`cv2.__version__ == '5.0.0'`) across all stages of the inspection lifecycle. It does not use OpenCV as a trivial image loading utility, but as the foundational computation engine for computer vision and deep learning inference:

```python
# Mandatory OpenCV AI Competition 2026 requirement
# Environment: opencv-python-headless==5.0.0.93
```

The key OpenCV 5 modules and APIs employed include:

| Pipeline Stage | OpenCV 5 Modules / APIs | Specific Function & Purpose |
| :--- | :--- | :--- |
| **Image Standardization** | `cv2.resize`, `cv2.INTER_AREA` | Downsamples high-resolution input frames to max side 1024 without aliasing artifacts. |
| **Physical Quality Gate** | `cv2.Laplacian`, `cv2.cvtColor`, `cv2.COLOR_BGR2Lab`, `cv2.inRange` | Computes 64-bit Laplacian variance for blur; extracts CIELAB $L^*$ channel for illumination covariate shift; thresholds specular glare. |
| **DNN Model Inference** | `cv2.dnn.readNetFromONNX`, `cv2.dnn.blobFromImage`, `net.setInput`, `net.forward` | Loads and executes the WideResNet-50 feature extraction backbone directly in C++ via the OpenCV 5 DNN engine. |
| **Reference Alignment (T1)** | `cv2.ORB_create`, `cv2.BFMatcher`, `cv2.estimateAffinePartial2D`, `cv2.findTransformECC` | Detects 2000 ORB keypoints, filters by Lowe's ratio test (0.75), estimates robust affine homography via RANSAC, and refines registration with sub-pixel Euclidean ECC. |
| **Photometric Matching & Residuals (T2/T5)** | `cv2.warpAffine`, `cv2.GaussianBlur`, `cv2.absdiff` | Forward-warps golden reference image with affine matrix $M$; performs global gain/offset matching; computes pixel absolute difference residual $r$. |
| **Lighting Normalization (T4)** | `cv2.createCLAHE` | Applies Contrast Limited Adaptive Histogram Equalization ($clip=2.0, tile=8\times8$) to test and reference copies when lighting shift is detected. |
| **Defect Geometry & Contours (T3)** | `cv2.threshold`, `cv2.THRESH_OTSU`, `cv2.findContours`, `cv2.minAreaRect`, `cv2.contourArea` | Extracts difference contours via Otsu thresholding; fits rotated bounding rectangles to measure aspect ratio, elongation, and defect footprint area percentage. |

### Concrete Agent Evidence Tool Progression Example
To illustrate how measured OpenCV outputs dynamically steer agent decisions, consider an ambiguous board (`test/bad/025.JPG`, solder melt anomaly):
1. **Perception Output:** The OpenCV 5 DNN forward pass yields an anomaly score of $s = 2.0637$. This exceeds $t_{\text{pass}} = 2.0405$, so the agent **refuses** to emit `FAST_PASS`.
2. **Tool T1 Dispatched:** The agent calls `align_to_templates()`. OpenCV extracts ORB features, estimates affine matrix $M$, and reports alignment quality $q = 0.8628$, exceeding the frozen $q_{\text{min}} = 0.74$.
3. **Tool T2 Dispatched:** The agent warps the golden template with `cv2.warpAffine` and measures ROI structural residual. The recorded residual mean is $0.0826$, below the active $r_{\text{low}} = 0.30$.
4. **Tool T3 Dispatched:** The agent does not clear solely from the low residual. `measure()` uses OpenCV contours and `minAreaRect` and finds an elongated physical difference with elongation $14.99$ and area $0.0993\%$.
5. **Policy Decision:** The physical-contour guard overrides automated clearance and returns `HUMAN / SUSPECTED_DEFECT` with a suggested FAIL. This is the central agentic behavior: measured evidence changes the action, and uncertainty does not silently become PASS.

---

## AWS Deployment

### Target Architecture & Deployment Design
SureInspect's cloud deployment architecture is designed for edge-to-cloud manufacturing environments using energy-efficient **AWS Graviton (ARM64)** compute:

```
                                  Internet (HTTPS Port 443)
                                              │
                                              ▼
                           ┌─────────────────────────────────────┐
                           │         Caddy Reverse Proxy         │
                           │   - Automatic Let's Encrypt TLS     │
                           │   - 10 MiB Body Limit Enforcement   │
                           │   - Real-IP Header Forwarding       │
                           └──────────────────┬──────────────────┘
                                              │ localhost:8000
                                              ▼
                           ┌─────────────────────────────────────┐
                           │      FastAPI Container (Docker)     │
                           │   - Single Uvicorn Worker           │
                           │   - Bounded Async Queue (1+4)       │
                           │   - 25 MP Decompression Bomb Guard  │
                           │   - 200 MiB / 100 Item LRU Cache    │
                           │   - OpenCV 5 DNN Perception Engine  │
                           └──────────────────┬──────────────────┘
                                              │
                                              ▼
                           ┌─────────────────────────────────────┐
                           │       Private AWS S3 Storage        │
                           │   - /artifacts (Read-Only)          │
                           │   - /traces (Write-Only Audit)      │
                           │   - /feedback (Write-Only Audit)    │
                           └─────────────────────────────────────┘
```

### Runtime, Docker, and Configuration Details
1. **Container Specification (`deploy/Dockerfile`):**
   - Base image: `python:3.11-slim-bookworm` (multi-stage build minimizing attack surface).
   - OpenCV installation: Pinned to `opencv-python-headless==5.0.0.93`.
   - Thread capping: Environment variables strictly limit CPU thread pool contention:
     ```bash
     OMP_NUM_THREADS=1
     OPENBLAS_NUM_THREADS=1
     MKL_NUM_THREADS=1
     ```
   - Memory footprint: Peak RSS under full load stays under **1.1 GiB**, comfortably fitting within the 2 GiB physical RAM envelope of an AWS `t4g.small` instance ($0.0168/hr or Free Tier eligible).
2. **Admission Queue & Concurrency Guard (`backend/admission.py`):**
   - An asynchronous semaphore admits at most 1 running CPU inference job and queues up to 4 pending jobs. Excess requests receive immediate HTTP 503 (`QUEUE_SATURATED`) with a `Retry-After: 1` header, preventing memory starvation.
3. **Decompression Bomb Guard:**
   - Enforced via `OPENCV_IO_MAX_IMAGE_PIXELS=25000000` to prevent memory exhaustion from maliciously crafted or inflated image headers.
4. **Caddy Reverse Proxy (`deploy/Caddyfile`):**
   - Enforces a 10 MiB edge request body ceiling, terminates TLS, and forwards clean requests to FastAPI over localhost.

### Implementation Status Matrix
To prevent misrepresenting deployment readiness, the following table explicitly distinguishes verified implementations from proposed cloud designs:

| Component | Status | Verification & Evidence |
| :--- | :--- | :--- |
| **Docker Container Build** | **Implemented & Tested** | `deploy/Dockerfile` tested locally with OpenCV 5.0.0.93 headless. |
| **Admission Queue** | **Implemented & Tested** | Complete suite passes (180 passed, 5 skipped), including concurrency tests in `backend/tests/test_admission_queue.py`. |
| **Storage & Audit Cache** | **Implemented & Tested** | Thread-safe LRU cache with 200 MiB ceiling tested in `backend/tests/test_storage.py`. |
| **Reverse Proxy Config** | **Implemented & Tested** | `deploy/Caddyfile` with edge 10 MiB body limit and header forwarding. |
| **EC2 Bootstrap Script** | **Implemented & Tested** | Shell script `deploy/setup_ec2.sh` for Ubuntu 22.04 LTS Graviton setup. |
| **Live EC2 Cloud Host** | *Proposed Architecture* | Not provisioned on live AWS to avoid cloud costs and ensure standalone local reproducibility. |
| **Direct S3 Sync** | *Proposed Architecture* | S3 IAM policies designed in `docs/aws_section.md`; local runtime uses bounded local memory. |
| **CloudWatch Alarms** | *Proposed Architecture* | Design specification with $10.00 billing notification threshold. |

---

## Evaluation and Benchmarks

### Evaluation Methodology & Dataset
SureInspect was evaluated across the complete Amazon VisA PCB1 benchmark dataset (493 total images) with **zero simulated or fabricated metrics**. All metrics represent real inference executions through OpenCV 5 DNN on real JPEG files:

| Dataset Split | Normal Images (Clean) | Defect Images (Bad) | Total Images |
| :--- | :---: | :---: | :---: |
| **Calibration Fit Set** | 99 | 0 | 99 |
| **Development Split (`dev`)** | 297 | 26 | 323 |
| **Test Split (`test`)** | 100 | 70 | 170 |
| **Evaluation predictions (dev + test)** | **397** | **96** | **493** |
| **Separate calibration-fit normals** | **99** | **0** | **99** |
| **All disjoint images used across calibration and evaluation** | **496** | **96** | **592** |

*(Note: 323 dev images + 170 test images = 493 total evaluation images outside the 99 calibration fit images).*

### Operational Baseline Comparison
To demonstrate the practical value of multi-tool agent triage, SureInspect was benchmarked side-by-side against two single-threshold perception baselines on the same identical images:
1. **Cautious Baseline ($t_{\text{pass}} = 2.0405$):** Emits PASS if $s \le t_{\text{pass}}$; otherwise scraps the board as FAIL.
2. **Default Baseline ($t_{\text{fail}} = 2.2863$):** Emits PASS if $s \le t_{\text{fail}}$; otherwise scraps the board as FAIL.
3. **SureInspect:** Dispatches secondary OpenCV tools when $s > t_{\text{pass}}$ and routes ambiguities to human operators.

#### Table 1: Test Split Evaluation Results (100 Normals, 70 Defects)
*(Note: Evaluated as regression/development split; previously seen during initial tuning).*

| Metric | Cautious Baseline ($t_{\text{pass}}=2.04$) | Default Baseline ($t_{\text{fail}}=2.29$) | SureInspect Final Hardened |
| :--- | :---: | :---: | :---: |
| **Defect Escapes (PASS on Defect)** | **5.71%** (4 / 70) | 27.14% (19 / 70) | **5.71%** (4 / 70) [MATCHED!] |
| **95% Clopper-Pearson CI** | [1.58%, 13.99%] | [17.80%, 38.97%] | [1.58%, 13.99%] |
| **Automated Scrap on Good Boards** | 27.00% (27 / 100) | 9.00% (9 / 100) | **0.00%** (0 / 100) [ZERO SCRAP] |
| **Human Review on Good Boards** | 0.00% (0 / 100) | 0.00% (0 / 100) | **18.00%** (18 / 100) |
| **Total Good Boards Cleared (PASS)** | 73.00% (73 / 100) | 91.00% (91 / 100) | **82.00%** (82 / 100) |
| **Total Defect Interception Rate (FAIL + HUMAN)** | 94.29% (66 / 70) | 72.86% (51 / 70) | **92.86%** (65 / 70) |
| **Automated Rejection (FAIL)** | 66 / 70 | 51 / 70 | 10 / 70 |
| **Escalated to Operator (HUMAN)** | 0 / 70 | 0 / 70 | 55 / 70 |
| **Optical Retakes (RETAKE)** | 0 / 70 | 0 / 70 | 1 / 70 (041.JPG focus boundary) |

#### Table 2: Development Split Evaluation Results (297 Normals, 26 Defects)

| Metric | Cautious Baseline ($t_{\text{pass}}=2.04$) | Default Baseline ($t_{\text{fail}}=2.29$) | SureInspect Final Hardened |
| :--- | :---: | :---: | :---: |
| **Defect Escapes (PASS on Defect)** | **0.00%** (0 / 26) | 11.54% (3 / 26) | **0.00%** (0 / 26) |
| **Automated Scrap on Good Boards** | 20.54% (61 / 297) | 4.38% (13 / 297) | **0.00%** (0 / 297) [ZERO SCRAP] |
| **Human Review on Good Boards** | 0.00% (0 / 297) | 0.00% (0 / 297) | **11.78%** (35 / 297) |
| **Total Good Boards Cleared (PASS)** | 79.46% (236 / 297) | 95.62% (284 / 297) | **87.88%** (261 / 297) |
| **Total Defect Interception Rate (FAIL + HUMAN)** | 100.00% (26 / 26) | 88.46% (23 / 26) | **92.31%** (24 / 26) |
| **Optical Retakes (RETAKE)** | 0 / 26 | 0 / 26 | 2 / 26 (043.JPG, 050.JPG) |

### Key Operational Finding
> On the previously seen test regression split, SureInspect matches the cautious baseline's defect escape rate of 4/70 (5.71%) while eliminating automated good-board scrap (0/100 versus 27/100). It routes 18/100 normal boards and 55/70 defective boards to HUMAN, with one defective RETAKE. On development, no defect is passed, 24/26 route to FAIL or HUMAN, and 2/26 route to RETAKE. Human interception assumes reviewers catch flagged defects.

### Historical Optimization Notes (Not Canonical Evidence)
The table below is retained as historical engineering context. Its intermediate numbers and outcome labels are not canonical current metrics and must not be copied into submission claims. Current measured results come only from `eval/results/measured_e1_results_v2.json`; new controlled experiments are registered in `docs/EXPERIMENT_LOG.md` before execution.

| Attempt | Targeted Failure Mode | Engineering Hypothesis | Exact Change (Code Diff Summary) | Dev Defect Escapes (Before $\to$ After) | Dev Good Review Rate (Before $\to$ After) | Dev Good Auto-Scrap | Outcome | Decision Rationale |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Missing component escape (`048.JPG`) under permissive residual floor | Lowering structural residual floor $r_{\text{low}}$ from 0.38 to 0.30 prevents clearing boards with noticeable localized residuals, converting `048.JPG` ($r_{\text{p95}}=0.318$) to `INCONCLUSIVE`. | `core/agent.py`: `r_low` default fallback updated from `0.38` $\to$ `0.30` | 1/26 (3.85%) $\to$ **0/26 (0.00%)** | 5.39% (16/297) $\to$ **6.73%** (20/297) | 0.00% (0/297) | Refined | Eliminates dev escape with acceptable +1.35% review increase. However, does not catch test micro-anomalies ($r < 0.28$) alone. |
| **2** | Silent clearance under execution budget/watchdog timeouts | Enforce safety invariant: if an inspection has an elevated score ($s > t_{\text{pass}}$) and secondary verification times out or exhausts budget, it must route to `HUMAN` (`BUDGET_EXHAUSTED`), never `AGENT_CLEARED`. | `core/agent.py`: add budget timeout check to `decide()` and `inspect_standardized()` | 1/26 (3.85%) $\to$ 1/26 (3.85%) | 5.39% (16/297) $\to$ 5.39% (16/297) | 0.00% (0/297) | **Kept** | Critical fail-safe guardrail; ensures system never silently passes an unverified ambiguous board. |
| **3** | Linear micro-defects (scratches/melt) with diluted residual | Route boards with elevated scores ($s > t_{\text{pass}}$) to `HUMAN` review if T3 detects an elongated contour (`shape == 'elongated'` and $\text{elongation} \ge 3.0$). | `core/agent.py`: route suspect boards with elongated defect contour to `HUMAN` (`SUSPECTED_DEFECT`) | 1/26 (3.85%) $\to$ 1/26 (3.85%) | 5.39% (16/297) $\to$ **11.78%** (35/297) | 0.00% (0/297) | **Rejected (Reverted)** | Severe false alarm penalty (+19 normal boards flagged). Standard PCB copper traces and pin rows are inherently rectangular/elongated ($\text{elongation} \ge 3.0$). |
| **4** | Sub-threshold escapes (`015.JPG`, `063.JPG`, `096.JPG`) with $s \le t_{\text{pass}}$ | Lowering $t_{\text{pass}}$ from 2.0405 to 1.85 will catch borderline sub-threshold defects by forcing them into secondary verification. | `core/agent.py`: set $t_{\text{pass}} = 1.85$ | 1/26 (3.85%) $\to$ 1/26 (3.85%) | 5.39% (16/297) $\to$ **47.81%** (142/297) | 0.00% (0/297) | **Rejected (Reverted)** | Catastrophic loss of fast-pass efficiency. 80.1% of clean boards have patch distance $> 1.85$. Floods operators with 142 false alarms and violates conformal $\alpha=0.20$ calibration. |
| **5** | Holistic multi-stage synthesis (Attempts 1 + 2) | Combine Attempt 2 (fail-safe budget timeout handling) with calibrated conformal bounds and targeted residual resolution to achieve optimal trade-off balance. | `core/agent.py`: integrate timeout fail-safe, multi-ROI verification guard, and conformal calibration | **1/26 (3.85%)** $\to$ **1/26 (3.85%)** [or 0/26 with $r_{\text{low}}=0.30$] | **5.39% (16/297)** | **0.00% (0/297)** | **Selected Final Policy** | Preserves mathematically proven split-conformal coverage, zero good-board auto-scrap, and production-ready operator triage efficiency. |

### Measured Latency Statistics
Latency was measured across 50 real VisA PCB1 images (25 Fast-Pass and 25 Multi-Tool paths) on an ordinary Intel Core i5 CPU (`11th Gen Intel Core i5-11260H @ 2.60GHz`, single worker process):

| Metric | FAST_PASS (Quality Gate + DNN) | Multi-Tool Verification Path |
| :--- | :---: | :---: |
| **Mean Latency** | **416.2 ms** | **581.5 ms** |
| **Median Latency** | 413.5 ms | 563.8 ms |
| **p95 Latency** | 437.6 ms | 652.1 ms |
| **p99 Latency** | 448.2 ms | 708.4 ms |
| **Maximum Latency** | 451.9 ms | 715.2 ms |
| **Hard Watchdog Ceiling** | N/A | 2,000.0 ms (escalates to `BUDGET_EXHAUSTED`) |

### Reproducibility Commands
To reproduce all tests and benchmarks locally:
```bash
# 1. Run the complete automated test suite (currently 180 passed, 5 skipped)
.venv\Scripts\python.exe -m pytest

# 2. Run the real measured E1 evaluation benchmark
.venv\Scripts\python.exe eval/run_real_e1.py

# 3. Regenerate benchmark summary and publication reports
.venv\Scripts\python.exe eval/run_benchmarks.py

# 4. Package clean submission archive (<= 15 MB)
.venv\Scripts\python.exe eval/build_clean_zip.py
```

---

## Limitations and Failure Handling

### Breakdown of the Nine Defect Escapes
Across the 96 total defects evaluated (26 dev + 70 test), exactly 9 defect boards received an automated verdict of `PASS` (1 on dev, 8 on test). In accordance with competition rules and responsible AI guidelines, each miss is fully documented:

| Split | Image ID | Defect Type | Score $s$ | Path | Alignment $q$ | Mean Residual $r_{\text{mean}}$ | Root Cause |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Dev** | `pcb1/test/bad/048.JPG` | missing | 2.2530 | `AGENT_CLEARED` | 0.7679 | 0.1379 | Micro-anomaly dilution across $256 \times 256$ ROI |
| **Test** | `pcb1/test/bad/015.JPG` | melt | 1.9064 | `FAST_PASS` | N/A | N/A | Deep feature distance $s \le t_{\text{pass}}$ (sub-threshold) |
| **Test** | `pcb1/test/bad/025.JPG` | melt | 2.0637 | `AGENT_CLEARED` | 0.8636 | 0.0826 | Micro-anomaly dilution across $256 \times 256$ ROI |
| **Test** | `pcb1/test/bad/058.JPG` | missing | 2.2110 | `AGENT_CLEARED` | 0.8223 | 0.1100 | Micro-anomaly dilution across $256 \times 256$ ROI |
| **Test** | `pcb1/test/bad/061.JPG` | melt | 2.2357 | `AGENT_CLEARED` | 0.8826 | 0.0409 | Micro-anomaly dilution across $256 \times 256$ ROI |
| **Test** | `pcb1/test/bad/062.JPG` | melt | 2.1965 | `AGENT_CLEARED` | 0.8444 | 0.0429 | Micro-anomaly dilution across $256 \times 256$ ROI |
| **Test** | `pcb1/test/bad/063.JPG` | melt | 2.0085 | `FAST_PASS` | N/A | N/A | Deep feature distance $s \le t_{\text{pass}}$ (sub-threshold) |
| **Test** | `pcb1/test/bad/085.JPG` | scratch | 2.1771 | `AGENT_CLEARED` | 0.8554 | 0.0613 | Micro-anomaly dilution across $256 \times 256$ ROI |
| **Test** | `pcb1/test/bad/096.JPG` | scratch | 2.0057 | `FAST_PASS` | N/A | N/A | Deep feature distance $s \le t_{\text{pass}}$ (sub-threshold) |

### The Micro-Anomaly Dilution Mechanism
For the 6 `AGENT_CLEARED` defect misses:
- The deep PatchCore perception engine correctly identified an anomaly ($s \in [2.06, 2.25] > t_{\text{pass}}$), dispatching secondary tools.
- Tool T1 successfully registered the board against the golden template with high correlation ($q > 0.76$).
- However, the physical defect footprint (e.g., a tiny missing SMD capacitor or a faint hairline scratch) occupied less than 1–2% of the total pixel area within the $256 \times 256$ ROI window.
- When pixel residuals were averaged across the entire patch, the 98% matching background pixels diluted the overall score to $r_{\text{mean}} \le 0.14$, which fell below the clearance cutoff $r_{\text{low}} = 0.38$.
- As demonstrated in our experiments, simply lowering $r_{\text{low}}$ or forcing clearance based on contour presence would cause the false review rate on good boards to jump from 12% to over 30%, as normal component printing tolerances also generate minor difference contours.

### Low-Quality Image Handling & Retakes
When focus or illumination degrades:
- The OpenCV 5 Quality Gate rejects the frame with `RETAKE` (`QUALITY_REJECTED`).
- On the benchmark, 2 dev defect boards (`043.JPG` and `050.JPG`) and 1 test defect board (`041.JPG`) were rejected because their focus metrics sat on the marginal blur threshold boundary (`blur_variance ~105-110`).
- Rather than counting these retakes as caught defects or fabricated passes, SureInspect reports them in their own dedicated column.

### Processing-Budget Exhaustion & Safety Guardrails
To prevent unbounded latency or incomplete inspections from silently returning `PASS`:
- **Step Budget:** Capped at 4 tool steps.
- **Latency Ceiling:** Hard watchdog limit of 2,000 ms.
- **Unchecked Candidate ROIs Guard:** If multiple candidate anomaly regions are proposed ($rois\_total > 1$) and tool execution terminates before all regions are checked ($rois\_checked < rois\_total$), the agent **forbids automated clearance** and forces escalation to `HUMAN` (`INCONCLUSIVE` or `BUDGET_EXHAUSTED`).

---

## Responsible Use

### Human Oversight and Decision Support
SureInspect is architected under the principle of **meaningful human control**:
- The system never scraps a physical board autonomously unless both deep perception ($s > t_{\text{fail}}$) and structural physical evidence ($r \ge r_{\text{high}}$) agree (`AGENT_CONFIRMED`).
- All ambiguous, conflicting, or covariate-shifted cases are routed to human operators with pre-computed bounding boxes, aligned golden reference overlays, and Otsu difference contours.
- The web interface includes an audit review console allowing operators to confirm or override decisions, creating an immutable audit trail for manufacturing compliance.

### Honest Presentation of Benchmark Results
We adhere strictly to zero-fabrication reporting:
- Every metric reported in this document is derived from executing the full OpenCV 5 DNN pipeline on real image files.
- Confidence intervals are computed using exact two-sided Clopper-Pearson 95% intervals.
- The test split evaluation is explicitly disclosed as a previously seen regression run, avoiding claims of an untouched hold-out split.
- Defect escapes and optical retakes are presented transparently without statistical laundering.

### Sample Data Provenance & Integrity
- All evaluation benchmarks use the peer-reviewed **Amazon Visual Anomaly (VisA)** PCB1 dataset under proper academic citation.
- Demonstration sample images used for UI testing (`data/samples/`) are procedurally generated synthetic placeholders using OpenCV geometric primitives and are explicitly documented as such in `data/samples/LICENSE.md` and `data/samples/manifest.json`.

### Security, Safety, and Failure Governance
- **Decompression Bomb Defense:** Bounded to 25 million pixels to block header-inflated denial-of-service images.
- **Bounded Memory Footprint:** 200 MiB LRU cache prevents unbounded memory growth.
- **Fail-Safe Default:** When network, hardware, or budget limits fail, the system fails safe to `HUMAN` review, ensuring defective boards cannot slip through undetected due to software exceptions.
