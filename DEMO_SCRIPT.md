# SureInspect: Competition Video Demo Script (5-Minute Maximum)

> **Event:** OpenCV AI Competition 2026  
> **Project:** SureInspect — Selective Inspection & Operator Triage System  
> **Team:** Biswabasini Prasad (@Biswabasini90), Gyantwik (@Gyantwik), Khushbu (@Khushbu656)  
> **Maximum Duration:** 5 minutes (300 seconds)  
> **Hardware / Environment Setup:** Local development laptop running FastAPI backend (`localhost:8000`) and React frontend (`localhost:5173`) with OpenCV 5.0.0 headless and real trained weights in `artifacts/snapshots/`.

---

## Pre-Recording Checklist & Staging Preparation

Before pressing Record (e.g. in OBS Studio or screen recorder):
1. **Start the Live Backend:**
   ```bash
   .venv\Scripts\python.exe -m backend.main --host 127.0.0.1 --port 8000
   ```
2. **Start the Frontend UI:**
   ```bash
   cd frontend && npm run dev
   ```
   Open `http://localhost:5173` in a clean browser window at 100% zoom (1920×1080).
3. **Verify Sample Image Availability:**
   - Normal board: `datasets/VisA_pytorch/1cls/pcb1/test/good/0000.JPG` (or sample demo image `data/samples/pcb1_good_01.png`)
   - Ambiguous defect board: `datasets/VisA_pytorch/1cls/pcb1/test/bad/025.JPG` or `048.JPG`
   - Blur/rejection board: `data/samples/pcb1_blur_01.png`
   - Real severe defect board: `datasets/VisA_pytorch/1cls/pcb1/test/bad/000.JPG` (spurious solder bridge)
4. **Open Terminal Window:** Keep a clean PowerShell terminal tab ready with `eval/results/BENCHMARK_REPORT.md` and test suite execution.

---

## Minute-by-Minute Script & Staging

### Segment 1: Introduction & The Core Industrial Problem (0:00 – 0:30)

* **Visual on Screen:**
  - Fullscreen view of the SureInspect web dashboard showing the PCB inspection workspace, with clean modern UI, status indicators, and camera feed ingest.
  - Picture-in-picture or voiceover of the presenter.
* **Spoken Script:**
  > "Hello, judges! Welcome to **SureInspect**, our entry for the OpenCV AI Competition 2026.
  >
  > In high-throughput electronics manufacturing, computer vision systems face an excruciating dilemma: automated false alarms halt the factory line, costing tens of thousands of dollars per hour, while an unflagged defect that escapes to a customer is catastrophic.
  >
  > Traditional deep learning classifiers output overconfident softmax probabilities that break under factory lighting shifts and minor alignment variations. SureInspect solves this by replacing fragile single-threshold classification with a **hybrid, defense-in-depth inspection system** combining OpenCV 5, split-conformal statistical calibration, and deterministic OpenCV evidence verification tools."
* **Presenter Action:** Mouse hover across the dashboard overview and model status pill (`pcb1-wide_resnet50_2-256-edc526f4da11`).

---

### Segment 2: Architecture & OpenCV 5 Substantive Integration (0:30 – 1:00)

* **Visual on Screen:**
  - Transition to `docs/architecture.mmd` flowchart diagram or slides highlighting the 5 sequential inspection stages:
    1. OpenCV 5 Quality Gate
    2. Conformal PatchCore DNN Perception
    3. Multi-Scale Alignment & Photometric Normalization
    4. ROI Structural Residual Analysis
    5. Pure Policy Decision Routing
* **Spoken Script:**
  > "Our architecture is powered by **OpenCV 5.0.0** at every critical juncture:
  >
  > First, an OpenCV 5 physical Quality Gate checks Laplacian blur variance, exposure balance, and specular glare before running neural inference.
  >
  > Next, OpenCV 5's DNN module runs our WideResNet-50 backbone ONNX model directly in C++, extracting localized patch embeddings that are matched against a coreset memory bank to produce calibrated anomaly heatmaps.
  >
  > When an anomaly score is ambiguous, the agent dispatches deterministic OpenCV tools: Tool T1 aligns the board against golden reference templates using ORB feature detection with RANSAC and sub-pixel Euclidean ECC refinement. Tool T2 normalizes lighting and computes structural pixel residuals over candidate ROIs, while Tool T3 measures contour geometry with minimum area bounding rectangles.
  >
  > Every step is bounded by a strict tool budget to ensure real-time deterministic performance on low-power CPU hardware."
* **Presenter Action:** Point to the OpenCV 5 DNN and ORB/ECC alignment blocks on the architecture diagram.

---

### Segment 3: Live Demonstration — Clean Normal Board & Fast-Pass (1:00 – 1:45)

* **Visual on Screen:**
  - Switch to the live SureInspect browser UI.
  - Select or drag-and-drop a known normal PCB image (`test/good/0000.JPG`).
  - Click **'Inspect Board'**.
  - Show the live results panel rendering:
    - Verdict: `PASS`
    - Decision Path: `FAST_PASS`
    - Perception Score: `~1.38` (well below $t_{\text{pass}} = 2.0405$)
    - Execution Trace: Step 1 Quality Gate (PASSED), Step 2 PatchCore Perception (COMPLETED)
    - Latency: `~410 ms` on Intel i5 CPU
* **Spoken Script:**
  > "Let's see SureInspect in action. Here we load a normal board from the Amazon VisA PCB1 test set.
  >
  > Within 416 milliseconds on CPU, the quality gate validates the frame, and the OpenCV 5 DNN engine extracts patch features.
  >
  > Because the PatchCore anomaly score of 1.38 falls comfortably below our mathematically calibrated pass threshold $t_{\text{pass}} = 2.04$, the agent issues an immediate **FAST_PASS**. Secondary verification tools are skipped entirely, saving compute cycles and keeping the production line moving at maximum throughput."
* **Presenter Action:** Scroll through the inspection timeline trace in the UI to demonstrate that only 2 steps were used and no unnecessary tools were executed.

---

### Segment 4: Live Demonstration — Ambiguous Defect & Tool Verification (1:45 – 2:45)

* **Visual on Screen:**
  - Load a defect board (`test/bad/025.JPG` or `test/bad/000.JPG`).
  - Click **'Inspect Board'**.
  - Watch the UI render the multi-step execution trace live:
    - Step 1: Quality Gate (PASSED)
    - Step 2: PatchCore Perception (Score `> 2.05`, anomaly heatmap overlay rendered in red/yellow)
    - Step 3: Tool T1 Alignment (Template `pcb1-t01`, quality `q = 0.864`, sub-pixel affine matrix computed)
    - Step 4: Tool T2 ROI Compare (Structural residual map computed over the candidate ROI)
    - Step 5: Tool T3 Zoom Measure (Difference contour isolated, aspect ratio & area percentage calculated)
    - Final Verdict: `FAIL` (`AGENT_CONFIRMED`) or `HUMAN` (`SUSPECTED_DEFECT` / `INCONCLUSIVE`)
* **Spoken Script:**
  > "Now, let's test a challenging defective board.
  >
  > The deep perception engine identifies an anomaly in the bottom trace and flags a borderline score above $t_{\text{pass}}$. Instead of making a premature guess, the agent initiates multi-tool visual verification:
  >
  > First, Tool T1 extracts ORB keypoints and computes homography against our golden reference template, refining it with OpenCV ECC to eliminate physical board vibration.
  >
  > Next, Tool T2 warps the golden template into the test coordinate frame, matches photometric illumination, and computes the structural residual across the region of interest.
  >
  > Finally, Tool T3 isolates the defect contour using Otsu thresholding, calculating its exact area and shape class. The agent combines this deep and physical evidence to issue an **AGENT_CONFIRMED FAIL** or route borderline ambiguities to human operators with pre-extracted bounding overlays."
* **Presenter Action:** Click the 'View Heatmap Overlay' and 'Toggle Golden Reference Alignment' buttons in the frontend to show judges the visual alignment and ROI difference contour.

---

### Segment 5: Live Demonstration — Quality Gate Rejection (Blur / Glare) (2:45 – 3:25)

* **Visual on Screen:**
  - Select the out-of-focus sample image (`pcb1_blur_01.png` or simulate blur).
  - Click **'Inspect Board'**.
  - Show immediate rejection:
    - Verdict: `RETAKE`
    - Decision Path: `QUALITY_REJECTED`
    - Quality Gate Status: `FAILED` (Laplacian variance `34.2 < 110.0` blur threshold)
    - Steps Used: `0` (DNN inference completely bypassed)
    - Latency: `< 15 ms`
* **Spoken Script:**
  > "What happens when physical conditions degrade on the factory floor?
  >
  > If a conveyor belt vibrates or lighting flickers, feeding a blurry image into a deep neural network produces meaningless hallucinated features.
  >
  > Here, an out-of-focus board triggers the OpenCV 5 Quality Gate. OpenCV calculates the Laplacian variance, detecting that focus is below our live-calibrated threshold.
  >
  > In under 15 milliseconds, the system halts and issues a **RETAKE** verdict. Crucially, the expensive DNN backbone is bypassed entirely. No defect score is fabricated, and the system prompts the line operator to clean optics or re-trigger the frame."
* **Presenter Action:** Highlight the Quality Gate metric card on the dashboard showing the blur variance and glare percentage.

---

### Segment 6: Benchmark Evaluation, Baseline Comparison & Honest Failure Analysis (3:25 – 4:20)

* **Visual on Screen:**
  - Display the benchmark comparison table from `eval/results/BENCHMARK_REPORT.md`:
    - Cautious Baseline vs Default Baseline vs SureInspect
    - Test Defect Interception: `87.14%` (61/70 caught)
    - Good-board review reduction: `27% -> 12%`
    - Escaped Defect Breakdown Table (Table 1.3)
* **Spoken Script:**
  > "We evaluated SureInspect across all 493 images of the Amazon VisA PCB1 benchmark with zero simulated or fabricated metrics.
  >
  > Compared to a single-threshold cautious baseline, SureInspect cuts operator review on good boards from **27% down to 12%**, while eliminating catastrophic automated scrap. Total defect interception on the test set is **87.14%** (61 of 70 defects intercepted).
  >
  > In the spirit of scientific integrity and responsible AI, we document all 9 defect escapes openly:
  > Three escapes occurred because faint micro-scratches fell just below the initial deep feature threshold. Six escapes occurred due to *micro-anomaly dilution*—where a defect occupying less than 2% of a 256×256 window was averaged down when computing mean residual.
  >
  > Rather than tuning thresholds post-hoc to hide these misses, we document them with exact image IDs, coordinates, and failure modes in our failure report."
* **Presenter Action:** Scroll through Table 1.3 in `eval/results/BENCHMARK_REPORT.md` showing the exact image filenames and defect categories.

---

### Segment 7: Human Oversight, Reproducibility & Real-World Impact (4:20 – 5:00)

* **Visual on Screen:**
  - Show the Operator Triage Queue in the web UI where human inspectors can approve, reject, or comment on flagged boards.
  - Show terminal running `pytest` (159 passing unit tests) and `eval/build_clean_zip.py`.
  - Closing title slide with team names, GitHub repository link, and OpenCV AI Competition 2026 logo.
* **Spoken Script:**
  > "SureInspect is designed as an **Operator Triage and Decision-Support System**, not an unchecked autonomous decision maker.
  >
  > Every uncertain inspection is accompanied by calibrated conformal confidence intervals, aligned reference overlays, and Otsu difference contours to empower human operators to make rapid, informed decisions.
  >
  > The entire repository is containerized with Docker, covered by 159 automated tests, runs locally on standard CPU hardware without cloud dependencies, and is fully reproducible with a single command.
  >
  > Thank you for reviewing SureInspect!"
* **Presenter Action:** Final smile and wave or transition to competition title card.

---

## Technical Notes for the Speaker

1. **Avoid Overpromising:** Never claim SureInspect achieves "100% defect detection" or "0% false alarms". Stick to the verified measured metrics: 87.14% defect interception (61/70 test defects caught), 12% good-board review, and 8 escapes on the seen test set.
2. **Emphasize CPU Performance:** Highlight that all benchmarks run on an ordinary Intel Core i5 CPU without requiring expensive GPUs, making it ideal for edge industrial deployment.
3. **OpenCV 5 Visibility:** Explicitly mention `cv2.dnn.readNetFromONNX`, `cv2.estimateAffinePartial2D`, `cv2.findTransformECC`, and `cv2.Laplacian`.
