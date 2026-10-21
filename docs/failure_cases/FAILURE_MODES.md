# SureInspect: Failure Modes, Edge Cases & Escalation Analysis

> **Author:** Person 1 (@Biswabasini90) — OpenCV & ML Lead  
> **Project:** SureInspect (OpenCV AI Competition 2026)  
> **Topic:** Safe Rejection, Conformal Bound Breakdown & Operator Escalation Protocols  

---

## 1. Executive Summary

In mission-critical industrial inspection, an unflagged defect (false escape) is catastrophic, while an automated false alarm halts production. Traditional deep-learning vision models produce uncalibrated softmax scores that silently fail under out-of-distribution conditions. 

SureInspect addresses this through a **multi-stage defense-in-depth architecture**:
1. **Physical Quality Gate (`core/quality_gate.py`):** Pre-inference rejection of corrupted frames.
2. **Conformal Perception (`core/perception.py` & `core/calibration.py`):** Finite-sample bounded distance scores.
3. **OpenCV Evidence Tool Validation (`core/tools.py`):** Deterministic physical checks (ORB+ECC alignment, photometric matching, pixel residuals, and shape extraction).
4. **Agent Policy Routing (`core/agent.py`):** Deterministic escalation to human experts (`HUMAN`) or line retakes (`RETAKE`) when statistical assumptions fail.

This document analyzes the 6 primary non-pass decision paths and failure recovery modes.

---

## 2. Failure Mode Taxonomy

```
                          Incoming Standardized Image
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Stage 1: Quality Gate   │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
       Gate Fails (Blur/Exposure/Glare)               Gate Passes
              ▼                                               ▼
       [QUALITY_REJECTED]                       ┌───────────────────────────┐
       Action: RETAKE                           │ Stage 2: PatchCore Scorer │
                                                └─────────────┬─────────────┘
                                                              │
                                     ┌────────────────────────┴────────────────────────┐
                             score <= t_pass                                    score > t_pass
                                     │                                                 │
                      ┌──────────────┴──────────────┐                           ┌──────┴──────┐
                Lighting OK                   Lighting Shift              Run Tool T1: Alignment
                      ▼                             ▼                           │             │
                 [FAST_PASS]            [LIGHTING_SHIFT_REVIEW]             Align Fails   Align Passes
                 Action: PASS                 Action: HUMAN                     ▼             ▼
                                                                         [ALIGN_FAILED] Run T2/T3
                                                                          Action: HUMAN       │
                                                               ┌──────────────────────────────┤
                                                       Residual >= r_high              Residual <= r_low
                                                               ▼                              ▼
                                                       [AGENT_CONFIRMED]             [AGENT_CLEARED] /
                                                         Action: FAIL                [EVIDENCE_CONFLICT]
```

---

## 3. Deep Dive: Failure Modes & Escalation Protocols

### 3.1 Failure Mode 1: Quality Gate Rejection (`QUALITY_REJECTED`)
* **Trigger:** Image Laplacian variance below `blur_thr`, underexposure (`dark_fraction > dark_max`), overexposure (`bright_fraction > bright_max`), or specular glare (`glare_fraction > glare_thr`).
* **Root Cause:** Vibration on conveyor belt, conveyor lighting flicker, or high-angle reflection from glossy solder points.
* **Agent Handling:**
  * Bypasses the DNN backbone completely to conserve CPU cycles.
  * Verdict: `RETAKE`.
  * User action: Trigger camera autofocus, clean optics, or re-acquire the frame.
  * Stored in audit storage but excluded from anomaly statistics.

### 3.2 Failure Mode 2: Illumination Covariate Shift (`LIGHTING_SHIFT_REVIEW`)
* **Trigger:** CIELAB $L^*$ mean differs from baseline normal by more than $3.0 \times \text{SD}$, or $L^*$ standard deviation ratio is outside $[0.7, 1.3]$.
* **Root Cause:** Factory ambient lighting change (daylight vs night shift, lamp degradation).
* **Statistical Implication:** Non-parametric conformal bounds assume exchangeability between calibration normals and live test images:
  $$P(\text{score} > t_{\text{fail}} \mid \text{Normal}) \le \alpha_{\text{fail}}$$
  When illumination shifts, exchangeability is violated, meaning $t_{\text{pass}}$ and $t_{\text{fail}}$ no longer provide rigorous coverage guarantees.
* **Agent Handling:**
  * Does **not** reject the frame automatically.
  * If score is in pass range, routes to `LIGHTING_SHIFT_REVIEW` with verdict `HUMAN` (suggested: `PASS`).
  * If secondary alignment and photometric matching verify normal structure, upgrades to `AGENT_CLEARED`.

### 3.3 Failure Mode 3: Alignment Failure (`ALIGN_FAILED`)
* **Trigger:** ORB keypoint matching fails ($<10$ inliers or $<25\%$ inlier ratio), rotation exceeds $45^\circ$, scale is outside $[0.8, 1.25]$, valid overlap $<60\%$, or alignment quality $q < q_{\text{min}}$.
* **Root Cause:** Board is severely misoriented on tray, warped, partially occluded, or completely fractured.
* **Agent Handling:**
  * Cannot construct a valid difference comparison image (`cmp.valid < MIN_VALID_PX`).
  * Avoids fabricating zero-residual numbers.
  * Verdict: `HUMAN` with `suggested_decision="FAIL"`.

### 3.4 Failure Mode 4: Evidence Conflict (`EVIDENCE_CONFLICT`)
* **Trigger:** Deep PatchCore feature distance exceeds $t_{\text{fail}}$ (indicating anomalous deep semantic representations), but optical difference residual $r_{95}$ is below $r_{\text{low}}$ (indicating no physical pixel-level deviation).
* **Root Cause:**
  1. Benign component lot variations (e.g. different text print or vendor logo on IC chip).
  2. High-frequency texture shift that triggers CNN filters without physical solder bridge or scratch.
* **Agent Handling:**
  * The system refuses to let perception override physical evidence, or vice versa.
  * Verdict: `HUMAN`.
  * Presents heatmap and side-by-side golden template comparison to the human operator for disambiguation.

### 3.5 Failure Mode 5: Inconclusive Secondary Evidence (`INCONCLUSIVE`)
* **Trigger:** Perception score is in the ambiguous margin $(t_{\text{pass}}, t_{\text{fail}}]$, and structural residual falls in the gray boundary $(r_{\text{low}}, r_{\text{high}})$.
* **Root Cause:** Borderline defect (e.g., minor solder bead or borderline line-width narrowing) that cannot be classified with statistical certainty.
* **Agent Handling:**
  * Escalates to human operator (`HUMAN`).
  * Provides Otsu contour bounding box and rotated rectangle aspect ratio to accelerate human triage.

### 3.6 Failure Mode 6: Budget Exhaustion (`BUDGET_EXHAUSTED`)
* **Trigger:** Step counter reaches `step_budget` (4 steps) before reaching an automated clear/confirm condition.
* **Root Cause:** Complex multi-defect board where successive ROIs require iterative alignment attempts without resolving confidence.
* **Agent Handling:**
  * Halts processing to enforce latency ceilings ($<200\text{ ms}$).
  * Verdict: `HUMAN`.

---

## 4. Summary Matrix

| Decision Path | Verdict | Primary Cause | Primary Action |
| :--- | :--- | :--- | :--- |
| `QUALITY_REJECTED` | `RETAKE` | Focus blur, exposure out of bounds, or glare | Request frame re-capture |
| `LIGHTING_SHIFT_REVIEW` | `HUMAN` | Factory ambient illumination shift | Operator review / lighting recalibration |
| `ALIGN_FAILED` | `HUMAN` | Board rotation $>45^\circ$ or low overlap | Inspect physical alignment & fixture |
| `EVIDENCE_CONFLICT` | `HUMAN` | Perception vs template difference contradiction | Disambiguate component batch variation |
| `INCONCLUSIVE` | `HUMAN` | Marginal anomaly in score & residual gray zone | Triage subtle defect |
| `BUDGET_EXHAUSTED` | `HUMAN` | Tool step budget limit reached | Operator manual decision |

---

## 5. Empirical Defect Escape Breakdown & Dilution Analysis

In benchmark experiment E1 across the 96 total defects (26 dev defects and 70 test defects), exactly 9 defect boards received a final automated verdict of `PASS` (1 on dev, 8 on test). This section provides the exhaustive empirical accounting of each escape alongside its root-cause mechanism.

### 5.1 Escaped Defect Breakdown Table

| Split | Image Path | Ground-Truth Defect Type (from VisA) | PatchCore Score | Decision | Decision Path | Alignment Quality ($q$) | Residual Mean ($r_{\text{mean}}$) | Secondary ROIs Checked |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dev** | `pcb1/test/bad/048.JPG` | `missing` | 2.2530 | `PASS` | `AGENT_CLEARED` | 0.7679 | 0.1379 | 1 / 1 |
| **Test** | `pcb1/test/bad/015.JPG` | `melt` | 1.9064 | `PASS` | `FAST_PASS` | N/A | N/A | N/A |
| **Test** | `pcb1/test/bad/025.JPG` | `melt` | 2.0637 | `PASS` | `AGENT_CLEARED` | 0.8636 | 0.0826 | 1 / 1 |
| **Test** | `pcb1/test/bad/058.JPG` | `missing` | 2.2110 | `PASS` | `AGENT_CLEARED` | 0.8223 | 0.1100 | 1 / 1 |
| **Test** | `pcb1/test/bad/061.JPG` | `melt` | 2.2357 | `PASS` | `AGENT_CLEARED` | 0.8826 | 0.0409 | 2 / 2 |
| **Test** | `pcb1/test/bad/062.JPG` | `melt` | 2.1965 | `PASS` | `AGENT_CLEARED` | 0.8444 | 0.0429 | 1 / 1 |
| **Test** | `pcb1/test/bad/063.JPG` | `melt` | 2.0085 | `PASS` | `FAST_PASS` | N/A | N/A | N/A |
| **Test** | `pcb1/test/bad/085.JPG` | `scratch` | 2.1771 | `PASS` | `AGENT_CLEARED` | 0.8554 | 0.0613 | 1 / 1 |
| **Test** | `pcb1/test/bad/096.JPG` | `scratch` | 2.0057 | `PASS` | `FAST_PASS` | N/A | N/A | N/A |

### 5.2 Micro-Anomaly Photometric Dilution Mechanism

An in-depth investigation into these 9 escapes reveals two distinct failure mechanisms:

#### Mechanism 1: Sub-Threshold Deep Perception ($s \le t_{\text{pass}}$)
- **Affected Samples:** 3 defect images (`pcb1/test/bad/015.JPG`, `063.JPG`, `096.JPG`).
- **Explanation:** In these three cases, the deep feature extraction distance produced by the WideResNet-50 backbone fell below the safe pass threshold $t_{\text{pass}} = 2.0405$. For subtle defects (a faint hairline scratch or shallow solder melt seam), the feature activations in layer-2 and layer-3 did not elevate the nearest-neighbor distance above normal calibration board variance. Under SureInspect's decision rule, any sample with $s \le t_{\text{pass}}$ and normal illumination routes directly to `FAST_PASS` to conserve computational budget, bypassing secondary evidence tools.

#### Mechanism 2: Micro-Anomaly Photometric Dilution Across ROI Window ($r_{\text{mean}} \le r_{\text{low}}$)
- **Affected Samples:** 6 defect images (`pcb1/test/bad/048.JPG` on dev, and `025.JPG`, `058.JPG`, `061.JPG`, `062.JPG`, `085.JPG` on test).
- **Explanation:** In all 6 cases, the deep PatchCore score correctly flagged the board as anomalous/ambiguous ($s \in (2.04, 2.29]$), appropriately dispatching Evidence Tools T1 (ORB+ECC alignment) and T3 (ROI residual measurement).
- Alignment succeeded with high confidence ($q \in [0.7679, 0.8826] \ge q_{\text{min}} = 0.65$), producing sub-pixel homography registration against the golden template.
- However, when the secondary tool extracted the $256 \times 256$ ROI patch centered on the anomaly heatmap peak and computed the mean absolute difference $r_{\text{mean}}$, the physical defect footprint (e.g., a tiny 0402 SMD missing, a single scratched trace, or minor solder bridge) occupied less than 1–3% of the total pixel area in the patch.
- Consequently, when the absolute differences were averaged across the entire $256 \times 256$ window, the residual was photometrically diluted by the 97–99% matching background pixels, yielding $r_{\text{mean}} \in [0.0409, 0.1379]$, which falls well below the clearance threshold $r_{\text{low}} = 0.38$.
- Because $r_{\text{mean}} \le r_{\text{low}}$, the agent policy determined that the deviation was benign registration or lighting variance rather than a structural fault, erroneously upgrading the board to `AGENT_CLEARED` (`PASS`).

### 5.3 Engineering Remediation Roadmap
To eliminate micro-anomaly dilution in future iterations:
1. **Adaptive Sub-Patch Bounding:** Instead of evaluating the mean residual over a fixed $256 \times 256$ window, compute connected-component bounding boxes directly on thresholded heatmap contours and restrict residual calculation strictly to the bounding hull.
2. **High-Percentile Residual Thresholding ($r_{95}$ / $r_{99}$):** Rather than evaluating $r_{\text{mean}}$, enforce dual criteria requiring both $r_{\text{mean}} \le r_{\text{low}}$ AND peak residual $r_{95} \le r_{\text{peak}}$ before permitting `AGENT_CLEARED`.
3. **Multi-Scale Template Differencing:** Evaluate difference maps across multiple pyramid levels to isolate localized high-frequency spatial discrepancies.
