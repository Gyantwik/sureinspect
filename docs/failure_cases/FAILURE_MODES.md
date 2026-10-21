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
