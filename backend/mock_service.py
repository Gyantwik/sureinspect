"""
backend/mock_service.py
Deterministic mock inspection engine generating realistic inspection traces,
asset payloads, and decision variants matching Section 4 of SureInspect v8.1.
"""
from __future__ import annotations

import base64
import time
import uuid
from typing import Dict, Any, Optional, List

from backend.schemas import (
    InspectionResponse,
    SnapshotInfo,
    ImageMetadata,
    ThresholdsInfo,
    CalibrationInfo,
    QualityGateResult,
    AlignmentInfo,
    MeasurementStats,
    AssetsInfo,
    TraceStep,
    TimingInfo,
    CounterfactualItem,
    DecisionEnum,
    DecisionPathEnum,
    ToolEnum,
    SamplesResponse,
    SampleItem,
    CalibrationSummaryResponse,
    FeedbackResponse,
)

# ---------------------------------------------------------------------------
# Pre-computed lightweight PNG asset placeholders (1x1 transparent PNG)
# ---------------------------------------------------------------------------
TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

TINY_PNG_BYTES = base64.b64decode(TINY_PNG_B64.split(",", 1)[1])

# ---------------------------------------------------------------------------
# Mock Catalog & Samples
# ---------------------------------------------------------------------------

MOCK_SAMPLES = [
    SampleItem(
        id="pcb1_good_01",
        category="pcb1",
        display_name="PCB 1 - Clean Board (Normal)",
        ground_truth="good",
        attribution="Amazon VisA Dataset (CC BY 4.0)",
        image_url="/api/samples/pcb1_good_01/image",
        role="FAST_PASS",
    ),
    SampleItem(
        id="pcb1_borderline_01",
        category="pcb1",
        display_name="PCB 1 - Borderline Variation (Agent Cleared)",
        ground_truth="good",
        attribution="Amazon VisA Dataset (CC BY 4.0)",
        image_url="/api/samples/pcb1_borderline_01/image",
        role="AGENT_CLEARED",
    ),
    SampleItem(
        id="pcb1_bad_missing_01",
        category="pcb1",
        display_name="PCB 1 - Missing IC Component",
        ground_truth="bad",
        attribution="Amazon VisA Dataset (CC BY 4.0)",
        image_url="/api/samples/pcb1_bad_missing_01/image",
        role="AGENT_CONFIRMED",
    ),
    SampleItem(
        id="pcb1_bad_scratch_01",
        category="pcb1",
        display_name="PCB 1 - Trace Scratch (Ambiguous)",
        ground_truth="bad",
        attribution="Amazon VisA Dataset (CC BY 4.0)",
        image_url="/api/samples/pcb1_bad_scratch_01/image",
        role="HUMAN_REVIEW",
    ),
    SampleItem(
        id="pcb1_blur_01",
        category="pcb1",
        display_name="PCB 1 - Camera Out-of-Focus (Blur)",
        ground_truth="good",
        attribution="Amazon VisA Dataset (CC BY 4.0)",
        image_url="/api/samples/pcb1_blur_01/image",
        role="QUALITY_REJECTED",
    ),
]

ALLOWLISTED_SAMPLE_IDS = {s.id for s in MOCK_SAMPLES}


def get_mock_samples(category: str = "pcb1") -> SamplesResponse:
    filtered = [s for s in MOCK_SAMPLES if s.category == category]
    return SamplesResponse(samples=filtered if filtered else MOCK_SAMPLES)


def get_mock_calibration(category: str = "pcb1") -> CalibrationSummaryResponse:
    return CalibrationSummaryResponse(
        category=category,
        scope="frozen_production",
        t_pass=1.21,
        t_fail=2.05,
        alpha_pass=0.20,
        alpha_fail=0.05,
        n_calib=300,
        report_bundle_id=None,
        evaluation_status="pending",
        evaluation_summary=None,
    )


# ---------------------------------------------------------------------------
# Master Mock Decision Path Builder
# ---------------------------------------------------------------------------

def build_mock_inspection(
    decision_path: DecisionPathEnum,
    *,
    category: str = "pcb1",
    alpha_fail: float = 0.05,
    skipped_t3: bool = False,
    no_contour: bool = False,
    insp_id: Optional[str] = None,
    req_id: Optional[str] = None,
    ts: Optional[str] = None,
) -> InspectionResponse:
    """
    Builds a fully conforming InspectionResponse for any of the 10 decision paths
    or special variants (skipped-T3, no-contour).
    """
    if insp_id is None:
        insp_id = f"insp_{uuid.uuid4().hex[:8]}"
    if req_id is None:
        req_id = f"req_{uuid.uuid4().hex[:8]}"
    if ts is None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    snapshot = SnapshotInfo(
        model_version="wrn50_v1_frozen",
        bank_version="coreset_5pct_v1",
        calibration_version="calib_alpha05_v1",
        policy_version="policy_v8_1",
    )
    image_meta = ImageMetadata(
        original_size=(1024, 1024),
        standardized_size=(1024, 1024),
        scale_x=1.0,
        scale_y=1.0,
    )

    t_pass = 1.21
    t_fail = 2.05

    # 1. QUALITY_REJECTED / RETAKE
    if decision_path == DecisionPathEnum.QUALITY_REJECTED:
        qg = QualityGateResult(
            status="FAILED",
            action="REQUEST_RETAKE",
            reason="Laplacian variance 42.3 is below category focus threshold 110.0 (image is out of focus)",
            blur_variance=42.3,
            blur_threshold=110.0,
            dark_fraction=0.02,
            bright_fraction=0.01,
            glare_fraction=0.00,
            glare_threshold=0.08,
            l_mean=128.5,
            exposure_ok=True,
            lighting_shift_detected=False,
        )
        return InspectionResponse(
            schema_version="1.0",
            inspection_id=insp_id,
            request_id=req_id,
            timestamp=ts,
            engine="mock",
            category=category,
            snapshot=snapshot,
            image=image_meta,
            decision=DecisionEnum.RETAKE,
            decision_path=DecisionPathEnum.QUALITY_REJECTED,
            suggested_decision=None,
            anomaly_score=None,
            thresholds=ThresholdsInfo(
                t_pass=None,
                t_fail=None,
                alpha_pass=0.20,
                alpha_fail=alpha_fail,
                n_calib=300,
            ),
            calibration=CalibrationInfo(
                status="not_evaluated",
                claim=None,
                assumptions="Quality gate rejected frame before anomaly inference",
            ),
            quality_gate=qg,
            alignment=None,
            measurements=None,
            assets=AssetsInfo(
                input_standardized=f"/api/assets/{insp_id}_input.png",
                reference_aligned=None,
                anomaly_heatmap=None,
                contour_mask=None,
            ),
            execution_trace=[
                TraceStep(
                    step=1,
                    tool=ToolEnum.QUALITY_GATE,
                    reason="Laplacian variance 42.3 < threshold 110.0",
                    status="FAILED",
                    counts_toward_budget=False,
                    duration_ms=6.8,
                    overlay_url=None,
                )
            ],
            steps_used=0,
            step_budget=4,
            timing_ms=TimingInfo(quality=6.8, perception=0.0, tools=0.0, total=6.8),
            grounded_report="Quality gate rejected frame due to severe blur (variance 42.3 vs threshold 110.0). No anomaly inference performed.",
            counterfactuals=[],
        )

    # 2. FAST_PASS
    if decision_path == DecisionPathEnum.FAST_PASS:
        qg = QualityGateResult(
            status="PASSED",
            action=None,
            reason="Focus and illumination within normal parameters",
            blur_variance=310.4,
            blur_threshold=110.0,
            dark_fraction=0.01,
            bright_fraction=0.02,
            glare_fraction=0.01,
            glare_threshold=0.08,
            l_mean=132.1,
            exposure_ok=True,
            lighting_shift_detected=False,
        )
        score = 0.88
        return InspectionResponse(
            schema_version="1.0",
            inspection_id=insp_id,
            request_id=req_id,
            timestamp=ts,
            engine="mock",
            category=category,
            snapshot=snapshot,
            image=image_meta,
            decision=DecisionEnum.PASS,
            decision_path=DecisionPathEnum.FAST_PASS,
            suggested_decision=None,
            anomaly_score=score,
            thresholds=ThresholdsInfo(
                t_pass=t_pass,
                t_fail=t_fail,
                alpha_pass=0.20,
                alpha_fail=alpha_fail,
                n_calib=300,
            ),
            calibration=CalibrationInfo(
                status="frozen_assumption_dependent",
                claim="normal_marginal_upper_tail",
                assumptions="Marginal false-FAIL bound holds under exchangeability with training normals",
            ),
            quality_gate=qg,
            alignment=None,
            measurements=None,
            assets=AssetsInfo(
                input_standardized=f"/api/assets/{insp_id}_input.png",
                reference_aligned=None,
                anomaly_heatmap=f"/api/assets/{insp_id}_heatmap.png",
                contour_mask=None,
            ),
            execution_trace=[
                TraceStep(
                    step=1,
                    tool=ToolEnum.QUALITY_GATE,
                    reason="Verify focus and illumination before perception",
                    status="PASSED",
                    counts_toward_budget=False,
                    duration_ms=7.1,
                    overlay_url=None,
                ),
                TraceStep(
                    step=2,
                    tool=ToolEnum.PATCHCORE_PERCEPTION,
                    reason=f"Anomaly score {score:.2f} <= t_pass ({t_pass:.2f}); immediate clear pass",
                    status="COMPLETED",
                    counts_toward_budget=False,
                    duration_ms=34.2,
                    overlay_url=f"/api/assets/{insp_id}_heatmap.png",
                ),
            ],
            steps_used=0,
            step_budget=4,
            timing_ms=TimingInfo(quality=7.1, perception=34.2, tools=0.0, total=41.3),
            grounded_report="Part scored comfortably within normal parameters (score 0.88 <= pass threshold 1.21). Immediate PASS emitted via fast path.",
            counterfactuals=[
                CounterfactualItem(alpha_fail=0.01, t_fail=2.48, decision=DecisionEnum.PASS, decision_path=DecisionPathEnum.FAST_PASS, needs_rerun=False),
                CounterfactualItem(alpha_fail=0.02, t_fail=2.25, decision=DecisionEnum.PASS, decision_path=DecisionPathEnum.FAST_PASS, needs_rerun=False),
                CounterfactualItem(alpha_fail=0.05, t_fail=2.05, decision=DecisionEnum.PASS, decision_path=DecisionPathEnum.FAST_PASS, needs_rerun=False),
                CounterfactualItem(alpha_fail=0.10, t_fail=1.82, decision=DecisionEnum.PASS, decision_path=DecisionPathEnum.FAST_PASS, needs_rerun=False),
            ],
        )

    # Standard Quality Gate for remaining paths
    is_lighting_shift = (decision_path == DecisionPathEnum.LIGHTING_SHIFT_REVIEW)
    qg = QualityGateResult(
        status="PASSED",
        action=None,
        reason="Lighting shift detected in factory illumination" if is_lighting_shift else "Focus and illumination within normal limits",
        blur_variance=298.6,
        blur_threshold=110.0,
        dark_fraction=0.02,
        bright_fraction=0.01,
        glare_fraction=0.01,
        glare_threshold=0.08,
        l_mean=145.2 if is_lighting_shift else 130.4,
        exposure_ok=True,
        lighting_shift_detected=is_lighting_shift,
    )

    # Path-specific parameters
    if decision_path == DecisionPathEnum.AGENT_CLEARED:
        decision = DecisionEnum.PASS
        suggested = None
        score = 1.45  # t_pass < s <= t_fail
        alignment_ok = True
        align_quality = 0.91
        residual_mean = 0.08  # <= r_low
        calib_status = "frozen_assumption_dependent"
        calib_claim = "normal_marginal_upper_tail"
        steps_used = 3
        report = "Candidate anomaly investigated across 1 ROI. Structural residual (0.08) is below r_low (0.15), confirming normal structural variation. Decision: PASS."
    elif decision_path == DecisionPathEnum.AGENT_CONFIRMED:
        decision = DecisionEnum.FAIL
        suggested = None
        score = 2.44  # s > t_fail
        alignment_ok = True
        align_quality = 0.89
        residual_mean = 0.52  # > r_high
        calib_status = "frozen_assumption_dependent"
        calib_claim = "normal_marginal_upper_tail"
        steps_used = 3
        report = "Anomalous region at [412, 180, 78, 64] occupies 0.48% surface area. Structural residual 0.52 exceeds r_high (0.35), confirming defect. Decision: FAIL."
    elif decision_path == DecisionPathEnum.SUSPECTED_DEFECT:
        decision = DecisionEnum.HUMAN
        suggested = "FAIL"
        score = 1.85  # s <= t_fail
        alignment_ok = True
        align_quality = 0.88
        residual_mean = 0.46  # > r_high
        calib_status = "frozen_assumption_dependent"
        calib_claim = None
        steps_used = 3
        report = "High structural residual (0.46 > r_high 0.35) detected, but image score 1.85 is within borderline zone (<= t_fail 2.05). Escalated to HUMAN with suggested decision FAIL."
    elif decision_path == DecisionPathEnum.EVIDENCE_CONFLICT:
        decision = DecisionEnum.HUMAN
        suggested = None
        score = 2.38  # s > t_fail
        alignment_ok = True
        align_quality = 0.90
        residual_mean = 0.09  # <= r_low
        calib_status = "frozen_assumption_dependent"
        calib_claim = None
        steps_used = 3
        report = "Perception score 2.38 exceeds upper threshold 2.05, but local ROI structural residual (0.09) is low. Conflicting evidence requires human adjudication. Decision: HUMAN."
    elif decision_path == DecisionPathEnum.LIGHTING_SHIFT_REVIEW:
        decision = DecisionEnum.HUMAN
        suggested = "PASS"
        score = 1.35
        alignment_ok = True
        align_quality = 0.86
        residual_mean = 0.10  # <= r_low
        calib_status = "not_applicable_lighting_shift"
        calib_claim = None
        steps_used = 3
        report = "Lighting shift detected during quality inspection. Although residual evidence (0.10) is low, automatic pass is forbidden under distribution shift. Escalated to HUMAN with suggested decision PASS."
    elif decision_path == DecisionPathEnum.ALIGN_FAILED:
        decision = DecisionEnum.HUMAN
        suggested = None
        score = 1.72
        alignment_ok = False
        align_quality = 0.22  # < q_min (0.45)
        residual_mean = 0.0
        calib_status = "frozen_assumption_dependent"
        calib_claim = None
        steps_used = 1
        report = "Registration to golden template failed (alignment quality 0.22 < q_min 0.45). Unable to run structural comparison. Escalated to HUMAN."
    elif decision_path == DecisionPathEnum.INCONCLUSIVE:
        decision = DecisionEnum.HUMAN
        suggested = None
        score = 1.65
        alignment_ok = True
        align_quality = 0.87
        residual_mean = 0.24  # between r_low (0.15) and r_high (0.35)
        calib_status = "frozen_assumption_dependent"
        calib_claim = None
        steps_used = 3
        report = "ROI residual mean 0.24 lies in ambiguous band between r_low (0.15) and r_high (0.35). Evidence is inconclusive; escalated to HUMAN."
    elif decision_path == DecisionPathEnum.BUDGET_EXHAUSTED:
        decision = DecisionEnum.HUMAN
        suggested = None
        score = 1.78
        alignment_ok = True
        align_quality = 0.85
        residual_mean = 0.22
        calib_status = "frozen_assumption_dependent"
        calib_claim = None
        steps_used = 4
        report = "Evidence tool budget (4 steps) exhausted before completing ROI metrology. Conservative fallback requires manual review. Escalated to HUMAN."
    else:
        raise ValueError(f"Unknown decision path: {decision_path}")

    # Build Alignment Info
    if alignment_ok:
        alignment = AlignmentInfo(
            template_id="template_pcb1_norm03",
            method="ORB+RANSAC_Affine",
            quality=align_quality,
            q_min=0.45,
            rotation_deg=0.42,
        )
    elif decision_path == DecisionPathEnum.ALIGN_FAILED:
        alignment = AlignmentInfo(
            template_id="template_pcb1_norm03",
            method="ORB+RANSAC_Affine",
            quality=align_quality,
            q_min=0.45,
            rotation_deg=18.4,
        )
    else:
        alignment = None

    # Build Measurements
    if decision_path == DecisionPathEnum.ALIGN_FAILED:
        measurements = None
    else:
        # Metrology handling: skipped-T3 or no-contour -> area/elongation/shape null, never zero
        if skipped_t3 or no_contour:
            area_pct = None
            elongation = None
            shape_class = None
        else:
            area_pct = 0.48
            elongation = 1.22
            shape_class = "compact"

        measurements = MeasurementStats(
            roi_bbox=(412, 180, 78, 64),
            roi_source="threshold",
            rois_checked=1 if decision_path != DecisionPathEnum.BUDGET_EXHAUSTED else 2,
            rois_total=1 if decision_path not in (DecisionPathEnum.INCONCLUSIVE, DecisionPathEnum.BUDGET_EXHAUSTED) else 3,
            residual_mean=residual_mean,
            r_low=0.15,
            r_high=0.35,
            area_pct=area_pct,
            area_denominator="image",
            elongation=elongation,
            shape_class=shape_class,
        )

    # Build Assets
    has_contour = not (no_contour or decision_path == DecisionPathEnum.ALIGN_FAILED)
    has_ref = alignment_ok

    assets = AssetsInfo(
        input_standardized=f"/api/assets/{insp_id}_input.png",
        reference_aligned=f"/api/assets/{insp_id}_ref.png" if has_ref else None,
        anomaly_heatmap=f"/api/assets/{insp_id}_heatmap.png",
        contour_mask=f"/api/assets/{insp_id}_mask.png" if has_contour else None,
    )

    # Build Trace
    trace_steps: List[TraceStep] = [
        TraceStep(
            step=1,
            tool=ToolEnum.QUALITY_GATE,
            reason=qg.reason,
            status=qg.status,
            counts_toward_budget=False,
            duration_ms=7.3,
            overlay_url=None,
        ),
        TraceStep(
            step=2,
            tool=ToolEnum.PATCHCORE_PERCEPTION,
            reason=f"Anomaly score {score:.2f} evaluated against thresholds",
            status="COMPLETED",
            counts_toward_budget=False,
            duration_ms=38.4,
            overlay_url=f"/api/assets/{insp_id}_heatmap.png",
        ),
    ]

    current_step = 3
    if is_lighting_shift:
        trace_steps.append(
            TraceStep(
                step=current_step,
                tool=ToolEnum.T4_LIGHT_NORMALIZE,
                reason="Lighting shift detected; compute normalized reference comparison",
                status="COMPLETED",
                counts_toward_budget=True,
                duration_ms=12.5,
                overlay_url=None,
            )
        )
        current_step += 1

    trace_steps.append(
        TraceStep(
            step=current_step,
            tool=ToolEnum.T1_ALIGN_REFERENCE,
            reason="Register input image with golden reference template",
            status="COMPLETED" if alignment_ok else "ALIGNMENT_FAILED",
            counts_toward_budget=True,
            duration_ms=28.1,
            overlay_url=f"/api/assets/{insp_id}_ref.png" if alignment_ok else None,
        )
    )
    current_step += 1

    if alignment_ok:
        trace_steps.append(
            TraceStep(
                step=current_step,
                tool=ToolEnum.T2_ROI_COMPARE,
                reason=f"Residual mean {residual_mean:.2f} compared with r_low=0.15, r_high=0.35",
                status="CONFIRMED_ANOMALY" if residual_mean > 0.35 else ("CLEARED" if residual_mean <= 0.15 else "AMBIGUOUS"),
                counts_toward_budget=True,
                duration_ms=18.5,
                overlay_url=f"/api/assets/{insp_id}_diff.png",
            )
        )
        current_step += 1

        if not skipped_t3 and decision_path != DecisionPathEnum.BUDGET_EXHAUSTED:
            trace_steps.append(
                TraceStep(
                    step=current_step,
                    tool=ToolEnum.T3_ROI_ZOOM_MEASURE,
                    reason="Contour metrology extracted" if has_contour else "Metrology contour empty; shape attributes unavailable",
                    status="COMPLETED",
                    counts_toward_budget=True,
                    duration_ms=14.2,
                    overlay_url=f"/api/assets/{insp_id}_mask.png" if has_contour else None,
                )
            )
            current_step += 1

        if decision_path == DecisionPathEnum.BUDGET_EXHAUSTED:
            trace_steps.append(
                TraceStep(
                    step=current_step,
                    tool=ToolEnum.T5_NEXT_ROI_COMPARE,
                    reason="Compare secondary proposed candidate ROI",
                    status="COMPLETED",
                    counts_toward_budget=True,
                    duration_ms=16.0,
                    overlay_url=None,
                )
            )
            current_step += 1

    total_tool_ms = sum(s.duration_ms for s in trace_steps if s.counts_toward_budget)
    timing = TimingInfo(quality=7.3, perception=38.4, tools=total_tool_ms, total=45.7 + total_tool_ms)

    counterfactuals = [
        CounterfactualItem(alpha_fail=0.01, t_fail=2.48, decision=decision, decision_path=decision_path, needs_rerun=False),
        CounterfactualItem(alpha_fail=0.02, t_fail=2.25, decision=decision, decision_path=decision_path, needs_rerun=False),
        CounterfactualItem(alpha_fail=0.05, t_fail=2.05, decision=decision, decision_path=decision_path, needs_rerun=False),
        CounterfactualItem(alpha_fail=0.10, t_fail=1.82, decision=decision, decision_path=decision_path, needs_rerun=False),
    ]

    return InspectionResponse(
        schema_version="1.0",
        inspection_id=insp_id,
        request_id=req_id,
        timestamp=ts,
        engine="mock",
        category=category,
        snapshot=snapshot,
        image=image_meta,
        decision=decision,
        decision_path=decision_path,
        suggested_decision=suggested,
        anomaly_score=score,
        thresholds=ThresholdsInfo(
            t_pass=t_pass,
            t_fail=t_fail,
            alpha_pass=0.20,
            alpha_fail=alpha_fail,
            n_calib=300,
        ),
        calibration=CalibrationInfo(
            status=calib_status,
            claim=calib_claim,
            assumptions="Marginal calibration bounds hold under exchangeability" if calib_claim else "Statistical calibration bounds not guaranteed under covariate or quality shift",
        ),
        quality_gate=qg,
        alignment=alignment,
        measurements=measurements,
        assets=assets,
        execution_trace=trace_steps,
        steps_used=steps_used,
        step_budget=4,
        timing_ms=timing,
        grounded_report=report,
        counterfactuals=counterfactuals,
    )


def generate_mock_inspection(
    category: str = "pcb1",
    simulate_blur: bool = False,
    simulate_darken: bool = False,
    simulate_glare: bool = False,
    sample_id: Optional[str] = None,
    alpha_fail: float = 0.05,
    path: Optional[str] = None,
) -> InspectionResponse:
    """Entrypoint for mock inspection route."""
    if path:
        try:
            target_path = DecisionPathEnum(path)
            return build_mock_inspection(target_path, category=category, alpha_fail=alpha_fail)
        except ValueError:
            pass

    if simulate_blur or sample_id == "pcb1_blur_01":
        return build_mock_inspection(DecisionPathEnum.QUALITY_REJECTED, category=category, alpha_fail=alpha_fail)

    if sample_id == "pcb1_good_01":
        return build_mock_inspection(DecisionPathEnum.FAST_PASS, category=category, alpha_fail=alpha_fail)

    if sample_id == "pcb1_borderline_01":
        return build_mock_inspection(DecisionPathEnum.AGENT_CLEARED, category=category, alpha_fail=alpha_fail)

    if sample_id == "pcb1_bad_missing_01":
        return build_mock_inspection(DecisionPathEnum.AGENT_CONFIRMED, category=category, alpha_fail=alpha_fail)
        return build_mock_inspection(DecisionPathEnum.SUSPECTED_DEFECT, category=category, alpha_fail=alpha_fail)

    if simulate_darken:
        return build_mock_inspection(DecisionPathEnum.LIGHTING_SHIFT_REVIEW, category=category, alpha_fail=alpha_fail)

    return build_mock_inspection(DecisionPathEnum.AGENT_CONFIRMED, category=category, alpha_fail=alpha_fail)


def handle_mock_feedback(inspection_id: str, user_label: str) -> FeedbackResponse:
    return FeedbackResponse(
        status="LOGGED",
        message=f"Operator label '{user_label}' logged to audit store for inspection {inspection_id}.",
        scope="audit_only",
        production_changed=False,
    )
