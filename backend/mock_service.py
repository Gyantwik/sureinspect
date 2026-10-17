"""
backend/mock_service.py
Deterministic mock inspection engine generating realistic inspection traces,
asset payloads, and decision variants matching Section 4 of SureInspect v8.1.
"""
from __future__ import annotations

import base64
import time
import uuid
from typing import Dict, Any, Optional

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
# Pre-computed lightweight PNG asset placeholders (1x1 or simple transparent)
# ---------------------------------------------------------------------------
TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# In-memory session asset storage for mock assets
MOCK_ASSET_STORE: Dict[str, bytes] = {}

def store_mock_asset(asset_id: str, data: bytes) -> str:
    MOCK_ASSET_STORE[asset_id] = data
    return f"/api/assets/{asset_id}"


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


def get_mock_samples(category: str) -> SamplesResponse:
    filtered = [s for s in MOCK_SAMPLES if s.category == category]
    return SamplesResponse(samples=filtered if filtered else MOCK_SAMPLES)


def get_mock_calibration(category: str) -> CalibrationSummaryResponse:
    return CalibrationSummaryResponse(
        category=category,
        scope="frozen_production",
        t_pass=1.21,
        t_fail=2.05,
        alpha_pass=0.20,
        alpha_fail=0.05,
        n_calib=300,
        report_bundle_id="report_v1_pcb1_frozen",
        evaluation_status="available",
        evaluation_summary={
            "auroc": 0.982,
            "fpr_target_alpha": 0.048,
            "defect_recall": 0.975,
            "human_escalation_rate": 0.082,
            "fast_pass_latency_p50_ms": 38.4,
            "agent_latency_p50_ms": 142.1,
        },
    )


# ---------------------------------------------------------------------------
# Mock Inspection Builder
# ---------------------------------------------------------------------------

def generate_mock_inspection(
    category: str = "pcb1",
    simulate_blur: bool = False,
    simulate_darken: bool = False,
    simulate_glare: bool = False,
    sample_id: Optional[str] = None,
    alpha_fail: float = 0.05,
) -> InspectionResponse:
    """Generate a realistic InspectionResponse following all v8.1 decision paths."""
    insp_id = f"insp_{uuid.uuid4().hex[:8]}"
    req_id = f"req_{uuid.uuid4().hex[:8]}"
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

    # 1. Check Quality Gate Rejection path
    if simulate_blur or sample_id == "pcb1_blur_01":
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
                t_pass=None, t_fail=None, alpha_pass=0.20, alpha_fail=alpha_fail, n_calib=300
            ),
            calibration=CalibrationInfo(
                status="not_applicable_lighting_shift",
                claim=None,
                assumptions="Quality gate rejected frame before inference",
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
        )

    # 2. Fast Pass Path
    if sample_id == "pcb1_good_01":
        qg = QualityGateResult(
            status="PASSED",
            action=None,
            reason="Focus and illumination within normal factory parameters",
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
        t_pass = 1.21
        t_fail = 2.05
        return InspectionResponse(
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
                t_pass=t_pass, t_fail=t_fail, alpha_pass=0.20, alpha_fail=alpha_fail, n_calib=300
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
        )

    # 3. Defect Confirmed Path (AGENT_CONFIRMED)
    qg = QualityGateResult(
        status="PASSED",
        action=None,
        reason="Focus and illumination within normal limits",
        blur_variance=298.6,
        blur_threshold=110.0,
        dark_fraction=0.02,
        bright_fraction=0.01,
        glare_fraction=0.01,
        glare_threshold=0.08,
        l_mean=130.4,
        exposure_ok=True,
        lighting_shift_detected=simulate_darken,
    )
    score = 2.44
    t_pass = 1.21
    t_fail = 2.05

    alignment = AlignmentInfo(
        template_id="template_pcb1_norm03",
        method="ORB+RANSAC_Affine",
        quality=0.89,
        q_min=0.45,
        rotation_deg=0.42,
    )
    measurements = MeasurementStats(
        roi_bbox=(412, 180, 78, 64),
        roi_source="threshold",
        rois_checked=1,
        rois_total=1,
        residual_mean=0.52,
        r_low=0.15,
        r_high=0.35,
        area_pct=0.48,
        area_denominator="image",
        elongation=1.22,
        shape_class="compact",
    )

    return InspectionResponse(
        inspection_id=insp_id,
        request_id=req_id,
        timestamp=ts,
        engine="mock",
        category=category,
        snapshot=snapshot,
        image=image_meta,
        decision=DecisionEnum.FAIL,
        decision_path=DecisionPathEnum.AGENT_CONFIRMED,
        suggested_decision=None,
        anomaly_score=score,
        thresholds=ThresholdsInfo(
            t_pass=t_pass, t_fail=t_fail, alpha_pass=0.20, alpha_fail=alpha_fail, n_calib=300
        ),
        calibration=CalibrationInfo(
            status="frozen_assumption_dependent",
            claim="normal_marginal_upper_tail",
            assumptions="Marginal false-FAIL bound holds under exchangeability with training normals",
        ),
        quality_gate=qg,
        alignment=alignment,
        measurements=measurements,
        assets=AssetsInfo(
            input_standardized=f"/api/assets/{insp_id}_input.png",
            reference_aligned=f"/api/assets/{insp_id}_ref.png",
            anomaly_heatmap=f"/api/assets/{insp_id}_heatmap.png",
            contour_mask=f"/api/assets/{insp_id}_mask.png",
        ),
        execution_trace=[
            TraceStep(
                step=1,
                tool=ToolEnum.QUALITY_GATE,
                reason="Focus and illumination within normal limits",
                status="PASSED",
                counts_toward_budget=False,
                duration_ms=7.3,
                overlay_url=None,
            ),
            TraceStep(
                step=2,
                tool=ToolEnum.PATCHCORE_PERCEPTION,
                reason=f"Anomaly score {score:.2f} > t_fail ({t_fail:.2f}); candidate defect detected",
                status="COMPLETED",
                counts_toward_budget=False,
                duration_ms=38.4,
                overlay_url=f"/api/assets/{insp_id}_heatmap.png",
            ),
            TraceStep(
                step=3,
                tool=ToolEnum.T1_ALIGN_REFERENCE,
                reason="Retrieve golden template and align via affine RANSAC",
                status="COMPLETED",
                counts_toward_budget=True,
                duration_ms=28.1,
                overlay_url=f"/api/assets/{insp_id}_ref.png",
            ),
            TraceStep(
                step=4,
                tool=ToolEnum.T2_ROI_COMPARE,
                reason="Residual mean 0.52 exceeds high cutoff r_high 0.35 on primary candidate ROI",
                status="CONFIRMED_ANOMALY",
                counts_toward_budget=True,
                duration_ms=18.5,
                overlay_url=f"/api/assets/{insp_id}_diff.png",
            ),
            TraceStep(
                step=5,
                tool=ToolEnum.T3_ROI_ZOOM_MEASURE,
                reason="Contour metrology extracted: bbox [412, 180, 78, 64], area 0.48%, elongation 1.22",
                status="COMPLETED",
                counts_toward_budget=True,
                duration_ms=14.2,
                overlay_url=f"/api/assets/{insp_id}_mask.png",
            ),
        ],
        steps_used=3,
        step_budget=4,
        timing_ms=TimingInfo(quality=7.3, perception=38.4, tools=60.8, total=106.5),
        grounded_report=(
            "Anomalous region at [412, 180, 78, 64] occupies 0.48% of the surface area with elongation 1.22 "
            "(compact shape). The structural residual difference of 0.52 exceeds r_high (0.35), confirming "
            "the PatchCore perception anomaly. Decision: FAIL."
        ),
        counterfactuals=[
            CounterfactualItem(alpha_fail=0.01, t_fail=2.48, decision=DecisionEnum.HUMAN, decision_path=DecisionPathEnum.SUSPECTED_DEFECT, needs_rerun=False),
            CounterfactualItem(alpha_fail=0.02, t_fail=2.25, decision=DecisionEnum.FAIL, decision_path=DecisionPathEnum.AGENT_CONFIRMED, needs_rerun=False),
            CounterfactualItem(alpha_fail=0.05, t_fail=2.05, decision=DecisionEnum.FAIL, decision_path=DecisionPathEnum.AGENT_CONFIRMED, needs_rerun=False),
            CounterfactualItem(alpha_fail=0.10, t_fail=1.82, decision=DecisionEnum.FAIL, decision_path=DecisionPathEnum.AGENT_CONFIRMED, needs_rerun=False),
        ],
    )


def handle_mock_feedback(inspection_id: str, user_label: str) -> FeedbackResponse:
    return FeedbackResponse(
        status="LOGGED",
        message=f"Operator label '{user_label}' logged to audit store for inspection {inspection_id}.",
        scope="audit_only",
        production_changed=False,
    )
