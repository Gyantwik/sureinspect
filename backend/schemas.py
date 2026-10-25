"""
backend/schemas.py
Pydantic v2 schemas strictly matching Section 4 of the SureInspect v8.1 specification.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Error Schemas
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Session Schemas
# ---------------------------------------------------------------------------

class SessionCreateResponse(BaseModel):
    session_token: str
    expires_at: str


class SessionResetResponse(BaseModel):
    status: str = "RESET"
    message: str = "Shadow state cleared"
    snapshot_id: str


# ---------------------------------------------------------------------------
# Health & Categories
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    engine: Literal["live", "mock"]
    model_version: str
    enabled_categories: List[str]
    loaded_categories: List[str]
    opencv_version: str
    python_version: str
    architecture: str
    deployment_provider: Literal["aws", "local_or_unspecified"]
    aws_region: Optional[str] = None


class CategoryInfo(BaseModel):
    id: str
    display_name: str
    enabled: bool
    alpha_pass: float
    alpha_fail_default: float
    alpha_fail_options: List[float]
    n_calib: int
    sample_count: int


class CategoriesResponse(BaseModel):
    categories: List[CategoryInfo]


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

class SampleItem(BaseModel):
    id: str
    category: str
    display_name: str
    ground_truth: Literal["good", "bad"]
    attribution: str
    image_url: str
    role: Optional[str] = None


class SamplesResponse(BaseModel):
    samples: List[SampleItem]


# ---------------------------------------------------------------------------
# Enums for Decisions & Tools
# ---------------------------------------------------------------------------

class DecisionEnum(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    HUMAN = "HUMAN"
    RETAKE = "RETAKE"


class DecisionPathEnum(str, Enum):
    QUALITY_REJECTED = "QUALITY_REJECTED"
    FAST_PASS = "FAST_PASS"
    AGENT_CLEARED = "AGENT_CLEARED"
    AGENT_CONFIRMED = "AGENT_CONFIRMED"
    SUSPECTED_DEFECT = "SUSPECTED_DEFECT"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    LIGHTING_SHIFT_REVIEW = "LIGHTING_SHIFT_REVIEW"
    ALIGN_FAILED = "ALIGN_FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ToolEnum(str, Enum):
    SIMULATED_DEGRADATION = "SIMULATED_DEGRADATION"
    ACQUISITION_NORMALIZE = "ACQUISITION_NORMALIZE"
    QUALITY_GATE = "QUALITY_GATE"
    PATCHCORE_PERCEPTION = "PATCHCORE_PERCEPTION"
    T1_ALIGN_REFERENCE = "T1_ALIGN_REFERENCE"
    T2_ROI_COMPARE = "T2_ROI_COMPARE"
    T3_ROI_ZOOM_MEASURE = "T3_ROI_ZOOM_MEASURE"
    T4_LIGHT_NORMALIZE = "T4_LIGHT_NORMALIZE"
    T5_NEXT_ROI_COMPARE = "T5_NEXT_ROI_COMPARE"


# ---------------------------------------------------------------------------
# Inspection Sub-schemas
# ---------------------------------------------------------------------------

class SnapshotInfo(BaseModel):
    model_version: str
    bank_version: str
    calibration_version: str
    policy_version: str


class ImageMetadata(BaseModel):
    original_size: Tuple[int, int]
    standardized_size: Tuple[int, int]
    scale_x: float
    scale_y: float


class ThresholdsInfo(BaseModel):
    t_pass: Optional[float] = None
    t_fail: Optional[float] = None
    alpha_pass: float
    alpha_fail: float
    n_calib: int


class CalibrationInfo(BaseModel):
    status: Literal["frozen_assumption_dependent", "not_applicable_lighting_shift", "not_evaluated"]
    claim: Optional[Literal["normal_marginal_upper_tail"]] = None
    assumptions: str


class QualityGateResult(BaseModel):
    status: Literal["PASSED", "FAILED"]
    action: Optional[Literal["REQUEST_RETAKE"]] = None
    reason: str
    blur_variance: float
    blur_threshold: float
    dark_fraction: float
    bright_fraction: float
    glare_fraction: float
    glare_threshold: float
    l_mean: float
    exposure_ok: bool
    lighting_shift_detected: bool


class AlignmentInfo(BaseModel):
    template_id: str
    method: str
    quality: float
    q_min: float
    rotation_deg: float
    scale: Optional[float] = None
    overlap: Optional[float] = None
    translation_px: Optional[Tuple[float, float]] = None


class MeasurementStats(BaseModel):
    roi_bbox: Tuple[int, int, int, int]
    roi_source: Literal["threshold", "peak_fallback"]
    rois_checked: int
    rois_total: int
    residual_mean: float
    r_low: float
    r_high: float
    area_pct: Optional[float] = None
    area_denominator: Literal["image", "part_mask"] = "image"
    elongation: Optional[float] = None
    shape_class: Optional[Literal["elongated", "compact", "irregular"]] = None


class AssetsInfo(BaseModel):
    input_standardized: str
    reference_aligned: Optional[str] = None
    anomaly_heatmap: Optional[str] = None
    contour_mask: Optional[str] = None


class TraceStep(BaseModel):
    step: int
    tool: ToolEnum
    reason: str
    status: str
    counts_toward_budget: bool
    duration_ms: float
    overlay_url: Optional[str] = None


class TimingInfo(BaseModel):
    quality: float
    perception: float
    tools: float
    total: float


class CounterfactualItem(BaseModel):
    alpha_fail: float
    t_fail: float
    decision: Optional[DecisionEnum] = None
    decision_path: Optional[DecisionPathEnum] = None
    needs_rerun: bool = False


class ExperimentalShadowInfo(BaseModel):
    status: Literal["experimental_no_calibration_claim"]
    shadow_version: str
    score: float
    shadow_t_pass: float
    shadow_t_fail: float
    approximate: bool
    note: str


class NextActionInfo(BaseModel):
    action: Literal[
        "CONTINUE_LINE",
        "QUARANTINE_PENDING_CONFIRMATION",
        "REVIEW_EVIDENCE",
        "REQUEST_RETAKE",
    ]
    execution_mode: Literal["recommendation_only"] = "recommendation_only"
    requires_human_confirmation: bool
    external_actuation_performed: bool = False
    reversible: bool = True
    reason: str


def next_action_for_decision(decision: str | DecisionEnum) -> dict[str, Any]:
    value = decision.value if isinstance(decision, DecisionEnum) else str(decision)
    if value == "PASS":
        return {
            "action": "CONTINUE_LINE",
            "requires_human_confirmation": False,
            "reason": "Agent recommends normal line flow; the service does not actuate factory equipment.",
        }
    if value == "FAIL":
        return {
            "action": "QUARANTINE_PENDING_CONFIRMATION",
            "requires_human_confirmation": True,
            "reason": "Agent recommends reversible quarantine; an operator must confirm disposition.",
        }
    if value == "RETAKE":
        return {
            "action": "REQUEST_RETAKE",
            "requires_human_confirmation": False,
            "reason": "Admission checks require a new frame before any defect decision.",
        }
    return {
        "action": "REVIEW_EVIDENCE",
        "requires_human_confirmation": True,
        "reason": "The agent abstained and routed its evidence trace to a trained reviewer.",
    }


# ---------------------------------------------------------------------------
# Master Inspection Response
# ---------------------------------------------------------------------------

class InspectionResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    inspection_id: str
    request_id: str
    timestamp: str
    engine: Literal["live", "mock"]
    category: str
    snapshot: SnapshotInfo
    image: ImageMetadata
    decision: DecisionEnum
    decision_path: DecisionPathEnum
    next_action: NextActionInfo
    suggested_decision: Optional[Literal["PASS", "FAIL"]] = None
    anomaly_score: Optional[float] = None
    thresholds: ThresholdsInfo
    calibration: CalibrationInfo
    quality_gate: QualityGateResult
    alignment: Optional[AlignmentInfo] = None
    measurements: Optional[MeasurementStats] = None
    assets: AssetsInfo
    execution_trace: List[TraceStep]
    steps_used: int
    step_budget: int = 4
    timing_ms: TimingInfo
    grounded_report: str
    counterfactuals: List[CounterfactualItem] = Field(default_factory=list)
    experimental_shadow: Optional[ExperimentalShadowInfo] = None

    @model_validator(mode="before")
    @classmethod
    def populate_next_action(cls, data: Any) -> Any:
        if isinstance(data, dict) and "next_action" not in data and data.get("decision") is not None:
            data = dict(data)
            data["next_action"] = next_action_for_decision(data["decision"])
        return data


# ---------------------------------------------------------------------------
# Feedback Schemas
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    inspection_id: str
    user_label: Literal["NORMAL", "DEFECT"]
    idempotency_key: str


class FeedbackResponse(BaseModel):
    status: Literal["LOGGED", "UPDATED", "DUPLICATE"] = "LOGGED"
    scope: Literal["audit_only", "session_shadow"] = "audit_only"
    production_changed: bool = False
    message: Optional[str] = None
    patches_added: Optional[int] = None
    new_t_pass: Optional[float] = None
    new_t_fail: Optional[float] = None
    rescore: Optional[float] = None


# ---------------------------------------------------------------------------
# Calibration Summary
# ---------------------------------------------------------------------------

class CalibrationSummaryResponse(BaseModel):
    category: str
    scope: Literal["frozen_production"] = "frozen_production"
    t_pass: float
    t_fail: float
    alpha_pass: float
    alpha_fail: float
    n_calib: int
    report_bundle_id: Optional[str] = None
    evaluation_status: Literal["pending", "available"] = "pending"
    evaluation_summary: Optional[Dict[str, Any]] = None
