"""
backend/tests/test_api.py
Comprehensive test suite verifying all SureInspect v8.1 API contracts:
- All 10 decision paths and variants validate against schema with exact nullability rules
- Non-owned inspection and asset 404 identical to unknown
- SESSION_EXPIRED 401
- Uniform error shape across 401/404/422/429
- Feedback idempotency, conflicting key 409, RETAKE rejection, and response body
- OpenAPI export drift check
"""
import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app_factory import create_app, SESSION_STORE, STORED_INSPECTIONS, STORED_ASSETS
from backend.mock_service import build_mock_inspection
from backend.schemas import (
    InspectionResponse,
    DecisionEnum,
    DecisionPathEnum,
)
from backend.export_openapi import export as export_openapi


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Decision Paths & Variants Tests
# ---------------------------------------------------------------------------

def test_all_ten_paths_validate_schema():
    """Verify all 10 decision paths validate against the Pydantic InspectionResponse schema."""
    for path in DecisionPathEnum:
        insp = build_mock_inspection(path)
        assert isinstance(insp, InspectionResponse)
        assert insp.decision_path == path
        assert insp.engine == "mock"
        assert insp.schema_version == "1.0"

        # Verify dump and re-parse
        dumped = insp.model_dump()
        reparsed = InspectionResponse.model_validate(dumped)
        assert reparsed.decision_path == path


def test_next_action_contract_preserves_human_control():
    passed = build_mock_inspection(DecisionPathEnum.AGENT_CLEARED)
    assert passed.next_action.action == "CONTINUE_LINE"
    assert passed.next_action.requires_human_confirmation is False
    assert passed.next_action.external_actuation_performed is False

    failed = build_mock_inspection(DecisionPathEnum.AGENT_CONFIRMED)
    assert failed.next_action.action == "QUARANTINE_PENDING_CONFIRMATION"
    assert failed.next_action.requires_human_confirmation is True
    assert failed.next_action.reversible is True

    review = build_mock_inspection(DecisionPathEnum.INCONCLUSIVE)
    assert review.next_action.action == "REVIEW_EVIDENCE"
    assert review.next_action.requires_human_confirmation is True


def test_quality_rejected_null_rules():
    """RETAKE: score, thresholds t_pass/t_fail, alignment, measurements null; input asset + quality trace exist."""
    insp = build_mock_inspection(DecisionPathEnum.QUALITY_REJECTED)
    assert insp.decision == DecisionEnum.RETAKE
    assert insp.suggested_decision is None
    assert insp.anomaly_score is None
    assert insp.thresholds.t_pass is None
    assert insp.thresholds.t_fail is None
    assert insp.alignment is None
    assert insp.measurements is None
    assert insp.assets.input_standardized.startswith("/api/assets/")
    assert insp.assets.reference_aligned is None
    assert insp.assets.anomaly_heatmap is None
    assert insp.assets.contour_mask is None
    assert len(insp.execution_trace) >= 1
    assert insp.execution_trace[0].tool.value == "QUALITY_GATE"
    assert insp.quality_gate.status == "FAILED"
    assert insp.quality_gate.action == "REQUEST_RETAKE"
    assert insp.calibration.status == "not_evaluated"


def test_fast_pass_null_rules():
    """FAST_PASS: alignment, measurements, reference, contour null; heatmap may exist."""
    insp = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    assert insp.decision == DecisionEnum.PASS
    assert insp.suggested_decision is None
    assert insp.anomaly_score is not None
    assert insp.thresholds.t_pass is not None
    assert insp.anomaly_score <= insp.thresholds.t_pass
    assert insp.alignment is None
    assert insp.measurements is None
    assert insp.assets.input_standardized.startswith("/api/assets/")
    assert insp.assets.reference_aligned is None
    assert insp.assets.contour_mask is None
    assert insp.assets.anomaly_heatmap is not None


def test_lighting_shift_review_rules():
    """Lighting-shift paths never yield auto PASS or FAIL except as HUMAN with suggested PASS."""
    insp = build_mock_inspection(DecisionPathEnum.LIGHTING_SHIFT_REVIEW)
    assert insp.decision == DecisionEnum.HUMAN
    assert insp.suggested_decision == "PASS"
    assert insp.quality_gate.lighting_shift_detected is True
    assert insp.calibration.status == "not_applicable_lighting_shift"
    assert insp.calibration.claim is None


def test_suspected_defect_rules():
    """SUSPECTED_DEFECT: suggested FAIL."""
    insp = build_mock_inspection(DecisionPathEnum.SUSPECTED_DEFECT)
    assert insp.decision == DecisionEnum.HUMAN
    assert insp.suggested_decision == "FAIL"
    assert insp.measurements is not None
    assert insp.measurements.residual_mean > insp.measurements.r_high


def test_skipped_t3_and_no_contour_variants():
    """Skipped-T3 and no-contour variants: shape/area/elongation null, never zero."""
    # Skipped T3
    skipped = build_mock_inspection(DecisionPathEnum.SUSPECTED_DEFECT, skipped_t3=True)
    assert skipped.measurements is not None
    assert skipped.measurements.area_pct is None
    assert skipped.measurements.elongation is None
    assert skipped.measurements.shape_class is None
    assert skipped.measurements.residual_mean is not None

    # No contour
    no_cnt = build_mock_inspection(DecisionPathEnum.AGENT_CLEARED, no_contour=True)
    assert no_cnt.measurements is not None
    assert no_cnt.measurements.area_pct is None
    assert no_cnt.measurements.elongation is None
    assert no_cnt.measurements.shape_class is None
    assert no_cnt.assets.contour_mask is None


# ---------------------------------------------------------------------------
# 2. Ownership & Session Tests
# ---------------------------------------------------------------------------

def test_session_ownership_and_identical_404(client):
    """Unknown and non-owned inspection/asset IDs return the identical 404."""
    # Session A
    res_a = client.post("/api/session")
    token_a = res_a.json()["session_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Session B
    res_b = client.post("/api/session")
    token_b = res_b.json()["session_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Create inspection with Session A
    insp_res = client.post(
        "/api/inspect",
        data={"sample_id": "pcb1_bad_missing_01", "category": "pcb1"},
        headers=headers_a,
    )
    assert insp_res.status_code == 200
    insp_id = insp_res.json()["inspection_id"]
    asset_id = insp_id + "_input.png"

    # 1. Query with Session A (owner) -> 200
    assert client.get(f"/api/trace/{insp_id}", headers=headers_a).status_code == 200
    assert client.get(f"/api/assets/{asset_id}", headers=headers_a).status_code == 200

    # 2. Query unknown ID with Session A -> 404
    unknown_trace_res = client.get("/api/trace/insp_unknown_999", headers=headers_a)
    assert unknown_trace_res.status_code == 404
    unknown_trace_data = unknown_trace_res.json()
    assert unknown_trace_data["error"]["code"] == "NOT_FOUND"
    assert unknown_trace_data["error"]["message"] == "Inspection not found"

    unknown_asset_res = client.get("/api/assets/asset_unknown_999", headers=headers_a)
    assert unknown_asset_res.status_code == 404
    unknown_asset_data = unknown_asset_res.json()
    assert unknown_asset_data["error"]["code"] == "NOT_FOUND"
    assert unknown_asset_data["error"]["message"] == "Asset not found"

    # 3. Query Session A's inspection with Session B (non-owned) -> IDENTICAL 404
    non_owned_trace_res = client.get(f"/api/trace/{insp_id}", headers=headers_b)
    assert non_owned_trace_res.status_code == 404
    non_owned_trace_data = non_owned_trace_res.json()
    assert non_owned_trace_data["error"]["code"] == unknown_trace_data["error"]["code"]
    assert non_owned_trace_data["error"]["message"] == unknown_trace_data["error"]["message"]

    # 4. Query Session A's asset with Session B (non-owned) -> IDENTICAL 404
    non_owned_asset_res = client.get(f"/api/assets/{asset_id}", headers=headers_b)
    assert non_owned_asset_res.status_code == 404
    non_owned_asset_data = non_owned_asset_res.json()
    assert non_owned_asset_data["error"]["code"] == unknown_asset_data["error"]["code"]
    assert non_owned_asset_data["error"]["message"] == unknown_asset_data["error"]["message"]


def test_session_expired_401(client):
    """Expired session returns 401 with code SESSION_EXPIRED."""
    res = client.post("/api/session")
    token = res.json()["session_token"]

    # Artificially expire session
    SESSION_STORE[token]["expires_at"] = time.time() - 10

    resp = client.post(
        "/api/inspect",
        data={"sample_id": "pcb1_good_01"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "SESSION_EXPIRED"
    assert data["error"]["message"] == "Session has expired or is invalid"
    assert "request_id" in data["error"]


# ---------------------------------------------------------------------------
# 3. Uniform Error Shape Tests
# ---------------------------------------------------------------------------

def test_uniform_error_shape_401(client):
    res = client.post("/api/inspect", data={"sample_id": "pcb1_good_01"})
    assert res.status_code == 401
    err = res.json()["error"]
    assert err["code"] == "UNAUTHORIZED"
    assert isinstance(err["message"], str)
    assert isinstance(err["request_id"], str)


def test_uniform_error_shape_404(client):
    # Public sample image allowlist check
    res = client.get("/api/samples/malicious_traversal_id/image")
    assert res.status_code == 404
    err = res.json()["error"]
    assert err["code"] == "NOT_FOUND"
    assert isinstance(err["message"], str)
    assert isinstance(err["request_id"], str)


def test_uniform_error_shape_422(client):
    # Malformed JSON to feedback
    sess = client.post("/api/session").json()["session_token"]
    res = client.post(
        "/api/feedback",
        json={"wrong_field": 123},
        headers={"Authorization": f"Bearer {sess}"},
    )
    assert res.status_code == 422
    err = res.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert isinstance(err["message"], str)
    assert isinstance(err["request_id"], str)


def test_uniform_error_shape_429(client):
    # Verify exception handler for 429
    from fastapi import HTTPException
    app = client.app

    @app.get("/api/test-rate-limit")
    def trigger_429():
        raise HTTPException(status_code=429, detail="Rate limit exceeded; max 10 requests per minute")

    res = client.get("/api/test-rate-limit")
    assert res.status_code == 429
    err = res.json()["error"]
    assert err["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Rate limit exceeded" in err["message"]
    assert isinstance(err["request_id"], str)


# ---------------------------------------------------------------------------
# 4. Feedback Endpoint Semantics Tests
# ---------------------------------------------------------------------------

def test_feedback_idempotency_and_response_body(client):
    """Feedback repeat is safe; conflicting reuse returns 409; response body is exact."""
    sess = client.post("/api/session").json()["session_token"]
    headers = {"Authorization": f"Bearer {sess}"}

    # Inspect a part first
    insp = client.post(
        "/api/inspect",
        data={"sample_id": "pcb1_bad_missing_01", "category": "pcb1"},
        headers=headers,
    ).json()
    insp_id = insp["inspection_id"]

    # 1. First submission
    fb1 = client.post(
        "/api/feedback",
        json={"inspection_id": insp_id, "user_label": "DEFECT", "idempotency_key": "key_001"},
        headers=headers,
    )
    assert fb1.status_code == 200
    assert fb1.json() == {
        "status": "LOGGED",
        "scope": "audit_only",
        "production_changed": False,
        "message": None,
        "patches_added": None,
        "new_t_pass": None,
        "new_t_fail": None,
        "rescore": None,
    }

    # 2. Repeated submission with identical key and payload (safe repeat)
    fb2 = client.post(
        "/api/feedback",
        json={"inspection_id": insp_id, "user_label": "DEFECT", "idempotency_key": "key_001"},
        headers=headers,
    )
    assert fb2.status_code == 200
    assert fb2.json()["status"] == "LOGGED"

    # 3. Conflicting reuse of key_001 with different label -> 409
    fb_conflict = client.post(
        "/api/feedback",
        json={"inspection_id": insp_id, "user_label": "NORMAL", "idempotency_key": "key_001"},
        headers=headers,
    )
    assert fb_conflict.status_code == 409
    err = fb_conflict.json()["error"]
    assert err["code"] == "CONFLICT"
    assert "reused with conflicting payload" in err["message"]


def test_feedback_retake_rejection(client):
    """Feedback must reject RETAKE inspection records."""
    sess = client.post("/api/session").json()["session_token"]
    headers = {"Authorization": f"Bearer {sess}"}

    # Inspect with blur -> RETAKE
    insp = client.post(
        "/api/inspect",
        data={"sample_id": "pcb1_blur_01", "category": "pcb1", "simulate_blur": "true"},
        headers=headers,
    ).json()
    assert insp["decision"] == "RETAKE"
    insp_id = insp["inspection_id"]

    # Attempt feedback on RETAKE
    res = client.post(
        "/api/feedback",
        json={"inspection_id": insp_id, "user_label": "NORMAL", "idempotency_key": "key_retake_1"},
        headers=headers,
    )
    assert res.status_code == 400
    err = res.json()["error"]
    assert err["code"] == "BAD_REQUEST"
    assert "RETAKE" in err["message"]


# ---------------------------------------------------------------------------
# 5. Missing Routes & Public Endpoints
# ---------------------------------------------------------------------------

def test_samples_image_allowlist(client):
    """GET /api/samples/{id}/image returns curated image bytes for allowlisted items, 404 for others."""
    # Allowlisted sample
    res = client.get("/api/samples/pcb1_good_01/image")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 0

    # Non-allowlisted sample
    res_bad = client.get("/api/samples/etc_passwd_exploit/image")
    assert res_bad.status_code == 404
    assert res_bad.json()["error"]["code"] == "NOT_FOUND"


def test_frozen_calibration_endpoint(client):
    """GET /api/calibration/pcb1 returns frozen production calibration with pending evaluation."""
    res = client.get("/api/calibration/pcb1")
    assert res.status_code == 200
    data = res.json()
    assert data["scope"] == "frozen_production"
    assert data["report_bundle_id"] is None
    assert data["evaluation_status"] == "pending"


# ---------------------------------------------------------------------------
# 6. OpenAPI Drift Test
# ---------------------------------------------------------------------------

def test_openapi_drift():
    """Exporting OpenAPI twice produces identical bytes without diff."""
    contracts_file = Path(__file__).resolve().parents[2] / "contracts" / "openapi.json"
    export_openapi()
    content_first = contracts_file.read_bytes()

    export_openapi()
    content_second = contracts_file.read_bytes()

    assert content_first == content_second, "OpenAPI schema drifted between consecutive exports"
