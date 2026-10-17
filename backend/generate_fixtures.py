"""
backend/generate_fixtures.py
Generates all required contract fixtures into contracts/fixtures/*.json.
"""
import json
from pathlib import Path

from backend.schemas import (
    CategoriesResponse,
    CategoryInfo,
    SessionCreateResponse,
    SessionResetResponse,
    FeedbackResponse,
    CalibrationSummaryResponse,
    ErrorResponse,
    ErrorDetail,
    DecisionPathEnum,
)
from backend.mock_service import (
    build_mock_inspection,
    get_mock_samples,
    get_mock_calibration,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "contracts" / "fixtures"


def save_json(filename: str, data: dict):
    filepath = FIXTURES_DIR / filename
    filepath.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {filepath.name}")


def generate_all():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ten Decision Paths
    paths = [
        ("path_quality_rejected.json", DecisionPathEnum.QUALITY_REJECTED),
        ("path_fast_pass.json", DecisionPathEnum.FAST_PASS),
        ("path_agent_cleared.json", DecisionPathEnum.AGENT_CLEARED),
        ("path_agent_confirmed.json", DecisionPathEnum.AGENT_CONFIRMED),
        ("path_suspected_defect.json", DecisionPathEnum.SUSPECTED_DEFECT),
        ("path_evidence_conflict.json", DecisionPathEnum.EVIDENCE_CONFLICT),
        ("path_lighting_shift_review.json", DecisionPathEnum.LIGHTING_SHIFT_REVIEW),
        ("path_align_failed.json", DecisionPathEnum.ALIGN_FAILED),
        ("path_inconclusive.json", DecisionPathEnum.INCONCLUSIVE),
        ("path_budget_exhausted.json", DecisionPathEnum.BUDGET_EXHAUSTED),
    ]

    for filename, p in paths:
        insp = build_mock_inspection(
            p,
            insp_id=f"insp_fixture_{p.value.lower()}",
            req_id="req_fixture_det",
            ts="2026-10-17T12:00:00Z",
        )
        save_json(filename, insp.model_dump())

    # 2. Variants
    skipped_t3 = build_mock_inspection(
        DecisionPathEnum.SUSPECTED_DEFECT,
        skipped_t3=True,
        insp_id="insp_fixture_skipped_t3",
        req_id="req_fixture_det",
        ts="2026-10-17T12:00:00Z",
    )
    save_json("variant_skipped_t3.json", skipped_t3.model_dump())

    no_contour = build_mock_inspection(
        DecisionPathEnum.AGENT_CLEARED,
        no_contour=True,
        insp_id="insp_fixture_no_contour",
        req_id="req_fixture_det",
        ts="2026-10-17T12:00:00Z",
    )
    save_json("variant_no_contour.json", no_contour.model_dump())

    # 3. Session create & Session expired
    sess_create = SessionCreateResponse(
        session_token="fixture_session_token_sec1234567890",
        expires_at="2026-10-17T12:30:00Z",
    )
    save_json("session_create.json", sess_create.model_dump())

    sess_expired = ErrorResponse(
        error=ErrorDetail(
            code="SESSION_EXPIRED",
            message="Session has expired or is invalid",
            request_id="req_fixture_expired",
        )
    )
    save_json("session_expired.json", sess_expired.model_dump())

    # 4. Error shapes (400, 401, 404, 413, 415, 422, 429, 500, 503)
    errors = [
        ("error_400.json", "BAD_REQUEST", "Either 'file' or 'sample_id' must be provided."),
        ("error_401.json", "UNAUTHORIZED", "Missing or malformed Authorization header with Bearer token"),
        ("error_404.json", "NOT_FOUND", "Inspection not found"),
        ("error_413.json", "PAYLOAD_TOO_LARGE", "Image payload exceeds maximum allowed size of 10 MiB"),
        ("error_415.json", "UNSUPPORTED_MEDIA_TYPE", "Unsupported image format; expected PNG or JPEG"),
        ("error_422.json", "VALIDATION_ERROR", "Invalid request parameters or payload"),
        ("error_429.json", "RATE_LIMIT_EXCEEDED", "Rate limit exceeded; please wait before retrying"),
        ("error_500.json", "INTERNAL_SERVER_ERROR", "Internal server error"),
        ("error_503.json", "SERVICE_UNAVAILABLE", "Model inference queue full or artifacts unavailable"),
    ]
    for filename, code, msg in errors:
        err = ErrorResponse(
            error=ErrorDetail(code=code, message=msg, request_id=f"req_fixture_{code.lower()}")
        )
        save_json(filename, err.model_dump())

    # 5. Categories
    cats = CategoriesResponse(
        categories=[
            CategoryInfo(
                id="pcb1",
                display_name="PCB 1",
                enabled=True,
                alpha_pass=0.20,
                alpha_fail_default=0.05,
                alpha_fail_options=[0.01, 0.02, 0.05, 0.10],
                n_calib=300,
                sample_count=4,
            ),
            CategoryInfo(
                id="cashew",
                display_name="Cashew (Disabled)",
                enabled=False,
                alpha_pass=0.20,
                alpha_fail_default=0.05,
                alpha_fail_options=[0.02, 0.05, 0.10],
                n_calib=150,
                sample_count=0,
            ),
        ]
    )
    save_json("categories.json", cats.model_dump())

    # 6. Samples
    samples = get_mock_samples("pcb1")
    save_json("samples.json", samples.model_dump())

    # 7. Frozen Calibration
    calib = get_mock_calibration("pcb1")
    save_json("calibration_frozen.json", calib.model_dump())

    # 8. Feedback Logged Response
    feedback = FeedbackResponse(
        status="LOGGED",
        scope="audit_only",
        production_changed=False,
    )
    # Ensure only non-null fields or exact fields per spec
    save_json("feedback_logged.json", feedback.model_dump(exclude_none=True))

    # 9. Session Reset Response
    sess_reset = SessionResetResponse(
        status="RESET",
        message="Session shadow adaptation reset; frozen baseline active.",
        snapshot_id="wrn50_v1_frozen",
    )
    save_json("session_reset.json", sess_reset.model_dump())


if __name__ == "__main__":
    generate_all()
