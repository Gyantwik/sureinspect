"""
backend/app_factory.py
FastAPI application factory implementing all SureInspect v8.1 routes in mock mode,
strictly adhering to error contracts, session ownership, and feedback semantics.
"""
from __future__ import annotations

import secrets
import time
import uuid
from typing import Dict, Optional, Tuple, Any

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Depends,
    Request,
    UploadFile,
    File,
    Form,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError

from backend.schemas import (
    ErrorResponse,
    SessionCreateResponse,
    SessionResetResponse,
    HealthResponse,
    CategoriesResponse,
    CategoryInfo,
    SamplesResponse,
    InspectionResponse,
    FeedbackRequest,
    FeedbackResponse,
    CalibrationSummaryResponse,
    DecisionEnum,
    DecisionPathEnum,
)
from backend.mock_service import (
    get_mock_samples,
    get_mock_calibration,
    generate_mock_inspection,
    build_mock_inspection,
    ALLOWLISTED_SAMPLE_IDS,
    TINY_PNG_BYTES,
)

# ---------------------------------------------------------------------------
# In-memory Storage & Session Management
# ---------------------------------------------------------------------------

SESSION_STORE: Dict[str, Dict[str, float]] = {}  # token -> {"expires_at": float, "last_active": float}
SESSION_TTL_SECONDS = 1800  # 30-minute idle TTL
MAX_ACTIVE_SESSIONS = 100   # Bounded active sessions

# Session-bound stores
# inspection_id -> {"session_token": str, "response": InspectionResponse, "category": str, "decision": str, "created_at": float}
STORED_INSPECTIONS: Dict[str, Dict[str, Any]] = {}

# asset_id -> {"session_token": str, "data": bytes, "created_at": float}
STORED_ASSETS: Dict[str, Dict[str, Any]] = {}

# (session_token, idempotency_key) -> {"inspection_id": str, "user_label": str, "category": str, "created_at": float}
FEEDBACK_IDEMPOTENCY: Dict[Tuple[str, str], Dict[str, Any]] = {}


def purge_expired_sessions() -> None:
    now = time.time()
    expired = [t for t, s in SESSION_STORE.items() if now > s["expires_at"]]
    for t in expired:
        SESSION_STORE.pop(t, None)


def create_new_session() -> SessionCreateResponse:
    purge_expired_sessions()
    if len(SESSION_STORE) >= MAX_ACTIVE_SESSIONS:
        # Evict oldest by last_active
        oldest_token = min(SESSION_STORE.keys(), key=lambda t: SESSION_STORE[t]["last_active"])
        SESSION_STORE.pop(oldest_token, None)

    token = secrets.token_urlsafe(32)
    now = time.time()
    expires_at = now + SESSION_TTL_SECONDS
    SESSION_STORE[token] = {"expires_at": expires_at, "last_active": now}
    expires_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at))
    return SessionCreateResponse(session_token=token, expires_at=expires_str)


def verify_session(authorization: Optional[str] = Header(None)) -> str:
    """Verifies Bearer token, enforces 30-minute idle TTL. Never logs the token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or malformed Authorization header with Bearer token"},
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    session = SESSION_STORE.get(token)
    if not session or time.time() > session["expires_at"]:
        SESSION_STORE.pop(token, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "SESSION_EXPIRED", "message": "Session has expired or is invalid"},
        )
    # Refresh idle TTL on valid request
    now = time.time()
    session["expires_at"] = now + SESSION_TTL_SECONDS
    session["last_active"] = now
    return token


# ---------------------------------------------------------------------------
# Status Code to Standard Error Code Mapping
# ---------------------------------------------------------------------------

DEFAULT_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_SERVER_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="SureInspect API",
        version="1.0.0",
        description="Explainable, Risk-Controlled Agentic Quality Inspection System",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Uniform Error Handlers strictly returning {"error": {"code", "message", "request_id"}}
    # -----------------------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            code = exc.detail["code"]
            message = exc.detail.get("message", "An error occurred")
        else:
            code = DEFAULT_ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
            message = str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message, "request_id": req_id}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters or payload",
                    "request_id": req_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error",
                    "request_id": req_id,
                }
            },
        )

    # -----------------------------------------------------------------------
    # Public Endpoints
    # -----------------------------------------------------------------------
    @app.get("/api/health", response_model=HealthResponse)
    def health_check():
        return HealthResponse(
            status="healthy",
            engine="mock",
            model_version="wrn50_v1_frozen",
            enabled_categories=["pcb1"],
            loaded_categories=["pcb1"],
        )

    @app.post("/api/session", response_model=SessionCreateResponse)
    def create_session():
        return create_new_session()

    @app.get("/api/categories", response_model=CategoriesResponse)
    def list_categories():
        return CategoriesResponse(
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

    @app.get("/api/samples", response_model=SamplesResponse)
    def list_samples(category: str = "pcb1"):
        return get_mock_samples(category=category)

    @app.get("/api/samples/{id}/image")
    def get_sample_image(id: str):
        """Allowlisted lookup for curated samples; rejects unknown IDs with 404 without touching filesystem."""
        if id not in ALLOWLISTED_SAMPLE_IDS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"Sample image '{id}' not found"},
            )
        return Response(content=TINY_PNG_BYTES, media_type="image/png")

    @app.get("/api/calibration/{category}", response_model=CalibrationSummaryResponse)
    def get_calibration(category: str):
        return get_mock_calibration(category=category)

    # -----------------------------------------------------------------------
    # Protected Endpoints (Require Bearer Session Token)
    # -----------------------------------------------------------------------
    @app.post("/api/session/reset", response_model=SessionResetResponse)
    def reset_session(session_token: str = Depends(verify_session)):
        return SessionResetResponse(
            status="RESET",
            message="Session shadow adaptation reset; frozen baseline active.",
            snapshot_id="wrn50_v1_frozen",
        )

    @app.post("/api/inspect", response_model=InspectionResponse)
    async def inspect_part(
        file: Optional[UploadFile] = File(None),
        sample_id: Optional[str] = Form(None),
        category: str = Form("pcb1"),
        simulate_blur: bool = Form(False),
        simulate_darken: bool = Form(False),
        simulate_glare: bool = Form(False),
        alpha_fail: float = Form(0.05),
        path: Optional[str] = Form(None),
        session_token: str = Depends(verify_session),
    ):
        if not file and not sample_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BAD_REQUEST", "message": "Either 'file' or 'sample_id' must be provided."},
            )

        inspection = generate_mock_inspection(
            category=category,
            simulate_blur=simulate_blur,
            simulate_darken=simulate_darken,
            simulate_glare=simulate_glare,
            sample_id=sample_id,
            alpha_fail=alpha_fail,
            path=path,
        )

        now = time.time()
        # Bind inspection to session
        STORED_INSPECTIONS[inspection.inspection_id] = {
            "session_token": session_token,
            "response": inspection,
            "category": category,
            "decision": inspection.decision.value,
            "created_at": now,
        }

        # Store generated assets bound to this session
        insp_id = inspection.inspection_id
        for asset_url in [
            inspection.assets.input_standardized,
            inspection.assets.reference_aligned,
            inspection.assets.anomaly_heatmap,
            inspection.assets.contour_mask,
        ]:
            if asset_url:
                asset_key = asset_url.split("/api/assets/", 1)[-1]
                STORED_ASSETS[asset_key] = {
                    "session_token": session_token,
                    "data": TINY_PNG_BYTES,
                    "created_at": now,
                }

        return inspection

    @app.get("/api/trace/{id}", response_model=InspectionResponse)
    def get_trace(id: str, session_token: str = Depends(verify_session)):
        """Protected trace lookup. Non-owned or unknown returns identical 404."""
        record = STORED_INSPECTIONS.get(id)
        if not record or record["session_token"] != session_token:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": "Inspection not found"},
            )
        return record["response"]

    @app.get("/api/assets/{id}")
    def get_asset(id: str, session_token: str = Depends(verify_session)):
        """Protected asset lookup. Non-owned or unknown returns identical 404."""
        record = STORED_ASSETS.get(id)
        if not record or record["session_token"] != session_token:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": "Asset not found"},
            )
        return Response(content=record["data"], media_type="image/png")

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def submit_feedback(
        feedback: FeedbackRequest,
        session_token: str = Depends(verify_session),
    ):
        """
        Operator feedback endpoint:
        - Resolves category from stored inspection.
        - Unknown or non-owned inspection returns 404.
        - Rejects RETAKE or expired records.
        - Idempotent: repeated identical key is safe, conflicting reuse is 409.
        - Returns {"status": "LOGGED", "scope": "audit_only", "production_changed": False}.
        """
        record = STORED_INSPECTIONS.get(feedback.inspection_id)
        if not record or record["session_token"] != session_token:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": "Inspection not found"},
            )

        # Check expired record
        if time.time() - record["created_at"] > SESSION_TTL_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BAD_REQUEST", "message": "Inspection record has expired"},
            )

        # Reject RETAKE
        if record["decision"] == DecisionEnum.RETAKE.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BAD_REQUEST", "message": "Cannot submit operator feedback on RETAKE inspection"},
            )

        category = record["category"]
        idempotency_key = (session_token, feedback.idempotency_key)

        if idempotency_key in FEEDBACK_IDEMPOTENCY:
            existing = FEEDBACK_IDEMPOTENCY[idempotency_key]
            if (
                existing["inspection_id"] == feedback.inspection_id
                and existing["user_label"] == feedback.user_label
            ):
                # Safe repeat
                return FeedbackResponse(
                    status="LOGGED",
                    scope="audit_only",
                    production_changed=False,
                )
            else:
                # Conflicting reuse
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "CONFLICT",
                        "message": f"Idempotency key '{feedback.idempotency_key}' reused with conflicting payload",
                    },
                )

        # Record new feedback event
        FEEDBACK_IDEMPOTENCY[idempotency_key] = {
            "inspection_id": feedback.inspection_id,
            "user_label": feedback.user_label,
            "category": category,
            "created_at": time.time(),
        }

        return FeedbackResponse(
            status="LOGGED",
            scope="audit_only",
            production_changed=False,
        )

    return app
