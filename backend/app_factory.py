"""
backend/app_factory.py
FastAPI application factory implementing all SureInspect v8.1 routes in mock mode.
"""
from __future__ import annotations

import secrets
import time
import uuid
from typing import Dict, Optional

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
    ErrorDetail,
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
)
from backend.mock_service import (
    get_mock_samples,
    get_mock_calibration,
    generate_mock_inspection,
    handle_mock_feedback,
    TINY_PNG_B64,
)

# ---------------------------------------------------------------------------
# In-memory Session Manager
# ---------------------------------------------------------------------------

SESSION_STORE: Dict[str, float] = {}  # token -> expires_at_timestamp
SESSION_TTL_SECONDS = 1800  # 30 minutes


def create_new_session() -> SessionCreateResponse:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + SESSION_TTL_SECONDS
    SESSION_STORE[token] = expires_at
    expires_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at))
    return SessionCreateResponse(session_token=token, expires_at=expires_str)


def verify_session(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header with Bearer token",
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    expires_at = SESSION_STORE.get(token)
    if not expires_at or time.time() > expires_at:
        SESSION_STORE.pop(token, None)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid",
        )
    # Refresh idle TTL on valid request
    SESSION_STORE[token] = time.time() + SESSION_TTL_SECONDS
    return token


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
    # Error Handlers strictly matching {"error": {"code", "message", "request_id"}}
    # -----------------------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        code = "HTTP_ERROR"
        if exc.status_code == 401:
            code = "UNAUTHORIZED"
        elif exc.status_code == 404:
            code = "NOT_FOUND"
        elif exc.status_code == 400:
            code = "BAD_REQUEST"
        elif exc.status_code == 413:
            code = "PAYLOAD_TOO_LARGE"
        elif exc.status_code == 415:
            code = "UNSUPPORTED_MEDIA_TYPE"
        elif exc.status_code == 429:
            code = "RATE_LIMIT_EXCEEDED"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail), "request_id": req_id}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request fields",
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

    @app.get("/api/samples/{sample_id}/image")
    def get_sample_image(sample_id: str):
        # Return 1x1 transparent png for mock
        import base64
        data = base64.b64decode(TINY_PNG_B64.split(",", 1)[1])
        return Response(content=data, media_type="image/png")

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
        session_token: str = Depends(verify_session),
    ):
        if not file and not sample_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or 'sample_id' must be provided.",
            )
        return generate_mock_inspection(
            category=category,
            simulate_blur=simulate_blur,
            simulate_darken=simulate_darken,
            simulate_glare=simulate_glare,
            sample_id=sample_id,
            alpha_fail=alpha_fail,
        )

    @app.get("/api/trace/{inspection_id}", response_model=InspectionResponse)
    def get_trace(inspection_id: str, session_token: str = Depends(verify_session)):
        return generate_mock_inspection(category="pcb1", sample_id="pcb1_bad_missing_01")

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str, session_token: str = Depends(verify_session)):
        import base64
        data = base64.b64decode(TINY_PNG_B64.split(",", 1)[1])
        return Response(content=data, media_type="image/png")

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def submit_feedback(
        feedback: FeedbackRequest,
        session_token: str = Depends(verify_session),
    ):
        return handle_mock_feedback(
            inspection_id=feedback.inspection_id,
            user_label=feedback.user_label,
        )

    return app
