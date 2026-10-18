"""
backend/app_factory.py
FastAPI application factory implementing all SureInspect v8.1 routes:
- Bounded streaming uploads (10 MiB limit, PNG/JPEG magic check).
- Decompression bomb protection (OPENCV_IO_MAX_IMAGE_PIXELS=25000000 ceiling, HTTP 413).
- Bounded async admission queue (max 1 running inference job, max 4 waiting, HTTP 503 on saturation).
- Bounded storage manager (200 MiB, 100 inspections, 30-min TTL, coherent LRU eviction).
- Session ownership enforcement and audit feedback idempotency.
- Allowlisted sample lookup from curated data/samples directory.
"""
from __future__ import annotations

import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.admission import (
    INSPECTION_ADMISSION,
    decode_and_validate_image,
    read_bounded_upload,
)
from backend.executor import EXECUTOR
from backend.mock_service import (
    ALLOWLISTED_SAMPLE_IDS,
    TINY_PNG_BYTES,
    get_mock_calibration,
    get_mock_samples,
)
from backend.schemas import (
    CalibrationSummaryResponse,
    CategoriesResponse,
    CategoryInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    InspectionResponse,
    SamplesResponse,
    SessionCreateResponse,
    SessionResetResponse,
)
from backend.storage import (
    STORAGE,
)

# ---------------------------------------------------------------------------
# Session Management & Backward Compatibility Dictionaries
# ---------------------------------------------------------------------------

SESSION_STORE: dict[str, dict[str, float]] = {}  # token -> {"expires_at": float, "last_active": float}
SESSION_TTL_SECONDS = 1800  # 30-minute idle TTL
MAX_ACTIVE_SESSIONS = 100   # Bounded active sessions

# Backward-compatibility alias references for existing tests
STORED_INSPECTIONS: dict[str, dict[str, Any]] = {}
STORED_ASSETS: dict[str, dict[str, Any]] = {}
FEEDBACK_IDEMPOTENCY: dict[tuple[str, str], dict[str, Any]] = {}


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


def verify_session(authorization: str | None = Header(None)) -> str:
    """Verifies Bearer token, enforces 30-minute idle TTL. Never logs the token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Missing or malformed Authorization header with Bearer token"},
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    now = time.time()
    purge_expired_sessions()
    record = SESSION_STORE.get(token)
    if not record or now > record["expires_at"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "SESSION_EXPIRED", "message": "Session has expired or is invalid"},
        )
    # Refresh idle TTL on valid request
    record["expires_at"] = now + SESSION_TTL_SECONDS
    record["last_active"] = now
    return token


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


def create_app() -> FastAPI:
    """Factory creating and configuring the SureInspect FastAPI application."""
    app = FastAPI(
        title="SureInspect API",
        version="1.0.0",
        description="Explainable, Risk-Controlled Agentic Quality Inspection System",
        docs_url="/docs",
        redoc_url="/redoc",
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
            engine=EXECUTOR.get_engine_mode(),
            model_version=EXECUTOR.model_version,
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
                    alpha_fail_options=[0.01, 0.02, 0.05, 0.10],
                    n_calib=0,
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
        sample_path = Path("data/samples") / f"{id}.png"
        if sample_path.is_file():
            return Response(content=sample_path.read_bytes(), media_type="image/png")
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
        file: UploadFile | None = File(None),  # noqa: B008
        sample_id: str | None = Form(None),
        category: str = Form("pcb1"),
        simulate_blur: bool = Form(False),
        simulate_darken: bool = Form(False),
        simulate_glare: bool = Form(False),
        alpha_fail: float = Form(0.05),
        path: str | None = Form(None),
        session_token: str = Depends(verify_session),
    ):
        if not file and not sample_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "BAD_REQUEST", "message": "Either 'file' or 'sample_id' must be provided."},
            )

        orig_dims = (256, 256)
        if file:
            raw_bytes, _media_type = await read_bounded_upload(file)
            decoded_img = decode_and_validate_image(raw_bytes)
            orig_dims = (decoded_img.shape[1], decoded_img.shape[0])

        # Acquire bounded admission slot (max 1 running, 4 queued, 503 if saturated)
        await INSPECTION_ADMISSION.acquire()
        try:
            inspection, assets_to_store = EXECUTOR.inspect(
                category=category,
                sample_id=sample_id,
                image_bytes=raw_bytes if file else None,
                decoded_img=decoded_img if file else None,
                simulate_blur=simulate_blur,
                simulate_darken=simulate_darken,
                simulate_glare=simulate_glare,
                alpha_fail=alpha_fail,
                path=path,
                orig_dimensions=orig_dims,
            )

            STORAGE.store_inspection(
                session_token=session_token,
                inspection=inspection,
                category=category,
                decision=inspection.decision.value,
                assets=assets_to_store,
                orig_dimensions=orig_dims,
            )

            # Keep backward compatibility dictionaries updated
            now = time.time()
            STORED_INSPECTIONS[inspection.inspection_id] = {
                "session_token": session_token,
                "response": inspection,
                "category": category,
                "decision": inspection.decision.value,
                "created_at": now,
            }
            for a_key, (content, _) in assets_to_store.items():
                STORED_ASSETS[a_key] = {
                    "session_token": session_token,
                    "data": content,
                    "created_at": now,
                }

            return inspection
        finally:
            await INSPECTION_ADMISSION.release()

    @app.get("/api/trace/{id}", response_model=InspectionResponse)
    def get_trace(id: str, session_token: str = Depends(verify_session)):
        """Protected trace lookup. Non-owned or unknown returns identical 404."""
        return STORAGE.get_trace(id, session_token)

    @app.get("/api/assets/{id}")
    def get_asset(id: str, session_token: str = Depends(verify_session)):
        """Protected asset lookup. Non-owned or unknown returns identical 404."""
        content, media_type = STORAGE.get_asset(id, session_token)
        return Response(content=content, media_type=media_type)

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
        # Retrieve target record to find category
        record = STORED_INSPECTIONS.get(feedback.inspection_id)
        category = record["category"] if record else "pcb1"

        return STORAGE.record_feedback(
            session_token=session_token,
            inspection_id=feedback.inspection_id,
            user_label=feedback.user_label.value if hasattr(feedback.user_label, "value") else str(feedback.user_label),
            category=category,
            idempotency_key=feedback.idempotency_key,
        )

    # Serve static frontend build if present
    dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static_assets")

    @app.get("/", include_in_schema=False)
    def serve_root():
        index_file = dist_dir / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return JSONResponse({"status": "healthy", "service": "SureInspect API", "docs": "/docs"})

    return app
