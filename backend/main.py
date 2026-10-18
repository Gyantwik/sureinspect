"""
backend/main.py
Runtime entrypoint for the SureInspect FastAPI application.

Enforces:
- OPENCV_IO_MAX_IMAGE_PIXELS=25000000 before any cv2/image import.
- Bounded thread counts for NumPy / OpenBLAS / OpenMP.
- Single uvicorn worker configuration.
- Localhost default binding with proxy header support.
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Strict Environment Pre-Initialization (before importing cv2 / numpy)
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "25000000")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import uvicorn

from backend.app_factory import create_app

# Global ASGI application instance for uvicorn workers
app = create_app()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SureInspect Backend Server (OpenCV AI Competition 2026)"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("RELOAD", "false").lower() in ("true", "1", "yes"),
        help="Enable auto-reload for local development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of uvicorn worker processes (default: 1)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        help="Trusted proxy IPs for X-Forwarded headers",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        forwarded_allow_ips=args.forwarded_allow_ips,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
