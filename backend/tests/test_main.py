"""
backend/tests/test_main.py
Unit and integration tests for backend/main.py runtime entrypoint.
"""
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app, parse_args


def test_app_instance():
    """Verify that backend.main exports a valid FastAPI instance."""
    assert isinstance(app, FastAPI)
    assert app.title == "SureInspect API"


def test_main_health_endpoint():
    """Verify that backend.main:app serves the health endpoint."""
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "mock"


def test_parse_args_defaults():
    """Verify parse_args default CLI parameters."""
    args = parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.reload is False
    assert args.workers == 1
    assert args.forwarded_allow_ips == "127.0.0.1"


def test_parse_args_custom():
    """Verify parse_args with custom CLI parameters."""
    args = parse_args(["--host", "0.0.0.0", "--port", "9000", "--reload", "--workers", "2", "--forwarded-allow-ips", "10.0.0.1"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.reload is True
    assert args.workers == 2
    assert args.forwarded_allow_ips == "10.0.0.1"


def test_opencv_pixel_limit_env_set():
    """Verify that OPENCV_IO_MAX_IMAGE_PIXELS is set to 25000000."""
    assert os.environ.get("OPENCV_IO_MAX_IMAGE_PIXELS") == "25000000"
