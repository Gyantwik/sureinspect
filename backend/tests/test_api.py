"""
backend/tests/test_api.py
Comprehensive unit and integration tests for SureInspect backend mock API.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app_factory import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "mock"
    assert "pcb1" in data["enabled_categories"]

def test_categories_endpoint(client):
    res = client.get("/api/categories")
    assert res.status_code == 200
    data = res.json()
    assert len(data["categories"]) >= 1
    pcb1 = next(c for c in data["categories"] if c["id"] == "pcb1")
    assert pcb1["enabled"] is True
    assert pcb1["alpha_pass"] == 0.20
    assert pcb1["alpha_fail_default"] == 0.05

def test_samples_endpoint(client):
    res = client.get("/api/samples?category=pcb1")
    assert res.status_code == 200
    data = res.json()
    assert len(data["samples"]) >= 1

def test_session_creation(client):
    res = client.post("/api/session")
    assert res.status_code == 200
    data = res.json()
    assert "session_token" in data
    assert "expires_at" in data
    assert len(data["session_token"]) > 20

def test_protected_inspect_without_auth_fails(client):
    res = client.post("/api/inspect", data={"sample_id": "pcb1_good_01"})
    assert res.status_code == 401
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "UNAUTHORIZED"

def test_protected_inspect_with_auth_succeeds(client):
    # 1. Create session
    session_res = client.post("/api/session")
    token = session_res.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Call inspect on clean part (Fast Pass)
    res = client.post(
        "/api/inspect",
        data={"sample_id": "pcb1_good_01", "category": "pcb1"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] == "PASS"
    assert data["decision_path"] == "FAST_PASS"
    assert data["engine"] == "mock"

def test_quality_gate_rejection(client):
    session_res = client.post("/api/session")
    token = session_res.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Simulate blur
    res = client.post(
        "/api/inspect",
        data={"sample_id": "pcb1_good_01", "category": "pcb1", "simulate_blur": "true"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] == "RETAKE"
    assert data["decision_path"] == "QUALITY_REJECTED"
    assert data["quality_gate"]["status"] == "FAILED"
    assert data["quality_gate"]["action"] == "REQUEST_RETAKE"

def test_feedback_logging(client):
    session_res = client.post("/api/session")
    token = session_res.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post(
        "/api/feedback",
        json={"inspection_id": "insp_test_123", "user_label": "NORMAL", "idempotency_key": "idem_1"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "LOGGED"
    assert data["scope"] == "audit_only"
    assert data["production_changed"] is False
