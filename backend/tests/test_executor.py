"""
backend/tests/test_executor.py
Unit and integration tests for InspectionExecutor pluggable engine architecture.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app_factory import create_app
from backend.executor import EXECUTOR, InspectionExecutor
from backend.mock_service import TINY_PNG_BYTES, generate_mock_inspection
from backend.schemas import InspectionResponse


def test_executor_defaults(tmp_path: Path):
    """Verify default auto mode and fallback to mock when artifacts are absent."""
    executor = InspectionExecutor(artifacts_dir=tmp_path)
    assert executor.configured_mode == "auto"
    assert executor.model_version == "wrn50_v1_frozen"
    # In default checkout without model weights, live is not ready
    assert executor.is_live_ready() is False
    assert executor.get_engine_mode() == "mock"


def test_executor_mock_inspection():
    """Verify mock inspection returns valid InspectionResponse and asset dict."""
    executor = InspectionExecutor(mode="mock")
    insp, assets = executor.inspect(
        category="pcb1",
        sample_id="pcb1_good_01",
        alpha_fail=0.05,
    )
    assert isinstance(insp, InspectionResponse)
    assert insp.category == "pcb1"
    assert insp.engine == "mock"
    assert isinstance(assets, dict)
    assert len(assets) >= 1
    # Check that input asset is in assets dictionary
    input_key = insp.assets.input_standardized.split("/api/assets/", 1)[-1]
    assert input_key in assets
    assert assets[input_key][1] == "image/png"


def test_executor_explicit_mock_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify explicit mock mode via environment variable."""
    monkeypatch.setenv("SUREINSPECT_ENGINE", "mock")
    executor = InspectionExecutor(artifacts_dir=tmp_path)
    assert executor.configured_mode == "mock"
    assert executor.get_engine_mode() == "mock"


def test_executor_explicit_live_mode_unavailable(tmp_path: Path):
    """Verify 503 is raised if live mode is requested without model artifacts."""
    executor = InspectionExecutor(mode="live", artifacts_dir=tmp_path)
    assert executor.get_engine_mode() == "live"
    assert executor.is_live_ready() is False

    with pytest.raises(HTTPException) as exc_info:
        executor.inspect(category="pcb1", sample_id="pcb1_good_01")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "SERVICE_UNAVAILABLE"


def test_executor_live_with_snapshot():
    """Verify that when snapshot exists on disk, live engine is ready and executes."""
    executor = InspectionExecutor(artifacts_dir="artifacts")
    if executor.is_live_ready():
        assert executor.get_engine_mode() == "live"
        insp, assets = executor.inspect(category="pcb1", sample_id="template_00")
        assert insp.engine == "live"
        assert insp.decision == "PASS"
        assert insp.decision_path == "FAST_PASS"
        assert len(assets) >= 3


def test_executor_live_runner_registration(tmp_path: Path):
    """Verify registering a live runner activates live engine mode."""
    executor = InspectionExecutor(mode="auto", artifacts_dir=tmp_path)

    def dummy_live_runner(**kwargs):
        base = generate_mock_inspection(category="pcb1", sample_id="pcb1_good_01")
        dumped = base.model_dump()
        dumped["engine"] = "live"
        live_insp = InspectionResponse.model_validate(dumped)
        assets = {"dummy_live_asset": (TINY_PNG_BYTES, "image/png")}
        return live_insp, assets

    assert executor.is_live_ready() is False
    executor.register_live_runner(dummy_live_runner)
    assert executor.is_live_ready() is True
    assert executor.get_engine_mode() == "live"

    insp, assets = executor.inspect(category="pcb1", sample_id="pcb1_good_01")
    assert insp.engine == "live"
    assert "dummy_live_asset" in assets


def test_executor_detects_artifacts(tmp_path: Path):
    """Verify detection of artifacts when weights file exists on disk."""
    weights_file = tmp_path / "backbone.onnx"
    weights_file.write_bytes(b"dummy_onnx_weights")

    executor = InspectionExecutor(artifacts_dir=tmp_path)
    assert executor.artifacts_dir == tmp_path
    has_artifacts = (tmp_path / "backbone.onnx").is_file()
    assert has_artifacts is True


def test_health_reflects_executor_mode():
    """Verify that /api/health reflects live/mock transition dynamically."""
    app = create_app()
    client = TestClient(app)

    initial_mode = EXECUTOR._explicit_mode
    try:
        EXECUTOR._explicit_mode = "mock"
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["engine"] == "mock"

        EXECUTOR._explicit_mode = "live"
        res_live = client.get("/api/health")
        assert res_live.status_code == 200
        assert res_live.json()["engine"] == "live"
    finally:
        EXECUTOR._explicit_mode = initial_mode
