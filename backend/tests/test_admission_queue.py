"""
backend/tests/test_admission_queue.py
Tests for bounded async admission queue cancellation, saturation, and uniform error envelopes.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.admission import MAX_UPLOAD_BYTES, AdmissionController
from backend.app_factory import create_app


def test_queue_cancellation_cleans_up_counts():
    """
    Cancel test:
    1 running + 2 waiting tasks.
    Cancel 1 of the waiting tasks before it acquires.
    Verify running == 0 and queued == 0 at the end, and verify a subsequent
    request is admitted cleanly.
    """
    async def _run():
        ctrl = AdmissionController(max_running=1, max_queued=4)

        # 1 running task
        await ctrl.acquire()
        assert ctrl.running == 1
        assert ctrl.queued == 0

        # 2 waiting tasks
        task_b = asyncio.create_task(ctrl.acquire())
        task_c = asyncio.create_task(ctrl.acquire())
        await asyncio.sleep(0.01)

        assert ctrl.running == 1
        assert ctrl.queued == 2

        # Cancel task_b while it is waiting in the queue
        task_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_b

        # task_b was cancelled before acquiring: queued decrements, running remains 1
        assert ctrl.running == 1
        assert ctrl.queued == 1

        # Release task_a; task_c should now acquire
        await ctrl.release()
        await task_c
        assert ctrl.running == 1
        assert ctrl.queued == 0

        # Release task_c; counts should return cleanly to 0
        await ctrl.release()
        assert ctrl.running == 0
        assert ctrl.queued == 0

        # Verify a subsequent request is admitted cleanly
        await ctrl.acquire()
        assert ctrl.running == 1
        assert ctrl.queued == 0
        await ctrl.release()
        assert ctrl.running == 0
        assert ctrl.queued == 0

    asyncio.run(_run())


def test_queue_saturation_and_recovery():
    """
    Saturation test:
    1 running + 4 queued (5 total allowed by defaults).
    The 6th request gets HTTP 503 SERVICE_UNAVAILABLE immediately without blocking.
    When running tasks finish, queued tasks promote to running, and new requests are admitted.
    """
    async def _run():
        ctrl = AdmissionController(max_running=1, max_queued=4)

        # Fill 1 running slot
        await ctrl.acquire()
        assert ctrl.running == 1
        assert ctrl.queued == 0

        # Fill 4 queued slots
        queued_tasks = [asyncio.create_task(ctrl.acquire()) for _ in range(4)]
        await asyncio.sleep(0.01)

        assert ctrl.running == 1
        assert ctrl.queued == 4

        # 6th request exceeds 1 + 4 capacity -> 503 immediately
        with pytest.raises(HTTPException) as exc_info:
            await ctrl.acquire()
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["code"] == "QUEUE_SATURATED"

        # Drain queued tasks as running tasks finish
        for i in range(4):
            await ctrl.release()
            await queued_tasks[i]
            assert ctrl.running == 1
            assert ctrl.queued == (3 - i)

        # Release final task
        await ctrl.release()
        assert ctrl.running == 0
        assert ctrl.queued == 0

        # New requests are admitted now that capacity has freed up
        await ctrl.acquire()
        assert ctrl.running == 1
        assert ctrl.queued == 0
        await ctrl.release()
        assert ctrl.running == 0
        assert ctrl.queued == 0

    asyncio.run(_run())


def test_uniform_error_shapes_through_app():
    """
    Error shape test:
    413, 415, and 503 responses through the app have the exact same top-level envelope:
    {"error": {"code": str, "message": str, "request_id": str}} matching the contract.
    """
    app = create_app()
    client = TestClient(app)

    # Helper to assert strict ErrorEnvelope shape
    def assert_error_envelope(response_json: dict, expected_code: str):
        assert "error" in response_json, f"Missing 'error' in response: {response_json}"
        err = response_json["error"]
        assert set(err.keys()) == {"code", "message", "request_id"}
        assert isinstance(err["code"], str)
        assert err["code"] == expected_code
        assert isinstance(err["message"], str)
        assert len(err["message"]) > 0
        assert isinstance(err["request_id"], str)
        assert err["request_id"].startswith("req_")

    # Create active session for authentication
    session_res = client.post("/api/session")
    token = session_res.json()["session_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. HTTP 413: upload > 10 MiB
    oversized_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_UPLOAD_BYTES + 1024)
    resp_413 = client.post(
        "/api/inspect",
        files={"file": ("bomb.png", oversized_bytes, "image/png")},
        data={"category": "pcb1"},
        headers=headers,
    )
    assert resp_413.status_code == 413
    assert_error_envelope(resp_413.json(), "PAYLOAD_TOO_LARGE")

    # 2. HTTP 415: unsupported media type / invalid magic bytes
    resp_415 = client.post(
        "/api/inspect",
        files={"file": ("test.txt", b"plain text is not an image", "text/plain")},
        data={"category": "pcb1"},
        headers=headers,
    )
    assert resp_415.status_code == 415
    assert_error_envelope(resp_415.json(), "UNSUPPORTED_MEDIA_TYPE")

    # 3. HTTP 503: saturated queue
    with patch(
        "backend.app_factory.INSPECTION_ADMISSION.acquire",
        new=AsyncMock(
            side_effect=HTTPException(
                status_code=503,
                detail={"code": "QUEUE_SATURATED", "message": "Inference queue saturated."},
            )
        ),
    ):
        resp_503 = client.post(
            "/api/inspect",
            data={"sample_id": "pcb1_bad_missing_01", "category": "pcb1"},
            headers=headers,
        )
        assert resp_503.status_code == 503
        assert_error_envelope(resp_503.json(), "QUEUE_SATURATED")
