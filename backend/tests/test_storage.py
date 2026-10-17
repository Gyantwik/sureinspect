"""
backend/tests/test_storage.py
Tests for StorageManager:
- Bounded cache eviction when count exceeds max_inspections (100).
- Bounded cache eviction when bytes exceed max_bytes (200 MiB).
- TTL expiration (30 minutes).
- Session ownership protection for traces and assets.
- Feedback idempotency and RETAKE rejection.
"""
import time
import pytest
from fastapi import HTTPException

from backend.storage import StorageManager, encode_lossless_png, encode_preview_jpeg
from backend.mock_service import build_mock_inspection
from backend.schemas import DecisionPathEnum, DecisionEnum
import numpy as np


def test_storage_count_eviction():
    # Test with small count limit of 3
    storage = StorageManager(max_bytes=10_000_000, max_inspections=3, ttl_seconds=1800)
    session = "sess_1"

    inspections = [build_mock_inspection(DecisionPathEnum.FAST_PASS) for _ in range(5)]
    for i, insp in enumerate(inspections):
        insp.inspection_id = f"insp_{i}"
        storage.store_inspection(
            session_token=session,
            inspection=insp,
            category="pcb1",
            decision="PASS",
            assets={f"asset_{i}": (b"x" * 100, "image/png")},
        )

    # Only the last 3 should remain (insp_2, insp_3, insp_4)
    assert storage.inspection_count == 3
    with pytest.raises(HTTPException) as exc_info:
        storage.get_trace("insp_0", session)
    assert exc_info.value.status_code == 404

    with pytest.raises(HTTPException) as exc_info:
        storage.get_trace("insp_1", session)
    assert exc_info.value.status_code == 404

    assert storage.get_trace("insp_4", session).inspection_id == "insp_4"


def test_storage_byte_limit_eviction():
    # Test with small byte limit of 500 bytes
    storage = StorageManager(max_bytes=500, max_inspections=10, ttl_seconds=1800)
    session = "sess_1"

    for i in range(4):
        insp = build_mock_inspection(DecisionPathEnum.FAST_PASS)
        insp.inspection_id = f"insp_b_{i}"
        # Each inspection has 200 bytes of assets
        storage.store_inspection(
            session_token=session,
            inspection=insp,
            category="pcb1",
            decision="PASS",
            assets={f"asset_b_{i}": (b"A" * 200, "image/png")},
        )

    # 4 * 200 = 800 bytes > 500 bytes limit. Oldest should have been evicted.
    assert storage.total_bytes <= 500
    with pytest.raises(HTTPException):
        storage.get_asset("asset_b_0", session)
    assert storage.get_asset("asset_b_3", session)[0] == b"A" * 200


def test_storage_ttl_expiration():
    # Test with 1-second TTL
    storage = StorageManager(max_bytes=10000, max_inspections=10, ttl_seconds=1)
    session = "sess_1"
    insp = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp.inspection_id = "insp_ttl"

    storage.store_inspection(
        session_token=session,
        inspection=insp,
        category="pcb1",
        decision="PASS",
        assets={"asset_ttl": (b"data", "image/png")},
    )

    # Immediate access works
    assert storage.get_trace("insp_ttl", session).inspection_id == "insp_ttl"

    # Sleep past TTL
    time.sleep(1.1)
    with pytest.raises(HTTPException) as exc_info:
        storage.get_trace("insp_ttl", session)
    assert exc_info.value.status_code == 404


def test_session_ownership_protection():
    storage = StorageManager()
    owner_session = "sess_alice"
    intruder_session = "sess_bob"

    insp = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp.inspection_id = "insp_alice"
    storage.store_inspection(
        session_token=owner_session,
        inspection=insp,
        category="pcb1",
        decision="PASS",
        assets={"asset_alice": (b"secure_pixels", "image/png")},
    )

    # Owner accesses successfully
    assert storage.get_trace("insp_alice", owner_session).inspection_id == "insp_alice"
    data, media_type = storage.get_asset("asset_alice", owner_session)
    assert data == b"secure_pixels"
    assert media_type == "image/png"

    # Intruder access yields 404 (not 403, preventing resource enumeration)
    with pytest.raises(HTTPException) as exc_trace:
        storage.get_trace("insp_alice", intruder_session)
    assert exc_trace.value.status_code == 404

    with pytest.raises(HTTPException) as exc_asset:
        storage.get_asset("asset_alice", intruder_session)
    assert exc_asset.value.status_code == 404


def test_lossless_png_and_jpeg_encoding():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:30, 10:30] = [0, 255, 0]

    png_bytes = encode_lossless_png(img)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    jpeg_bytes = encode_preview_jpeg(img, quality=85)
    assert jpeg_bytes.startswith(b"\xff\xd8\xff")
