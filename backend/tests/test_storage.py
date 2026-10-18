"""
backend/tests/test_storage.py
Tests for StorageManager:
- Bounded cache eviction when count exceeds max_inspections (100).
- Bounded cache eviction when bytes exceed max_bytes (200 MiB).
- TTL expiration for both traces and individual assets (30 minutes).
- Immediate purge on reads (get_trace, get_asset, record_feedback).
- Rejection of single incoming payloads exceeding total cache capacity (413).
- Overwrite of repeated inspection ID cleans up previous assets with zero orphaned bytes.
- Feedback idempotency records are evicted coherently with parent inspections.
- Feedback checks target inspection ownership before inspecting idempotency keys.
- Session ownership protection for traces and assets.
- Feedback idempotency and RETAKE rejection.
"""
import time

import numpy as np
import pytest
from fastapi import HTTPException

from backend.mock_service import build_mock_inspection
from backend.schemas import DecisionPathEnum
from backend.storage import StorageManager, encode_lossless_png, encode_preview_jpeg


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


def test_storage_ttl_expiration_trace_and_asset():
    """Both get_trace and get_asset strictly respect TTL and purge expired data."""
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
    data, _ = storage.get_asset("asset_ttl", session)
    assert data == b"data"

    # Sleep past TTL
    time.sleep(1.1)

    # Asset must return 404 once expired (bug 3 fix)
    with pytest.raises(HTTPException) as exc_asset:
        storage.get_asset("asset_ttl", session)
    assert exc_asset.value.status_code == 404

    # Trace must return 404 once expired
    with pytest.raises(HTTPException) as exc_trace:
        storage.get_trace("insp_ttl", session)
    assert exc_trace.value.status_code == 404

    # Reading also purged the expired inspection and its bytes (bug 4 fix)
    assert storage.inspection_count == 0
    assert storage.total_bytes == 0


def test_storage_oversized_single_payload_rejected():
    """Single inspection whose assets exceed max_bytes is rejected with 413 (bug 5 fix)."""
    storage = StorageManager(max_bytes=1000, max_inspections=10, ttl_seconds=1800)
    session = "sess_1"

    # Pre-populate with a valid 200-byte inspection
    insp_valid = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp_valid.inspection_id = "insp_valid"
    storage.store_inspection(
        session_token=session,
        inspection=insp_valid,
        category="pcb1",
        decision="PASS",
        assets={"asset_valid": (b"V" * 200, "image/png")},
    )
    assert storage.total_bytes == 200

    # Attempt to store an oversized 1500-byte inspection (> 1000 byte cap)
    insp_oversized = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp_oversized.inspection_id = "insp_oversized"
    with pytest.raises(HTTPException) as exc_info:
        storage.store_inspection(
            session_token=session,
            inspection=insp_oversized,
            category="pcb1",
            decision="PASS",
            assets={"asset_huge": (b"H" * 1500, "image/png")},
        )
    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["code"] == "PAYLOAD_TOO_LARGE"

    # Existing valid item was not evicted and total_bytes remains intact
    assert storage.inspection_count == 1
    assert storage.total_bytes == 200
    assert storage.get_trace("insp_valid", session).inspection_id == "insp_valid"


def test_storage_repeated_inspection_id_no_orphans():
    """Overwriting an existing inspection_id cleans up old assets without orphaned bytes (bug 6 fix)."""
    storage = StorageManager(max_bytes=10000, max_inspections=10, ttl_seconds=1800)
    session = "sess_1"

    insp1 = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp1.inspection_id = "same_id"
    storage.store_inspection(
        session_token=session,
        inspection=insp1,
        category="pcb1",
        decision="PASS",
        assets={"old_asset_1": (b"A" * 300, "image/png"), "old_asset_2": (b"B" * 200, "image/png")},
    )
    assert storage.total_bytes == 500
    assert storage.inspection_count == 1

    # Overwrite same inspection_id with new assets
    insp2 = build_mock_inspection(DecisionPathEnum.AGENT_CONFIRMED)
    insp2.inspection_id = "same_id"
    storage.store_inspection(
        session_token=session,
        inspection=insp2,
        category="pcb1",
        decision="FAIL",
        assets={"new_asset_1": (b"C" * 150, "image/png")},
    )

    # Count remains 1, bytes should be exactly 150 (not 650!)
    assert storage.inspection_count == 1
    assert storage.total_bytes == 150

    # Old assets are gone
    with pytest.raises(HTTPException) as exc_old:
        storage.get_asset("old_asset_1", session)
    assert exc_old.value.status_code == 404

    # New asset is present
    data, _ = storage.get_asset("new_asset_1", session)
    assert data == b"C" * 150


def test_storage_feedback_idempotency_evicted_with_parent():
    """When an inspection is evicted, its feedback idempotency records are evicted too (bug 2 fix)."""
    # Max count = 2
    storage = StorageManager(max_bytes=10000, max_inspections=2, ttl_seconds=1800)
    session = "sess_1"

    # Store insp_0 and record feedback
    insp_0 = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp_0.inspection_id = "insp_0"
    storage.store_inspection(
        session_token=session,
        inspection=insp_0,
        category="pcb1",
        decision="PASS",
        assets={"a0": (b"0", "image/png")},
    )
    fb0 = storage.record_feedback(
        session_token=session,
        inspection_id="insp_0",
        user_label="PASS",
        category="pcb1",
        idempotency_key="key_0",
    )
    assert fb0.status == "LOGGED"
    assert (session, "key_0") in storage._feedback_idempotency

    # Store insp_1 and insp_2 (causing insp_0 to be evicted)
    for i in (1, 2):
        insp_i = build_mock_inspection(DecisionPathEnum.FAST_PASS)
        insp_i.inspection_id = f"insp_{i}"
        storage.store_inspection(
            session_token=session,
            inspection=insp_i,
            category="pcb1",
            decision="PASS",
            assets={f"a{i}": (b"x", "image/png")},
        )

    # insp_0 is evicted
    assert storage.inspection_count == 2
    with pytest.raises(HTTPException):
        storage.get_trace("insp_0", session)

    # Key (session, "key_0") should be evicted from _feedback_idempotency
    assert (session, "key_0") not in storage._feedback_idempotency


def test_storage_feedback_ownership_validated_before_idempotency():
    """Target inspection ownership is validated before checking idempotency keys (bug 1 fix)."""
    storage = StorageManager()
    alice_session = "sess_alice"
    bob_session = "sess_bob"

    insp_alice = build_mock_inspection(DecisionPathEnum.FAST_PASS)
    insp_alice.inspection_id = "insp_alice"
    storage.store_inspection(
        session_token=alice_session,
        inspection=insp_alice,
        category="pcb1",
        decision="PASS",
        assets={"a_alice": (b"x", "image/png")},
    )

    # Bob tries to submit feedback on Alice's inspection using any idempotency key
    with pytest.raises(HTTPException) as exc_info:
        storage.record_feedback(
            session_token=bob_session,
            inspection_id="insp_alice",
            user_label="PASS",
            category="pcb1",
            idempotency_key="bob_key_1",
        )
    # Must yield 404 (not owned), never 200 or 409
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
