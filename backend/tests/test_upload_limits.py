"""
backend/tests/test_upload_limits.py
Subprocess and direct tests for upload limits, magic byte validation, decompression bomb guard,
and bounded async queue saturation.
"""
import asyncio
import io
import os
import subprocess
import sys
import zlib
import pytest
from fastapi import UploadFile, HTTPException
from fastapi.testclient import TestClient

from backend.admission import (
    read_bounded_upload,
    decode_and_validate_image,
    AdmissionController,
    MAX_UPLOAD_BYTES,
    PNG_MAGIC,
    JPEG_MAGIC,
)
from backend.app_factory import create_app


def create_decompression_bomb_png(width: int = 20000, height: int = 20000) -> bytes:
    """
    Creates a tiny valid PNG header (~60 bytes) declaring extreme dimensions (e.g. 20000x20000 = 400 MP).
    """
    signature = PNG_MAGIC
    # IHDR chunk: length (13), type (IHDR), data (width, height, depth=8, color=2, comp=0, filt=0, inter=0)
    ihdr_data = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
    )
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data).to_bytes(4, "big")
    ihdr_chunk = len(ihdr_data).to_bytes(4, "big") + b"IHDR" + ihdr_data + ihdr_crc

    # Minimal dummy IDAT and IEND chunks
    idat_payload = zlib.compress(b"\x00" * 100)
    idat_crc = zlib.crc32(b"IDAT" + idat_payload).to_bytes(4, "big")
    idat_chunk = len(idat_payload).to_bytes(4, "big") + b"IDAT" + idat_payload + idat_crc

    iend_crc = zlib.crc32(b"IEND").to_bytes(4, "big")
    iend_chunk = (0).to_bytes(4, "big") + b"IEND" + iend_crc

    return signature + ihdr_chunk + idat_chunk + iend_chunk


def test_streaming_upload_exceeds_10mib():
    """Upload exceeding 10 MiB is aborted and raises HTTP 413."""
    async def _run():
        oversized = PNG_MAGIC + b"\x00" * (MAX_UPLOAD_BYTES + 1024)
        file = UploadFile(file=io.BytesIO(oversized), filename="large.png")
        with pytest.raises(HTTPException) as exc_info:
            await read_bounded_upload(file)
        assert exc_info.value.status_code == 413
        assert exc_info.value.detail["code"] == "PAYLOAD_TOO_LARGE"

    asyncio.run(_run())


def test_magic_byte_validation():
    """Non-PNG / non-JPEG files are rejected with HTTP 415."""
    async def _run():
        invalid_file = UploadFile(file=io.BytesIO(b"BM\x00\x00\x00\x00\x00\x00"), filename="test.bmp")
        with pytest.raises(HTTPException) as exc_info:
            await read_bounded_upload(invalid_file)
        assert exc_info.value.status_code == 415
        assert exc_info.value.detail["code"] == "UNSUPPORTED_MEDIA_TYPE"

    asyncio.run(_run())


def test_decompression_bomb_20000x20000():
    """Tiny PNG declaring 20000x20000 fails before large memory allocation."""
    bomb_bytes = create_decompression_bomb_png(20000, 20000)
    assert len(bomb_bytes) < 200  # Extremely small file size

    with pytest.raises(HTTPException) as exc_info:
        decode_and_validate_image(bomb_bytes)
    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["code"] == "PIXEL_LIMIT_EXCEEDED"


def test_queue_saturation_503():
    """
    Admission controller admits at most 1 running + 4 queued (5 total).
    The 6th request raises HTTP 503 SERVICE_UNAVAILABLE immediately.
    """
    async def _run():
        admission = AdmissionController(max_running=1, max_queued=4)
        # Acquire 1 running slot
        await admission.acquire()

        # Queue 4 waiting tasks
        queue_tasks = [asyncio.create_task(admission.acquire()) for _ in range(4)]
        # Give event loop a moment to enqueue them
        await asyncio.sleep(0.01)

        assert admission.running == 1
        assert admission.queued == 4

        # 6th task must be rejected with 503
        with pytest.raises(HTTPException) as exc_info:
            await admission.acquire()
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["code"] == "QUEUE_SATURATED"

        # Cleanup
        await admission.release()
        for task in queue_tasks:
            await task
            await admission.release()

    asyncio.run(_run())


def test_subprocess_pixel_limit_guard():
    """
    Required subprocess test:
    Launches a fresh Python subprocess with OPENCV_IO_MAX_IMAGE_PIXELS=25000000
    set in the environment BEFORE importing OpenCV, testing that the bomb is rejected.
    """
    sub_code = """
import os, sys
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = "25000000"
from backend.tests.test_upload_limits import create_decompression_bomb_png
from backend.admission import decode_and_validate_image
from fastapi import HTTPException

bomb = create_decompression_bomb_png(20000, 20000)
try:
    decode_and_validate_image(bomb)
    print("FAILED_NO_EXCEPTION")
    sys.exit(1)
except HTTPException as e:
    if e.status_code == 413:
        print("PASS_413_REJECTED")
        sys.exit(0)
    else:
        print(f"FAILED_WRONG_CODE_{e.status_code}")
        sys.exit(2)
"""
    env = os.environ.copy()
    env["OPENCV_IO_MAX_IMAGE_PIXELS"] = "25000000"
    res = subprocess.run(
        [sys.executable, "-c", sub_code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Subprocess failed: {res.stdout} {res.stderr}"
    assert "PASS_413_REJECTED" in res.stdout
