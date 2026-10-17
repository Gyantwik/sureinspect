"""
backend/admission.py
Streaming upload byte limiter, magic-byte media type validator, OpenCV decompression
bomb guard (OPENCV_IO_MAX_IMAGE_PIXELS=25000000), and bounded async admission queue.

Guarantees:
- Compressed upload bytes strictly bounded to 10 MiB while streaming.
- Magic signatures checked before any decoding: PNG (b"\\x89PNG\\r\\n\\x1a\\n"), JPEG (b"\\xff\\xd8\\xff").
- Decompression bomb guard enforces 25 MP ceiling; header/pixel limit violations yield HTTP 413.
- Malformed decodes yield HTTP 415.
- Admission controller admits at most 1 running inference job and 4 queued waiting jobs.
  Any excess request is immediately rejected with HTTP 503 without entering the queue.
"""
from __future__ import annotations

import asyncio
import os
import io
from typing import Tuple, Optional
import cv2
import numpy as np
from fastapi import UploadFile, HTTPException, status

# 10 MiB hard limit on compressed upload size
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10,485,760 bytes
STREAM_CHUNK_SIZE = 64 * 1024        # 64 KiB chunks for streaming validation

# OpenCV max pixels environment variable must be set
OPENCV_MAX_PIXELS = 25_000_000
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_MAX_PIXELS)

# Magic bytes
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


class AdmissionController:
    """
    Bounded async admission mechanism.
    Admits at most max_running (1) active inference/decode tasks,
    and at most max_queued (4) waiting tasks.
    Additional requests are rejected immediately with HTTP 503.
    """
    def __init__(self, max_running: int = 1, max_queued: int = 4):
        self.max_running = max_running
        self.max_queued = max_queued
        self._lock = asyncio.Lock()
        self._running_count = 0
        self._queued_count = 0
        self._worker_semaphore = asyncio.Semaphore(max_running)

    @property
    def running(self) -> int:
        return self._running_count

    @property
    def queued(self) -> int:
        return self._queued_count

    async def acquire(self) -> None:
        """
        Attempts to join the queue and then acquire the execution slot.
        Raises HTTP 503 if queue is saturated.
        """
        async with self._lock:
            if self._running_count + self._queued_count >= self.max_running + self.max_queued:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "code": "QUEUE_SATURATED",
                        "message": (
                            f"Inference queue saturated: {self._running_count} running, "
                            f"{self._queued_count} queued (limit: {self.max_running}+{self.max_queued}). "
                            "Please retry shortly."
                        ),
                    },
                )
            self._queued_count += 1

        # Wait for the worker slot
        try:
            await self._worker_semaphore.acquire()
        finally:
            async with self._lock:
                self._queued_count -= 1
                self._running_count += 1

    async def release(self) -> None:
        """Releases the execution slot."""
        async with self._lock:
            self._running_count = max(0, self._running_count - 1)
        self._worker_semaphore.release()


# Global singleton admission controller
INSPECTION_ADMISSION = AdmissionController(max_running=1, max_queued=4)


async def read_bounded_upload(file: UploadFile) -> Tuple[bytes, str]:
    """
    Streams file bytes in 64 KiB chunks, enforcing the 10 MiB limit during reading.
    Validates magic bytes for PNG or JPEG.
    Returns: (raw_bytes, media_type)
    Raises:
        HTTP 413: If total stream size exceeds 10 MiB.
        HTTP 415: If magic bytes do not match PNG or JPEG.
    """
    buffer = io.BytesIO()
    total_bytes = 0
    header_checked = False
    media_type = "application/octet-stream"

    while True:
        chunk = await file.read(STREAM_CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": f"Upload exceeds maximum compressed size of {MAX_UPLOAD_BYTES} bytes (10 MiB).",
                },
            )
        buffer.write(chunk)

        # Check magic bytes on first chunk
        if not header_checked and total_bytes >= 8:
            raw_head = buffer.getvalue()[:8]
            if raw_head.startswith(PNG_MAGIC):
                media_type = "image/png"
                header_checked = True
            elif raw_head.startswith(JPEG_MAGIC):
                media_type = "image/jpeg"
                header_checked = True
            else:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail={
                        "code": "UNSUPPORTED_MEDIA_TYPE",
                        "message": "Unsupported file format. Image must be a valid PNG or JPEG.",
                    },
                )

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Uploaded file is empty (0 bytes)."},
        )

    if not header_checked:
        raw_head = buffer.getvalue()[:8]
        if raw_head.startswith(PNG_MAGIC):
            media_type = "image/png"
        elif raw_head.startswith(JPEG_MAGIC):
            media_type = "image/jpeg"
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": "File is too small to contain a valid PNG or JPEG header.",
                },
            )

    return buffer.getvalue(), media_type


def decode_and_validate_image(raw_bytes: bytes) -> np.ndarray:
    """
    Decodes image bytes with OpenCV, strictly enforcing pixel limits and valid 3-channel geometry.
    Guarantees:
    - Pixel-limit violation or decompression bomb -> HTTP 413.
    - Malformed or corrupted bytes -> HTTP 415.
    - Non-3-channel or empty image -> HTTP 415.
    """
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    except cv2.error as e:
        err_msg = str(e).lower()
        if "pixel" in err_msg or "size" in err_msg or "dimension" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "PIXEL_LIMIT_EXCEEDED",
                    "message": f"Image dimensions violate the 25 MP pixel ceiling (OPENCV_IO_MAX_IMAGE_PIXELS={OPENCV_MAX_PIXELS}).",
                },
            )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "MALFORMED_IMAGE", "message": f"OpenCV failed to decode image: {e}"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "DECODE_ERROR", "message": f"Unexpected image decode error: {e}"},
        )

    if img is None:
        # Check if the PNG/JPEG header specified dimensions larger than 25 MP
        if len(raw_bytes) >= 24 and raw_bytes.startswith(PNG_MAGIC):
            # Read width & height from PNG IHDR chunk (bytes 16-24)
            w = int.from_bytes(raw_bytes[16:20], byteorder="big")
            h = int.from_bytes(raw_bytes[20:24], byteorder="big")
            if w * h > OPENCV_MAX_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "code": "PIXEL_LIMIT_EXCEEDED",
                        "message": f"PNG dimensions ({w}x{h} = {w*h} pixels) exceed 25 MP ceiling.",
                    },
                )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "DECODE_FAILED", "message": "Failed to decode image bytes into valid pixels."},
        )

    if img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "INVALID_CHANNELS",
                "message": f"Expected 3-channel RGB/BGR image, got shape {img.shape}.",
            },
        )

    h, w = img.shape[:2]
    if h * w > OPENCV_MAX_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "PIXEL_LIMIT_EXCEEDED",
                "message": f"Image resolution {w}x{h} ({w*h} pixels) exceeds the 25 MP limit.",
            },
        )

    return img
