"""
backend/admission.py
Streaming upload byte limiter, magic-byte media type validator, OpenCV decompression
bomb guard (OPENCV_IO_MAX_IMAGE_PIXELS=25000000), and bounded async admission queue.

Guarantees:
- Compressed upload bytes strictly bounded to 10 MiB while streaming (413 runs before magic-byte 415 check).
- Magic signatures checked before decoding: PNG (b"\\x89PNG\\r\\n\\x1a\\n"), JPEG (b"\\xff\\xd8\\xff").
- Decompression bomb guard enforces 25 MP ceiling via header pre-check (PNG IHDR / JPEG SOF);
  header dimension violations yield HTTP 413 PIXEL_LIMIT_EXCEEDED.
- Malformed decodes and unhandled OpenCV errors yield HTTP 415 MALFORMED_IMAGE.
- Admission controller admits at most 1 running inference job and 4 queued waiting jobs.
  Excess requests immediately return HTTP 503 QUEUE_SATURATED without entering the queue.
  Cancelled waits decrement queued count cleanly and never increment running count.
"""
from __future__ import annotations

import asyncio
import io
import os

# Must explicitly set OPENCV_IO_MAX_IMAGE_PIXELS in os.environ before importing cv2.
# OpenCV checks this environment variable only once during C++ extension initialization;
# setting it after cv2 is loaded has no effect on internal decompression bounds.
OPENCV_MAX_PIXELS = 25_000_000
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_MAX_PIXELS)

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

# 10 MiB hard limit on compressed upload size
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10,485,760 bytes
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KiB chunks for streaming validation

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
        Only transitions from queued to running if the semaphore wait succeeds.
        If cancelled or failed during wait, decrements queued count and leaves running unchanged.
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

        acquired = False
        try:
            await self._worker_semaphore.acquire()
            acquired = True
        finally:
            async with self._lock:
                self._queued_count -= 1
                if acquired:
                    self._running_count += 1

    async def release(self) -> None:
        """
        Releases the execution slot for a task that successfully acquired it.
        Decrements running count and releases the worker semaphore.
        """
        async with self._lock:
            if self._running_count > 0:
                self._running_count -= 1
                self._worker_semaphore.release()


# Global singleton admission controller
INSPECTION_ADMISSION = AdmissionController(max_running=1, max_queued=4)


async def read_bounded_upload(file: UploadFile) -> tuple[bytes, str]:
    """
    Streams file bytes in 64 KiB chunks, enforcing size and format bounds.

    Order of verification:
    1. Size limit check (HTTP 413): runs on every chunk, aborting immediately if total
       accumulated bytes exceed 10 MiB, strictly before any magic-byte validation.
    2. Magic-byte signature check (HTTP 415): verifies PNG or JPEG magic bytes once at
       least 8 bytes are read.
    3. Empty file check (HTTP 400): verifies file contains at least 1 byte.

    Returns:
        (raw_bytes, media_type)
    Raises:
        HTTP 413: If total stream size exceeds 10 MiB.
        HTTP 415: If magic bytes do not match PNG or JPEG.
        HTTP 400: If uploaded file is empty (0 bytes).
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
        # Size limit check (413) runs before magic-byte check (415)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": f"Upload exceeds maximum compressed size of {MAX_UPLOAD_BYTES} bytes (10 MiB).",
                },
            )
        buffer.write(chunk)

        # Check magic bytes on first chunk once size is checked
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


def parse_image_dimensions(raw_bytes: bytes) -> tuple[int, int] | None:
    """
    Parses (width, height) directly from PNG (IHDR) or JPEG (SOF markers) headers.
    Returns None if headers are truncated, malformed, or marker is not found.
    """
    # 1. PNG Header Pre-check: IHDR chunk (bytes 16..24)
    if len(raw_bytes) >= 24 and raw_bytes.startswith(PNG_MAGIC) and raw_bytes[12:16] == b"IHDR":
        width = int.from_bytes(raw_bytes[16:20], byteorder="big")
        height = int.from_bytes(raw_bytes[20:24], byteorder="big")
        return width, height

    # 2. JPEG Header Pre-check: scan SOF markers (SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15)
    if len(raw_bytes) >= 4 and raw_bytes.startswith(JPEG_MAGIC):
        offset = 2
        n = len(raw_bytes)
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }
        while offset < n - 1:
            if raw_bytes[offset] != 0xFF:
                offset += 1
                continue
            # Skip fill 0xFF bytes
            while offset < n - 1 and raw_bytes[offset] == 0xFF:
                offset += 1
            if offset >= n:
                break
            marker = raw_bytes[offset]
            offset += 1

            # Standalone markers with no length payload
            if marker in (0xD8, 0xD9, 0x01) or (0xD0 <= marker <= 0xD7):
                continue
            if marker == 0x00:
                continue

            if offset + 2 > n:
                break
            segment_len = int.from_bytes(raw_bytes[offset:offset + 2], byteorder="big")
            if segment_len < 2:
                break

            if marker in sof_markers:
                # Format: [length (2)][precision (1)][height (2)][width (2)]
                if offset + 2 + 5 <= n:
                    height = int.from_bytes(raw_bytes[offset + 3:offset + 5], byteorder="big")
                    width = int.from_bytes(raw_bytes[offset + 5:offset + 7], byteorder="big")
                    return width, height
                break

            offset += segment_len

    return None


def decode_and_validate_image(raw_bytes: bytes) -> np.ndarray:
    """
    Decodes image bytes with OpenCV, strictly enforcing pixel limits and valid 3-channel geometry.

    Guarantees:
    - Pre-checks header dimensions for PNG (IHDR) and JPEG (SOF markers); if width * height > 25 MP,
      raises HTTP 413 PIXEL_LIMIT_EXCEEDED before large buffer allocation.
    - Malformed or truncated headers fall through to cv2.imdecode.
    - Any cv2.error or decode failure raises HTTP 415 MALFORMED_IMAGE.
    - Non-3-channel images raise HTTP 415 INVALID_CHANNELS.
    """
    # Header pre-check for PNG / JPEG dimensions
    header_dims = parse_image_dimensions(raw_bytes)
    if header_dims is not None:
        width, height = header_dims
        if width * height > OPENCV_MAX_PIXELS:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "PIXEL_LIMIT_EXCEEDED",
                    "message": f"Image dimensions ({width}x{height} = {width * height} pixels) exceed 25 MP ceiling.",
                },
            )

    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    except cv2.error as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "MALFORMED_IMAGE", "message": f"OpenCV failed to decode image: {e}"},
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "DECODE_ERROR", "message": f"Unexpected image decode error: {e}"},
        ) from e

    if img is None:
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
            status_code=413,
            detail={
                "code": "PIXEL_LIMIT_EXCEEDED",
                "message": f"Image resolution {w}x{h} ({w * h} pixels) exceeds the 25 MP limit.",
            },
        )

    return img
