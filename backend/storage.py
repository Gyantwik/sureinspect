"""
backend/storage.py
Bounded storage and session cache managing:
- Standardized inputs and contour masks as lossless PNG.
- Photographic overlays, reference previews, and anomaly heatmaps as display-quality JPEG (quality=85).
- Bounded LRU/TTL cache: 200 MiB total asset bytes, 100 max stored inspections, 30-minute idle TTL.
- Coherent eviction: inspection metadata, associated asset payloads, and idempotency records are evicted together.
- Audit feedback tracking with idempotency safety, RETAKE rejection, and session ownership enforcement.
"""
from __future__ import annotations

import collections
import threading
import time
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, status

from backend.schemas import (
    FeedbackResponse,
    InspectionResponse,
)

# Cache limits
MAX_CACHE_BYTES = 200 * 1024 * 1024   # 200 MiB
MAX_INSPECTIONS = 100                 # 100 inspections
CACHE_TTL_SECONDS = 1800              # 30 minutes
JPEG_DISPLAY_QUALITY = 85             # Display JPEG quality


class StorageManager:
    """
    Thread-safe storage manager holding bounded assets and inspection traces in memory.
    """
    def __init__(
        self,
        max_bytes: int = MAX_CACHE_BYTES,
        max_inspections: int = MAX_INSPECTIONS,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        self.max_bytes = max_bytes
        self.max_inspections = max_inspections
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

        # inspection_id -> dict with keys:
        # {
        #   "session_token": str,
        #   "response": InspectionResponse,
        #   "category": str,
        #   "decision": str,
        #   "created_at": float,
        #   "expires_at": float,
        #   "asset_ids": list[str],
        #   "orig_dimensions": tuple[int, int],
        #   "feedback_keys": list[tuple[str, str]]
        # }
        self._inspections: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()

        # asset_id -> dict with keys:
        # {
        #   "session_token": str,
        #   "data": bytes,
        #   "media_type": str,
        #   "inspection_id": str,
        #   "created_at": float,
        #   "expires_at": float,
        #   "size_bytes": int
        # }
        self._assets: dict[str, dict[str, Any]] = {}
        self._total_asset_bytes: int = 0

        # (session_token, idempotency_key) -> feedback record
        self._feedback_idempotency: dict[tuple[str, str], dict[str, Any]] = {}

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_asset_bytes

    @property
    def inspection_count(self) -> int:
        with self._lock:
            return len(self._inspections)

    def _purge_expired_locked(self, now: float) -> None:
        """Removes expired inspections and their orphaned assets and feedback records."""
        expired_ids = [
            iid for iid, data in self._inspections.items()
            if now > data["expires_at"]
        ]
        for iid in expired_ids:
            self._evict_inspection_locked(iid)

    def _evict_inspection_locked(self, inspection_id: str) -> None:
        """Evicts an inspection, its associated asset bytes, and its feedback idempotency entries."""
        record = self._inspections.pop(inspection_id, None)
        if not record:
            return
        # Evict all assets and deduct byte size
        for asset_id in record.get("asset_ids", []):
            asset_record = self._assets.pop(asset_id, None)
            if asset_record:
                self._total_asset_bytes = max(0, self._total_asset_bytes - asset_record["size_bytes"])
        # Evict all feedback idempotency keys bound to this inspection
        for cache_key in record.get("feedback_keys", []):
            self._feedback_idempotency.pop(cache_key, None)

    def _ensure_capacity_locked(self, incoming_bytes: int) -> None:
        """Evicts oldest inspections until both count limit and byte limit are satisfied."""
        while len(self._inspections) >= self.max_inspections and self._inspections:
            oldest_id = next(iter(self._inspections))
            self._evict_inspection_locked(oldest_id)

        while (self._total_asset_bytes + incoming_bytes > self.max_bytes) and self._inspections:
            oldest_id = next(iter(self._inspections))
            self._evict_inspection_locked(oldest_id)

    def store_inspection(
        self,
        session_token: str,
        inspection: InspectionResponse,
        category: str,
        decision: str,
        assets: dict[str, tuple[bytes, str]],
        orig_dimensions: tuple[int, int] | None = None,
    ) -> None:
        """
        Stores an inspection and its associated assets bound to a session.
        assets is a dict mapping asset_id -> (bytes_content, media_type).
        """
        incoming_bytes = sum(len(content) for content, _ in assets.values())
        if incoming_bytes > self.max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "PAYLOAD_TOO_LARGE",
                    "message": (
                        f"Incoming inspection assets ({incoming_bytes} bytes) exceed "
                        f"maximum cache capacity of {self.max_bytes} bytes."
                    ),
                },
            )

        now = time.time()
        expires_at = now + self.ttl_seconds

        with self._lock:
            # Overwrite protection: if inspection_id exists, evict old record and assets first
            if inspection.inspection_id in self._inspections:
                self._evict_inspection_locked(inspection.inspection_id)

            self._purge_expired_locked(now)
            self._ensure_capacity_locked(incoming_bytes)

            asset_ids = []
            for asset_id, (content, media_type) in assets.items():
                size = len(content)
                self._assets[asset_id] = {
                    "session_token": session_token,
                    "data": content,
                    "media_type": media_type,
                    "inspection_id": inspection.inspection_id,
                    "created_at": now,
                    "expires_at": expires_at,
                    "size_bytes": size,
                }
                self._total_asset_bytes += size
                asset_ids.append(asset_id)

            self._inspections[inspection.inspection_id] = {
                "session_token": session_token,
                "response": inspection,
                "category": category,
                "decision": decision,
                "created_at": now,
                "expires_at": expires_at,
                "asset_ids": asset_ids,
                "orig_dimensions": orig_dimensions or (0, 0),
                "feedback_keys": [],
            }

    def get_trace(self, inspection_id: str, session_token: str) -> InspectionResponse:
        """
        Retrieves stored inspection trace for owning session.
        Returns 404 for unknown, non-owned, or expired inspection.
        """
        now = time.time()
        with self._lock:
            self._purge_expired_locked(now)
            record = self._inspections.get(inspection_id)
            if not record or now > record["expires_at"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Inspection not found"},
                )
            if record["session_token"] != session_token:
                # Security rule: Non-owned returns identical 404
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Inspection not found"},
                )
            # Move to end for LRU freshness
            self._inspections.move_to_end(inspection_id)
            return record["response"]

    def get_asset(self, asset_id: str, session_token: str) -> tuple[bytes, str]:
        """
        Retrieves stored asset bytes for owning session.
        Enforces TTL expiration and returns (data_bytes, media_type) or raises 404.
        """
        now = time.time()
        with self._lock:
            self._purge_expired_locked(now)
            record = self._assets.get(asset_id)
            if not record or now > record.get("expires_at", 0):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Asset not found"},
                )
            if record["session_token"] != session_token:
                # Security rule: Non-owned returns identical 404
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Asset not found"},
                )
            return record["data"], record["media_type"]

    def record_feedback(
        self,
        session_token: str,
        inspection_id: str,
        user_label: str,
        category: str,
        idempotency_key: str,
    ) -> FeedbackResponse:
        """
        Records human feedback for an inspection.
        Enforces:
        - Target inspection ownership and existence checked FIRST.
        - Safe repeated idempotency keys.
        - HTTP 409 for conflicting label under same idempotency key.
        - HTTP 400 rejection if user_label is applied on RETAKE or expired inspections.
        - Audit-only scope with production_changed=False.
        """
        now = time.time()
        with self._lock:
            self._purge_expired_locked(now)

            # 1. Validate inspection existence and ownership FIRST
            record = self._inspections.get(inspection_id)
            if not record or now > record["expires_at"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Target inspection record not found or expired"},
                )

            if record["session_token"] != session_token:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "NOT_FOUND", "message": "Target inspection record not found"},
                )

            # 2. Check idempotency for this session
            cache_key = (session_token, idempotency_key)
            if cache_key in self._feedback_idempotency:
                prior = self._feedback_idempotency[cache_key]
                if prior["inspection_id"] == inspection_id and prior["user_label"] == user_label:
                    return FeedbackResponse(
                        status="LOGGED",
                        message=None,
                        scope="audit_only",
                        production_changed=False,
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "CONFLICT",
                            "message": f"Idempotency key '{idempotency_key}' reused with conflicting payload",
                        },
                    )

            # 3. Rule: RETAKE inspections cannot receive normal/defect feedback
            if record["decision"] == "RETAKE":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "BAD_REQUEST",
                        "message": "Cannot apply PASS/FAIL human label to a RETAKE (quality rejected) inspection.",
                    },
                )

            # 4. Generate collision-resistant unique feedback ID
            feedback_id = f"fb_{uuid.uuid4().hex}"
            self._feedback_idempotency[cache_key] = {
                "feedback_id": feedback_id,
                "inspection_id": inspection_id,
                "user_label": user_label,
                "category": category,
                "created_at": now,
            }
            record.setdefault("feedback_keys", []).append(cache_key)

            return FeedbackResponse(
                status="LOGGED",
                message=None,
                scope="audit_only",
                production_changed=False,
            )


# Global storage manager singleton
STORAGE = StorageManager()


def encode_lossless_png(image: np.ndarray) -> bytes:
    """Encodes standardized pixels or masks as lossless PNG."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to lossless PNG")
    return buf.tobytes()


def encode_preview_jpeg(image: np.ndarray, quality: int = JPEG_DISPLAY_QUALITY) -> bytes:
    """Encodes photographic overlays and heatmaps as display-quality JPEG."""
    success, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image to preview JPEG")
    return buf.tobytes()
