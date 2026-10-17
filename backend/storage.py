"""
backend/storage.py
Bounded storage and session cache managing:
- Standardized inputs and contour masks as lossless PNG.
- Photographic overlays, reference previews, and anomaly heatmaps as display-quality JPEG (quality=85).
- Bounded LRU/TTL cache: 200 MiB total asset bytes, 100 max stored inspections, 30-minute idle TTL.
- Coherent eviction: inspection metadata and all associated asset payloads are evicted together.
- Audit feedback tracking with idempotency safety, RETAKE rejection, and session ownership enforcement.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Dict, Optional, Tuple, Any, List
import cv2
import numpy as np
from fastapi import HTTPException, status

from backend.schemas import (
    InspectionResponse,
    FeedbackResponse,
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
        #   "asset_ids": List[str],
        #   "orig_dimensions": Tuple[int, int]
        # }
        self._inspections: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()

        # asset_id -> dict with keys:
        # {
        #   "session_token": str,
        #   "data": bytes,
        #   "media_type": str,
        #   "inspection_id": str,
        #   "created_at": float,
        #   "size_bytes": int
        # }
        self._assets: Dict[str, Dict[str, Any]] = {}
        self._total_asset_bytes: int = 0

        # (session_token, idempotency_key) -> FeedbackResponse
        self._feedback_idempotency: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_asset_bytes

    @property
    def inspection_count(self) -> int:
        with self._lock:
            return len(self._inspections)

    def _purge_expired_locked(self, now: float) -> None:
        """Removes expired inspections and their orphaned assets."""
        expired_ids = [
            iid for iid, data in self._inspections.items()
            if now > data["expires_at"]
        ]
        for iid in expired_ids:
            self._evict_inspection_locked(iid)

    def _evict_inspection_locked(self, inspection_id: str) -> None:
        """Evicts an inspection and all of its associated asset bytes."""
        record = self._inspections.pop(inspection_id, None)
        if not record:
            return
        for asset_id in record.get("asset_ids", []):
            asset_record = self._assets.pop(asset_id, None)
            if asset_record:
                self._total_asset_bytes = max(0, self._total_asset_bytes - asset_record["size_bytes"])

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
        assets: Dict[str, Tuple[bytes, str]],
        orig_dimensions: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Stores an inspection and its associated assets bound to a session.
        assets is a dict mapping asset_id -> (bytes_content, media_type).
        """
        now = time.time()
        incoming_bytes = sum(len(content) for content, _ in assets.values())

        with self._lock:
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
                "expires_at": now + self.ttl_seconds,
                "asset_ids": asset_ids,
                "orig_dimensions": orig_dimensions or (0, 0),
            }

    def get_trace(self, inspection_id: str, session_token: str) -> InspectionResponse:
        """
        Retrieves stored inspection trace for owning session.
        Returns 404 for unknown or non-owned inspection.
        """
        now = time.time()
        with self._lock:
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

    def get_asset(self, asset_id: str, session_token: str) -> Tuple[bytes, str]:
        """
        Retrieves stored asset bytes for owning session.
        Returns (data_bytes, media_type) or raises 404.
        """
        with self._lock:
            record = self._assets.get(asset_id)
            if not record:
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
        - Safe repeated idempotency keys.
        - HTTP 409 for conflicting label under same idempotency key.
        - HTTP 400 rejection if user_label is applied on RETAKE or expired inspections.
        - Audit-only scope with production_changed=False.
        """
        now = time.time()
        with self._lock:
            # Check idempotency
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

            # Check inspection existence and ownership
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

            # Rule: RETAKE inspections cannot receive normal/defect feedback
            if record["decision"] == "RETAKE":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "BAD_REQUEST",
                        "message": "Cannot apply PASS/FAIL human label to a RETAKE (quality rejected) inspection.",
                    },
                )

            feedback_id = f"fb_{int(now)}_{idempotency_key[:8]}"
            self._feedback_idempotency[cache_key] = {
                "feedback_id": feedback_id,
                "inspection_id": inspection_id,
                "user_label": user_label,
                "category": category,
                "created_at": now,
            }

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
