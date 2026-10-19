"""
backend/executor.py
Unified, pluggable inspection executor for SureInspect.

Supports two execution modes:
1. "mock": Deterministic mock engine generating realistic synthetic traces,
   calibrations, and asset placeholders matching the SureInspect v8.1 contract.
2. "live": Real OpenCV 5 DNN + Agentic inspection pipeline.

Mode selection:
- Configured via environment variable SUREINSPECT_ENGINE ("auto", "mock", "live").
- Defaults to "auto": uses "live" if model artifacts and live runner are available,
  otherwise cleanly falls back to "mock".
- If explicitly set to "live" and artifacts/code are unavailable, raises a 503 HTTP exception.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, status

from backend.mock_service import (
    TINY_PNG_BYTES,
    generate_mock_inspection,
)
from backend.schemas import InspectionResponse


class InspectionExecutor:
    """Pluggable inspection executor orchestrating mock and live inspection pipelines."""

    def __init__(
        self,
        mode: str | None = None,
        artifacts_dir: Path | str | None = None,
        live_runner: Callable[..., Any] | None = None,
        model_version: str = "wrn50_v1_frozen",
    ) -> None:
        self._explicit_mode = mode
        self.artifacts_dir = Path(artifacts_dir or "artifacts")
        self.live_runner = live_runner
        self.model_version = model_version
        try:
            from backend.live_runner import find_snapshot_dir
            snap = find_snapshot_dir(self.artifacts_dir)
            if snap is not None:
                self.model_version = snap.name
        except Exception:
            pass

    @property
    def configured_mode(self) -> str:
        if self._explicit_mode is not None:
            return self._explicit_mode.lower()
        return os.environ.get("SUREINSPECT_ENGINE", "auto").lower()

    def is_live_ready(self) -> bool:
        """
        Returns True if live inference prerequisites are satisfied:
        - A live runner callable has been explicitly registered, OR
        - Model artifacts (snapshots or backbone.onnx) exist on disk AND
          a live pipeline in backend.live_runner / core.perception can be resolved.
        """
        if self.live_runner is not None:
            return True

        from backend.live_runner import find_snapshot_dir
        has_artifacts = (
            find_snapshot_dir(self.artifacts_dir) is not None
            or (self.artifacts_dir / "backbone.onnx").is_file()
            or (self.artifacts_dir / "memory_bank.npy").is_file()
        )
        if not has_artifacts:
            return False

        try:
            import importlib.util
            perception_spec = importlib.util.find_spec("core.perception")
            live_runner_spec = importlib.util.find_spec("backend.live_runner")
            return perception_spec is not None or live_runner_spec is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def get_engine_mode(self) -> str:
        """Resolves active engine mode: 'mock' or 'live'."""
        cfg = self.configured_mode
        if cfg == "mock":
            return "mock"
        if cfg == "live":
            return "live"
        # "auto"
        return "live" if self.is_live_ready() else "mock"

    def register_live_runner(self, runner: Callable[..., Any] | None) -> None:
        """Programmatic hook for registering the live agent pipeline."""
        self.live_runner = runner

    def inspect(
        self,
        category: str = "pcb1",
        sample_id: str | None = None,
        image_bytes: bytes | None = None,
        decoded_img: np.ndarray | None = None,
        simulate_blur: bool = False,
        simulate_darken: bool = False,
        simulate_glare: bool = False,
        alpha_fail: float = 0.05,
        path: str | None = None,
        orig_dimensions: tuple[int, int] | None = None,
    ) -> tuple[InspectionResponse, dict[str, tuple[bytes, str]]]:
        """
        Executes inspection using the active engine.
        Returns:
            (InspectionResponse, assets_to_store)
        """
        active_mode = self.get_engine_mode()

        if active_mode == "live":
            if not self.is_live_ready():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "Live inference engine is not ready: model weights or perception pipeline not found.",
                    },
                )
            return self._execute_live(
                category=category,
                sample_id=sample_id,
                image_bytes=image_bytes,
                decoded_img=decoded_img,
                simulate_blur=simulate_blur,
                simulate_darken=simulate_darken,
                simulate_glare=simulate_glare,
                alpha_fail=alpha_fail,
                path=path,
                orig_dimensions=orig_dimensions,
            )

        # Mock Mode
        return self._execute_mock(
            category=category,
            sample_id=sample_id,
            simulate_blur=simulate_blur,
            simulate_darken=simulate_darken,
            simulate_glare=simulate_glare,
            alpha_fail=alpha_fail,
            path=path,
        )

    def _execute_mock(
        self,
        category: str,
        sample_id: str | None,
        simulate_blur: bool,
        simulate_darken: bool,
        simulate_glare: bool,
        alpha_fail: float,
        path: str | None,
    ) -> tuple[InspectionResponse, dict[str, tuple[bytes, str]]]:
        inspection = generate_mock_inspection(
            category=category,
            simulate_blur=simulate_blur,
            simulate_darken=simulate_darken,
            simulate_glare=simulate_glare,
            sample_id=sample_id,
            alpha_fail=alpha_fail,
            path=path,
        )

        assets_to_store: dict[str, tuple[bytes, str]] = {}
        for asset_url in [
            inspection.assets.input_standardized,
            inspection.assets.reference_aligned,
            inspection.assets.anomaly_heatmap,
            inspection.assets.contour_mask,
        ]:
            if asset_url:
                asset_key = asset_url.split("/api/assets/", 1)[-1]
                if "mask" in asset_key or "input" in asset_key:
                    assets_to_store[asset_key] = (TINY_PNG_BYTES, "image/png")
                else:
                    assets_to_store[asset_key] = (TINY_PNG_BYTES, "image/jpeg")

        for step in inspection.execution_trace:
            if step.overlay_url:
                overlay_key = step.overlay_url.split("/api/assets/", 1)[-1]
                if overlay_key not in assets_to_store:
                    assets_to_store[overlay_key] = (TINY_PNG_BYTES, "image/jpeg")

        return inspection, assets_to_store

    def _execute_live(
        self,
        category: str,
        sample_id: str | None,
        image_bytes: bytes | None,
        decoded_img: np.ndarray | None,
        simulate_blur: bool,
        simulate_darken: bool,
        simulate_glare: bool,
        alpha_fail: float,
        path: str | None,
        orig_dimensions: tuple[int, int] | None,
    ) -> tuple[InspectionResponse, dict[str, tuple[bytes, str]]]:
        if self.live_runner is not None:
            return self.live_runner(
                category=category,
                sample_id=sample_id,
                image_bytes=image_bytes,
                decoded_img=decoded_img,
                simulate_blur=simulate_blur,
                simulate_darken=simulate_darken,
                simulate_glare=simulate_glare,
                alpha_fail=alpha_fail,
                path=path,
                orig_dimensions=orig_dimensions,
            )

        # Execute through backend.live_runner
        try:
            from backend.live_runner import get_live_engine
            engine = get_live_engine(self.artifacts_dir)
            if engine is not None:
                return engine.inspect(
                    category=category,
                    sample_id=sample_id,
                    image_bytes=image_bytes,
                    decoded_img=decoded_img,
                    simulate_blur=simulate_blur,
                    simulate_darken=simulate_darken,
                    simulate_glare=simulate_glare,
                    alpha_fail=alpha_fail,
                    path=path,
                    orig_dimensions=orig_dimensions,
                )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"Failed executing live pipeline: {exc}",
                },
            ) from exc

        # Fallback to importing core.agent if available
        try:
            import importlib
            agent_mod = importlib.import_module("core.agent")
            if hasattr(agent_mod, "run_inspection"):
                return agent_mod.run_inspection(
                    category=category,
                    sample_id=sample_id,
                    image_bytes=image_bytes,
                    decoded_img=decoded_img,
                    simulate_blur=simulate_blur,
                    simulate_darken=simulate_darken,
                    simulate_glare=simulate_glare,
                    alpha_fail=alpha_fail,
                    orig_dimensions=orig_dimensions,
                )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "Live inspection pipeline is not yet implemented.",
            },
        )


# Global singleton executor
EXECUTOR = InspectionExecutor()
