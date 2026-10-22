"""Smoke-test the runtime strictly from an extracted submission directory."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from backend.live_runner import LiveEngine, find_snapshot_dir


def main() -> int:
    root = Path.cwd()
    snapshot = find_snapshot_dir(root / "artifacts")
    if snapshot is None:
        raise SystemExit("No runnable snapshot found in extracted archive")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("readiness") != "agent_ready":
        raise SystemExit("Extracted runtime did not select an agent-ready snapshot")
    engine = LiveEngine(snapshot)
    normal, _ = engine.inspect(sample_id="pcb1_good_01")
    retake, _ = engine.inspect(sample_id="pcb1_good_01", simulate_blur=True)
    if normal.decision.value not in {"PASS", "FAIL", "HUMAN", "RETAKE"}:
        raise SystemExit("Invalid normal-sample decision")
    if retake.decision.value != "RETAKE":
        raise SystemExit(f"Blur quality gate did not RETAKE: {retake.decision.value}")
    if not cv2.__version__.startswith("5."):
        raise SystemExit(f"Expected OpenCV 5, found {cv2.__version__}")
    print(f"PACKAGED RUNTIME PASSED: {manifest['snapshot_id']}; OpenCV {cv2.__version__}; normal={normal.decision.value}; blurred=RETAKE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
