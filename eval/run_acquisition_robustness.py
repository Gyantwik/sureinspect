"""Measure file-upload and screenshot-like acquisition robustness on real samples.

This is a narrow, reproducible safety benchmark. It does not claim robustness
outside the declared transforms and never feeds transformed pixels back into
calibration or model training.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.live_runner import get_live_engine

RESULT_PATH = ROOT / "eval" / "results" / "acquisition_robustness.json"
REPORT_PATH = ROOT / "docs" / "ACQUISITION_ROBUSTNESS.md"
SAMPLE_DIR = ROOT / "data" / "samples"

SAMPLES = (
    ("pcb1_borderline_01", "good", "PASS"),
    ("pcb1_bad_missing_01", "bad", "FAIL"),
    ("pcb1_bad_scratch_01", "bad", "HUMAN"),
)


def png_reencode(image: np.ndarray) -> np.ndarray:
    ok, payload = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encode failed")
    decoded = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("PNG decode failed")
    return decoded


def browser_resize_0729(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=0.729, fy=0.729, interpolation=cv2.INTER_AREA)


def jpeg_q90(image: np.ndarray) -> np.ndarray:
    ok, payload = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    decoded = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed")
    return decoded


TRANSFORMS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("png_reencode", png_reencode),
    ("browser_resize_0729", browser_resize_0729),
    ("jpeg_q90", jpeg_q90),
)


def response_record(sample_id: str, label: str, variant: str, response: Any) -> dict[str, Any]:
    alignment = response.alignment
    return {
        "sample_id": sample_id,
        "ground_truth": label,
        "variant": variant,
        "decision": response.decision.value,
        "decision_path": response.decision_path.value,
        "anomaly_score": response.anomaly_score,
        "quality_gate": response.quality_gate.status,
        "alignment": alignment.model_dump(mode="json") if alignment else None,
        "steps_used": response.steps_used,
        "step_budget": response.step_budget,
        "timing_ms": response.timing_ms.total,
    }


def evaluate_records(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["sample_id"], {})[record["variant"]] = record

    for sample_id, label, expected in SAMPLES:
        variants = grouped.get(sample_id, {})
        sample_route = variants.get("sample_id_route")
        byte_route = variants.get("exact_image_bytes")
        if not sample_route or not byte_route:
            failures.append(f"{sample_id}: missing exact-route evidence")
            continue
        parity_fields = ("decision", "decision_path", "anomaly_score")
        if any(sample_route[field] != byte_route[field] for field in parity_fields):
            failures.append(f"{sample_id}: sample-id and exact-byte routes diverged")
        if sample_route["decision"] != expected:
            failures.append(
                f"{sample_id}: expected curated decision {expected}, got {sample_route['decision']}"
            )

        for variant, record in variants.items():
            if label == "good" and record["decision"] == "FAIL":
                failures.append(f"{sample_id}/{variant}: good part was automatically failed")
            if label == "bad" and record["decision"] == "PASS":
                failures.append(f"{sample_id}/{variant}: bad part escaped as PASS")
            if record["steps_used"] > record["step_budget"]:
                failures.append(f"{sample_id}/{variant}: agent exceeded step budget")

    regression = grouped.get("pcb1_borderline_01", {}).get("browser_resize_0729")
    if not regression or regression["decision"] != "PASS":
        failures.append("0.729x screenshot regression: expected safe recovery to PASS")
    elif not regression["alignment"]:
        failures.append("0.729x screenshot regression: missing alignment evidence")
    else:
        scale = regression["alignment"].get("scale")
        overlap = regression["alignment"].get("overlap")
        if scale is None or not 0.70 <= scale <= 0.76:
            failures.append(f"0.729x screenshot regression: unexpected measured scale {scale}")
        if overlap is None or overlap < 0.60:
            failures.append(f"0.729x screenshot regression: unsafe overlap {overlap}")
    return failures


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Acquisition Robustness Evidence",
        "",
        "This measured benchmark exercises the real frozen OpenCV/PatchCore agent through both "
        "sample-ID and uploaded-byte paths. Transforms are evaluation-only and never update the model.",
        "",
        f"**Status:** {'PASS' if payload['passed'] else 'FAIL'}  ",
        f"**Generated:** {payload['generated_at']}  ",
        f"**Cases:** {len(payload['records'])}",
        "",
        "| Sample | Truth | Acquisition variant | Decision | Path | Scale | Overlap |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for record in payload["records"]:
        alignment = record.get("alignment") or {}
        scale = "—" if alignment.get("scale") is None else f"{alignment['scale']:.4f}"
        overlap = "—" if alignment.get("overlap") is None else f"{alignment['overlap']:.4f}"
        lines.append(
            f"| `{record['sample_id']}` | {record['ground_truth']} | `{record['variant']}` | "
            f"**{record['decision']}** | `{record['decision_path']}` | {scale} | {overlap} |"
        )
    lines.extend(
        [
            "",
            "## Safety assertions",
            "",
            "- Exact uploaded bytes must match the built-in sample route in decision, path, and score.",
            "- A known-good part may be passed or conservatively reviewed, but never automatically failed.",
            "- A known-bad part may fail or be routed to a human, but never escape as PASS.",
            "- Every run must remain inside the four-step agent budget.",
            "- The reported 0.729× browser-resize regression must recover to PASS with explicit scale and overlap evidence.",
            "",
        ]
    )
    if payload["failures"]:
        lines.extend(["## Failures", ""] + [f"- {failure}" for failure in payload["failures"]])
    return "\n".join(lines) + "\n"


def run() -> int:
    engine = get_live_engine(ROOT / "artifacts")
    if engine is None:
        raise RuntimeError("Live artifact snapshot is unavailable")

    records: list[dict[str, Any]] = []
    for sample_id, label, _expected in SAMPLES:
        source = SAMPLE_DIR / f"{sample_id}.png"
        image_bytes = source.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {source}")

        response, _ = engine.inspect(category="pcb1", sample_id=sample_id)
        records.append(response_record(sample_id, label, "sample_id_route", response))
        response, _ = engine.inspect(category="pcb1", image_bytes=image_bytes)
        records.append(response_record(sample_id, label, "exact_image_bytes", response))

        for variant, transform in TRANSFORMS:
            transformed = transform(image)
            height, width = transformed.shape[:2]
            response, _ = engine.inspect(
                category="pcb1",
                decoded_img=transformed,
                orig_dimensions=(width, height),
            )
            records.append(response_record(sample_id, label, variant, response))

    failures = evaluate_records(records)
    payload = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "curated real VisA PCB samples under declared acquisition transforms",
        "passed": not failures,
        "failures": failures,
        "records": records,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(payload), encoding="utf-8")
    print(f"{'PASS' if not failures else 'FAIL'}: {len(records)} acquisition cases")
    for failure in failures:
        print(f"- {failure}")
    print(f"Evidence: {RESULT_PATH.relative_to(ROOT)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())
