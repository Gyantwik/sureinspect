"""Evaluate the frozen PCB1 pipeline on the official VisA PCB2 test split.

This is an out-of-distribution, cross-category stress test. It measures whether
the agent fails safely on a different PCB design; it is not evidence that PCB2
is a supported or calibrated production category.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.live_runner import get_live_engine
from eval.run_real_e1 import get_git_commit_hash, get_pipeline_fingerprint

DATA_ROOT = ROOT / "data" / "visa" / "pcb2"
MANIFEST_PATH = DATA_ROOT / "acquisition_manifest.json"
RESULT_PATH = ROOT / "eval" / "results" / "pcb2_cross_category.json"
REPORT_PATH = ROOT / "docs" / "PCB2_CROSS_CATEGORY_EVALUATION.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in ("normal", "anomaly"):
        selected = [record for record in records if record["ground_truth"] == label]
        decisions = Counter(record["decision"] for record in selected)
        paths = Counter(record["decision_path"] for record in selected)
        summary[label] = {
            "n": len(selected),
            "decisions": dict(sorted(decisions.items())),
            "paths": dict(sorted(paths.items())),
            "automated": decisions["PASS"] + decisions["FAIL"],
            "human_or_retake": decisions["HUMAN"] + decisions["RETAKE"],
        }
    anomaly_n = max(summary["anomaly"]["n"], 1)
    normal_n = max(summary["normal"]["n"], 1)
    summary["safety"] = {
        "anomaly_pass_escapes": summary["anomaly"]["decisions"].get("PASS", 0),
        "anomaly_pass_escape_rate": summary["anomaly"]["decisions"].get("PASS", 0) / anomaly_n,
        "anomaly_intercepted_fail_human_retake": (
            summary["anomaly"]["decisions"].get("FAIL", 0)
            + summary["anomaly"]["decisions"].get("HUMAN", 0)
            + summary["anomaly"]["decisions"].get("RETAKE", 0)
        ),
        "normal_automatic_failures": summary["normal"]["decisions"].get("FAIL", 0),
        "normal_automatic_failure_rate": summary["normal"]["decisions"].get("FAIL", 0) / normal_n,
        "normal_abstention_rate": (
            summary["normal"]["decisions"].get("HUMAN", 0)
            + summary["normal"]["decisions"].get("RETAKE", 0)
        ) / normal_n,
    }
    return summary


def markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    safety = summary["safety"]
    lines = [
        "# PCB2 Cross-Category Evaluation",
        "",
        (
            "This is a zero-shot out-of-distribution stress test: the frozen PCB1 model, "
            "calibration, and PCB1 reference templates were applied to the official VisA PCB2 test split. "
            "PCB2 is not a supported category, so this result measures safe abstention—not PCB2 accuracy."
        ),
        "",
        "- Official registry: https://registry.opendata.aws/visa/",
        "- Official one-class split: https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv",
        "- Dataset license: CC BY 4.0",
        f"- Images: {len(payload['records'])} (100 normal, 100 anomaly)",
        f"- Pipeline fingerprint: `{payload['pipeline_fingerprint']}`",
        f"- PCB2 acquisition manifest SHA-256: `{payload['acquisition_manifest_sha256']}`",
        "",
        "| Ground truth | PASS | FAIL | HUMAN | RETAKE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("normal", "anomaly"):
        counts = summary[label]["decisions"]
        lines.append(
            f"| {label} | {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | "
            f"{counts.get('HUMAN', 0)} | {counts.get('RETAKE', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Safety interpretation",
            "",
            f"- Anomaly PASS escapes: **{safety['anomaly_pass_escapes']}/100**.",
            f"- Anomalies intercepted by FAIL/HUMAN/RETAKE: **{safety['anomaly_intercepted_fail_human_retake']}/100**.",
            f"- Normal automatic failures: **{safety['normal_automatic_failures']}/100**.",
            f"- Normal abstention rate: **{safety['normal_abstention_rate']:.1%}**.",
            "",
            (
                "A high HUMAN rate is correct behavior here: PCB2 geometry is outside the PCB1 calibration and template set. "
                "Supporting PCB2 automation would require a separate PCB2 bank, calibration split, reference templates, "
                "quality thresholds, and a new independent evaluation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run() -> int:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}; run `python -m eval.download_visa_pcb2` first"
        )
    acquisition = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = []
    for record in acquisition["records"]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Missing or hash-mismatched PCB2 image: {path}")
        files.append((path, str(record["label"]), str(record["sha256"])))
    if len(files) != 200:
        raise RuntimeError(f"Expected 200 official PCB2 test images, found {len(files)}")

    engine = get_live_engine(ROOT / "artifacts")
    if engine is None:
        raise RuntimeError("Live PCB1 snapshot is unavailable")
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (path, label, image_sha256) in enumerate(files, 1):
        response, _assets = engine.inspect(category="pcb1", image_bytes=path.read_bytes())
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "image_sha256": image_sha256,
                "ground_truth": label,
                "decision": response.decision.value,
                "decision_path": response.decision_path.value,
                "anomaly_score": response.anomaly_score,
                "steps_used": response.steps_used,
                "step_budget": response.step_budget,
                "alignment": response.alignment.model_dump(mode="json") if response.alignment else None,
                "timing_ms": response.timing_ms.total,
            }
        )
        if index % 25 == 0 or index == len(files):
            print(f"Evaluated {index}/{len(files)} PCB2 images", flush=True)

    budget_violations = [record["path"] for record in records if record["steps_used"] > record["step_budget"]]
    commit = get_git_commit_hash()
    payload = {
        "schema_version": "sureinspect.pcb2-cross-category.v1",
        "evaluation_type": "zero_shot_cross_category_out_of_distribution",
        "supported_production_category": False,
        "source_model_category": "pcb1",
        "evaluation_category": "pcb2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "pipeline_fingerprint": get_pipeline_fingerprint(commit),
        "acquisition_manifest_sha256": sha256_file(MANIFEST_PATH),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "budget_violations": budget_violations,
        "summary": summarize(records),
        "records": records,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(payload), encoding="utf-8")
    safety = payload["summary"]["safety"]
    print(json.dumps(safety, indent=2))
    print(f"Evidence: {RESULT_PATH.relative_to(ROOT)}")
    return 1 if budget_violations else 0


if __name__ == "__main__":
    raise SystemExit(run())
