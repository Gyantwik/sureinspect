"""Run the PCB1 engine on the external, rights-cleared development robustness set."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.live_runner import get_live_engine
from eval.run_real_e1 import get_git_commit_hash, get_pipeline_fingerprint

SNAPSHOT = ROOT / "artifacts" / "snapshots" / "pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1"
DATA_ROOT = ROOT / "data" / "real_tests" / "pcb1_commons"
MANIFEST_PATH = DATA_ROOT / "acquisition_manifest.json"
RESULT_PATH = ROOT / "eval" / "results" / "pcb1_external_normal_audit.json"
REPORT_PATH = ROOT / "docs" / "PCB1_EXTERNAL_NORMAL_AUDIT.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_snapshot_hashes(node: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "sha256" and isinstance(value, str):
                hashes.add(value)
            else:
                hashes.update(collect_snapshot_hashes(value))
    elif isinstance(node, list):
        for value in node:
            hashes.update(collect_snapshot_hashes(value))
    return hashes


def collect_snapshot_dhashes(node: Any) -> set[int]:
    hashes: set[int] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "dhash" and isinstance(value, str):
                hashes.add(int(value, 16))
            else:
                hashes.update(collect_snapshot_dhashes(value))
    elif isinstance(node, list):
        for value in node:
            hashes.update(collect_snapshot_dhashes(value))
    return hashes


def difference_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def summarize_cohort(records: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(record["decision"] for record in records)
    paths = Counter(record["decision_path"] for record in records)
    quality_reasons = Counter(record["quality_gate"]["reason"] for record in records)
    total = max(len(records), 1)
    return {
        "n": len(records),
        "decisions": dict(sorted(decisions.items())),
        "paths": dict(sorted(paths.items())),
        "quality_reasons": dict(sorted(quality_reasons.items())),
        "normal_automatic_failures": decisions["FAIL"],
        "normal_automatic_failure_rate": decisions["FAIL"] / total,
        "normal_pass_rate": decisions["PASS"] / total,
        "abstention_rate": (decisions["HUMAN"] + decisions["RETAKE"]) / total,
        "automated_coverage": (decisions["PASS"] + decisions["FAIL"]) / total,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [record for record in records if record["cohort"] == "primary_front_blue_hc_sr04"]
    supplemental = [record for record in records if record["cohort"] != "primary_front_blue_hc_sr04"]
    return {
        "primary_front_blue_hc_sr04": summarize_cohort(primary),
        "supplemental_views_and_variant": summarize_cohort(supplemental),
        "all": summarize_cohort(records),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]["primary_front_blue_hc_sr04"]
    supplemental = payload["summary"]["supplemental_views_and_variant"]
    counts = summary["decisions"]
    return "\n".join(
        [
            "# PCB1 External Normal-Image Development Robustness Audit",
            "",
            (
                "This is an externally sourced, normal-only development robustness audit using six rights-cleared "
                "Wikimedia Commons photographs of HC-SR04 modules. None is an exact or near-duplicate match for "
                "any image in the frozen VisA PCB1 split."
            ),
            "",
            (
                "It is **not independent final evidence** because these images have informed robustness development. "
                "The boards are only visibly normal, electrical function is unverified, labels were not blinded, "
                "and there are no defective images."
            ),
            "",
            f"- Candidate commit: `{payload['git_commit']}`",
            f"- Pipeline fingerprint: `{payload['pipeline_fingerprint']}`",
            f"- Acquisition manifest SHA-256: `{payload['acquisition_manifest_sha256']}`",
            f"- Primary eligible images (front-view blue HC-SR04): {summary['n']}",
            f"- Supplemental challenge images (back view or red product variant): {supplemental['n']}",
            "- Source: https://commons.wikimedia.org/wiki/Category:Ultrasonic_range_finding",
            "",
            "| PASS | FAIL | HUMAN | RETAKE |",
            "|---:|---:|---:|---:|",
            (
                f"| {counts.get('PASS', 0)} | {counts.get('FAIL', 0)} | "
                f"{counts.get('HUMAN', 0)} | {counts.get('RETAKE', 0)} |"
            ),
            "",
            f"- Normal automatic failure rate: **{summary['normal_automatic_failure_rate']:.1%}**",
            f"- Normal pass rate: **{summary['normal_pass_rate']:.1%}**",
            f"- Abstention rate: **{summary['abstention_rate']:.1%}**",
            f"- Automated coverage: **{summary['automated_coverage']:.1%}**",
            f"- Supplemental outcomes: `{supplemental['decisions']}`",
            f"- Stable decision and route across three runs: **{payload['repeatability']['stable_records']}/{payload['repeatability']['records_checked']}**",
            "",
            "## Interpretation",
            "",
            (
                "This audit measures behavior on externally photographed examples of the supported board family "
                "under uncontrolled backgrounds, viewpoints, lighting, and product variants. It can reveal unsafe "
                "normal-board rejection and acquisition brittleness, but it cannot measure defect interception."
            ),
            "",
            (
                "The primary set is reported separately from the back-side view and red-board product variant so "
                "unsupported views do not distort the supported front-view result."
            ),
            "",
            "All records are replayed three times and exact decision/path stability is reported above.",
            "",
        ]
    )


def run() -> int:
    acquisition = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    snapshot_splits = json.loads((SNAPSHOT / "splits.json").read_text(encoding="utf-8"))
    used_hashes = collect_snapshot_hashes(snapshot_splits)
    used_dhashes = collect_snapshot_dhashes(snapshot_splits)
    records: list[dict[str, Any]] = []
    files: list[tuple[Path, dict[str, Any]]] = []
    for source_record in acquisition["records"]:
        path = ROOT / source_record["path"]
        image_hash = sha256_file(path)
        if image_hash != source_record["sha256"]:
            raise RuntimeError(f"Hash mismatch for {path}")
        if image_hash in used_hashes:
            raise RuntimeError(f"External image duplicates a frozen VisA PCB1 image: {path}")
        image_dhash = difference_hash(path)
        minimum_dhash_distance = min(hamming(image_dhash, known) for known in used_dhashes)
        if minimum_dhash_distance <= 2:
            raise RuntimeError(f"External image is a near-duplicate of frozen VisA PCB1 data: {path}")
        source_record = dict(source_record)
        source_record["dhash"] = f"{image_dhash:016x}"
        source_record["minimum_frozen_visa_dhash_distance"] = minimum_dhash_distance
        files.append((path, source_record))

    engine = get_live_engine(ROOT / "artifacts")
    if engine is None:
        raise RuntimeError("Live PCB1 snapshot is unavailable")
    started = time.perf_counter()
    for path, source_record in files:
        image_bytes = path.read_bytes()
        responses = [engine.inspect(category="pcb1", image_bytes=image_bytes)[0] for _ in range(3)]
        response = responses[0]
        observed_routes = [
            {"decision": replay.decision.value, "decision_path": replay.decision_path.value}
            for replay in responses
        ]
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "title": source_record["title"],
                "description_url": source_record["description_url"],
                "license": source_record["license"],
                "cohort": source_record["cohort"],
                "image_sha256": source_record["sha256"],
                "image_dhash": source_record["dhash"],
                "minimum_frozen_visa_dhash_distance": source_record["minimum_frozen_visa_dhash_distance"],
                "ground_truth": "visible_normal_unverified_function",
                "decision": response.decision.value,
                "decision_path": response.decision_path.value,
                "anomaly_score": response.anomaly_score,
                "steps_used": response.steps_used,
                "step_budget": response.step_budget,
                "quality_gate": response.quality_gate.model_dump(mode="json"),
                "alignment": response.alignment.model_dump(mode="json") if response.alignment else None,
                "next_action": response.next_action.model_dump(mode="json"),
                "execution_trace": [step.model_dump(mode="json") for step in response.execution_trace],
                "timing_ms": response.timing_ms.total,
                "repeatability": {
                    "runs": len(observed_routes),
                    "observed": observed_routes,
                    "stable": len({(item['decision'], item['decision_path']) for item in observed_routes}) == 1,
                },
            }
        )

    budget_violations = [record["path"] for record in records if record["steps_used"] > record["step_budget"]]
    commit = get_git_commit_hash()
    payload = {
        "schema_version": "sureinspect.pcb1-external-development-robustness.v2",
        "evaluation_type": "external_normal_development_robustness_audit",
        "complete_independent_effectiveness_evaluation": False,
        "limitations": acquisition["limitations"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "pipeline_fingerprint": get_pipeline_fingerprint(commit),
        "acquisition_manifest_sha256": sha256_file(MANIFEST_PATH),
        "exact_overlap_with_frozen_visa_pcb1": 0,
        "near_duplicate_overlap_at_dhash_distance_2": 0,
        "frozen_visa_pcb1_hashes_checked": len(used_hashes),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "budget_violations": budget_violations,
        "repeatability": {
            "runs_per_record": 3,
            "records_checked": len(records),
            "stable_records": sum(record["repeatability"]["stable"] for record in records),
        },
        "summary": summarize(records),
        "records": records,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Evidence: {RESULT_PATH.relative_to(ROOT)}")
    return 1 if budget_violations else 0


if __name__ == "__main__":
    raise SystemExit(run())
