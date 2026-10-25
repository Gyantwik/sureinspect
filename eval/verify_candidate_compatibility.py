"""Prove whether cached inference remains compatible with the current candidate."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from eval.run_real_e1 import get_pipeline_fingerprint

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "eval/results/measured_e1_results_v2.json"
CRITICAL = ["core", "artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11"]


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    source_commit = result["commit"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    command = ["git", "diff", "--name-only", f"{source_commit}..{head}", "--", *CRITICAL]
    changed = [line for line in subprocess.check_output(command, cwd=ROOT, text=True).splitlines() if line]
    working_command = ["git", "diff", "--name-only", "HEAD", "--", *CRITICAL]
    working_changed = [line for line in subprocess.check_output(working_command, cwd=ROOT, text=True).splitlines() if line]
    measured_fingerprint = result.get("pipeline_fingerprint")
    candidate_fingerprint = get_pipeline_fingerprint(head)
    fingerprint_match = bool(measured_fingerprint) and measured_fingerprint == candidate_fingerprint
    payload = {
        "schema_version": "sureinspect.candidate-compatibility.v1",
        "measured_inference_commit": source_commit,
        "candidate_commit": head,
        "critical_scopes": CRITICAL,
        "critical_files_changed": changed,
        "uncommitted_critical_files_changed": working_changed,
        "measured_pipeline_fingerprint": measured_fingerprint,
        "candidate_pipeline_fingerprint": candidate_fingerprint,
        "compatible": fingerprint_match,
        "interpretation": (
            "Measured inference is candidate-compatible only when the content hash of inference code, "
            "snapshot artifacts, and templates exactly matches; uncommitted changes are included."
        ),
    }
    out = ROOT / "eval/results/candidate_compatibility.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
