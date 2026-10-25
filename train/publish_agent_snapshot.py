"""Promote an immutable perception snapshot into a complete agent-ready bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11")
    parser.add_argument("--version", default="agent-v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/snapshots")
    args = parser.parse_args()
    parent = args.parent.resolve()
    out = (args.out_dir / f"{parent.name}-{args.version}").resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite immutable snapshot: {out}")
    required = ["model.onnx", "bank.npz", "calibration.json", "quality_gate.json", "splits.json", "manifest.json"]
    missing = [name for name in required if not (parent / name).is_file()]
    if missing:
        raise SystemExit(f"Parent snapshot is incomplete: {', '.join(missing)}")
    shutil.copytree(parent, out)
    templates = ROOT / "data/templates/pcb1"
    if not templates.is_dir():
        raise SystemExit("Missing evidence templates: data/templates/pcb1")
    shutil.copytree(templates, out / "templates")
    results = ROOT / "eval/results/measured_e1_results_v2.json"
    policy = {
        "policy_version": "sureinspect_agentic_v1",
        "decision_outcomes": ["PASS", "FAIL", "HUMAN", "RETAKE"],
        "evidence_thresholds": {"q_min": 0.74, "r_low": 0.30, "r_high": 0.48},
        "limits": {"tool_step_budget": 4, "latency_ceiling_seconds": 2.0},
        "safety_invariants": [
            "quality rejection stops ordinary perception",
            "budget exhaustion or unresolved evidence cannot return PASS",
            "feedback is audit-only and cannot change production decisions",
        ],
        "implementation": {"agent": "core/agent.py", "tools": "core/tools.py"},
    }
    write_json(out / "policy.json", policy)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["snapshot_id"] = out.name
    manifest["parent_snapshot_id"] = parent.name
    manifest["readiness"] = "agent_ready"
    manifest["versions"]["policy_version"] = policy["policy_version"]
    manifest["tool_policy"] = policy
    manifest["evaluation_status"] = "regression_evaluated_not_independent"
    manifest["evaluation"] = {"artifact": "eval/results/measured_e1_results_v2.json", "sha256": sha256(results)}
    manifest["files"] = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name not in {"manifest.json", "SHA256SUMS"}):
        manifest["files"][path.relative_to(out).as_posix()] = sha256(path)
    write_json(out / "manifest.json", manifest)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(out).as_posix()}\n" for path in files),
        encoding="utf-8", newline="\n",
    )
    print(f"Published immutable agent-ready snapshot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
