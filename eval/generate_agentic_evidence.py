"""Generate a deterministic, judge-readable contract for every decision path."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from core.agent import decide

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "eval/results/agentic_policy_matrix.json"
OUT_MD = ROOT / "docs/AGENTIC_POLICY_MATRIX.md"

TP, TF = 2.0, 2.3
ALIGNED_LOW = {"alignment_valid": True, "residual_p95": 0.20, "r_low": 0.30, "r_high": 0.48, "rois_total": 1, "rois_checked": 1}

CASES = [
    ("quality_rejected", None, False, None, {}, "QUALITY_REJECTED"),
    ("fast_pass", 1.8, False, None, {}, "FAST_PASS"),
    ("agent_cleared", 2.1, False, ALIGNED_LOW, {}, "AGENT_CLEARED"),
    ("agent_confirmed", 2.5, False, {**ALIGNED_LOW, "residual_p95": 0.60}, {}, "AGENT_CONFIRMED"),
    ("suspected_defect", 2.1, False, {**ALIGNED_LOW, "residual_p95": 0.60}, {}, "SUSPECTED_DEFECT"),
    ("evidence_conflict", 2.5, False, ALIGNED_LOW, {}, "EVIDENCE_CONFLICT"),
    ("lighting_shift_review", 1.8, True, None, {}, "LIGHTING_SHIFT_REVIEW"),
    ("alignment_failed", 2.1, False, {"alignment_valid": False, "reason": "insufficient inliers"}, {}, "ALIGN_FAILED"),
    ("inconclusive", 2.1, False, {**ALIGNED_LOW, "residual_p95": 0.40}, {}, "INCONCLUSIVE"),
    ("budget_exhausted", 2.1, False, None, {"steps": 4, "budget": 4, "exhausted": True}, "BUDGET_EXHAUSTED"),
]


def build_matrix() -> dict:
    rows = []
    for fixture, score, lighting, evidence, budget, expected in CASES:
        result = decide(score, lighting, evidence, TP, TF, budget)
        if result.decision_path != expected:
            raise AssertionError(f"{fixture}: expected {expected}, got {result.decision_path}")
        rows.append({
            "fixture": fixture,
            "inputs": {"score": score, "lighting_shift": lighting, "evidence": evidence, "budget_state": budget},
            "decision": result.decision,
            "decision_path": result.decision_path,
            "suggested_decision": result.suggested_decision,
            "reason": result.reason,
        })
    return {
        "schema_version": "sureinspect.agentic-policy-matrix.v1",
        "opencv_version": cv2.__version__,
        "thresholds": {"t_pass": TP, "t_fail": TF},
        "note": "Deterministic policy-contract fixtures; live-image evidence is documented separately.",
        "paths": rows,
    }


def main() -> int:
    matrix = build_matrix()
    OUT_JSON.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Agentic Policy Decision Matrix", "",
        "This artifact executes the frozen pure policy and proves that every advertised decision path is reachable under an explicit, reviewable input contract. It complements—not replaces—the live-image traces in `docs/AGENTIC_TRACE_EVIDENCE.md`.", "",
        f"OpenCV runtime: `{matrix['opencv_version']}`", "",
        "| Fixture | Decision | Path | Why |", "|---|---|---|---|",
    ]
    for row in matrix["paths"]:
        reason = row["reason"].replace("|", "\\|")
        lines.append(f"| `{row['fixture']}` | **{row['decision']}** | `{row['decision_path']}` | {reason} |")
    lines.extend(["", "Regenerate with:", "", "```powershell", ".venv\\Scripts\\python.exe -m eval.generate_agentic_evidence", "```", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {len(matrix['paths'])} policy-path fixtures with OpenCV {matrix['opencv_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
