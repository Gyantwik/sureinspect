from __future__ import annotations

import copy
import json
from pathlib import Path

from eval.validate_measured_results import validate_file, validate_payload

ROOT = Path(__file__).resolve().parents[2]
MEASURED = ROOT / "eval" / "results" / "measured_e1_results_v2.json"


def _payload() -> dict:
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def test_current_measured_result_is_internally_consistent():
    errors, summary = validate_file(MEASURED)
    # In clean CI checkouts where large binary model weights are git-ignored,
    # skip physical provenance artifact checks if snapshot directory is not mounted.
    if not (ROOT / "artifacts" / "snapshots").exists():
        errors = [e for e in errors if not e.startswith("provenance artifact is missing")]
    assert errors == []
    final = summary["splits"]["final_test_split"]
    assert final["defect_escapes"] == 4
    assert final["defect_total"] == 70
    assert final["normal_human"] == 18
    assert final["defect_human"] == 55
    assert final["normal_fail"] == 0


def test_detects_escape_arithmetic_drift():
    payload = _payload()
    payload["final_test_split"]["agent"]["defect_escapes"]["numerator"] = 5
    errors, _ = validate_payload(payload)
    assert any("defect_escapes.numerator=5, expected 4" in error for error in errors)


def test_detects_count_total_drift():
    payload = _payload()
    payload["dev_split"]["agent"]["counts"]["normal_human"] += 1
    errors, _ = validate_payload(payload)
    assert any("n_normal=297, expected 298" in error for error in errors)


def test_expected_commit_is_enforced():
    payload = _payload()
    expected = "0" * 40
    errors, _ = validate_payload(payload, expected_commit=expected)
    assert any("expected current commit" in error for error in errors)


def test_current_disclosure_cannot_be_silently_removed():
    payload = copy.deepcopy(_payload())
    payload["final_test_split"]["label"] = "held-out test"
    errors, _ = validate_payload(payload)
    assert any("previously-seen disclosure" in error for error in errors)


def test_invalid_provenance_digest_is_rejected():
    payload = _payload()
    payload["provenance"]["model_sha256"] = "not-a-digest"
    errors, _ = validate_payload(payload)
    assert "provenance.model_sha256 must be a SHA-256 digest" in errors
