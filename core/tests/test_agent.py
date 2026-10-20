"""
core/tests/test_agent.py
Unit tests for the live agent decision engine (core/agent.py).
"""
import numpy as np
import pytest

from core import agent, interfaces as I
from core.agent import DecisionResult, decide, inspect_standardized, run_inspection


def test_decide_fast_pass():
    res = decide(score=1.0, lighting_shift=False, evidence=None, t_pass=1.2, t_fail=2.0, budget_state={"steps": 0, "budget": 4})
    assert isinstance(res, DecisionResult)
    assert res.decision == "PASS"
    assert res.decision_path == "FAST_PASS"


def test_decide_lighting_shift_review():
    res = decide(score=1.0, lighting_shift=True, evidence=None, t_pass=1.2, t_fail=2.0, budget_state={"steps": 0, "budget": 4})
    assert res.decision == "HUMAN"
    assert res.decision_path == "LIGHTING_SHIFT_REVIEW"
    assert res.suggested_decision == "PASS"


def test_decide_budget_exhausted():
    res = decide(score=1.5, lighting_shift=False, evidence=None, t_pass=1.2, t_fail=2.0, budget_state={"steps": 4, "budget": 4})
    assert res.decision == "HUMAN"
    assert res.decision_path == "BUDGET_EXHAUSTED"


def test_decide_quality_rejected_when_score_none():
    res = decide(score=None, lighting_shift=False, evidence=None, t_pass=1.2, t_fail=2.0, budget_state={"steps": 0, "budget": 4})
    assert res.decision == "RETAKE"
    assert res.decision_path == "QUALITY_REJECTED"


def test_decide_agent_confirmed_defect():
    evidence = {"alignment_valid": True, "residual_p95": 0.55, "r_low": 0.38, "r_high": 0.48}
    res = decide(score=2.2, lighting_shift=False, evidence=evidence, t_pass=1.2, t_fail=2.0, budget_state={"steps": 2, "budget": 4})
    assert res.decision == "FAIL"
    assert res.decision_path == "AGENT_CONFIRMED"


def test_decide_evidence_conflict():
    evidence = {"alignment_valid": True, "residual_p95": 0.20, "r_low": 0.38, "r_high": 0.48}
    res = decide(score=2.2, lighting_shift=False, evidence=evidence, t_pass=1.2, t_fail=2.0, budget_state={"steps": 2, "budget": 4})
    assert res.decision == "HUMAN"
    assert res.decision_path == "EVIDENCE_CONFLICT"


def test_decide_alignment_failed():
    evidence = {"alignment_valid": False, "reason": "No inliers found"}
    res = decide(score=2.2, lighting_shift=False, evidence=evidence, t_pass=1.2, t_fail=2.0, budget_state={"steps": 1, "budget": 4})
    assert res.decision == "HUMAN"
    assert res.decision_path == "ALIGN_FAILED"


def test_decide_agent_cleared_ambiguous():
    evidence = {"alignment_valid": True, "residual_p95": 0.30, "r_low": 0.38, "r_high": 0.48}
    res = decide(score=1.5, lighting_shift=False, evidence=evidence, t_pass=1.2, t_fail=2.0, budget_state={"steps": 2, "budget": 4})
    assert res.decision == "PASS"
    assert res.decision_path == "AGENT_CLEARED"


def test_interfaces_delegation_no_longer_raises():
    """Verify that core.interfaces successfully delegates to core.agent without NotImplementedError."""
    res = I.decide(1.0, False, None, 1.2, 2.0, {"steps": 0, "budget": 4})
    assert res.decision == "PASS"
    assert res.decision_path == "FAST_PASS"


def test_inspect_standardized_quality_failure():
    """Severely blurred image must be caught and rejected by the quality gate."""
    blurred = np.full((128, 128, 3), 128, dtype=np.uint8)
    gate = {
        "blur_thr": 1000.0,  # Impossible threshold
        "dark_max": 0.05,
        "bright_max": 0.05,
        "glare_thr": 0.08,
        "l_mean_ref": 128.0,
        "l_mean_sd": 5.0,
        "l_std_ref": 20.0,
        "n_fit": 50,
        "gate_version": "v1.0",
    }
    dummy_bundle = {
        "net": None,
        "bank": None,
        "calibration": {"t_pass": {"threshold": 1.0}, "t_fail_default": 2.0},
        "quality_gate": gate,
        "input_size": 256,
    }
    out = inspect_standardized(blurred, category="pcb1", snapshot=dummy_bundle)
    assert out["decision"] == "RETAKE"
    assert out["decision_path"] == "QUALITY_REJECTED"
    assert out["score"] is None
    assert len(out["trace"]) == 1
    assert out["trace"][0]["status"] == "FAILED"


def test_run_inspection_live_mock_sample():
    """Verify run_inspection end-to-end execution on a sample board."""
    resp, assets = run_inspection(sample_id="pcb1_good_01", alpha_fail=0.05)
    assert resp.engine == "live"
    assert resp.category == "pcb1"
    assert resp.decision.value in ("PASS", "FAIL", "HUMAN", "RETAKE")
    assert len(assets) >= 1
    # Standardized input asset is encoded as PNG
    assert any(k.endswith("_input.png") for k in assets)
