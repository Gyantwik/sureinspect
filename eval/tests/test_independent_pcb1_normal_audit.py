from eval.run_independent_pcb1_normal_audit import (
    collect_snapshot_dhashes,
    collect_snapshot_hashes,
    hamming,
    summarize,
)


def test_collect_snapshot_hashes_recurses_through_manifest():
    payload = {"splits": {"a": [{"sha256": "first"}], "b": {"items": [{"sha256": "second"}]}}}
    assert collect_snapshot_hashes(payload) == {"first", "second"}


def test_collect_snapshot_dhashes_and_hamming():
    payload = {"items": [{"dhash": "0000000000000000"}, {"dhash": "0000000000000003"}]}
    assert collect_snapshot_dhashes(payload) == {0, 3}
    assert hamming(0, 3) == 2


def test_normal_audit_summary_keeps_abstention_separate():
    records = [
        {"cohort": "primary_front_blue_hc_sr04", "decision": "PASS", "decision_path": "FAST_PASS", "quality_gate": {"reason": "ok"}},
        {"cohort": "primary_front_blue_hc_sr04", "decision": "FAIL", "decision_path": "AGENT_CONFIRMED", "quality_gate": {"reason": "ok"}},
        {"cohort": "other", "decision": "HUMAN", "decision_path": "ALIGN_FAILED", "quality_gate": {"reason": "shift"}},
        {"cohort": "other", "decision": "RETAKE", "decision_path": "QUALITY_REJECTED", "quality_gate": {"reason": "blur"}},
    ]
    result = summarize(records)
    assert result["primary_front_blue_hc_sr04"]["normal_automatic_failure_rate"] == 0.5
    assert result["primary_front_blue_hc_sr04"]["normal_pass_rate"] == 0.5
    assert result["supplemental_views_and_variant"]["abstention_rate"] == 1.0
    assert result["all"]["automated_coverage"] == 0.5
