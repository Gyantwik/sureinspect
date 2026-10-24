from eval.run_pcb2_cross_category import summarize


def test_cross_category_summary_keeps_abstentions_separate():
    records = [
        {"ground_truth": "normal", "decision": "HUMAN", "decision_path": "ALIGN_FAILED"},
        {"ground_truth": "normal", "decision": "FAIL", "decision_path": "AGENT_CONFIRMED"},
        {"ground_truth": "anomaly", "decision": "HUMAN", "decision_path": "ALIGN_FAILED"},
        {"ground_truth": "anomaly", "decision": "PASS", "decision_path": "FAST_PASS"},
    ]
    result = summarize(records)
    assert result["normal"]["decisions"] == {"FAIL": 1, "HUMAN": 1}
    assert result["anomaly"]["decisions"] == {"HUMAN": 1, "PASS": 1}
    assert result["safety"]["anomaly_pass_escapes"] == 1
    assert result["safety"]["normal_automatic_failures"] == 1
    assert result["safety"]["normal_abstention_rate"] == 0.5
