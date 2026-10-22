from eval.generate_agentic_evidence import build_matrix


def test_policy_matrix_covers_every_contract_path():
    matrix = build_matrix()
    paths = {row["decision_path"] for row in matrix["paths"]}
    assert paths == {
        "QUALITY_REJECTED", "FAST_PASS", "AGENT_CLEARED", "AGENT_CONFIRMED",
        "SUSPECTED_DEFECT", "EVIDENCE_CONFLICT", "LIGHTING_SHIFT_REVIEW",
        "ALIGN_FAILED", "INCONCLUSIVE", "BUDGET_EXHAUSTED",
    }
    assert matrix["opencv_version"].startswith("5.")
