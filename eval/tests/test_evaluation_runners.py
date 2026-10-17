"""
eval/tests/test_evaluation_runners.py
Tests verifying baseline evaluation, agent evaluation, calibration splits, and corruption robustness.
"""
import pytest
import numpy as np

from eval.run_baseline import evaluate_baseline
from eval.run_agent import evaluate_agent
from eval.run_calibration_splits import evaluate_calibration_split, compute_conformal_threshold
from eval.run_corruptions import evaluate_corruption_robustness


def test_baseline_evaluation_metrics():
    # 5 normals (scores 0.5, 0.8, 1.1, 1.2, 1.5) and 5 defects (1.0, 1.3, 1.8, 2.0, 2.5)
    scores = [0.5, 0.8, 1.1, 1.2, 1.5, 1.0, 1.3, 1.8, 2.0, 2.5]
    labels = ["good"] * 5 + ["bad"] * 5
    t_fail = 1.25

    res = evaluate_baseline(scores, labels, t_fail)
    assert res["n_total"] == 10
    assert res["n_normal"] == 5
    assert res["n_defective"] == 5

    # Normals: 4 <= 1.25 (pass), 1 > 1.25 (fail)
    assert res["counts"]["normal_pass"] == 4
    assert res["counts"]["normal_fail"] == 1
    assert res["false_fail"]["rate"] == 1 / 5

    # Defects: 1 <= 1.25 (pass - escape!), 4 > 1.25 (fail - recall)
    assert res["counts"]["defective_pass"] == 1
    assert res["counts"]["defective_fail"] == 4
    assert res["defect_escapes"]["rate"] == 1 / 5
    assert res["defect_recall"]["rate"] == 4 / 5

    assert res["coverage"]["rate"] == 1.0


def test_agent_evaluation_metrics():
    decisions = ["PASS", "PASS", "FAIL", "HUMAN", "RETAKE", "PASS", "FAIL", "FAIL", "HUMAN", "RETAKE"]
    paths = ["FAST_PASS", "AGENT_CLEARED", "SUSPECTED_DEFECT", "BORDERLINE_REVIEW", "QUALITY_REJECTED",
             "AGENT_CLEARED", "CONFIRMED_DEFECT", "CONFIRMED_DEFECT", "BORDERLINE_REVIEW", "QUALITY_REJECTED"]
    labels = ["good"] * 5 + ["bad"] * 5

    res = evaluate_agent(decisions, paths, labels)
    assert res["n_total"] == 10
    assert res["n_normal"] == 5
    assert res["n_defective"] == 5

    # Automated decisions: 2 normal pass, 1 normal fail, 1 defective pass, 2 defective fail = 6
    assert res["coverage"]["rate"] == 6 / 10
    # Defect escape: 1 defective pass out of 5 defects
    assert res["defect_escapes"]["rate"] == 1 / 5
    # Review burden: 2 out of 10
    assert res["review_burden"]["rate"] == 2 / 10
    # Retake rate: 2 out of 10
    assert res["retake_rate"]["rate"] == 2 / 10


def test_conformal_threshold_computation():
    calib = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thr_01 = compute_conformal_threshold(calib, alpha=0.10)
    assert thr_01 >= 0.9


def test_calibration_split_evaluation():
    calib = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    test_normals = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    res = evaluate_calibration_split(calib, test_normals, alphas=[0.10, 0.20])
    assert "alpha_0.10" in res["alpha_evaluations"]
    assert "alpha_0.20" in res["alpha_evaluations"]


def test_corruption_robustness_evaluation():
    # 3 dummy synthetic images
    images = [np.full((64, 64, 3), 120, dtype=np.uint8) for _ in range(3)]
    labels = ["good", "good", "bad"]

    res = evaluate_corruption_robustness(images, labels)
    assert res["n_original_images"] == 3
    assert "clean_baseline" in res
    assert "joint_corruptions" in res
