"""
eval/run_baseline.py
E1: Baseline inspection decision runner and evaluation metrics.
Rule:
- Decision: PASS if anomaly_score <= t_fail, else FAIL.
- Computes coverage, defect recall, defect escapes, normal false-FAIL, and selective accuracy.
- Computes two-sided 95% Wilson confidence intervals using eval.stats.wilson.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from eval.stats import wilson, clopper_pearson


def evaluate_baseline(
    scores: List[float],
    ground_truth: List[str],  # "good" or "bad"
    t_fail: float,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Evaluates the single-threshold baseline on paired test scores and ground truth labels.
    """
    if len(scores) != len(ground_truth):
        raise ValueError(f"Scores ({len(scores)}) and labels ({len(ground_truth)}) length mismatch")

    n_total = len(scores)
    if n_total == 0:
        return {
            "n_total": 0,
            "coverage": None,
            "defect_escapes": None,
            "defect_recall": None,
            "false_fail_rate": None,
            "selective_accuracy": None,
        }

    normal_pass = 0
    normal_fail = 0
    defective_pass = 0
    defective_fail = 0

    for score, label in zip(scores, ground_truth):
        is_pass = score <= t_fail
        if label.lower() in ("good", "normal"):
            if is_pass:
                normal_pass += 1
            else:
                normal_fail += 1
        else:
            if is_pass:
                defective_pass += 1
            else:
                defective_fail += 1

    n_normal = normal_pass + normal_fail
    n_defective = defective_pass + defective_fail
    n_decided = normal_pass + normal_fail + defective_pass + defective_fail

    # Rates
    coverage_rate = n_decided / n_total if n_total > 0 else 0.0
    defect_escapes_rate = defective_pass / n_defective if n_defective > 0 else None
    defect_recall_rate = defective_fail / n_defective if n_defective > 0 else None
    false_fail_rate = normal_fail / n_normal if n_normal > 0 else None
    selective_acc_rate = (normal_pass + defective_fail) / n_decided if n_decided > 0 else None

    # Wilson intervals
    defect_escapes_ci = wilson(defective_pass, n_defective, confidence) if n_defective > 0 else None
    defect_recall_ci = wilson(defective_fail, n_defective, confidence) if n_defective > 0 else None
    false_fail_ci = wilson(normal_fail, n_normal, confidence) if n_normal > 0 else None
    selective_acc_ci = wilson(normal_pass + defective_fail, n_decided, confidence) if n_decided > 0 else None

    return {
        "n_total": n_total,
        "n_normal": n_normal,
        "n_defective": n_defective,
        "counts": {
            "normal_pass": normal_pass,
            "normal_fail": normal_fail,
            "defective_pass": defective_pass,
            "defective_fail": defective_fail,
        },
        "coverage": {
            "rate": coverage_rate,
            "ci_95": wilson(n_decided, n_total, confidence),
        },
        "defect_escapes": {
            "rate": defect_escapes_rate,
            "numerator": defective_pass,
            "denominator": n_defective,
            "ci_95": defect_escapes_ci,
        },
        "defect_recall": {
            "rate": defect_recall_rate,
            "numerator": defective_fail,
            "denominator": n_defective,
            "ci_95": defect_recall_ci,
        },
        "false_fail": {
            "rate": false_fail_rate,
            "numerator": normal_fail,
            "denominator": n_normal,
            "ci_95": false_fail_ci,
        },
        "selective_accuracy": {
            "rate": selective_acc_rate,
            "numerator": normal_pass + defective_fail,
            "denominator": n_decided,
            "ci_95": selective_acc_ci,
        },
    }
