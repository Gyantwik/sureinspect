"""
eval/run_agent.py
E1: Official frozen agent evaluation harness and comparison metrics.
Computes:
- Coverage: (PASS + FAIL) / n_total
- Rejection rate: RETAKE / n_total
- Review burden: HUMAN / n_total
- Defect escapes: defective PASS / n_defective (split by FAST_PASS vs AGENT_CLEARED)
- Automatic defect recall: defective FAIL / n_defective
- Normal false-FAIL rate: normal FAIL / n_normal
- Selective accuracy: correct automated decisions / (PASS + FAIL)
- Wilson 95% intervals and Clopper-Pearson exact confidence intervals.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from eval.stats import wilson, clopper_pearson


def evaluate_agent(
    decisions: List[str],        # "PASS", "FAIL", "HUMAN", "RETAKE"
    paths: List[str],            # DecisionPathEnum string or path identifier
    ground_truth: List[str],     # "good" or "bad"
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Evaluates agent outcomes against ground-truth labels.
    """
    n = len(decisions)
    if len(ground_truth) != n or len(paths) != n:
        raise ValueError("Decisions, paths, and ground_truth lists must have identical lengths")

    if n == 0:
        return {"n_total": 0, "coverage": None, "defect_escapes": None}

    counts = {
        "normal_pass_fast": 0,
        "normal_pass_cleared": 0,
        "normal_fail": 0,
        "normal_human": 0,
        "normal_retake": 0,
        "defective_pass_fast": 0,
        "defective_pass_cleared": 0,
        "defective_fail": 0,
        "defective_human": 0,
        "defective_retake": 0,
    }

    for dec, path, gt in zip(decisions, paths, ground_truth):
        is_good = gt.lower() in ("good", "normal")
        prefix = "normal" if is_good else "defective"
        dec_upper = dec.upper()
        path_upper = str(path).upper()

        if dec_upper == "PASS":
            if "FAST" in path_upper:
                counts[f"{prefix}_pass_fast"] += 1
            else:
                counts[f"{prefix}_pass_cleared"] += 1
        elif dec_upper == "FAIL":
            counts[f"{prefix}_fail"] += 1
        elif dec_upper == "HUMAN":
            counts[f"{prefix}_human"] += 1
        elif dec_upper == "RETAKE":
            counts[f"{prefix}_retake"] += 1

    n_normal = sum(counts[k] for k in counts if k.startswith("normal"))
    n_defective = sum(counts[k] for k in counts if k.startswith("defective"))

    normal_pass = counts["normal_pass_fast"] + counts["normal_pass_cleared"]
    defective_pass = counts["defective_pass_fast"] + counts["defective_pass_cleared"]
    automated_decisions = normal_pass + counts["normal_fail"] + defective_pass + counts["defective_fail"]

    # Coverage: (PASS + FAIL) / n_total
    cov_rate = automated_decisions / n
    # Defect Escapes: defective PASS / n_defective
    esc_rate = defective_pass / n_defective if n_defective > 0 else None
    # Defect Recall: defective FAIL / n_defective
    recall_rate = counts["defective_fail"] / n_defective if n_defective > 0 else None
    # Normal False-FAIL: normal FAIL / n_normal
    false_fail_rate = counts["normal_fail"] / n_normal if n_normal > 0 else None
    # Review Burden: HUMAN / n_total
    human_rate = (counts["normal_human"] + counts["defective_human"]) / n
    # Retake rate: RETAKE / n_total
    retake_rate = (counts["normal_retake"] + counts["defective_retake"]) / n
    # Selective Accuracy: (normal_pass + defective_fail) / automated_decisions
    selective_acc = (normal_pass + counts["defective_fail"]) / automated_decisions if automated_decisions > 0 else None

    return {
        "n_total": n,
        "n_normal": n_normal,
        "n_defective": n_defective,
        "counts": counts,
        "coverage": {
            "rate": cov_rate,
            "numerator": automated_decisions,
            "denominator": n,
            "ci_95": wilson(automated_decisions, n, confidence),
        },
        "defect_escapes": {
            "rate": esc_rate,
            "numerator": defective_pass,
            "denominator": n_defective,
            "fast_pass_escapes": counts["defective_pass_fast"],
            "agent_cleared_escapes": counts["defective_pass_cleared"],
            "ci_95": wilson(defective_pass, n_defective, confidence) if n_defective > 0 else None,
            "exact_ci": clopper_pearson(defective_pass, n_defective, confidence) if n_defective > 0 else None,
        },
        "defect_recall": {
            "rate": recall_rate,
            "numerator": counts["defective_fail"],
            "denominator": n_defective,
            "ci_95": wilson(counts["defective_fail"], n_defective, confidence) if n_defective > 0 else None,
        },
        "false_fail": {
            "rate": false_fail_rate,
            "numerator": counts["normal_fail"],
            "denominator": n_normal,
            "ci_95": wilson(counts["normal_fail"], n_normal, confidence) if n_normal > 0 else None,
        },
        "review_burden": {
            "rate": human_rate,
            "numerator": counts["normal_human"] + counts["defective_human"],
            "denominator": n,
            "ci_95": wilson(counts["normal_human"] + counts["defective_human"], n, confidence),
        },
        "retake_rate": {
            "rate": retake_rate,
            "numerator": counts["normal_retake"] + counts["defective_retake"],
            "denominator": n,
            "ci_95": wilson(counts["normal_retake"] + counts["defective_retake"], n, confidence),
        },
        "selective_accuracy": {
            "rate": selective_acc,
            "numerator": normal_pass + counts["defective_fail"],
            "denominator": automated_decisions,
            "ci_95": wilson(normal_pass + counts["defective_fail"], automated_decisions, confidence) if automated_decisions > 0 else None,
        },
    }
