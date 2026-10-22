"""
eval/run_calibration_splits.py
E2: Calibration sensitivity analysis and alpha coverage across held-out splits.
Evaluates:
- Calibration normal scores vs test normal scores for preselected alpha values [0.01, 0.02, 0.05, 0.10].
- Computes empirical exceedance rate: count(test_normal > t_alpha) / n_test_normal.
- Compares empirical rate against nominal alpha with Wilson 95% intervals and Clopper-Pearson exact bounds.
"""
from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from core.calibration import conformal_threshold
from eval.stats import wilson, clopper_pearson


def compute_conformal_threshold(calib_scores: List[float], alpha: float) -> float | None:
    """
    Computes standard conformal quantile threshold by delegating to core.calibration:
    ceil((n + 1) * (1 - alpha)) / n quantile.
    Returns None if k > n (finite sample bound cannot be guaranteed).
    """
    res = conformal_threshold(calib_scores, alpha)
    if res is None:
        return None
    return float(res["threshold"])


def evaluate_calibration_split(
    calib_scores: List[float],
    test_normal_scores: List[float],
    alphas: List[float] = (0.01, 0.02, 0.05, 0.10),
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Evaluates empirical coverage of conformal thresholds on test normals.
    Omits alphas where k > n rather than clipping to n.
    """
    n_calib = len(calib_scores)
    n_test = len(test_normal_scores)
    if n_calib == 0 or n_test == 0:
        raise ValueError("Calibration and test score lists must be non-empty")

    results_by_alpha = {}
    omitted_alphas: List[float] = []
    for a in alphas:
        thr = compute_conformal_threshold(calib_scores, a)
        if thr is None:
            omitted_alphas.append(a)
            continue

        exceedances = sum(1 for s in test_normal_scores if s > thr)
        exceed_rate = exceedances / n_test

        results_by_alpha[f"alpha_{a:.2f}"] = {
            "nominal_alpha": a,
            "threshold": float(thr),
            "n_calib": n_calib,
            "n_test": n_test,
            "exceedances": exceedances,
            "empirical_exceedance_rate": float(exceed_rate),
            "wilson_ci_95": wilson(exceedances, n_test, confidence),
            "clopper_pearson_ci": clopper_pearson(exceedances, n_test, confidence),
        }

    return {
        "n_calib": n_calib,
        "n_test": n_test,
        "alpha_evaluations": results_by_alpha,
        "omitted_alphas": omitted_alphas,
    }
