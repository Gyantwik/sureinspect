"""
core/calibration.py
Split-conformal thresholds from held-out NORMAL scores.

For n calibration scores and a target level alpha:
    k = ceil((n + 1) * (1 - alpha))
    threshold = the k-th smallest score  (sorted index k - 1)
    if k > n the alpha is OMITTED (never clipped to the maximum).

Decision convention (full precision, never rounded):
    fail      : score >  t_fail
    fast pass : score <= t_pass

Claim (assumption dependent): for a new normal image exchangeable with the calibration
normals, P(score > t) <= alpha. It does not hold under lighting shift or other distribution shift.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def conformal_k(n: int, alpha: float) -> int:
    if n <= 0:
        raise ValueError("need at least one calibration score")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    # round() guards float noise such as 100 * 0.99 = 99.00000000000001
    return int(math.ceil(round((n + 1) * (1.0 - alpha), 9)))


def conformal_threshold(scores: Iterable[float], alpha: float) -> dict | None:
    """-> {"alpha", "k", "index", "threshold", "n"} or None when k > n (alpha omitted)."""
    s = np.sort(np.asarray(list(scores), dtype=np.float64))
    n = len(s)
    k = conformal_k(n, alpha)
    if k > n:
        return None
    return {"alpha": float(alpha), "k": k, "index": k - 1, "threshold": float(s[k - 1]), "n": n}


def fit_thresholds(scores: Sequence[float], alpha_pass: float, alpha_fail_options: Sequence[float]) -> dict:
    """Fit t_pass and every allowed t_fail option; omitted options are listed, not clipped."""
    tp = conformal_threshold(scores, alpha_pass)
    if tp is None:
        raise ValueError(f"alpha_pass={alpha_pass} is not available with n={len(list(scores))} calibration scores")
    fail, omitted = {}, []
    for a in alpha_fail_options:
        t = conformal_threshold(scores, a)
        if t is None:
            omitted.append(float(a))
        else:
            fail[f"{a:g}"] = t
    return {"t_pass": tp, "t_fail_options": fail, "omitted_alpha_fail": omitted}
