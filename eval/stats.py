"""Two-sided binomial intervals; n=0 is unavailable, not a fabricated zero rate."""
import math
from scipy.stats import beta, norm


def _validate(k, n, confidence):
    if not isinstance(k, int) or not isinstance(n, int):
        raise ValueError("k and n must be integers")
    if n < 0 or k < 0 or k > n or not 0 < confidence < 1:
        raise ValueError("Invalid count or confidence")


def wilson(k, n, confidence=0.95):
    _validate(k, n, confidence)
    if n == 0:
        return None
    z = float(norm.ppf((1 + confidence) / 2))
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson(k, n, confidence=0.95):
    _validate(k, n, confidence)
    if n == 0:
        return None
    tail = (1 - confidence) / 2
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lower, upper
