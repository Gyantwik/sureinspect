"""
core/interfaces.py  --  the shared engine contract (P1 <-> P2).

Frozen together with the API contract. Changing a signature here needs the
`contract-ok` label on the pull request and a heads-up to the whole team.

Functions
---------
standardize(image_bgr)                       -> StandardizedImage      (core.preprocess)
to_blob(image_std, size)                     -> float32 NCHW array     (core.preprocess)
fit_gate_thresholds(images_std)              -> threshold dict         (core.quality_gate, P2)
check_quality(image_std, thresholds)         -> API quality object     (core.quality_gate, P2)
inspect_standardized(image_std, category,
                     flags, alpha_fail,
                     snapshot)               -> inspection result      (core.agent, P1, later)
decide(score, lighting_shift, evidence,
       t_pass, t_fail, budget_state)         -> decision               (core.agent, P1, later)

Rules
-----
* inspect_standardized() receives an ALREADY standardized BGR uint8 image. It never
  resizes it a second time. The backend keeps the original size, supplies the
  standardized size and assembles the metadata. CLI and training code must call
  standardize() first.
* The score is a raw distance, not a probability.
* Fail is `score > t_fail`; fast pass is `score <= t_pass` (full precision, never rounded).
* CLAHE, sharpened or zoomed images are never fed back into the scorer.
* No live-ready artifact may contain stub thresholds (gate_version == "stub").

The two engine functions below are stubs until core/agent.py exists. They raise
NotImplementedError with a clear message instead of returning fake results.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.preprocess import StandardizedImage, standardize, to_blob
from core.quality_gate import check_quality, fit_gate_thresholds

__all__ = [
    "StandardizedImage",
    "standardize",
    "to_blob",
    "fit_gate_thresholds",
    "check_quality",
    "inspect_standardized",
    "decide",
]


def inspect_standardized(
    image_std: np.ndarray,
    category: str,
    flags: Mapping[str, Any],
    alpha_fail: float,
    snapshot: Any,
) -> Any:
    """Run the full inspection on an already standardized BGR uint8 image.

    Delegates to core.agent.inspect_standardized once it exists.
    """
    try:
        from core import agent  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - exercised until agent lands
        raise NotImplementedError(
            "inspect_standardized: the live engine (core/agent.py) is not implemented yet"
        ) from exc
    return agent.inspect_standardized(image_std, category, flags, alpha_fail, snapshot)


def decide(
    score: float,
    lighting_shift: bool,
    evidence: Any,
    t_pass: float,
    t_fail: float,
    budget_state: Any,
) -> Any:
    """Pure decision rule: map score, lighting flag and evidence to the next action/outcome.

    Delegates to core.agent.decide once it exists.
    """
    try:
        from core import agent  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - exercised until agent lands
        raise NotImplementedError(
            "decide: the live engine (core/agent.py) is not implemented yet"
        ) from exc
    return agent.decide(score, lighting_shift, evidence, t_pass, t_fail, budget_state)
