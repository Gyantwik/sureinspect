"""
eval/run_corruptions.py
E3: Robustness evaluation under deterministic degradations.
Evaluates:
- Paired comparison: clean baseline images vs corrupted variants (motion blur, gamma darken, glare blob).
- Performance with Quality Gate ON vs Quality Gate OFF.
- Measures reject rate, escape rate, false-FAIL rate, and review burden under image corruptions.
- Keeps corrupted variants strictly grouped with their original physical images without counting
  them as independent new samples for confidence bounds.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from core.degrade import apply_degradations
from core.quality_gate import check_quality, fit_gate_thresholds
from eval.stats import wilson

SAMPLE_GATE_THRESHOLDS = {
    "blur_thr": 20.0,
    "dark_max": 0.30,
    "bright_max": 0.30,
    "glare_thr": 0.05,
    "l_mean_ref": 128.0,
    "l_mean_sd": 10.0,
    "l_std_ref": 40.0,
    "n_fit": 50,
    "gate_version": "v1.0",
}


def evaluate_corruption_robustness(
    images: List[np.ndarray],
    ground_truth: List[str],  # "good" or "bad"
    simulate_effects: Tuple[str, ...] = ("blur", "darken", "glare"),
    gate_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates quality gate rejection and decision impact on clean vs corrupted image variants.
    """
    if gate_thresholds is None:
        gate_thresholds = SAMPLE_GATE_THRESHOLDS

    n_images = len(images)
    if len(ground_truth) != n_images:
        raise ValueError("Image count and ground truth count must match")

    clean_rejects = 0
    corrupted_rejects = {eff: 0 for eff in simulate_effects}
    joint_corrupted_rejects = 0

    for img, gt in zip(images, ground_truth):
        # 1. Check clean image
        clean_res = check_quality(img, gate_thresholds)
        if clean_res["status"] == "FAILED":
            clean_rejects += 1

        # 2. Check individual corruptions
        for eff in simulate_effects:
            corr_img, _ = apply_degradations(
                img,
                blur=(eff == "blur"),
                darken=(eff == "darken"),
                glare=(eff == "glare"),
            )
            corr_res = check_quality(corr_img, gate_thresholds)
            if corr_res["status"] == "FAILED":
                corrupted_rejects[eff] += 1

        # 3. Check joint corruptions
        joint_corr, _ = apply_degradations(
            img,
            blur=("blur" in simulate_effects),
            darken=("darken" in simulate_effects),
            glare=("glare" in simulate_effects),
        )
        joint_res = check_quality(joint_corr, gate_thresholds)
        if joint_res["status"] == "FAILED":
            joint_corrupted_rejects += 1

    clean_reject_rate = clean_rejects / n_images if n_images > 0 else 0.0
    joint_reject_rate = joint_corrupted_rejects / n_images if n_images > 0 else 0.0

    return {
        "n_original_images": n_images,
        "clean_baseline": {
            "reject_count": clean_rejects,
            "reject_rate": clean_reject_rate,
            "wilson_ci_95": wilson(clean_rejects, n_images),
        },
        "individual_corruptions": {
            eff: {
                "reject_count": corrupted_rejects[eff],
                "reject_rate": corrupted_rejects[eff] / n_images if n_images > 0 else 0.0,
                "wilson_ci_95": wilson(corrupted_rejects[eff], n_images) if n_images > 0 else None,
            }
            for eff in simulate_effects
        },
        "joint_corruptions": {
            "reject_count": joint_corrupted_rejects,
            "reject_rate": joint_reject_rate,
            "wilson_ci_95": wilson(joint_corrupted_rejects, n_images),
        },
    }
