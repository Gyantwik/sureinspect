from __future__ import annotations

import cv2
import numpy as np

from eval.run_acquisition_robustness import (
    browser_resize_0729,
    evaluate_records,
    jpeg_q90,
    png_reencode,
)


def test_declared_transforms_have_expected_shapes_and_types():
    image = np.random.default_rng(7).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    assert png_reencode(image).shape == image.shape
    assert jpeg_q90(image).shape == image.shape
    assert browser_resize_0729(image).shape == (73, 146, 3)


def test_png_reencode_is_pixel_exact():
    image = np.random.default_rng(8).integers(0, 256, (31, 47, 3), dtype=np.uint8)
    assert np.array_equal(png_reencode(image), image)


def test_evaluator_rejects_bad_part_escape():
    def row(sample_id, label, variant, decision, expected_path="FAST_PASS"):
        return {
            "sample_id": sample_id,
            "ground_truth": label,
            "variant": variant,
            "decision": decision,
            "decision_path": expected_path,
            "anomaly_score": 1.0,
            "alignment": {"scale": 0.729, "overlap": 0.95},
            "steps_used": 1,
            "step_budget": 4,
        }

    records = []
    for sample_id, label, decision in (
        ("pcb1_borderline_01", "good", "PASS"),
        ("pcb1_bad_missing_01", "bad", "FAIL"),
        ("pcb1_bad_scratch_01", "bad", "HUMAN"),
    ):
        records.extend(
            [
                row(sample_id, label, "sample_id_route", decision),
                row(sample_id, label, "exact_image_bytes", decision),
                row(sample_id, label, "browser_resize_0729", decision),
            ]
        )
    records[-1]["decision"] = "PASS"
    assert any("escaped as PASS" in failure for failure in evaluate_records(records))
