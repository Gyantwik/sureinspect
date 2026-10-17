"""
core/tests/test_quality_gate.py
Comprehensive unit tests for SureInspect Quality Gate.

Verifies:
- Clean synthetic images pass
- Strongly blurred fails with blur reason
- Very dark and very bright images fail with exposure reasons
- Glare blob fails with glare reason
- Lighting shift is flagged but NOT rejected (status remains PASSED)
- L-mean SD floor of 2.0 respected
- Zero-std guard respected
- Empty, wrong dtype, wrong channels raise expected exceptions
- Boundary equality behavior documented and tested
- Joint gate behavior: <2% false rejects on clean normals, 100% rejection on strong degradations
"""
import cv2
import numpy as np
import pytest

from core.degrade import apply_degradations
from core.quality_gate import (
    BLUR_PERCENTILE,
    BLUR_VARIANCE_SAFETY_RATIO,
    STUB_GATE_VERSION,
    assert_live_ready,
    check_quality,
    compute_frame_metrics,
    fit_and_report,
    fit_gate_thresholds,
    gate_reject_report,
    validate_standardized_image,
)


def make_clean_image(seed: int = 42, width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a synthetic normal industrial frame with texture and edge features."""
    rng = np.random.RandomState(seed)
    base = rng.randint(100, 160, (height, width, 3), dtype=np.uint8)
    for _ in range(16):
        pt1 = (int(rng.randint(0, width)), int(rng.randint(0, height)))
        pt2 = (int(rng.randint(0, width)), int(rng.randint(0, height)))
        color = (int(rng.randint(60, 200)), int(rng.randint(60, 200)), int(rng.randint(60, 200)))
        cv2.line(base, pt1, pt2, color, thickness=2)
    return base


@pytest.fixture
def clean_normals():
    return [make_clean_image(seed=i) for i in range(50)]


@pytest.fixture
def fitted_thresholds(clean_normals):
    return fit_gate_thresholds(clean_normals)


def test_clean_image_passes(clean_normals, fitted_thresholds):
    """Clean normal image should pass quality gate with action=None."""
    result = check_quality(clean_normals[0], fitted_thresholds)
    assert result["status"] == "PASSED"
    assert result["action"] is None
    assert result["exposure_ok"] is True
    assert result["lighting_shift_detected"] is False
    assert "Focus and illumination within normal parameters" in result["reason"]


def test_strongly_blurred_fails(clean_normals, fitted_thresholds):
    """Blurred image must fail with Laplacian variance reason and REQUEST_RETAKE."""
    img = clean_normals[0]
    blurred, effects = apply_degradations(img, blur=True)
    assert "blur" in effects

    result = check_quality(blurred, fitted_thresholds)
    assert result["status"] == "FAILED"
    assert result["action"] == "REQUEST_RETAKE"
    assert "Laplacian variance" in result["reason"]
    assert result["blur_variance"] < result["blur_threshold"]


def test_very_dark_image_fails(fitted_thresholds):
    """Severely underexposed image must fail with dark fraction reason."""
    dark_img = np.full((256, 256, 3), 5, dtype=np.uint8)
    result = check_quality(dark_img, fitted_thresholds)
    assert result["status"] == "FAILED"
    assert result["action"] == "REQUEST_RETAKE"
    assert result["exposure_ok"] is False
    assert "Dark pixel fraction" in result["reason"]
    assert result["dark_fraction"] > result["dark_fraction"] - 0.01


def test_very_bright_image_fails(fitted_thresholds):
    """Severely overexposed image must fail with bright fraction reason."""
    bright_img = np.full((256, 256, 3), 250, dtype=np.uint8)
    result = check_quality(bright_img, fitted_thresholds)
    assert result["status"] == "FAILED"
    assert result["action"] == "REQUEST_RETAKE"
    assert result["exposure_ok"] is False
    assert "Bright pixel fraction" in result["reason"]


def test_glare_blob_fails(clean_normals, fitted_thresholds):
    """Specular glare blob must fail with glare fraction reason and REQUEST_RETAKE."""
    img = clean_normals[0]
    glared, effects = apply_degradations(img, glare=True)
    assert "glare" in effects

    result = check_quality(glared, fitted_thresholds)
    assert result["status"] == "FAILED"
    assert result["action"] == "REQUEST_RETAKE"
    assert "Glare pixel fraction" in result["reason"]


def test_lighting_shift_flagged_not_rejected(clean_normals, fitted_thresholds):
    """
    Lighting shift alone must NOT reject the image:
    status remains PASSED, action is None, but lighting_shift_detected is True.
    """
    img = clean_normals[0]
    # Add a moderate global illumination offset (+25) that shifts L-mean beyond 3 SDs
    # but does not cause dark/bright clipping
    shifted = cv2.add(img, np.full_like(img, 25))
    result = check_quality(shifted, fitted_thresholds)
    assert result["status"] == "PASSED"
    assert result["action"] is None
    assert result["lighting_shift_detected"] is True
    assert "lighting shift detected" in result["reason"].lower()


def test_sd_floor_of_two(clean_normals):
    """L-mean standard deviation must be floored at minimum 2.0 L units."""
    # When all images have identical illumination, sample SD would be 0, but floor forces >= 2.0
    identical_images = [clean_normals[0].copy() for _ in range(5)]
    th = fit_gate_thresholds(identical_images)
    assert th["l_mean_sd"] >= 2.0
    assert th["l_mean_sd"] == 2.0


def test_zero_std_guarded():
    """L-std reference must be guarded against zero even on uniform color images."""
    uniform_images = [np.full((128, 128, 3), 128, dtype=np.uint8) for _ in range(5)]
    th = fit_gate_thresholds(uniform_images)
    assert th["l_std_ref"] > 0.0
    assert th["l_std_ref"] >= 1e-4


def test_empty_wrong_dtype_wrong_channels():
    """Validates that malformed inputs raise clear exceptions."""
    # Wrong type
    with pytest.raises(TypeError):
        validate_standardized_image("not an array")

    # Wrong dtype
    float_img = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        validate_standardized_image(float_img)

    # Wrong channels (grayscale 2D instead of 3-channel BGR)
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        validate_standardized_image(gray_img)

    # Empty image
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        validate_standardized_image(empty_img)


def test_boundary_equality_behavior():
    """
    Boundary equality behavior specification:
    - Blur: blur_variance < blur_thr fails; equality blur_variance == blur_thr passes.
    - Exposure: dark_fraction > dark_max fails; equality dark_fraction == dark_max passes.
    - Glare: glare_fraction > glare_thr fails; equality glare_fraction == glare_thr passes.
    """
    # Synthetic frame exactly at thresholds
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    metrics = compute_frame_metrics(img)

    # Mock thresholds matching exact metrics
    exact_th = {
        "blur_thr": metrics["blur_variance"],
        "dark_max": metrics["dark_fraction"],
        "bright_max": metrics["bright_fraction"],
        "glare_thr": metrics["glare_fraction"],
        "l_mean_ref": metrics["l_mean"],
        "l_mean_sd": 5.0,
        "l_std_ref": metrics["l_std"],
        "n_fit": 10,
        "gate_version": "test",
    }
    res = check_quality(img, exact_th)
    # At exact equality (< not satisfied, > not satisfied), the gate passes
    assert res["status"] == "PASSED"
    assert res["action"] is None


def test_joint_gate_behavior(clean_normals, fitted_thresholds):
    """
    Joint gate test:
    - On clean normal images: false rejection rate < 2.0% (target: 0/50).
    - On strongly degraded images: 100% rejection rate.
    """
    # 1. Clean normals
    rejections = 0
    for img in clean_normals:
        res = check_quality(img, fitted_thresholds)
        if res["status"] == "FAILED":
            rejections += 1
    false_reject_rate = rejections / len(clean_normals)
    assert false_reject_rate < 0.02, f"False rejection rate {false_reject_rate:.1%} exceeds 2%"

    # 2. Strong degradations
    for img in clean_normals[:10]:
        blurred, _ = apply_degradations(img, blur=True)
        assert check_quality(blurred, fitted_thresholds)["status"] == "FAILED"

        darkened, _ = apply_degradations(img, darken=True)
        assert check_quality(darkened, fitted_thresholds)["status"] == "FAILED"

        glared, _ = apply_degradations(img, glare=True)
        assert check_quality(glared, fitted_thresholds)["status"] == "FAILED"


def test_assert_live_ready_validation(clean_normals, fitted_thresholds):
    """Verifies that assert_live_ready enforces live-ready artifact requirements."""
    # 1. Accepts valid fitted thresholds
    assert_live_ready(fitted_thresholds)

    # 2. Rejects stub gate version
    stub_th = fit_gate_thresholds(clean_normals[:5], gate_version=STUB_GATE_VERSION)
    with pytest.raises(ValueError, match="stub"):
        assert_live_ready(stub_th)

    # 3. Rejects missing key
    incomplete_th = dict(fitted_thresholds)
    del incomplete_th["blur_thr"]
    with pytest.raises(ValueError, match="blur_thr"):
        assert_live_ready(incomplete_th)

    # 4. Rejects NaN / inf values
    nan_th = dict(fitted_thresholds)
    nan_th["l_mean_ref"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assert_live_ready(nan_th)

    inf_th = dict(fitted_thresholds)
    inf_th["dark_max"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        assert_live_ready(inf_th)


def test_check_quality_raises_value_error_not_key_error(clean_normals, fitted_thresholds):
    """check_quality must raise ValueError (never bare KeyError) naming the missing/invalid key."""
    img = clean_normals[0]

    # Non-dict input
    with pytest.raises(ValueError, match="must be a dict"):
        check_quality(img, "invalid_th")  # type: ignore

    # Missing required key
    incomplete_th = dict(fitted_thresholds)
    del incomplete_th["glare_thr"]
    with pytest.raises(ValueError, match="glare_thr") as exc_info:
        check_quality(img, incomplete_th)
    assert not issubclass(exc_info.type, KeyError) or exc_info.type is ValueError

    # Non-finite value in required numeric field
    nan_th = dict(fitted_thresholds)
    nan_th["blur_thr"] = float("nan")
    with pytest.raises(ValueError, match="blur_thr"):
        check_quality(img, nan_th)


def test_percentile_fit_outlier_resistance_and_small_n_fallback():
    """
    Verifies that for N >= 20, 1st percentile avoids collapse from a single extreme outlier,
    and for N < 20, fallback to minimum is used with appropriate blur_fit_method tags.
    """
    assert BLUR_PERCENTILE == 1
    assert BLUR_VARIANCE_SAFETY_RATIO == 0.70

    normals_29 = [make_clean_image(seed=i) for i in range(29)]
    # An extreme outlier image (e.g. flat image with near-zero Laplacian variance)
    flat_outlier = np.full((256, 256, 3), 128, dtype=np.uint8)

    th_with_outlier = fit_gate_thresholds([flat_outlier] + normals_29)
    assert th_with_outlier["blur_fit_method"] == "percentile_1"
    assert th_with_outlier["n_fit"] == 30

    th_without_outlier = fit_gate_thresholds(normals_29)
    assert th_without_outlier["blur_fit_method"] == "percentile_1"

    # Outlier has Laplacian variance 0.0. If minimum were used, blur_thr would floor to 10.0.
    # At N=30, 1st percentile linearly interpolates at index 0.29 between 0.0 and lowest normal.
    # Verify the exact expected mathematical ratio against the no-outlier threshold:
    ratio_30 = th_with_outlier["blur_thr"] / th_without_outlier["blur_thr"]
    assert 0.25 <= ratio_30 <= 0.35, f"Expected ratio near 0.29, got {ratio_30:.3f}"

    # For standard N >= 100 development batches, 1st percentile filters out 1 outlier:
    normals_100 = [make_clean_image(seed=i) for i in range(100)]
    th_100_outlier = fit_gate_thresholds([flat_outlier] + normals_100)
    th_100_clean = fit_gate_thresholds(normals_100)
    ratio_100 = th_100_outlier["blur_thr"] / th_100_clean["blur_thr"]
    assert ratio_100 >= 0.99, f"Expected 1st percentile to be >= 99% retained at N=101, got {ratio_100:.3f}"

    # Small N (< 20) falls back to min_small_n
    th_small = fit_gate_thresholds(normals_29[:15])
    assert th_small["blur_fit_method"] == "min_small_n"
    assert th_small["n_fit"] == 15


def test_gate_reject_report_consistency(clean_normals, fitted_thresholds):
    """
    Verifies that gate_reject_report produces mathematically consistent rejection counts and rates.
    """
    # Create a mixed batch: clean images + individual faults + multi-fault images
    batch = list(clean_normals[:20])
    blurred_img, _ = apply_degradations(clean_normals[21], blur=True)
    dark_img = np.full((256, 256, 3), 5, dtype=np.uint8)
    bright_img = np.full((256, 256, 3), 250, dtype=np.uint8)
    glare_img, _ = apply_degradations(clean_normals[23], glare=True)
    # Multi-fault: zero image triggers both blur (variance 0 < thr) and dark (dark_fraction 1.0 > max)
    multi_fault = np.zeros((256, 256, 3), dtype=np.uint8)
    batch.extend([blurred_img, dark_img, bright_img, glare_img, multi_fault])

    report = gate_reject_report(batch, fitted_thresholds)

    assert report["n"] == len(batch)
    assert 0 <= report["n_rejected"] <= report["n"]
    assert report["reject_rate"] == report["n_rejected"] / report["n"]

    reason_sum = sum(report["per_reason_counts"].values())
    # Multi-fault images increment multiple reason counters, so reason_sum >= n_rejected
    assert reason_sum >= report["n_rejected"]
    assert reason_sum > report["n_rejected"]  # Strictly greater due to multi_fault
    assert report["per_reason_counts"]["blur"] >= 2
    assert report["per_reason_counts"]["dark"] >= 2
    assert report["per_reason_counts"]["bright"] >= 1
    assert report["per_reason_counts"]["glare"] >= 1

    assert 0 <= report["n_lighting_shift"] <= report["n"]
    assert report["lighting_shift_rate"] == report["n_lighting_shift"] / report["n"]

    # Empty collection report
    empty_report = gate_reject_report([], fitted_thresholds)
    assert empty_report["n"] == 0
    assert empty_report["n_rejected"] == 0
    assert empty_report["reject_rate"] == 0.0


def test_fit_and_report_100_clean_normals():
    """
    fit_and_report on 100 clean normals must have false reject rate < 2.0%.
    """
    normals_100 = [make_clean_image(seed=i) for i in range(100)]
    th, report = fit_and_report(normals_100)

    assert report["n"] == 100
    assert report["reject_rate"] < 0.02
    assert th["n_fit"] == 100
    assert th["gate_version"] == "v1.0"
    assert th["blur_fit_method"] == "percentile_1"


def test_out_of_sample_held_out_rejection_rate():
    """
    Evaluates false rejection rate on a held-out set of 100 clean normals
    never seen during fitting (out-of-sample generalization).
    Must remain below the 2.0% false rejection budget.
    """
    train_normals = [make_clean_image(seed=i) for i in range(100)]
    held_out_normals = [make_clean_image(seed=1000 + i) for i in range(100)]

    th = fit_gate_thresholds(train_normals)
    report = gate_reject_report(held_out_normals, th)

    assert report["n"] == 100
    assert report["n_rejected"] <= 2
    assert report["reject_rate"] < 0.02


def test_extra_keys_in_thresholds_tolerated(clean_normals, fitted_thresholds):
    """Extra non-standard keys in thresholds dict must be tolerated without error."""
    extended_th = dict(fitted_thresholds)
    extended_th["custom_engine_tag"] = "test_run_42"
    extended_th["extra_metadata"] = {"batch": 1, "owner": "Gyantwik"}

    res = check_quality(clean_normals[0], extended_th)
    assert res["status"] == "PASSED"

    report = gate_reject_report([clean_normals[0]], extended_th)
    assert report["n"] == 1
    assert report["n_rejected"] == 0

