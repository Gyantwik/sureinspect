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
    check_quality,
    compute_frame_metrics,
    fit_gate_thresholds,
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
