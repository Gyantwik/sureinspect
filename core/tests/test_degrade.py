"""
core/tests/test_degrade.py
Comprehensive unit tests for SureInspect image degradation module.

Verifies:
- Determinism: same image + same flags -> byte-identical output
- Input unmodified: input image array is never mutated
- Each effect measurably changes the image
- Glare seed depends strictly on image content hash
- Applying effects once only
- Degrade -> quality gate integration: strong blur, darken, glare each trigger RETAKE
- No external network or dataset dependencies
"""
import cv2
import numpy as np

from core.degrade import apply_degradations, degrade
from core.quality_gate import check_quality, fit_gate_thresholds


def make_test_frame(seed: int = 100) -> np.ndarray:
    """Generates a synthetic textured frame."""
    rng = np.random.RandomState(seed)
    base = rng.randint(90, 170, (256, 256, 3), dtype=np.uint8)
    for _ in range(10):
        pt1 = (int(rng.randint(0, 256)), int(rng.randint(0, 256)))
        pt2 = (int(rng.randint(0, 256)), int(rng.randint(0, 256)))
        color = (int(rng.randint(50, 220)), int(rng.randint(50, 220)), int(rng.randint(50, 220)))
        cv2.line(base, pt1, pt2, color, thickness=2)
    return base


def test_determinism_twice_produces_identical_bytes():
    """Applying degradations twice to identical input produces byte-identical output."""
    img = make_test_frame(seed=101)

    out1, eff1 = apply_degradations(img, blur=True, darken=True, glare=True)
    out2, eff2 = apply_degradations(img, blur=True, darken=True, glare=True)

    assert eff1 == eff2 == ["blur", "darken", "glare"]
    assert np.array_equal(out1, out2)
    assert out1.tobytes() == out2.tobytes()


def test_input_unmodified():
    """Degrade operations must never mutate the input numpy array."""
    img = make_test_frame(seed=102)
    original_bytes = img.tobytes()

    _out, _eff = apply_degradations(img, blur=True, darken=True, glare=True)

    assert img.tobytes() == original_bytes, "Input image array was mutated during degradation"


def test_each_effect_measurably_changes_image():
    """Each individual degradation flag measurably alters image pixels."""
    img = make_test_frame(seed=103)

    # Blur
    blurred, eff_b = apply_degradations(img, blur=True)
    assert eff_b == ["blur"]
    assert not np.array_equal(blurred, img)
    assert np.mean(np.abs(blurred.astype(float) - img.astype(float))) > 2.0

    # Darken
    darkened, eff_d = apply_degradations(img, darken=True)
    assert eff_d == ["darken"]
    assert not np.array_equal(darkened, img)
    assert np.mean(darkened) < np.mean(img)

    # Glare
    glared, eff_g = apply_degradations(img, glare=True)
    assert eff_g == ["glare"]
    assert not np.array_equal(glared, img)
    assert np.max(glared) == 255


def test_glare_seed_depends_on_content():
    """Glare position and mask depend on image content SHA-256 hash."""
    img1 = np.full((256, 256, 3), 100, dtype=np.uint8)
    img2 = np.full((256, 256, 3), 150, dtype=np.uint8)

    glared1, _ = apply_degradations(img1, glare=True)
    glared2, _ = apply_degradations(img2, glare=True)

    # Residuals from original image isolate the glare overlay
    diff1 = glared1.astype(float) - img1.astype(float)
    diff2 = glared2.astype(float) - img2.astype(float)

    assert not np.array_equal(diff1, diff2), "Glare blob pattern should vary with image content hash"


def test_applying_effects_once_only():
    """All requested effects are applied cleanly in a single pass."""
    img = make_test_frame(seed=104)

    out, effects = degrade(img, blur=True, darken=False, glare=True)
    assert effects == ["blur", "glare"]
    assert not np.array_equal(out, img)


def test_degrade_to_gate_integration_retake():
    """Strong blur, darken, and glare each trigger RETAKE (status=FAILED) from the quality gate."""
    normals = [make_test_frame(seed=i) for i in range(30)]
    thresholds = fit_gate_thresholds(normals)

    test_img = normals[0]

    # 1. Blur -> RETAKE
    blurred, _ = degrade(test_img, blur=True)
    q_blur = check_quality(blurred, thresholds)
    assert q_blur["status"] == "FAILED"
    assert q_blur["action"] == "REQUEST_RETAKE"
    assert "Laplacian variance" in q_blur["reason"]

    # 2. Darken -> RETAKE
    darkened, _ = degrade(test_img, darken=True)
    q_dark = check_quality(darkened, thresholds)
    assert q_dark["status"] == "FAILED"
    assert q_dark["action"] == "REQUEST_RETAKE"

    # 3. Glare -> RETAKE
    glared, _ = degrade(test_img, glare=True)
    q_glare = check_quality(glared, thresholds)
    assert q_glare["status"] == "FAILED"
    assert q_glare["action"] == "REQUEST_RETAKE"
    assert "Glare pixel fraction" in q_glare["reason"]
