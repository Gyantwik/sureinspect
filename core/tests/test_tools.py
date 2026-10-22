"""Tests for core/tools.py (evidence tools). Synthetic images only; no model needed."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import tools as T


def textured(h=600, w=800, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)
    for _ in range(500):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 20)),
                   int(rng.integers(30, 255)), -1)
    for _ in range(60):
        cv2.line(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                 (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(60, 255)), 2)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def true_transform(angle=30.0, shift=(25.0, -15.0), shape=(600, 800)):
    """Reference -> test: rotate about the image centre, then translate."""
    c = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    r = cv2.getRotationMatrix2D(c, angle, 1.0)
    r[:, 2] += shift
    return T.as_h(r)


def make_pair(angle=30.0, shift=(25.0, -15.0)):
    ref = textured()
    m = true_transform(angle, shift, ref.shape)
    test = cv2.warpAffine(ref, m[:2], (ref.shape[1], ref.shape[0]))
    return ref, test, m


def mean_point_error(m_est, m_true, shape=(600, 800)):
    pts = np.array([[x, y, 1.0] for x in range(100, shape[1] - 100, 150) for y in range(100, shape[0] - 100, 150)]).T
    return float(np.linalg.norm((m_est @ pts - m_true @ pts)[:2], axis=0).mean())


# ------------------------------------------------------------------ transforms

def test_to_full_inverts_the_reduction_formula():
    rng = np.random.default_rng(0)
    m_full = T.as_h(np.array([[0.9, -0.3, 12.0], [0.3, 0.9, -7.0]]))
    s_t, s_r = T.scale_matrix(0.25), T.scale_matrix(0.5)
    m_small = s_t @ m_full @ np.linalg.inv(s_r)
    assert np.allclose(T.to_full(m_small, s_t, s_r), m_full)
    assert rng is not None


def test_transform_direction_reference_to_test_reproduces_test_image():
    ref, test, m = make_pair()
    warped, valid = T.warp_reference(ref, m, test.shape)
    assert T.alignment_quality(test, warped, valid) > 0.95
    # the inverse direction must NOT reproduce it
    wrong, vw = T.warp_reference(ref, np.linalg.inv(m), test.shape)
    assert T.alignment_quality(test, wrong, vw) < 0.5


# ------------------------------------------------------------------------ T1

def test_orb_recovers_30_degree_rotation_plus_translation():
    ref, test, m_true = make_pair()
    r = T.align(test, ref)
    assert r.valid, r.reason
    assert r.method == "orb_ransac" and r.n_inliers >= T.MIN_INLIERS
    assert mean_point_error(r.M, m_true) < 1.5
    # rotation_deg is atan2(M10, M00) in image coordinates (y down); OpenCV's +30 deg getRotationMatrix2D gives -30
    assert abs(r.rotation_deg + 30.0) < 1.0 and abs(r.scale - 1.0) < 0.02
    assert r.quality > 0.95


def test_rotation_search_plus_ecc_fallback_recovers_the_same_transform():
    ref, test, m_true = make_pair()
    r = T._align_rotation_ecc(test, ref)
    assert r.valid, r.reason
    assert mean_point_error(r.M, m_true) < 3.0
    assert r.quality > 0.9


def test_alignment_is_deterministic():
    ref, test, _ = make_pair()
    a, b = T.align(test, ref), T.align(test, ref)
    assert np.array_equal(a.M, b.M)


def test_resolution_scaling_gives_a_full_resolution_transform():
    ref, test, m_true = make_pair()
    ref_s = cv2.resize(ref, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    test_s = cv2.resize(test, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    r = T.align(test_s, ref_s)
    assert r.valid, r.reason
    full = T.to_full(r.M, T.scale_matrix(0.5), T.scale_matrix(0.5))
    assert mean_point_error(full, m_true) < 3.0


def test_alignment_recovers_browser_screenshot_scale():
    """A display-resampled 0.729x frame remains inside the bounded recovery envelope."""
    ref = textured()
    test = cv2.resize(ref, None, fx=0.729, fy=0.729, interpolation=cv2.INTER_AREA)
    r = T.align(test, ref)
    assert r.valid, r.reason
    assert r.scale == pytest.approx(0.729, abs=0.02)
    assert r.quality is not None and r.quality > 0.90
    assert r.overlap is not None and r.overlap > 0.95


@pytest.mark.parametrize("scale", [0.59, 1.61])
def test_alignment_rejects_scale_outside_recovery_envelope(scale):
    ref = textured()
    m = T.as_h(cv2.getRotationMatrix2D(((ref.shape[1] - 1) / 2, (ref.shape[0] - 1) / 2), 0, scale))
    result = T._finalize(m, ref, ref, "test", q_min=None)
    assert not result.valid
    assert "scale" in result.reason


def test_low_overlap_is_rejected_with_a_reason():
    ref = textured()
    far = T.as_h(np.array([[1, 0, 700.0], [0, 1, 0.0]]))
    res = T._finalize(far, ref, ref, "test")
    assert not res.valid and "overlap" in res.reason


def test_implausible_scale_rotation_and_nonfinite_are_rejected():
    ref = textured()
    big = T.as_h(np.array([[2.0, 0, 0], [0, 2.0, 0]]))
    assert "scale" in T._finalize(big, ref, ref, "t").reason
    rot = T.as_h(cv2.getRotationMatrix2D((400, 300), 80.0, 1.0))
    assert "rotation" in T._finalize(rot, ref, ref, "t").reason
    bad = np.eye(3)
    bad[0, 2] = np.nan
    assert "non-finite" in T._finalize(bad, ref, ref, "t").reason
    assert not T._finalize(None, ref, ref, "t").valid


def test_featureless_images_fail_gracefully_without_exceptions():
    blank = np.full((300, 400), 90, np.uint8)
    r = T.align(blank, blank)
    assert isinstance(r.valid, bool)               # either result is acceptable; no crash, reason recorded
    assert r.valid or r.reason


def test_align_to_templates_prefers_the_valid_template_and_applies_q_min():
    ref, test, _ = make_pair()
    junk = np.full(ref.shape, 128, np.uint8)
    templates = [{"id": "bad", "gray": junk}, {"id": "good", "gray": ref}]
    r, tried = T.align_to_templates(test, templates, [0, 1], q_min=0.5)
    assert r.template_id == "good" and r.valid and len(tried) == 2
    r2, _ = T.align_to_templates(test, templates, [1], q_min=1.01)        # impossible cut-off
    assert not r2.valid and "q_min" in r2.reason


# --------------------------------------------------------------------- T2 / T5

def test_residual_is_low_for_good_part_and_high_on_a_defect():
    ref, test, m = make_pair()
    test = test.copy()
    cv2.rectangle(test, (300, 250), (340, 290), 255, -1)           # a defect on the test image
    cmp = T.prepare_comparison(test, ref, m)
    on_defect = T.roi_residual(cmp, (290, 240, 60, 60))
    elsewhere = T.roi_residual(cmp, (500, 400, 60, 60))
    assert on_defect["available"] and elsewhere["available"]
    assert on_defect["residual_mean"] > 5 * max(elsewhere["residual_mean"], 0.005)


def test_padding_is_excluded_from_residuals():
    ref, test, m = make_pair()
    cmp = T.prepare_comparison(test, ref, m)
    # a ROI straddling the black corner padding created by the rotation
    corner = T.roi_residual(cmp, (0, 0, 120, 120))
    if corner["available"]:
        assert corner["residual_mean"] < 0.03 and corner["valid_fraction"] < 1.0
    entirely_padding = T.roi_residual(cmp, (0, 0, 8, 8))
    assert not entirely_padding["available"]


# ------------------------------------------------------------------------ T3

def _cmp_with_patch(patch):
    ref = textured()
    test = ref.copy()
    x, y, w, h = patch
    test[y:y + h, x:x + w] = 255 - test[y:y + h, x:x + w]
    return T.prepare_comparison(test, ref, np.eye(3))


def test_measure_compact_and_elongated_defects():
    m1 = T.measure(_cmp_with_patch((300, 250, 24, 24)), (280, 230, 64, 64), 600 * 800)
    assert m1["available"] and m1["shape_class"] == "compact"
    assert 300 < m1["area_px"] < 900 and m1["area_pct"] == pytest.approx(100 * m1["area_px"] / 480000)
    m2 = T.measure(_cmp_with_patch((300, 250, 90, 8)), (280, 230, 130, 50), 600 * 800)
    assert m2["available"] and m2["shape_class"] == "elongated" and m2["elongation"] >= 3.0


def test_measure_reports_unavailable_instead_of_fabricating_numbers():
    ref = textured()
    clean = T.prepare_comparison(ref, ref, np.eye(3))
    m = T.measure(clean, (100, 100, 60, 60), 480000)
    assert m["available"] is False and "contour" in m["reason"]
    assert T._shape_from_contour(np.array([[[0, 0]], [[0, 10]]], dtype=np.int32)) is None   # zero-width


def test_measure_with_no_valid_overlap_is_unavailable():
    ref = textured()
    cmp = T.prepare_comparison(ref, ref, T.as_h(np.array([[1, 0, 700.0], [0, 1, 0.0]])))
    assert T.measure(cmp, (10, 10, 40, 40), 480000)["available"] is False


# ------------------------------------------------------------------------ T4

def test_clahe_raises_contrast_on_a_flat_image_and_returns_a_copy():
    rng = np.random.default_rng(0)
    flat = (120 + rng.normal(0, 2, (200, 300))).clip(0, 255).astype(np.uint8)
    out = T.normalize_lighting(flat)
    assert out.std() > 2 * flat.std() and out is not flat and out.dtype == np.uint8


# ------------------------------------------------------------------ templates

def test_nearest_order_and_template_loading(tmp_path):
    for i, v in enumerate((0, 100, 200)):
        cv2.imwrite(str(tmp_path / f"t{i}.png"), np.full((20, 30, 3), v, np.uint8))
    emb = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], np.float32)
    np.savez(tmp_path / "embeddings.npz", embeddings=emb)
    (tmp_path / "templates.json").write_text(json.dumps({"templates": [
        {"id": f"t{i}", "file": f"t{i}.png", "embedding_row": i} for i in range(3)]}))
    t = T.load_templates(tmp_path)
    assert [x["id"] for x in t["items"]] == ["t0", "t1", "t2"] and t["items"][0]["gray"].ndim == 2
    assert T.nearest_order(np.array([6.0, 0.0], np.float32), t["embeddings"]) == [1, 2, 0]
    assert T.nearest_order(np.array([5.0, 0.0], np.float32), np.array([[5.0, 0], [5.0, 0]], np.float32)) == [0, 1]


def test_photometric_matching_removes_an_exposure_difference_but_keeps_a_defect():
    ref = textured()
    test = np.clip(ref.astype(np.float32) * 0.75 + 25, 0, 255).astype(np.uint8)       # darker, offset photo
    plain = T.roi_residual(T.prepare_comparison(test, ref, np.eye(3)), (100, 100, 120, 120))
    matched = T.roi_residual(T.prepare_comparison(test, ref, np.eye(3), match=True), (100, 100, 120, 120))
    assert plain["residual_mean"] > 0.05 and matched["residual_mean"] < 0.02
    test2 = test.copy()
    cv2.rectangle(test2, (140, 140), (180, 180), 255, -1)
    d = T.roi_residual(T.prepare_comparison(test2, ref, np.eye(3), match=True), (120, 120, 80, 80))
    assert d["residual_mean"] > 4 * matched["residual_mean"]


def test_180_degree_board_orientation_pairwise_alignment():
    """Verify robust pairwise alignment between upright templates and 180-deg templates (A5)."""
    t_dir = Path("data/templates/pcb1")
    if not (t_dir / "template_00.png").is_file() or not (t_dir / "template_01.png").is_file():
        pytest.skip("Template files not available for alignment test")

    t00 = cv2.imread(str(t_dir / "template_00.png"), cv2.IMREAD_GRAYSCALE)
    t01 = cv2.imread(str(t_dir / "template_01.png"), cv2.IMREAD_GRAYSCALE)

    res = T.align(t00, t01)
    assert res.valid is True
    assert res.method == "orb_ransac"
    assert res.quality is not None and res.quality >= 0.65
    rot = abs(res.rotation_deg)
    assert abs(rot - 180.0) < 10.0 or rot < 10.0

