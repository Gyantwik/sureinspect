"""Tests for core/preprocess.py."""
import numpy as np
import pytest

from core.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_SIDE,
    standardize,
    to_blob,
)


def make_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------- standardize ----------------

def test_small_image_is_never_upscaled():
    img = make_image(300, 500)
    r = standardize(img)
    assert r.image.shape == (300, 500, 3)
    assert r.original_size == (500, 300)
    assert r.standardized_size == (500, 300)
    assert r.scale_x == 1.0 and r.scale_y == 1.0
    assert np.array_equal(r.image, img)


def test_exactly_max_side_is_unchanged():
    img = make_image(1024, 700)
    r = standardize(img)
    assert r.standardized_size == (700, 1024)
    assert r.scale_x == 1.0 and r.scale_y == 1.0


def test_landscape_is_shrunk_with_aspect_ratio_kept():
    img = make_image(1000, 2000)
    r = standardize(img)
    assert r.standardized_size == (MAX_SIDE, 512)
    assert r.image.shape == (512, MAX_SIDE, 3)
    assert r.original_size == (2000, 1000)
    assert r.scale_x == pytest.approx(2000 / 1024)
    assert r.scale_y == pytest.approx(1000 / 512)


def test_portrait_is_shrunk_with_aspect_ratio_kept():
    img = make_image(3000, 1500)
    r = standardize(img)
    assert r.standardized_size == (512, MAX_SIDE)
    assert max(r.image.shape[:2]) == MAX_SIDE


def test_scale_maps_a_box_back_to_original():
    img = make_image(2048, 2048)
    r = standardize(img)
    assert r.standardized_size == (1024, 1024)
    x_std = 100
    assert x_std * r.scale_x == pytest.approx(200.0)


def test_input_is_not_modified_and_output_is_a_copy():
    img = make_image(400, 400)
    before = img.copy()
    r = standardize(img)
    r.image[0, 0] = 255 - r.image[0, 0]
    assert np.array_equal(img, before)


def test_standardize_is_deterministic():
    img = make_image(1800, 1200, seed=3)
    a = standardize(img).image
    b = standardize(img).image
    assert np.array_equal(a, b)
    assert a.flags["C_CONTIGUOUS"]


def test_standardize_rejects_bad_input():
    with pytest.raises(TypeError):
        standardize("not an array")
    with pytest.raises(ValueError):
        standardize(np.zeros((10, 10), dtype=np.uint8))          # gray
    with pytest.raises(ValueError):
        standardize(np.zeros((10, 10, 3), dtype=np.float32))     # wrong dtype
    with pytest.raises(ValueError):
        standardize(make_image(10, 10), max_side=0)


# ---------------- to_blob ----------------

def test_blob_shape_dtype_and_layout():
    blob = to_blob(make_image(500, 300), 256)
    assert blob.shape == (1, 3, 256, 256)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]


def test_blob_converts_bgr_to_rgb_and_normalizes():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 255                     # pure BLUE in BGR
    blob = to_blob(img, 32)
    r = (0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]   # R channel is 0
    g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    b = (1.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]   # B channel is 1
    assert np.allclose(blob[0, 0], r, atol=1e-5)
    assert np.allclose(blob[0, 1], g, atol=1e-5)
    assert np.allclose(blob[0, 2], b, atol=1e-5)


def test_blob_is_deterministic_and_does_not_mutate_input():
    img = make_image(333, 777, seed=5)
    before = img.copy()
    a = to_blob(img, 128)
    b = to_blob(img, 128)
    assert np.array_equal(a, b)
    assert np.array_equal(img, before)


def test_blob_rejects_bad_input():
    with pytest.raises(ValueError):
        to_blob(make_image(10, 10), 0)
    with pytest.raises(ValueError):
        to_blob(make_image(10, 10), -4)
    with pytest.raises(ValueError):
        to_blob(np.zeros((10, 10), dtype=np.uint8), 32)
