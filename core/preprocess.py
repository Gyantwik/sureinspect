"""
core/preprocess.py
Image preparation shared by training, evaluation and the live service.

Two functions, both deterministic:

* standardize(image_bgr)    -> StandardizedImage
    Keeps the aspect ratio, makes the longest side at most MAX_SIDE (1024),
    uses INTER_AREA when shrinking, never upscales, and returns the exact sizes
    and scale factors so boxes can be mapped back to the original image.

* to_blob(image_std, size)  -> float32 NCHW array (1, 3, size, size)
    BGR -> RGB, fixed whole-image square resize (no centre crop), divide by 255,
    subtract the ImageNet mean, divide by the ImageNet std. Normalisation is done
    here in NumPy, so it must NOT also be baked into the ONNX model.

Scale convention (matches the API contract):
    scale_x = original_width  / standardized_width
    scale_y = original_height / standardized_height
    original_x = standardized_x * scale_x
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from core.quality_gate import validate_standardized_image

MAX_SIDE = 1024

# One fixed interpolation policy for the backbone input. Reference building and
# runtime must both use this exact constant.
BLOB_INTERPOLATION = cv2.INTER_AREA

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class StandardizedImage(NamedTuple):
    image: np.ndarray                  # BGR uint8, contiguous
    original_size: tuple[int, int]     # (width, height) before standardizing
    standardized_size: tuple[int, int]  # (width, height) after standardizing
    scale_x: float
    scale_y: float


def standardize(image_bgr: np.ndarray, max_side: int = MAX_SIDE) -> StandardizedImage:
    """Return a standardized copy of a BGR uint8 image plus exact size metadata."""
    validate_standardized_image(image_bgr)  # uint8, 3 channels, non-empty
    if not isinstance(max_side, int) or max_side <= 0:
        raise ValueError(f"max_side must be a positive int, got {max_side!r}")

    h, w = image_bgr.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        out = image_bgr.copy()
        new_w, new_h = w, h
    else:
        ratio = max_side / longest
        if w >= h:
            new_w, new_h = max_side, max(1, int(round(h * ratio)))
        else:
            new_w, new_h = max(1, int(round(w * ratio))), max_side
        out = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return StandardizedImage(
        image=np.ascontiguousarray(out),
        original_size=(int(w), int(h)),
        standardized_size=(int(new_w), int(new_h)),
        scale_x=float(w) / float(new_w),
        scale_y=float(h) / float(new_h),
    )


def to_blob(image_std: np.ndarray, size: int) -> np.ndarray:
    """Convert a standardized BGR image to a contiguous float32 NCHW network input."""
    validate_standardized_image(image_std)
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"size must be a positive int, got {size!r}")

    rgb = cv2.cvtColor(image_std, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=BLOB_INTERPOLATION)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # HWC -> NCHW
    return np.ascontiguousarray(x, dtype=np.float32)
