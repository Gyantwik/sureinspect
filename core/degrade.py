"""
core/degrade.py
Deterministic image degradation module applied after standardization.

Supports:
- blur: fixed diagonal motion blur kernel (kernel size 19)
- darken: fixed gamma curve (gamma=3.5)
- glare: feathered specular white blob seeded from image content SHA-256 hash

Strict rules:
- Fully deterministic: identical input + flags produces byte-identical output.
- Never mutates the input image array.
- Applied after standardization, once only.
- Returns (degraded_image, applied_effects_list).
- Fixed effect execution order is strictly blur, darken, glare.
- The returned applied list records this exact order: ['blur', 'darken', 'glare'].
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from core.quality_gate import validate_standardized_image

FIXED_MOTION_KERNEL_SIZE = 19
FIXED_DARKEN_GAMMA = 3.5


def _motion_blur(img: np.ndarray) -> np.ndarray:
    """Applies a fixed diagonal motion blur kernel."""
    kernel = np.zeros((FIXED_MOTION_KERNEL_SIZE, FIXED_MOTION_KERNEL_SIZE), dtype=np.float32)
    np.fill_diagonal(kernel, 1.0 / FIXED_MOTION_KERNEL_SIZE)
    return cv2.filter2D(img, -1, kernel)


def _gamma_darken(img: np.ndarray, gamma: float = FIXED_DARKEN_GAMMA) -> np.ndarray:
    """Applies fixed nonlinear gamma darkening via lookup table."""
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _glare_blob(img: np.ndarray) -> np.ndarray:
    """
    Applies a feathered specular white glare blob.
    The blob position is seeded deterministically from the SHA-256 hash of the image content.
    """
    # Deterministic seed from image bytes
    digest = hashlib.sha256(img.tobytes()).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="big")

    h, w = img.shape[:2]
    # Bound blob center inside the central 50% region of the frame
    cx = int(0.25 * w + (seed_int % max(1, int(0.50 * w))))
    cy = int(0.25 * h + ((seed_int >> 16) % max(1, int(0.50 * h))))
    radius = max(32, int(0.25 * min(h, w)))
    core_radius = int(0.6 * radius)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[dist <= core_radius] = 1.0
    feather_mask = (dist > core_radius) & (dist <= radius)
    alpha[feather_mask] = 0.5 * (1.0 + np.cos(np.pi * (dist[feather_mask] - core_radius) / (radius - core_radius)))
    alpha = alpha[:, :, np.newaxis]

    white = np.full_like(img, 255, dtype=np.float32)
    result = img.astype(np.float32) * (1.0 - alpha) + white * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_degradations(
    image_std: np.ndarray,
    blur: bool = False,
    darken: bool = False,
    glare: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Applies requested simulated degradations to a standardized BGR uint8 image.

    Returns:
      (degraded_image, applied_effects)
    """
    validate_standardized_image(image_std)

    out = image_std.copy()
    applied: list[str] = []

    if blur:
        out = _motion_blur(out)
        applied.append("blur")

    if darken:
        out = _gamma_darken(out)
        applied.append("darken")

    if glare:
        out = _glare_blob(out)
        applied.append("glare")

    return out, applied


# Alias for concise import
degrade = apply_degradations
