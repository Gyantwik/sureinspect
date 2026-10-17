"""
core/quality_gate.py
SureInspect Quality Gate Engine.

Implements threshold fitting from development normals and single-frame quality checks:
- Focus/blur detection via Laplacian variance
- Underexposure and overexposure pixel fraction limits
- Specular glare detection in HSV color space
- Lighting covariate shift detection in CIELAB color space (L mean and std ratio)

Adheres strictly to SureInspect v8.1 specification:
- Input is an already standardized BGR uint8 image (never resized).
- Floor L-mean SD at 2.0 L units; guard L-std denominator.
- Lighting shift flags lighting_shift_detected but does NOT reject the image alone.
- Rejection (FAILED + REQUEST_RETAKE) occurs only for blur, exposure, or glare violations.
- Concrete recorded margins used for all fitted thresholds.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

# Concrete safety margins for threshold fitting from development normals
BLUR_VARIANCE_SAFETY_RATIO = 0.70  # Floor set at 70% of lowest clean normal variance
DARK_FRACTION_MARGIN = 0.05        # 5 percentage points headroom above max normal dark fraction
BRIGHT_FRACTION_MARGIN = 0.05      # 5 percentage points headroom above max normal bright fraction
GLARE_FRACTION_MARGIN = 0.03       # 3 percentage points headroom above max normal glare fraction

# Pixel intensity thresholds for feature extraction
DARK_INTENSITY_THRESHOLD = 15      # Pixels < 15 considered dark
BRIGHT_INTENSITY_THRESHOLD = 240   # Pixels > 240 considered bright
GLARE_V_THRESHOLD = 240            # HSV Value >= 240
GLARE_S_THRESHOLD = 35             # HSV Saturation <= 35


def validate_standardized_image(image_std: Any) -> None:
    """Validates that input is a non-empty 3-channel BGR uint8 numpy array."""
    if not isinstance(image_std, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_std).__name__}")
    if image_std.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {image_std.dtype}")
    if image_std.ndim != 3 or image_std.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image_std.shape}")
    if image_std.shape[0] == 0 or image_std.shape[1] == 0 or image_std.size == 0:
        raise ValueError(f"Image has empty dimensions: {image_std.shape}")


def compute_frame_metrics(image_std: np.ndarray) -> dict[str, float]:
    """Computes full-precision quality metrics on a standardized BGR image."""
    validate_standardized_image(image_std)

    # 1. Grayscale for blur and exposure
    gray = cv2.cvtColor(image_std, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_variance = float(laplacian.var())

    dark_fraction = float(np.mean(gray < DARK_INTENSITY_THRESHOLD))
    bright_fraction = float(np.mean(gray > BRIGHT_INTENSITY_THRESHOLD))

    # 2. HSV for specular glare
    hsv = cv2.cvtColor(image_std, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    glare_mask = (val >= GLARE_V_THRESHOLD) & (sat <= GLARE_S_THRESHOLD)
    glare_fraction = float(np.mean(glare_mask))

    # 3. LAB for illumination statistics
    lab = cv2.cvtColor(image_std, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    l_mean = float(np.mean(l_channel))
    l_std = float(np.std(l_channel))

    return {
        "blur_variance": blur_variance,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "glare_fraction": glare_fraction,
        "l_mean": l_mean,
        "l_std": l_std,
    }


def fit_gate_thresholds(
    images_std: Sequence[np.ndarray],
    gate_version: str = "v1.0",
) -> dict[str, Any]:
    """
    Fits gate thresholds from a collection of standardized development normal images.

    Returns dict matching Section 3 specification:
      blur_thr, dark_max, bright_max, glare_thr,
      l_mean_ref, l_mean_sd (floored at 2.0 L units),
      l_std_ref (guarded against zero), n_fit, gate_version
    """
    if not images_std:
        raise ValueError("Cannot fit gate thresholds from an empty collection of images.")

    if gate_version == "stub":
        return {
            "blur_thr": 100.0,
            "dark_max": 0.05,
            "bright_max": 0.05,
            "glare_thr": 0.08,
            "l_mean_ref": 128.0,
            "l_mean_sd": 2.0,
            "l_std_ref": 20.0,
            "n_fit": len(images_std),
            "gate_version": "stub",
        }

    blur_vars = []
    dark_fracs = []
    bright_fracs = []
    glare_fracs = []
    l_means = []
    l_stds = []

    for img in images_std:
        metrics = compute_frame_metrics(img)
        blur_vars.append(metrics["blur_variance"])
        dark_fracs.append(metrics["dark_fraction"])
        bright_fracs.append(metrics["bright_fraction"])
        glare_fracs.append(metrics["glare_fraction"])
        l_means.append(metrics["l_mean"])
        l_stds.append(metrics["l_std"])

    # Blur threshold: set at 70% of min clean normal variance (min floor 10.0)
    min_blur = float(np.min(blur_vars))
    blur_thr = max(10.0, min_blur * BLUR_VARIANCE_SAFETY_RATIO)

    # Exposure & glare limits with concrete recorded margins
    dark_max = min(1.0, float(np.max(dark_fracs)) + DARK_FRACTION_MARGIN)
    bright_max = min(1.0, float(np.max(bright_fracs)) + BRIGHT_FRACTION_MARGIN)
    glare_thr = min(1.0, float(np.max(glare_fracs)) + GLARE_FRACTION_MARGIN)

    # Lighting reference statistics
    l_mean_ref = float(np.mean(l_means))
    raw_sd = float(np.std(l_means))
    # Spec requirement: l_mean_sd minimum 2.0 L units
    l_mean_sd = max(2.0, raw_sd)

    # Spec requirement: l_std_ref guarded against zero
    mean_std = float(np.mean(l_stds))
    l_std_ref = max(1e-4, mean_std)

    return {
        "blur_thr": blur_thr,
        "dark_max": dark_max,
        "bright_max": bright_max,
        "glare_thr": glare_thr,
        "l_mean_ref": l_mean_ref,
        "l_mean_sd": l_mean_sd,
        "l_std_ref": l_std_ref,
        "n_fit": len(images_std),
        "gate_version": gate_version,
    }


def check_quality(
    image_std: np.ndarray,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluates standardized image against fitted quality thresholds.

    Returns dict matching the API QualityGateResult schema:
      status ("PASSED"|"FAILED"), action ("REQUEST_RETAKE"|None), reason (str),
      blur_variance, blur_threshold, dark_fraction, bright_fraction,
      glare_fraction, glare_threshold, l_mean,
      exposure_ok (bool), lighting_shift_detected (bool)
    """
    metrics = compute_frame_metrics(image_std)

    blur_variance = metrics["blur_variance"]
    dark_fraction = metrics["dark_fraction"]
    bright_fraction = metrics["bright_fraction"]
    glare_fraction = metrics["glare_fraction"]
    l_mean = metrics["l_mean"]
    l_std = metrics["l_std"]

    blur_thr = float(thresholds["blur_thr"])
    dark_max = float(thresholds["dark_max"])
    bright_max = float(thresholds["bright_max"])
    glare_thr = float(thresholds["glare_thr"])
    l_mean_ref = float(thresholds["l_mean_ref"])
    l_mean_sd = max(2.0, float(thresholds["l_mean_sd"]))
    l_std_ref = max(1e-4, float(thresholds["l_std_ref"]))

    failures = []

    # 1. Focus / Blur check
    if blur_variance < blur_thr:
        failures.append(
            f"Laplacian variance {blur_variance:.1f} is below focus threshold {blur_thr:.1f}"
        )

    # 2. Exposure checks
    dark_failed = dark_fraction > dark_max
    bright_failed = bright_fraction > bright_max
    if dark_failed:
        failures.append(
            f"Dark pixel fraction {dark_fraction:.3f} exceeds threshold {dark_max:.3f}"
        )
    if bright_failed:
        failures.append(
            f"Bright pixel fraction {bright_fraction:.3f} exceeds threshold {bright_max:.3f}"
        )
    exposure_ok = not (dark_failed or bright_failed)

    # 3. Specular glare check
    if glare_fraction > glare_thr:
        failures.append(
            f"Glare pixel fraction {glare_fraction:.3f} exceeds threshold {glare_thr:.3f}"
        )

    # 4. Lighting shift check (LAB L mean > 3 SDs OR std ratio outside [0.7, 1.3])
    # Note: Lighting shift alone does NOT reject the image; it only sets the flag.
    mean_shift = abs(l_mean - l_mean_ref) > (3.0 * l_mean_sd)
    std_ratio = l_std / l_std_ref
    std_shift = (std_ratio < 0.7) or (std_ratio > 1.3)
    lighting_shift_detected = bool(mean_shift or std_shift)

    if failures:
        status = "FAILED"
        action = "REQUEST_RETAKE"
        reason = "; ".join(failures)
    else:
        status = "PASSED"
        action = None
        if lighting_shift_detected:
            reason = "Focus and exposure within limits; lighting shift detected in illumination"
        else:
            reason = "Focus and illumination within normal parameters"

    return {
        "status": status,
        "action": action,
        "reason": reason,
        "blur_variance": blur_variance,
        "blur_threshold": blur_thr,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "glare_fraction": glare_fraction,
        "glare_threshold": glare_thr,
        "l_mean": l_mean,
        "exposure_ok": exposure_ok,
        "lighting_shift_detected": lighting_shift_detected,
    }
