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
- Live artifacts must call assert_live_ready at load time to prevent stub or corrupt
  thresholds from serving production inference.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

# Stub marker and public constants
STUB_GATE_VERSION = "stub"
BLUR_PERCENTILE = 1
BLUR_VARIANCE_SAFETY_RATIO = 0.70  # Safety ratio applied to 1st percentile of normal blur variances
DARK_FRACTION_MARGIN = 0.05        # 5 percentage points headroom above max normal dark fraction
BRIGHT_FRACTION_MARGIN = 0.05      # 5 percentage points headroom above max normal bright fraction
GLARE_FRACTION_MARGIN = 0.03       # 3 percentage points headroom above max normal glare fraction

# Pixel intensity thresholds for feature extraction
DARK_INTENSITY_THRESHOLD = 15      # Pixels < 15 considered dark
BRIGHT_INTENSITY_THRESHOLD = 240   # Pixels > 240 considered bright
GLARE_V_THRESHOLD = 240            # HSV Value >= 240
GLARE_S_THRESHOLD = 35             # HSV Saturation <= 35
GLARE_TINTED_V_THRESHOLD = 250     # HSV Value >= 250 for tinted glare (v1.1)
GLARE_TINTED_S_THRESHOLD = 120     # HSV Saturation <= 120 for tinted glare (v1.1)

REQUIRED_THRESHOLD_KEYS = (
    "blur_thr",
    "dark_max",
    "bright_max",
    "glare_thr",
    "l_mean_ref",
    "l_mean_sd",
    "l_std_ref",
    "n_fit",
    "gate_version",
)

NUMERIC_THRESHOLD_KEYS = (
    "blur_thr",
    "dark_max",
    "bright_max",
    "glare_thr",
    "l_mean_ref",
    "l_mean_sd",
    "l_std_ref",
)


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


def validate_thresholds(thresholds: Any) -> None:
    """
    Validates that thresholds dictionary contains all required keys with finite numeric values.
    Raises ValueError naming any missing or non-finite key. Extra keys are tolerated.
    """
    if not isinstance(thresholds, dict):
        raise ValueError(f"Thresholds must be a dict, got {type(thresholds).__name__}")  # noqa: TRY004

    for key in REQUIRED_THRESHOLD_KEYS:
        if key not in thresholds:
            raise ValueError(f"Missing required threshold key: '{key}'")

    for key in NUMERIC_THRESHOLD_KEYS:
        val = thresholds[key]
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise ValueError(f"Threshold '{key}' must be numeric, got {type(val).__name__}")  # noqa: TRY004
        if not math.isfinite(float(val)):
            raise ValueError(f"Threshold '{key}' must be finite, got {val}")

    n_fit = thresholds["n_fit"]
    if not isinstance(n_fit, (int, np.integer)) or isinstance(n_fit, bool) or n_fit < 0:
        raise ValueError(f"Threshold 'n_fit' must be a non-negative integer, got {n_fit}")

    gate_version = thresholds["gate_version"]
    if not isinstance(gate_version, str) or not gate_version.strip():
        raise ValueError(f"Threshold 'gate_version' must be a non-empty string, got {gate_version}")


def assert_live_ready(thresholds: dict[str, Any]) -> None:
    """
    Asserts that the threshold dictionary is valid for production/live deployment.
    Raises ValueError if gate_version == 'stub', any key is missing, or any numeric value is non-finite.

    Note: Live artifacts must call assert_live_ready at load time.
    """
    validate_thresholds(thresholds)
    if thresholds["gate_version"] == STUB_GATE_VERSION:
        raise ValueError(
            f"Gate version is '{STUB_GATE_VERSION}'; stub thresholds cannot be used in a live-ready artifact."
        )


def compute_frame_metrics(
    image_std: np.ndarray,
    use_v1_1: bool = False,
) -> dict[str, float]:
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
    if use_v1_1:
        # D3: Support warm/tinted specular glare
        glare_mask = (val >= GLARE_V_THRESHOLD) & (
            (sat <= GLARE_S_THRESHOLD)
            | ((sat <= GLARE_TINTED_S_THRESHOLD) & (val >= GLARE_TINTED_V_THRESHOLD))
        )
    else:
        glare_mask = (val >= GLARE_V_THRESHOLD) & (sat <= GLARE_S_THRESHOLD)
    glare_fraction = float(np.mean(glare_mask))

    # 3. LAB for illumination statistics
    lab = cv2.cvtColor(image_std, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    l_mean = float(np.mean(l_channel))
    l_std = float(np.std(l_channel))

    metrics: dict[str, float] = {
        "blur_variance": blur_variance,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "glare_fraction": glare_fraction,
        "l_mean": l_mean,
        "l_std": l_std,
    }

    if use_v1_1:
        # D1: Center tile localized defocus check
        h, w = gray.shape
        center_gray = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        metrics["center_blur_variance"] = float(cv2.Laplacian(center_gray, cv2.CV_64F).var())

        # D2: Pre-filtered median blur for high-frequency sensor noise rejection
        denoised_gray = cv2.medianBlur(gray, 3)
        metrics["denoised_blur_variance"] = float(cv2.Laplacian(denoised_gray, cv2.CV_64F).var())

    return metrics


def fit_gate_thresholds(
    images_std: Sequence[np.ndarray],
    gate_version: str = "v1.0",
) -> dict[str, Any]:
    """
    Fits gate thresholds from a collection of standardized development normal images.

    Returns dict matching Section 3 specification:
      blur_thr, dark_max, bright_max, glare_thr,
      l_mean_ref, l_mean_sd (floored at 2.0 L units),
      l_std_ref (guarded against zero), n_fit, gate_version,
      blur_fit_method ("percentile_1" or "min_small_n")
      When gate_version == "v1.1", also includes center_blur_thr and denoised_blur_thr.
    """
    if not images_std:
        raise ValueError("Cannot fit gate thresholds from an empty collection of images.")

    if gate_version == STUB_GATE_VERSION:
        return {
            "blur_thr": 100.0,
            "dark_max": 0.05,
            "bright_max": 0.05,
            "glare_thr": 0.08,
            "l_mean_ref": 128.0,
            "l_mean_sd": 2.0,
            "l_std_ref": 20.0,
            "n_fit": len(images_std),
            "gate_version": STUB_GATE_VERSION,
            "blur_fit_method": "stub",
        }

    is_v1_1 = (gate_version == "v1.1")
    blur_vars = []
    center_blur_vars = []
    denoised_blur_vars = []
    dark_fracs = []
    bright_fracs = []
    glare_fracs = []
    l_means = []
    l_stds = []

    for img in images_std:
        metrics = compute_frame_metrics(img, use_v1_1=is_v1_1)
        blur_vars.append(metrics["blur_variance"])
        if is_v1_1:
            center_blur_vars.append(metrics["center_blur_variance"])
            denoised_blur_vars.append(metrics["denoised_blur_variance"])
        dark_fracs.append(metrics["dark_fraction"])
        bright_fracs.append(metrics["bright_fraction"])
        glare_fracs.append(metrics["glare_fraction"])
        l_means.append(metrics["l_mean"])
        l_stds.append(metrics["l_std"])

    n_images = len(images_std)
    # Robust blur threshold fitting: 1st percentile for N >= 20, min for N < 20
    if n_images >= 20:
        blur_fit_method = "percentile_1"
        base_blur = float(np.percentile(blur_vars, BLUR_PERCENTILE))
    else:
        blur_fit_method = "min_small_n"
        base_blur = float(np.min(blur_vars))

    blur_thr = max(10.0, base_blur * BLUR_VARIANCE_SAFETY_RATIO)

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

    out: dict[str, Any] = {
        "blur_thr": blur_thr,
        "dark_max": dark_max,
        "bright_max": bright_max,
        "glare_thr": glare_thr,
        "l_mean_ref": l_mean_ref,
        "l_mean_sd": l_mean_sd,
        "l_std_ref": l_std_ref,
        "n_fit": n_images,
        "gate_version": gate_version,
        "blur_fit_method": blur_fit_method,
    }

    if is_v1_1:
        if n_images >= 20:
            base_center = float(np.percentile(center_blur_vars, BLUR_PERCENTILE))
            base_denoised = float(np.percentile(denoised_blur_vars, BLUR_PERCENTILE))
        else:
            base_center = float(np.min(center_blur_vars))
            base_denoised = float(np.min(denoised_blur_vars))
        out["center_blur_thr"] = max(10.0, base_center * BLUR_VARIANCE_SAFETY_RATIO)
        out["denoised_blur_thr"] = max(10.0, base_denoised * BLUR_VARIANCE_SAFETY_RATIO)

    return out


def check_quality(
    image_std: np.ndarray,
    thresholds: dict[str, Any],
    use_v1_1: bool | None = None,
) -> dict[str, Any]:
    """
    Evaluates standardized image against quality thresholds.
    Validates thresholds dictionary first. Extra keys are tolerated.

    Returns dict matching the API QualityGateResult schema:
      status ("PASSED"|"FAILED"), action ("REQUEST_RETAKE"|None), reason (str),
      blur_variance, blur_threshold, dark_fraction, bright_fraction,
      glare_fraction, glare_threshold, l_mean,
      exposure_ok (bool), lighting_shift_detected (bool)
    """
    validate_thresholds(thresholds)
    is_v1_1 = use_v1_1 if use_v1_1 is not None else (thresholds.get("gate_version") == "v1.1")
    metrics = compute_frame_metrics(image_std, use_v1_1=is_v1_1)

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

    if is_v1_1:
        center_blur_thr = float(thresholds.get("center_blur_thr", blur_thr))
        center_blur_var = metrics.get("center_blur_variance", blur_variance)
        if center_blur_var < center_blur_thr:
            failures.append(
                f"Center Laplacian variance {center_blur_var:.1f} is below focus threshold {center_blur_thr:.1f}"
            )

        denoised_blur_thr = float(thresholds.get("denoised_blur_thr", blur_thr * 0.40))
        denoised_blur_var = metrics.get("denoised_blur_variance", blur_variance)
        if denoised_blur_var < denoised_blur_thr:
            failures.append(
                f"Denoised Laplacian variance {denoised_blur_var:.1f} is below focus threshold {denoised_blur_thr:.1f}"
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

    res_dict: dict[str, Any] = {
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
    if is_v1_1:
        res_dict["center_blur_variance"] = metrics.get("center_blur_variance")
        res_dict["denoised_blur_variance"] = metrics.get("denoised_blur_variance")

    return res_dict


def gate_reject_report(
    images_std: Sequence[np.ndarray],
    thresholds: dict[str, Any],
    use_v1_1: bool | None = None,
) -> dict[str, Any]:
    """
    Evaluates a collection of images against quality thresholds and summarizes rejection rates.

    Pure function, returns:
      n: total images evaluated
      n_rejected: number of rejected images (status == 'FAILED')
      reject_rate: fraction of rejected images
      per_reason_counts: dict of reason counts {'blur': int, 'dark': int, 'bright': int, 'glare': int}
      n_lighting_shift: number of images with lighting shift detected
      lighting_shift_rate: fraction of images with lighting shift detected
    """
    validate_thresholds(thresholds)
    is_v1_1 = use_v1_1 if use_v1_1 is not None else (thresholds.get("gate_version") == "v1.1")
    n = len(images_std)
    if n == 0:
        return {
            "n": 0,
            "n_rejected": 0,
            "reject_rate": 0.0,
            "per_reason_counts": {"blur": 0, "dark": 0, "bright": 0, "glare": 0},
            "n_lighting_shift": 0,
            "lighting_shift_rate": 0.0,
        }

    blur_thr = float(thresholds["blur_thr"])
    dark_max = float(thresholds["dark_max"])
    bright_max = float(thresholds["bright_max"])
    glare_thr = float(thresholds["glare_thr"])

    n_rejected = 0
    n_lighting_shift = 0
    per_reason_counts = {"blur": 0, "dark": 0, "bright": 0, "glare": 0}

    for img in images_std:
        res = check_quality(img, thresholds, use_v1_1=is_v1_1)
        if res["status"] == "FAILED":
            n_rejected += 1

        if res["blur_variance"] < blur_thr:
            per_reason_counts["blur"] += 1
        if res["dark_fraction"] > dark_max:
            per_reason_counts["dark"] += 1
        if res["bright_fraction"] > bright_max:
            per_reason_counts["bright"] += 1
        if res["glare_fraction"] > glare_thr:
            per_reason_counts["glare"] += 1

        if res["lighting_shift_detected"]:
            n_lighting_shift += 1

    return {
        "n": n,
        "n_rejected": n_rejected,
        "reject_rate": float(n_rejected / n),
        "per_reason_counts": per_reason_counts,
        "n_lighting_shift": n_lighting_shift,
        "lighting_shift_rate": float(n_lighting_shift / n),
    }


def fit_and_report(
    images_std: Sequence[np.ndarray],
    gate_version: str = "v1.0",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fits gate thresholds and computes the rejection report over the same development collection.
    Returns (thresholds, report).
    """
    thresholds = fit_gate_thresholds(images_std, gate_version=gate_version)
    report = gate_reject_report(images_std, thresholds)
    return thresholds, report
