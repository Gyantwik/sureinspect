"""
core/tools.py
Deterministic OpenCV evidence tools used by the agent. None of these ever feed the scorer.

    T1  align()                  register a defect-free template to the test image
    T2  prepare_comparison() +   compare the same ROI in test vs aligned template
        roi_residual()
    T3  measure()                padded crop -> difference contour -> rotated rectangle
    T4  normalize_lighting()     CLAHE comparison copies (never used for scoring)
    T5  roi_residual() on ROI2   (next ROI, same comparison function)

Transform convention (ONE only)
    A 3x3 homogeneous matrix M that maps REFERENCE-standardized pixel coordinates to
    TEST-standardized pixel coordinates:   p_test = M @ p_ref.
    Reduced-resolution transforms are converted with
        M_full = inverse(S_test) @ M_small @ S_ref
    where S_x = diag(f_x, f_x, 1) maps full coordinates to the reduced image.
    cv2.estimateAffinePartial2D(from=reference pts, to=test pts) already returns this direction.
    cv2.findTransformECC(template=test, input=reference) returns the INVERSE (test -> reference),
    so its output is inverted before use. Warping uses cv2.warpAffine(reference, M[:2]) (forward map).

Padding: an all-ones validity mask is warped with the same transform; residuals use only the
valid overlap. Unavailable measurements are reported as unavailable, never as fabricated zeros.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---- fixed, documented constants -------------------------------------------------
ORB_FEATURES = 2000
RATIO = 0.75
MIN_GOOD_MATCHES = 12
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.25
RANSAC_THRESH = 3.0
SCALE_RANGE = (0.8, 1.25)
MAX_ROT_DEG = 45.0
MIN_OVERLAP = 0.6
BLUR_KSIZE = 5
BLUR_SIGMA = 1.0
COARSE_MAX_SIDE = 128
ECC_MAX_SIDE = 512
ROT_STEP_DEG = 10
ECC_ITERS = 60
ECC_EPS = 1e-5
MIN_VALID_PX = 64
ELONGATED_RATIO = 3.0
COMPACT_RATIO = 1.5
CLAHE_CLIP = 2.0
CLAHE_TILE = 8


# ------------------------------------------------------------------ utilities

def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def scale_matrix(f: float) -> np.ndarray:
    return np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], dtype=np.float64)


def as_h(a: np.ndarray) -> np.ndarray:
    """2x3 -> 3x3 homogeneous (float64)."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape == (3, 3):
        return a
    return np.vstack([a, [0.0, 0.0, 1.0]])


def to_full(m_small: np.ndarray, s_test: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
    """M_full = inverse(S_test) @ M_small @ S_ref  (spec formula)."""
    return np.linalg.inv(s_test) @ m_small @ s_ref


def decompose(m: np.ndarray) -> tuple[float, float]:
    """-> (scale, rotation in degrees) of the linear part."""
    a = m[:2, :2]
    scale = math.sqrt(abs(float(np.linalg.det(a))))
    rot = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    return scale, rot


def _blur(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)


def warp_reference(ref: np.ndarray, m: np.ndarray, out_shape) -> tuple[np.ndarray, np.ndarray]:
    """Warp the reference into the test frame. Returns (warped, valid_mask uint8 0/255)."""
    h, w = out_shape[:2]
    warped = cv2.warpAffine(ref, m[:2], (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.full(ref.shape[:2], 255, np.uint8)
    valid = cv2.warpAffine(ones, m[:2], (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))          # drop interpolated border pixels
    return warped, valid


def alignment_quality(test_gray: np.ndarray, warped_ref: np.ndarray, valid: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation over the valid overlap (blurred). -1 if unavailable."""
    m = valid > 0
    if int(m.sum()) < MIN_VALID_PX:
        return -1.0
    a = _blur(test_gray)[m].astype(np.float64)
    b = _blur(warped_ref)[m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else -1.0


# ------------------------------------------------------------------------ T1

@dataclass
class AlignResult:
    M: np.ndarray | None            # 3x3 reference -> test, or None
    method: str                     # "orb_ransac" | "rotation_ecc"
    valid: bool
    reason: str | None = None
    n_matches: int | None = None
    n_inliers: int | None = None
    inlier_ratio: float | None = None
    rotation_deg: float | None = None
    scale: float | None = None
    overlap: float | None = None
    quality: float | None = None
    template_id: str | None = None
    ms: float = 0.0
    attempts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("M", "attempts")}
        d["M"] = None if self.M is None else [[float(x) for x in row] for row in self.M]
        return d


def _finalize(m, test_gray, ref_gray, method, **extra) -> AlignResult:
    """Validate a candidate transform: finite, plausible scale/rotation, overlap, quality."""
    res = AlignResult(M=None, method=method, valid=False, **extra)
    if m is None:
        res.reason = res.reason or "no transform"
        return res
    m = as_h(m)
    res.M = m
    if not np.isfinite(m).all():
        res.reason, res.valid = "non-finite transform", False
        return res
    scale, rot = decompose(m)
    res.scale, res.rotation_deg = scale, rot
    warped, valid = warp_reference(ref_gray, m, test_gray.shape)
    res.overlap = float((valid > 0).mean())
    res.quality = alignment_quality(test_gray, warped, valid)
    if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        res.reason = f"implausible scale {scale:.3f}"
    elif abs(rot) > MAX_ROT_DEG:
        res.reason = f"implausible rotation {rot:.1f} deg"
    elif res.overlap < MIN_OVERLAP:
        res.reason = f"low overlap {res.overlap:.2f}"
    else:
        res.valid = True
    return res


def _align_orb(test_gray, ref_gray) -> AlignResult:
    orb = cv2.ORB_create(ORB_FEATURES)
    kt, dt = orb.detectAndCompute(test_gray, None)
    kr, dr = orb.detectAndCompute(ref_gray, None)
    if dt is None or dr is None or len(kt) < 8 or len(kr) < 8:
        return _finalize(None, test_gray, ref_gray, "orb_ransac", reason="too few ORB features")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dr, dt, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_GOOD_MATCHES:
        return _finalize(None, test_gray, ref_gray, "orb_ransac", n_matches=len(good),
                         reason=f"only {len(good)} good matches")
    src = np.float32([kr[g.queryIdx].pt for g in good])      # reference points
    dst = np.float32([kt[g.trainIdx].pt for g in good])      # test points
    a, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_THRESH, maxIters=2000, confidence=0.99)
    if a is None or inl is None:
        return _finalize(None, test_gray, ref_gray, "orb_ransac", n_matches=len(good), reason="RANSAC failed")
    n_in = int(inl.sum())
    ratio = n_in / len(good)
    res = _finalize(a, test_gray, ref_gray, "orb_ransac", n_matches=len(good), n_inliers=n_in, inlier_ratio=ratio)
    if res.valid and (n_in < MIN_INLIERS or ratio < MIN_INLIER_RATIO):
        res.valid, res.reason = False, f"weak inliers {n_in} ({ratio:.2f})"
    return res


def _reduce(gray: np.ndarray, max_side: int):
    f = min(1.0, max_side / max(gray.shape[:2]))
    if f >= 1.0:
        return gray, 1.0
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return small, f


def _align_rotation_ecc(test_gray, ref_gray) -> AlignResult:
    """Coarse 10-degree rotation search on reduced images, then ECC refinement."""
    ts, ft = _reduce(test_gray, COARSE_MAX_SIDE)
    rs, fr = _reduce(ref_gray, COARSE_MAX_SIDE)
    s_t, s_r = scale_matrix(ft), scale_matrix(fr)
    c_ref = ((rs.shape[1] - 1) / 2.0, (rs.shape[0] - 1) / 2.0)
    c_test = np.array([(ts.shape[1] - 1) / 2.0, (ts.shape[0] - 1) / 2.0])
    best_q, best_m = -2.0, None
    for ang in range(0, 360, ROT_STEP_DEG):
        r = cv2.getRotationMatrix2D(c_ref, float(ang), 1.0)
        r[:, 2] += c_test - np.array(c_ref)                   # centre of ref lands on centre of test
        warped, valid = warp_reference(rs, as_h(r), ts.shape)
        if (valid > 0).mean() < 0.3:
            continue
        q = alignment_quality(ts, warped, valid)
        if q > best_q:
            best_q, best_m = q, as_h(r)
    if best_m is None:
        return _finalize(None, test_gray, ref_gray, "rotation_ecc", reason="rotation search found no overlap")
    m_coarse = to_full(best_m, s_t, s_r)                       # explicit: coarse -> full coordinates

    # ECC refinement at a higher (but bounded) resolution
    tm, ftm = _reduce(test_gray, ECC_MAX_SIDE)
    rm, frm = _reduce(ref_gray, ECC_MAX_SIDE)
    s_tm, s_rm = scale_matrix(ftm), scale_matrix(frm)
    m_mid0 = s_tm @ m_coarse @ np.linalg.inv(s_rm)             # full -> mid coordinates
    w0 = np.linalg.inv(m_mid0)[:2].astype(np.float32)          # ECC wants test -> reference
    m_final = m_coarse
    note = None
    try:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, ECC_ITERS, ECC_EPS)
        _, w = cv2.findTransformECC(tm, rm, w0, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        m_mid = np.linalg.inv(as_h(w))                         # invert back to reference -> test
        m_final = to_full(m_mid, s_tm, s_rm)
    except cv2.error:
        note = "ECC did not converge; coarse rotation estimate used"
    res = _finalize(m_final, test_gray, ref_gray, "rotation_ecc")
    if note and res.reason is None:
        res.reason = note
    return res


def align(test_gray: np.ndarray, ref_gray: np.ndarray) -> AlignResult:
    """T1: ORB + RANSAC first; deterministic rotation search + ECC as fallback."""
    t0 = time.time()
    a = _align_orb(test_gray, ref_gray)
    if a.valid:
        a.ms = (time.time() - t0) * 1000
        return a
    b = _align_rotation_ecc(test_gray, ref_gray)
    best = b if (b.valid or (b.quality or -2) > (a.quality or -2)) else a
    best.attempts = [{"method": r.method, "valid": r.valid, "reason": r.reason, "quality": r.quality}
                     for r in (a, b)]
    best.ms = (time.time() - t0) * 1000
    return best


def align_to_templates(test_gray, templates, order, q_min: float | None = None, max_templates: int = 2):
    """Try up to `max_templates` nearest templates; accept the first valid one with quality >= q_min."""
    best, tried = None, []
    for idx in list(order)[:max_templates]:
        t = templates[idx]
        r = align(test_gray, t["gray"])
        r.template_id = t["id"]
        tried.append({"template_id": t["id"], "valid": r.valid, "quality": r.quality, "reason": r.reason})
        ok = r.valid and (q_min is None or (r.quality is not None and r.quality >= q_min))
        if ok:
            return r, tried
        if best is None or (r.quality or -2) > (best.quality or -2):
            best = r
    if best is not None and best.valid and q_min is not None and (best.quality or -2) < q_min:
        best.valid, best.reason = False, f"alignment quality {best.quality:.3f} below q_min {q_min:.3f}"
    return best, tried


# --------------------------------------------------------------------- T2 / T5

@dataclass
class Comparison:
    diff: np.ndarray       # float32 absdiff/255 of blurred aligned grayscale images
    valid: np.ndarray      # bool validity mask (padding excluded)


def prepare_comparison(test_cmp_gray: np.ndarray, ref_cmp_gray: np.ndarray, m: np.ndarray,
                       match: bool = False) -> Comparison:
    """Warp the reference with M and build the residual image. Inputs may be CLAHE copies (T4).

    match=True first matches the warped reference's mean/std to the test image over the valid
    overlap (a global gain/offset), so an exposure difference between two photos does not swamp
    a small local defect. It is applied identically when fitting and at runtime.
    """
    warped, valid = warp_reference(ref_cmp_gray, m, test_cmp_gray.shape)
    tb = _blur(test_cmp_gray).astype(np.float32)
    rb = _blur(warped).astype(np.float32)
    if match:
        v = valid > 0
        if int(v.sum()) >= MIN_VALID_PX:
            mt, st = float(tb[v].mean()), float(tb[v].std())
            mr, sr = float(rb[v].mean()), float(rb[v].std())
            if sr > 1e-6:
                rb = np.clip((rb - mr) * (st / sr) + mt, 0.0, 255.0)
    return Comparison(diff=np.abs(tb - rb) / 255.0, valid=valid > 0)


def roi_residual(cmp: Comparison, roi) -> dict:
    x, y, w, h = [int(v) for v in roi]
    d = cmp.diff[y:y + h, x:x + w]
    v = cmp.valid[y:y + h, x:x + w]
    n = int(v.sum())
    if n < MIN_VALID_PX:
        return {"available": False, "reason": "ROI has too little valid overlap", "valid_px": n}
    vals = d[v]
    return {"available": True, "residual_mean": float(vals.mean()), "residual_p95": float(np.percentile(vals, 95)),
            "valid_px": n, "valid_fraction": n / float(max(w * h, 1))}


# ------------------------------------------------------------------------ T3

def _shape_from_contour(c) -> dict | None:
    area = float(cv2.contourArea(c))
    if area <= 0:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    mn, mx = min(rw, rh), max(rw, rh)
    if mn <= 0:
        return None                                    # zero-width rectangle: unavailable, not infinity
    elong = mx / mn
    shape = "elongated" if elong >= ELONGATED_RATIO else ("compact" if elong < COMPACT_RATIO else "irregular")
    return {"area_px": area, "elongation": float(elong), "shape_class": shape}


def measure(cmp: Comparison, roi, image_area: float, pad_frac: float = 0.25) -> dict:
    """Padded crop -> Otsu on the valid residual -> largest difference contour -> rotated rectangle.
    Area is reported as a percentage of the standardized IMAGE area (no part mask is used)."""
    x, y, w, h = [int(v) for v in roi]
    pad = int(round(pad_frac * max(w, h)))
    H, W = cmp.diff.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    d8 = (np.clip(cmp.diff[y0:y1, x0:x1], 0, 1) * 255.0).round().astype(np.uint8)
    v = cmp.valid[y0:y1, x0:x1]
    vals = d8[v]
    base = {"roi_bbox": [x0, y0, x1 - x0, y1 - y0]}
    if vals.size < MIN_VALID_PX:
        return {"available": False, "reason": "too little valid overlap", **base}
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((d8 > t) & v).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"available": False, "reason": "no difference contour", **base}
    c = max(contours, key=cv2.contourArea)
    s = _shape_from_contour(c)
    if s is None:
        return {"available": False, "reason": "degenerate contour or zero-width rectangle", **base}
    bx, by, bw, bh = cv2.boundingRect(c)
    return {"available": True, "area_px": s["area_px"], "area_pct": 100.0 * s["area_px"] / float(image_area),
            "elongation": s["elongation"], "shape_class": s["shape_class"],
            "contour_bbox": [x0 + bx, y0 + by, bw, bh], "otsu_threshold": int(t), **base}


# ------------------------------------------------------------------------ T4

def normalize_lighting(gray: np.ndarray) -> np.ndarray:
    """CLAHE comparison copy. For comparison and display only; NEVER passed to the scorer."""
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE)).apply(gray)


# ------------------------------------------------------------------ templates

def load_templates(directory: str | Path) -> dict:
    """Read data/templates/<category>/ -> {"items": [{id, gray, bgr, embedding}], "embeddings": (k, D)}."""
    d = Path(directory)
    meta = json.loads((d / "templates.json").read_text(encoding="utf-8"))
    emb = np.load(d / "embeddings.npz")["embeddings"].astype(np.float32)
    items = []
    for i, t in enumerate(meta["templates"]):
        bgr = cv2.imread(str(d / t["file"]))
        if bgr is None:
            raise FileNotFoundError(f"missing template image {d / t['file']}")
        items.append({"id": t["id"], "bgr": bgr, "gray": to_gray(bgr), "embedding": emb[t["embedding_row"]]})
    return {"items": items, "embeddings": emb, "meta": meta}


def nearest_order(embedding: np.ndarray, embeddings: np.ndarray) -> list[int]:
    """Template indices ordered by embedding distance (ties by index)."""
    d = np.sqrt(((embeddings - embedding[None, :]) ** 2).sum(axis=1))
    return [int(i) for i in np.lexsort((np.arange(len(d)), d))]
