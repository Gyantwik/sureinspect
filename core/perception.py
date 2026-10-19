"""
core/perception.py
PatchCore-style perception running on OpenCV's cv.dnn backbone.

Pipeline for one standardized BGR image:
    to_blob -> cv.dnn (layer2_out, layer3_out) -> upsample layer3 to layer2's grid ->
    concatenate -> 3x3 stride-1 average pool -> patch descriptors (h*w, D)
    -> distance of every patch to its nearest memory-bank patch -> patch map.

    image score = MAX raw patch distance      (a distance, NOT a probability)
    heat-map    = patch map upsampled to the standardized image size + fixed Gaussian blur

Explicit policies (recorded in the snapshot manifest):
    * Upsampling: bilinear, half-pixel centres (same as torch align_corners=False).
    * Average pool: 3x3, stride 1, ZERO padding, divisor 9 (count_include_pad=True),
      i.e. the torch AvgPool2d(3, 1, 1) default used by PatchCore.
    * Distances use squared norms + blocked matrix multiplication (queries AND bank),
      negatives from rounding are clipped before the square root.
    * Heat-map blur: sigma = HEATMAP_SIGMA_FRAC * longest standardized side.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.preprocess import to_blob

OUT_NAMES = ("layer2_out", "layer3_out")
HEATMAP_SIGMA_FRAC = 4.0 / 256.0   # sigma = 4 px at a 256 px side, scaled with image size


# ------------------------------------------------------------------ backbone

def load_backbone(onnx_path, engine: str = "auto"):
    """Load the exported backbone with cv.dnn. engine: auto | new | classic."""
    engines = {
        "auto": getattr(cv2.dnn, "ENGINE_AUTO", None),
        "new": getattr(cv2.dnn, "ENGINE_NEW", None),
        "classic": getattr(cv2.dnn, "ENGINE_CLASSIC", None),
    }
    eng = engines.get(engine)
    if engine != "auto" and eng is None:
        raise RuntimeError(f"This OpenCV build has no '{engine}' DNN engine")
    if eng is not None:
        return cv2.dnn.readNetFromONNX(str(onnx_path), eng)
    return cv2.dnn.readNetFromONNX(str(onnx_path))


def run_backbone(net, blob: np.ndarray) -> list[np.ndarray]:
    net.setInput(blob)
    outs = net.forward(list(OUT_NAMES))
    if isinstance(outs, np.ndarray):
        outs = [outs]
    return [np.asarray(o, dtype=np.float32) for o in outs]


# --------------------------------------------------------- feature building

def _axis_weights(in_size: int, out_size: int):
    scale = in_size / out_size
    src = np.clip((np.arange(out_size) + 0.5) * scale - 0.5, 0.0, None)
    i0 = np.minimum(np.floor(src).astype(np.int64), in_size - 1)
    i1 = np.minimum(i0 + 1, in_size - 1)
    w1 = (src - i0).astype(np.float32)
    return i0, i1, (1.0 - w1), w1


def resize_bilinear_chw(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of a (C, H, W) array with half-pixel centres. Any channel count."""
    c, h, w = x.shape
    if (h, w) == (out_h, out_w):
        return x.astype(np.float32, copy=True)
    y0, y1, wy0, wy1 = _axis_weights(h, out_h)
    x0, x1, wx0, wx1 = _axis_weights(w, out_w)
    t = x[:, y0, :] * wy0[None, :, None] + x[:, y1, :] * wy1[None, :, None]
    out = t[:, :, x0] * wx0[None, None, :] + t[:, :, x1] * wx1[None, None, :]
    return out.astype(np.float32, copy=False)


def avg_pool3x3_chw(x: np.ndarray) -> np.ndarray:
    """3x3, stride 1, zero padding, divide by 9 (count_include_pad=True)."""
    c, h, w = x.shape
    p = np.zeros((c, h + 2, w + 2), dtype=np.float32)
    p[:, 1:-1, 1:-1] = x
    acc = np.zeros((c, h, w), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += p[:, dy:dy + h, dx:dx + w]
    return acc / 9.0


def combine_layers(layer2: np.ndarray, layer3: np.ndarray) -> np.ndarray:
    """(1,C2,h,w) + (1,C3,h3,w3) -> (h*w, C2+C3) patch descriptors."""
    f2, f3 = layer2[0], layer3[0]
    _, h, w = f2.shape
    f3 = resize_bilinear_chw(f3, h, w)
    f = avg_pool3x3_chw(np.concatenate([f2, f3], axis=0))
    return np.ascontiguousarray(f.reshape(f.shape[0], h * w).T, dtype=np.float32)


def extract_patch_features(net, blob: np.ndarray):
    """-> (descriptors (h*w, D) float32, grid (h, w)). Dimensions are derived, not hard-coded."""
    l2, l3 = run_backbone(net, blob)
    grid = (int(l2.shape[2]), int(l2.shape[3]))
    return combine_layers(l2, l3), grid


# ------------------------------------------------------------------- coreset

def random_projection(dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((dim, out_dim)) / np.sqrt(out_dim)).astype(np.float32)


def greedy_coreset(proj: np.ndarray, k: int, seed: int, progress: bool = False) -> np.ndarray:
    """Greedy k-center (farthest-point) selection on projected descriptors.

    Deterministic for a given seed (the seed only chooses the first centre).
    Returns the selected row indices in selection order.
    """
    n = proj.shape[0]
    k = int(min(max(k, 1), n))
    rng = np.random.default_rng(seed)
    sq = np.einsum("ij,ij->i", proj, proj)
    first = int(rng.integers(n))
    selected = np.empty(k, dtype=np.int64)
    selected[0] = first
    min_d = np.maximum(sq + sq[first] - 2.0 * (proj @ proj[first]), 0.0)
    for i in range(1, k):
        nxt = int(np.argmax(min_d))
        selected[i] = nxt
        d = np.maximum(sq + sq[nxt] - 2.0 * (proj @ proj[nxt]), 0.0)
        np.minimum(min_d, d, out=min_d)
        if progress and i % 1000 == 0:
            print(f"  coreset {i}/{k}", flush=True)
    return selected


# ---------------------------------------------------------------- scoring

def nn_distances(queries: np.ndarray, bank: np.ndarray, q_block: int = 1024, b_block: int = 4096) -> np.ndarray:
    """Distance from each query row to its nearest bank row (blocked, squared-norm trick)."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    b = np.ascontiguousarray(bank, dtype=np.float32)
    q_sq = np.einsum("ij,ij->i", q, q)
    b_sq = np.einsum("ij,ij->i", b, b)
    out = np.empty(len(q), dtype=np.float32)
    for qs in range(0, len(q), q_block):
        qb = q[qs:qs + q_block]
        best = np.full(len(qb), np.inf, dtype=np.float32)
        for bs in range(0, len(b), b_block):
            d2 = q_sq[qs:qs + q_block, None] + b_sq[None, bs:bs + b_block] - 2.0 * (qb @ b[bs:bs + b_block].T)
            np.minimum(best, d2.min(axis=1), out=best)
        out[qs:qs + q_block] = np.sqrt(np.maximum(best, 0.0))
    return out


@dataclass
class ScoreResult:
    score: float                 # max raw patch distance
    patch_map: np.ndarray        # (h, w) float32 raw patch distances
    grid: tuple[int, int]        # (h, w)


def score_image(net, image_std: np.ndarray, bank: np.ndarray, size: int) -> ScoreResult:
    blob = to_blob(image_std, size)
    feats, grid = extract_patch_features(net, blob)
    d = nn_distances(feats, bank)
    pm = d.reshape(grid).astype(np.float32)
    return ScoreResult(score=float(pm.max()), patch_map=pm, grid=grid)


def make_heatmap(patch_map: np.ndarray, std_size: tuple[int, int]) -> np.ndarray:
    """Upsample the patch map to the standardized image (w, h) and apply the fixed blur."""
    w, h = std_size
    up = cv2.resize(patch_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    sigma = HEATMAP_SIGMA_FRAC * max(w, h)
    return cv2.GaussianBlur(up, (0, 0), sigma)


# ---------------------------------------------------------------------- ROIs

def find_rois(heat: np.ndarray, patch_map: np.ndarray, patch_thr: float, max_rois: int = 3) -> list[dict]:
    """Up to `max_rois` boxes (x, y, w, h) in STANDARDIZED coordinates, ranked by peak heat.

    Boxes are the bounding rectangles of connected regions where heat > patch_thr.
    Ties are broken deterministically (larger area, then top-left). If no region passes the
    threshold, ROI1 is a clipped 3x3 feature-patch box around the patch-map peak, with the
    stride mapped independently per axis (standardized width / grid width, height / grid height).
    """
    h_img, w_img = heat.shape[:2]
    mask = (heat > patch_thr).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        peak = float(heat[y:y + h, x:x + w].max())
        cands.append((-peak, -(w * h), y, x, [int(x), int(y), int(w), int(h)], peak))
    cands.sort(key=lambda t: t[:4])
    rois = [{"box": c[4], "peak": c[5], "source": "threshold"} for c in cands[:max_rois]]
    if rois:
        return rois

    gh, gw = patch_map.shape
    gy, gx = np.unravel_index(int(np.argmax(patch_map)), patch_map.shape)
    cell_w, cell_h = w_img / gw, h_img / gh
    x0 = max(0, int(np.floor((gx - 1) * cell_w)))
    x1 = min(w_img, int(np.ceil((gx + 2) * cell_w)))
    y0 = max(0, int(np.floor((gy - 1) * cell_h)))
    y1 = min(h_img, int(np.ceil((gy + 2) * cell_h)))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    peak = float(heat[y0:y1, x0:x1].max())
    return [{"box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], "peak": peak, "source": "peak_fallback"}]
