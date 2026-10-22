"""Execute and record the substantive OpenCV 5 operations used by SureInspect."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from core.perception import extract_patch_features, load_backbone
from core.preprocess import standardize, to_blob

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1"


def main() -> int:
    if int(cv2.__version__.split(".", 1)[0]) < 5:
        raise SystemExit(f"OpenCV 5 required, found {cv2.__version__}")
    sample = cv2.imread(str(ROOT / "data/samples/pcb1_good_01.png"))
    if sample is None:
        raise SystemExit("Missing deterministic OpenCV sample")
    std = standardize(sample).image
    blob = to_blob(std, 256)
    net = load_backbone(SNAPSHOT / "model.onnx")
    features, grid = extract_patch_features(net, blob)
    gray = cv2.cvtColor(std, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.ORB_create(nfeatures=500).detectAndCompute(gray, None)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    model_sha = hashlib.sha256((SNAPSHOT / "model.onnx").read_bytes()).hexdigest()
    result = {
        "schema_version": "sureinspect.opencv5-runtime-proof.v1",
        "opencv_version": cv2.__version__,
        "opencv_major_required": 5,
        "model_sha256": model_sha,
        "dnn_backend": "cv2.dnn.readNetFromONNX + net.forward",
        "blob_shape": list(blob.shape),
        "feature_grid": list(grid),
        "feature_shape": list(features.shape),
        "feature_finite": bool(np.isfinite(features).all()),
        "orb_keypoints": len(keypoints),
        "orb_descriptor_shape": list(descriptors.shape) if descriptors is not None else None,
        "laplacian_variance": laplacian_variance,
        "otsu_contours": len(contours),
        "operations_executed": [
            "cv2.imread", "cv2.resize", "cv2.cvtColor", "cv2.dnn.readNetFromONNX",
            "cv2.dnn.Net.forward", "cv2.ORB.detectAndCompute", "cv2.Laplacian",
            "cv2.threshold(OTSU)", "cv2.findContours",
        ],
    }
    if not result["feature_finite"] or not features.size:
        raise SystemExit("OpenCV DNN produced invalid features")
    out = ROOT / "eval/results/opencv5_runtime_proof.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
