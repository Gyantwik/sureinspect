"""
eval/run_5_attempts.py
Bounded 5-Attempt Optimization Controller on Development Split (297 normals, 26 defects).
Evaluates 5 controlled hypotheses against baseline.
"""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.preprocess import standardize
from core.agent import inspect_standardized, decide, DecisionResult

splits = json.loads((ROOT / "artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11/splits.json").read_text())
dev_bad_items = splits["splits"]["dev_anomalies"]
dev_good_items = splits["splits"]["dev_eval_normals"]

print(f"Loaded {len(dev_bad_items)} dev defects, {len(dev_good_items)} dev normals.")

def run_evaluation_on_dev(custom_decide_fn=None, custom_inspect_fn=None):
    """Evaluates pipeline on dev split with given policy function."""
    bad_results = []
    good_results = []
    
    t0_all = time.time()
    latencies = []

    # Evaluate all 26 dev bad
    for item in dev_bad_items:
        p = item["path"] if isinstance(item, dict) else item
        img = cv2.imread(f"datasets/VisA_pytorch/1cls/{p}")
        t0 = time.time()
        std = standardize(img)
        if custom_inspect_fn:
            res = custom_inspect_fn(std.image)
        else:
            res = inspect_standardized(std.image)
        dt = (time.time() - t0) * 1000.0
        latencies.append(dt)
        bad_results.append({
            "path": p,
            "decision": res["decision"],
            "decision_path": res["decision_path"],
            "score": res["score"],
            "dt": dt,
        })

    # Evaluate 297 dev good
    for item in dev_good_items:
        p = item["path"] if isinstance(item, dict) else item
        img = cv2.imread(f"datasets/VisA_pytorch/1cls/{p}")
        t0 = time.time()
        std = standardize(img)
        if custom_inspect_fn:
            res = custom_inspect_fn(std.image)
        else:
            res = inspect_standardized(std.image)
        dt = (time.time() - t0) * 1000.0
        latencies.append(dt)
        good_results.append({
            "path": p,
            "decision": res["decision"],
            "decision_path": res["decision_path"],
            "score": res["score"],
            "dt": dt,
        })

    # Metrics
    defect_escapes = [r for r in bad_results if r["decision"] == "PASS"]
    defect_intercepted = [r for r in bad_results if r["decision"] in ("FAIL", "HUMAN")]
    defect_retakes = [r for r in bad_results if r["decision"] == "RETAKE"]

    good_scrapped = [r for r in good_results if r["decision"] == "FAIL"]
    good_reviewed = [r for r in good_results if r["decision"] == "HUMAN"]
    good_cleared = [r for r in good_results if r["decision"] == "PASS"]
    good_retakes = [r for r in good_results if r["decision"] == "RETAKE"]

    metrics = {
        "n_bad": len(bad_results),
        "n_good": len(good_results),
        "defect_escapes": len(defect_escapes),
        "defect_escape_rate": len(defect_escapes) / len(bad_results),
        "defect_interception": len(defect_intercepted),
        "defect_interception_rate": len(defect_intercepted) / len(bad_results),
        "good_auto_scrap": len(good_scrapped),
        "good_review": len(good_reviewed),
        "good_review_rate": len(good_reviewed) / len(good_results),
        "good_cleared": len(good_cleared),
        "good_clear_rate": len(good_cleared) / len(good_results),
        "mean_latency_ms": round(float(np.mean(latencies)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
        "escaped_paths": [r["path"] for r in defect_escapes],
    }
    return metrics

if __name__ == "__main__":
    print("Testing baseline evaluation run on dev...")
    baseline_metrics = run_evaluation_on_dev()
    print("Baseline Metrics:", json.dumps(baseline_metrics, indent=2))
