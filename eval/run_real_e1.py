"""
eval/run_real_e1.py
Executes measured real-image inference for Experiment E1 across the Amazon VisA pcb1 splits:
1. Development split (297 normals, 26 defects)
2. Final held-out test split (100 normals, 70 defects)

Replaces all heuristic simulations with empirical OpenCV 5 DNN PatchCore inference
and real agent evidence tools T1-T5.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.live_runner import get_live_engine
from eval.run_agent import evaluate_agent
from eval.run_baseline import evaluate_baseline
from core.preprocess import standardize
from core.perception import score_image
import cv2

CACHE_PATH = Path("eval/results/cached_e1_predictions.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "latest"


def load_cache(current_commit: str) -> Dict[str, Any]:
    if CACHE_PATH.is_file():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("__commit__") == current_commit:
                return data.get("predictions", {})
            else:
                old_commit = data.get("__commit__") if isinstance(data, dict) else "unknown"
                print(f"[CACHE] Invalidated old prediction cache (cache commit {old_commit} != current {current_commit})")
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Any], current_commit: str):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "__commit__": current_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions": cache,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_image_split(
    engine,
    image_items: List[Dict[str, Any]],
    true_label: str,
    root: Path,
    t_fail: float,
    cache: Dict[str, Any],
    commit_hash: str,
    desc: str = "",
) -> Dict[str, Any]:
    """Runs live inference over a list of image items and records predictions."""
    raw_scores = []
    baseline_preds = []
    agent_decisions = []
    agent_paths = []
    ground_truth = []
    
    total = len(image_items)
    print(f"\n--- Evaluating {desc} ({total} images) ---")
    t0 = time.time()
    
    for i, it in enumerate(image_items):
        rel_p = it["path"]
        img_path = root / rel_p
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image file: {img_path}")
            
        if rel_p in cache:
            entry = cache[rel_p]
            score = entry["score"]
            dec = entry["decision"]
            path_val = entry["decision_path"]
            # Fix: Never coerce RETAKE scores to 0.0 for perception baselines
            if dec == "RETAKE" and (score == 0.0 or score is None):
                img_bgr = cv2.imread(str(img_path))
                std_img = standardize(img_bgr, 1024).image
                bundle = engine.bundle
                pc_res = score_image(bundle["net"], std_img, bundle["bank"], bundle.get("input_size", 256))
                score = float(pc_res.score)
                entry["score"] = score
                entry["raw_score"] = score
        else:
            resp, _ = engine.inspect(category="pcb1", image_bytes=img_path.read_bytes())
            dec = resp.decision.value  # PASS, FAIL, HUMAN, RETAKE
            path_val = resp.decision_path.value if resp.decision_path else "UNKNOWN"
            if resp.anomaly_score is not None:
                score = float(resp.anomaly_score)
            else:
                # RETAKE image: single-threshold perception baseline has no quality gate,
                # so compute true PatchCore score on standardized frame
                img_bgr = cv2.imread(str(img_path))
                std_img = standardize(img_bgr, 1024).image
                bundle = engine.bundle
                pc_res = score_image(bundle["net"], std_img, bundle["bank"], bundle.get("input_size", 256))
                score = float(pc_res.score)

            cache[rel_p] = {
                "score": score,
                "raw_score": score,
                "decision": dec,
                "decision_path": path_val,
                "ground_truth": true_label,
                "alignment_quality": resp.alignment.quality if resp.alignment else None,
                "alignment_method": resp.alignment.method if resp.alignment else None,
                "residual_mean": resp.measurements.residual_mean if resp.measurements else None,
                "rois_checked": resp.measurements.rois_checked if resp.measurements else 0,
                "rois_total": resp.measurements.rois_total if resp.measurements else 0,
                "timing_ms": resp.timing_ms.model_dump() if resp.timing_ms else None,
            }
            if (i + 1) % 10 == 0:
                save_cache(cache, commit_hash)
                
        raw_scores.append(score)
        ground_truth.append(true_label)
        
        # Baseline prediction: fail if score > t_fail else pass
        base_pred = "FAIL" if score > t_fail else "PASS"
        baseline_preds.append(base_pred)
        
        # Agent prediction
        agent_decisions.append(dec)
        agent_paths.append(path_val)
        
        if (i + 1) % 25 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            print(f"  [{i+1}/{total}] processed ({elapsed:.1f}s, {rate:.1f} img/s)")
            
    save_cache(cache, commit_hash)
    return {
        "scores": raw_scores,
        "baseline_preds": baseline_preds,
        "agent_decisions": agent_decisions,
        "agent_paths": agent_paths,
        "ground_truth": ground_truth,
    }


def run_measured_e1():
    print("=" * 70)
    print("      SUREINSPECT: MEASURED REAL-IMAGE EXPERIMENT E1 BENCHMARK        ")
    print("=" * 70)
    
    commit_hash = get_git_commit_hash()
    print(f"Active Git Commit: {commit_hash}")
    
    engine = get_live_engine("artifacts")
    if engine is None:
        raise RuntimeError("Cannot load live OpenCV engine from artifacts snapshot!")
        
    root = Path("datasets/VisA_pytorch/1cls")
    snapshot_dir = Path("artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11")
    splits_path = snapshot_dir / "splits.json"
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
        
    calib_path = snapshot_dir / "calibration.json"
    with open(calib_path, encoding="utf-8") as f:
        calib = json.load(f)
        
    t_pass = float(calib["t_pass"]["threshold"])
    t_fail = float(calib.get("t_fail_default", 2.2862818241119385))
    print(f"Conformal Thresholds: t_pass={t_pass:.4f}, t_fail={t_fail:.4f}")
    
    cache = load_cache(commit_hash)
    print(f"Loaded {len(cache)} existing cached predictions for commit {commit_hash[:8]}")
    
    # =========================================================================
    # Part 1: Development Split (297 normals, 26 anomalies)
    # =========================================================================
    dev_normals = splits["splits"]["dev_eval_normals"]
    dev_anomalies = splits["splits"]["dev_anomalies"]
    
    dev_norm_res = evaluate_image_split(
        engine, dev_normals, true_label="good", root=root, t_fail=t_fail, cache=cache, commit_hash=commit_hash, desc="Dev Normals"
    )
    dev_anom_res = evaluate_image_split(
        engine, dev_anomalies, true_label="bad", root=root, t_fail=t_fail, cache=cache, commit_hash=commit_hash, desc="Dev Anomalies"
    )
    
    dev_all_scores = dev_norm_res["scores"] + dev_anom_res["scores"]
    dev_all_gt = dev_norm_res["ground_truth"] + dev_anom_res["ground_truth"]
    dev_all_decisions = dev_norm_res["agent_decisions"] + dev_anom_res["agent_decisions"]
    dev_all_paths = dev_norm_res["agent_paths"] + dev_anom_res["agent_paths"]
    
    dev_baseline_metrics = evaluate_baseline(dev_all_scores, dev_all_gt, t_fail)
    dev_baseline_t_pass = evaluate_baseline(dev_all_scores, dev_all_gt, t_pass)
    dev_agent_metrics = evaluate_agent(dev_all_decisions, dev_all_paths, dev_all_gt)
    
    dev_base_fa = dev_baseline_metrics["counts"]["normal_fail"]
    dev_agent_fa = dev_agent_metrics["counts"]["normal_fail"]
    dev_fa_reduct = 100.0 * (dev_base_fa - dev_agent_fa) / max(1, dev_base_fa)
    
    # =========================================================================
    # Part 2: Final Test Split (100 normals, 70 anomalies)
    # Label: "re-run after fixes; split previously seen; not a clean hold-out"
    # =========================================================================
    test_normals = splits["splits"]["final_test"]["normals"]
    test_anomalies = splits["splits"]["final_test"]["anomalies"]
    
    test_norm_res = evaluate_image_split(
        engine, test_normals, true_label="good", root=root, t_fail=t_fail, cache=cache, commit_hash=commit_hash, desc="Test Normals (re-run after fixes)"
    )
    test_anom_res = evaluate_image_split(
        engine, test_anomalies, true_label="bad", root=root, t_fail=t_fail, cache=cache, commit_hash=commit_hash, desc="Test Anomalies (re-run after fixes)"
    )
    
    test_all_scores = test_norm_res["scores"] + test_anom_res["scores"]
    test_all_gt = test_norm_res["ground_truth"] + test_anom_res["ground_truth"]
    test_all_decisions = test_norm_res["agent_decisions"] + test_anom_res["agent_decisions"]
    test_all_paths = test_norm_res["agent_paths"] + test_anom_res["agent_paths"]
    
    test_baseline_metrics = evaluate_baseline(test_all_scores, test_all_gt, t_fail)
    test_baseline_t_pass = evaluate_baseline(test_all_scores, test_all_gt, t_pass)
    test_agent_metrics = evaluate_agent(test_all_decisions, test_all_paths, test_all_gt)
    
    test_base_fa = test_baseline_metrics["counts"]["normal_fail"]
    test_agent_fa = test_agent_metrics["counts"]["normal_fail"]
    test_fa_reduct = 100.0 * (test_base_fa - test_agent_fa) / max(1, test_base_fa)
    
    results_v2 = {
        "schema_version": "sureinspect.measured_e1.v1",
        "benchmark_type": "measured_real_images",
        "pipeline_version": "v2_unified",
        "commit": commit_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "generator": "eval.run_real_e1",
            "dataset": "Amazon VisA pcb1",
            "dataset_root": "datasets/VisA_pytorch/1cls",
            "snapshot_id": snapshot_dir.name,
            "snapshot_manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
            "splits_sha256": sha256_file(splits_path),
            "model_sha256": sha256_file(snapshot_dir / "model.onnx"),
            "bank_sha256": sha256_file(snapshot_dir / "bank.npz"),
            "calibration_sha256": sha256_file(calib_path),
            "quality_gate_sha256": sha256_file(snapshot_dir / "quality_gate.json"),
            "prediction_cache_commit": commit_hash,
            "prediction_count": len(cache),
        },
        "thresholds": {"t_pass": t_pass, "t_fail": t_fail},
        "dev_split": {
            "n_normals": len(dev_normals),
            "n_anomalies": len(dev_anomalies),
            "baseline": dev_baseline_metrics,
            "baseline_t_fail": dev_baseline_metrics,
            "baseline_t_pass": dev_baseline_t_pass,
            "agent": dev_agent_metrics,
            "false_alarm_reduction_pct": round(dev_fa_reduct, 2),
        },
        "final_test_split": {
            "n_normals": len(test_normals),
            "n_anomalies": len(test_anomalies),
            "label": "re-run after fixes; split previously seen; not a clean hold-out",
            "baseline": test_baseline_metrics,
            "baseline_t_fail": test_baseline_metrics,
            "baseline_t_pass": test_baseline_t_pass,
            "agent": test_agent_metrics,
            "false_alarm_reduction_pct": round(test_fa_reduct, 2),
        },
    }
    
    out_path_v2 = Path("eval/results/measured_e1_results_v2.json")
    out_path_v2.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_v2, "w", encoding="utf-8") as f:
        json.dump(results_v2, f, indent=2)
        
    print("\n" + "=" * 75)
    print("        REAL MEASURED RESULTS: SIDE-BY-SIDE COMPARISON (v1 vs v2)      ")
    print("=" * 75)
    
    v1_path = Path("eval/results/measured_e1_results.json")
    v1_results = None
    if v1_path.is_file():
        try:
            v1_results = json.loads(v1_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def get_summary(r, split_key):
        if not r or split_key not in r:
            return {}
        sp = r[split_key]
        ag = sp["agent"]
        ba = sp["baseline"]
        c = ag["counts"]
        cov = ag["coverage"]["rate"] * 100.0 if ag["coverage"]["rate"] is not None else 0.0
        esc = (c["defective_pass_fast"] + c["defective_pass_cleared"])
        fa = c["normal_fail"]
        human = (c["normal_human"] + c["defective_human"])
        retake = (c["normal_retake"] + c["defective_retake"])
        return {
            "cov": cov, "esc": esc, "fa": fa, "human": human, "retake": retake,
            "reduct": sp.get("false_alarm_reduction_pct", 0.0),
            "esc_ci": ag["defect_escapes"]["ci_95"],
            "fa_rate": (ag["false_fail"]["rate"] or 0.0) * 100.0,
        }

    v1_dev = get_summary(v1_results, "dev_split")
    v2_dev = get_summary(results_v2, "dev_split")
    v1_test = get_summary(v1_results, "final_test_split")
    v2_test = get_summary(results_v2, "final_test_split")

    print(f"{'Metric':<35} | {'v1 (Before Fixes)':<18} | {'v2 (Unified Pipeline)':<20}")
    print("-" * 75)
    print(f"{'[DEV] Coverage':<35} | {v1_dev.get('cov', 0.0):>17.1f}% | {v2_dev.get('cov', 0.0):>19.1f}%")
    print(f"{'[DEV] Defect Escapes (count)':<35} | {v1_dev.get('esc', 0):>18} | {v2_dev.get('esc', 0):>20}")
    print(f"{'[DEV] Normal False Alarms':<35} | {v1_dev.get('fa', 0):>18} | {v2_dev.get('fa', 0):>20}")
    print(f"{'[DEV] Operator Escalations (HUMAN)':<35} | {v1_dev.get('human', 0):>18} | {v2_dev.get('human', 0):>20}")
    print(f"{'[DEV] Retakes':<35} | {v1_dev.get('retake', 0):>18} | {v2_dev.get('retake', 0):>20}")
    print(f"{'[DEV] False Alarm Scrap Reduction':<35} | {v1_dev.get('reduct', 0.0):>17.1f}% | {v2_dev.get('reduct', 0.0):>19.1f}%")
    print("-" * 75)
    print(f"{'[TEST*] Coverage':<35} | {v1_test.get('cov', 0.0):>17.1f}% | {v2_test.get('cov', 0.0):>19.1f}%")
    print(f"{'[TEST*] Defect Escapes (count)':<35} | {v1_test.get('esc', 0):>18} | {v2_test.get('esc', 0):>20}")
    print(f"{'[TEST*] Normal False Alarms':<35} | {v1_test.get('fa', 0):>18} | {v2_test.get('fa', 0):>20}")
    print(f"{'[TEST*] Operator Escalations (HUMAN)':<35} | {v1_test.get('human', 0):>18} | {v2_test.get('human', 0):>20}")
    print(f"{'[TEST*] Retakes':<35} | {v1_test.get('retake', 0):>18} | {v2_test.get('retake', 0):>20}")
    print(f"{'[TEST*] False Alarm Scrap Reduction':<35} | {v1_test.get('reduct', 0.0):>17.1f}% | {v2_test.get('reduct', 0.0):>19.1f}%")
    print("=" * 75)
    print("* Note on Test Split: re-run after fixes; split previously seen; not a clean hold-out.")
    print(f"Results saved to: {out_path_v2}")
    return results_v2


if __name__ == "__main__":
    run_measured_e1()
