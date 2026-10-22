"""
eval/repro/reproduce_issues.py
Deterministic reproduction script for issues A1, A2, A3, A5, A6, A7, A8, D1, D2, D3.
Does not require the full VisA dataset or neural network weights.
Uses committed reference data and data/templates/pcb1/template_0*.png.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

results: Dict[str, Dict[str, Any]] = {}


def test_repro_a1():
    """A1: core/tools.py AlignResult attribute errors (M vs m, missing translation_px)."""
    from core.tools import AlignResult
    
    ar = AlignResult(M=np.eye(3), method="orb_ransac", valid=True)
    has_m = hasattr(ar, "m")
    has_translation_px = hasattr(ar, "translation_px")
    
    # In core/agent.py:
    # line 469: align_res.translation_px
    # line 476: align_res.m
    reproduced = (not has_m) and (not has_translation_px)
    results["A1"] = {
        "reproduced": reproduced,
        "detail": f"AlignResult has uppercase M: {hasattr(ar, 'M')}, lowercase m: {has_m}, translation_px: {has_translation_px}",
    }


def test_repro_a2():
    """A2: AlignmentInfo and MeasurementStats schema mismatches and suggested_decision='HUMAN'."""
    from backend.schemas import AlignmentInfo, MeasurementStats, InspectionResponse
    from core.agent import decide
    
    # Test decide() setting suggested_decision='HUMAN'
    dec_res = decide(
        score=2.15,
        lighting_shift=False,
        evidence=None,
        t_pass=2.04,
        t_fail=2.28,
        budget_state={"steps": 4, "budget": 4, "exhausted": True},
    )
    suggested_human = dec_res.suggested_decision == "HUMAN"
    
    # Test AlignmentInfo schema construction without q_min
    alignment_error = False
    try:
        AlignmentInfo(
            template_id="template_00",
            quality=0.85,
            method="orb_ecc",
            rotation_deg=0.0,
            # Omitting required q_min
        )
    except Exception as e:
        alignment_error = True
        
    reproduced = suggested_human and alignment_error
    results["A2"] = {
        "reproduced": reproduced,
        "detail": f"decide() returned suggested_decision='{dec_res.suggested_decision}', AlignmentInfo requires q_min (raised: {alignment_error})",
    }


def test_repro_a3():
    """A3: Split-brain and decide() spec violations."""
    from core.agent import decide
    
    # 1. Lighting shift auto-PASS bug: low residual under lighting shift returns PASS
    res_ls = decide(
        score=1.90,
        lighting_shift=True,
        evidence={"alignment_valid": True, "residual_p95": 0.02, "r_low": 0.05, "r_high": 0.12},
        t_pass=2.04,
        t_fail=2.28,
        budget_state={"steps": 2, "budget": 4},
    )
    ls_auto_pass = res_ls.decision == "PASS" and res_ls.decision_path == "AGENT_CLEARED"
    
    # 2. SUSPECTED_DEFECT inversion: score > t_fail with no secondary evidence returns FAIL instead of HUMAN
    res_sd = decide(
        score=2.35,
        lighting_shift=False,
        evidence=None,
        t_pass=2.04,
        t_fail=2.28,
        budget_state={"steps": 0, "budget": 4},
    )
    sd_fail = res_sd.decision == "FAIL" and res_sd.decision_path == "SUSPECTED_DEFECT"
    
    # 3. Budget exhausted false trigger on step 4
    res_b4 = decide(
        score=2.15,
        lighting_shift=False,
        evidence={"alignment_valid": True, "residual_p95": 0.02, "r_low": 0.05, "r_high": 0.12},
        t_pass=2.04,
        t_fail=2.28,
        budget_state={"steps": 4, "budget": 4, "exhausted": False},
    )
    premature_exhaust = res_b4.decision_path == "BUDGET_EXHAUSTED"
    
    reproduced = ls_auto_pass and sd_fail and premature_exhaust
    results["A3"] = {
        "reproduced": reproduced,
        "detail": f"Lighting-shift auto-PASS: {ls_auto_pass}, SUSPECTED_DEFECT returned FAIL: {sd_fail}, Premature budget exhaustion on step 4: {premature_exhaust}",
    }


def test_repro_a5():
    """A5: 180-degree PCB orientation failure in core/tools.py."""
    from core.tools import align_to_templates, load_templates, align
    
    t0 = cv2.imread(str(ROOT / "data/templates/pcb1/template_00.png"), cv2.IMREAD_GRAYSCALE)
    t1 = cv2.imread(str(ROOT / "data/templates/pcb1/template_01.png"), cv2.IMREAD_GRAYSCALE)
    
    if t0 is None or t1 is None:
        results["A5"] = {"reproduced": False, "detail": "Missing template images"}
        return
        
    res = align(t0, t1)
    # Bug A5: ORB fails due to abs(rot) > 45, ECC locks onto a false peak at ~25 deg with low quality (<0.50),
    # but marks valid=True because _finalize doesn't check q_min!
    reproduced = res.valid and (res.quality is not None and res.quality < 0.65) and (res.method == "rotation_ecc")
    results["A5"] = {
        "reproduced": reproduced,
        "detail": f"Aligning upright template_00 to 180-deg template_01 locked onto false peak: valid={res.valid}, method='{res.method}', rot={res.rotation_deg:.1f} deg, quality={res.quality:.3f} (< 0.65)",
    }


def test_repro_a6():
    """A6: find_default_snapshot() missing weight file existence check."""
    from core.agent import find_default_snapshot
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_snap = Path(tmp_dir) / "snapshots" / "pcb1-fake-snap"
        fake_snap.mkdir(parents=True)
        # Directory exists, but no model.onnx or bank.npz
        try:
            found_dir = find_default_snapshot("pcb1", root_dir=tmp_dir)
            reproduced = (found_dir == fake_snap)
        except FileNotFoundError:
            reproduced = False
        
    results["A6"] = {
        "reproduced": reproduced,
        "detail": f"find_default_snapshot() returned snapshot directory without verifying model.onnx/bank.npz: {reproduced}",
    }


def test_repro_a7():
    """A7: core/tests/test_bootstrap.py OpenCV 5 version hard assertion."""
    from pathlib import Path
    boot_test = (ROOT / "core/tests/test_bootstrap.py").read_text(encoding="utf-8")
    reproduced = 'startswith("5.")' in boot_test
    major = int(cv2.__version__.split(".")[0])
    results["A7"] = {
        "reproduced": reproduced,
        "detail": f"core/tests/test_bootstrap.py hardcodes startswith('5.'): {reproduced} (installed OpenCV is {cv2.__version__})",
    }


def test_repro_a8():
    """A8: Duplicate and clipping conformal logic in eval/run_calibration_splits.py."""
    from eval.run_calibration_splits import compute_conformal_threshold
    from core.calibration import conformal_threshold
    
    # n=10 calibration scores. For alpha=0.01: (n+1)*(1-0.01) = 11*0.99 = 10.89 -> rank 11 > 10.
    scores = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    core_thr = conformal_threshold(scores, 0.01)  # Should return None (k > n)
    eval_thr = compute_conformal_threshold(scores, 0.01)  # Clips to max score (2.8)
    
    reproduced = (core_thr is None) and (eval_thr == 2.8)
    results["A8"] = {
        "reproduced": reproduced,
        "detail": f"For n=10, alpha=0.01: core.calibration returned {core_thr} (k > n omitted), while eval returned {eval_thr} (clipped to n)",
    }


def test_repro_d1():
    """D1: Localized central defocus passes global Laplacian variance."""
    from core.quality_gate import check_quality
    t0_bgr = cv2.imread(str(ROOT / "data/templates/pcb1/template_00.png"))
    if t0_bgr is None:
        t0_bgr = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    
    h, w, _ = t0_bgr.shape
    blurred = t0_bgr.copy()
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    blurred[y1:y2, x1:x2] = cv2.GaussianBlur(blurred[y1:y2, x1:x2], (31, 31), 10.0)
    
    th = {
        "blur_thr": 189.95,
        "center_blur_thr": 200.0,
        "dark_max": 0.05,
        "bright_max": 0.12,
        "glare_thr": 0.10,
        "l_mean_ref": 137.0,
        "l_mean_sd": 2.5,
        "l_std_ref": 41.0,
        "n_fit": 99,
        "gate_version": "v1.1",
    }
    res = check_quality(blurred, th)
    reproduced = (res["status"] == "PASSED")
    results["D1"] = {
        "reproduced": reproduced,
        "detail": f"Centrally blurred image under v1.1 quality gate: status={res['status']}, reason='{res['reason']}'",
    }


def test_repro_d2():
    """D2: Sensor noise spoofing Laplacian variance on blurred image."""
    from core.quality_gate import check_quality
    t0_bgr = cv2.imread(str(ROOT / "data/templates/pcb1/template_00.png"))
    if t0_bgr is None:
        t0_bgr = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        
    blurred = cv2.GaussianBlur(t0_bgr, (19, 19), 0)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 12, blurred.shape).astype(np.float64)
    noisy_blurred = np.clip(blurred.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    th = {
        "blur_thr": 189.95,
        "denoised_blur_thr": 80.0,
        "dark_max": 0.05,
        "bright_max": 0.12,
        "glare_thr": 0.10,
        "l_mean_ref": 137.0,
        "l_mean_sd": 2.5,
        "l_std_ref": 41.0,
        "n_fit": 99,
        "gate_version": "v1.1",
    }
    res = check_quality(noisy_blurred, th)
    reproduced = (res["status"] == "PASSED")
    results["D2"] = {
        "reproduced": reproduced,
        "detail": f"Noisy blurred image under v1.1 quality gate: status={res['status']}, reason='{res['reason']}'",
    }


def test_repro_d3():
    """D3: Warm/tinted specular glare misses HSV (V >= 240, S <= 35) gate."""
    from core.quality_gate import check_quality
    bgr_tinted = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr_tinted[:, :] = [180, 255, 255]
    th = {
        "blur_thr": 10.0,
        "dark_max": 0.05,
        "bright_max": 0.12,
        "glare_thr": 0.10,
        "l_mean_ref": 137.0,
        "l_mean_sd": 2.5,
        "l_std_ref": 41.0,
        "n_fit": 99,
        "gate_version": "v1.1",
    }
    res = check_quality(bgr_tinted, th)
    reproduced = (res["status"] == "PASSED")
    results["D3"] = {
        "reproduced": reproduced,
        "detail": f"Tinted LED glare under v1.1 quality gate: status={res['status']}, reason='{res['reason']}'",
    }


def main():
    print("=" * 70)
    print("      SureInspect Phase 0: Issue Reproduction Suite                 ")
    print("=" * 70)
    
    test_repro_a1()
    test_repro_a2()
    test_repro_a3()
    test_repro_a5()
    test_repro_a6()
    test_repro_a7()
    test_repro_a8()
    test_repro_d1()
    test_repro_d2()
    test_repro_d3()
    
    all_repro = True
    for k, v in results.items():
        status = "REPRODUCED" if v["reproduced"] else "NOT REPRODUCED"
        print(f"[{status}] {k}: {v['detail']}")
        if not v["reproduced"]:
            all_repro = False
            
    repro_count = sum(1 for v in results.values() if v["reproduced"])
    print("=" * 70)
    print(f"Summary: {repro_count}/{len(results)} issues reproduced. ({len(results) - repro_count} verified resolved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
