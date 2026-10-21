"""
core/agent.py
Live Agentic Inspection Engine for SureInspect v8.1.

Implements the complete live decision cycle:
1. Standardization: converts BGR uint8 input to standardized resolution (max side 1024, INTER_AREA).
2. Quality Gate: checks blur, dark/bright exposure, specular glare, and lighting covariate shift.
3. PatchCore Perception: runs OpenCV 5 DNN backbone, extracts feature maps, computes nearest-neighbor distances against coreset memory bank, generates heatmaps and extracts candidate ROIs.
4. Conformal Calibration: evaluates distance score against non-parametric split-conformal thresholds (t_pass, t_fail).
5. Evidence Tools (T1-T5): executes multiscale template alignment (ORB+ECC), photometric matching, difference residuals, and rotated rectangle defect geometry measurements.
6. Pure Policy Decision: maps perception score, lighting flag, evidence residuals, and step budgets to one of the 10 frozen decision paths.

Rules:
- inspect_standardized() receives an ALREADY standardized BGR uint8 image. Never resizes twice.
- The score is a raw distance, not a probability.
- Fail is `score > t_fail`; fast pass is `score <= t_pass` (full precision, never rounded).
- CLAHE, sharpened or zoomed images are never fed back into the scorer.
- No live-ready artifact may contain stub thresholds (gate_version == "stub").
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from core.calibration import conformal_threshold
from core.degrade import apply_degradations
from core.perception import extract_patch_features, load_backbone, make_heatmap, find_rois, nn_distances
from core.preprocess import StandardizedImage, standardize, to_blob
from core.quality_gate import assert_live_ready, check_quality, validate_standardized_image
from core.tools import (
    align_to_templates,
    load_templates,
    measure,
    nearest_order,
    normalize_lighting,
    prepare_comparison,
    roi_residual,
)


@dataclass
class DecisionResult:
    decision: str            # "PASS" | "FAIL" | "HUMAN" | "RETAKE"
    decision_path: str       # One of the 10 frozen DecisionPathEnum values
    reason: str
    suggested_decision: str | None = None


# ---------------------------------------------------------------------------
# Pure Decision Rule: decide()
# ---------------------------------------------------------------------------

def decide(
    score: float | None,
    lighting_shift: bool,
    evidence: Any,
    t_pass: float,
    t_fail: float,
    budget_state: Any,
) -> DecisionResult:
    """Pure decision rule: maps score, lighting shift, tool evidence and budget state to decision and path."""
    # 0. Quality gate rejection / missing score
    if score is None:
        return DecisionResult(
            decision="RETAKE",
            decision_path="QUALITY_REJECTED",
            reason="Quality gate rejected frame before perception scoring",
            suggested_decision=None,
        )

    # 1. Check budget exhaustion
    if isinstance(budget_state, dict):
        steps = budget_state.get("steps", 0)
        budget = budget_state.get("budget", 4)
        is_exhausted = budget_state.get("exhausted", False) or (budget > 0 and steps >= budget and not evidence)
        if is_exhausted:
            return DecisionResult(
                decision="HUMAN",
                decision_path="BUDGET_EXHAUSTED",
                reason=f"Step budget ({budget}) exhausted before reaching a conclusive automated verdict",
                suggested_decision=None,
            )

    # 2. Fast pass zone: score <= t_pass and normal illumination
    if score <= t_pass and not lighting_shift:
        return DecisionResult(
            decision="PASS",
            decision_path="FAST_PASS",
            reason=f"Perception score {score:.4f} <= t_pass ({t_pass:.4f}) with normal illumination",
            suggested_decision=None,
        )

    # Lighting shift with score <= t_pass and no evidence routes to LIGHTING_SHIFT_REVIEW
    if score <= t_pass and lighting_shift and not evidence:
        return DecisionResult(
            decision="HUMAN",
            decision_path="LIGHTING_SHIFT_REVIEW",
            reason=f"Perception score {score:.4f} <= t_pass ({t_pass:.4f}), but factory illumination covariate shift requires review",
            suggested_decision="PASS",
        )

    # Defect score with no secondary evidence routes to SUSPECTED_DEFECT
    if score > t_fail and not evidence:
        return DecisionResult(
            decision="HUMAN",
            decision_path="SUSPECTED_DEFECT",
            reason=f"Perception score {score:.4f} > t_fail ({t_fail:.4f}); secondary evidence not evaluated",
            suggested_decision="FAIL",
        )

    # Ambiguous score with no secondary evidence routes to INCONCLUSIVE
    if not evidence:
        return DecisionResult(
            decision="HUMAN",
            decision_path="INCONCLUSIVE",
            reason=f"Perception score {score:.4f} in ambiguous range ({t_pass:.4f}, {t_fail:.4f}] with no secondary evidence",
            suggested_decision=None,
        )

    # 3. Any path past FAST_PASS requires evidence with alignment_valid=True
    if not isinstance(evidence, dict) or not evidence.get("alignment_valid", False):
        reason_msg = "template alignment failed"
        if isinstance(evidence, dict) and evidence.get("reason"):
            reason_msg = f"template alignment failed ({evidence.get('reason')})"
        return DecisionResult(
            decision="HUMAN",
            decision_path="ALIGN_FAILED",
            reason=f"Secondary verification required but {reason_msg}",
            suggested_decision=None,
        )

    # 4. Resolve active residual thresholds (CLAHE copy thresholds under lighting shift)
    if lighting_shift:
        r_low = float(evidence.get("clahe_r_low", evidence.get("r_low", 0.38)))
        r_high = float(evidence.get("clahe_r_high", evidence.get("r_high", 0.48)))
    else:
        r_low = float(evidence.get("r_low", 0.38))
        r_high = float(evidence.get("r_high", 0.48))

    r_val = float(evidence.get("residual_p95", evidence.get("residual_mean", 0.0)))
    rois_total = int(evidence.get("rois_total", 1))
    rois_checked = int(evidence.get("rois_checked", 1))

    # 5. High residual branch (r_val >= r_high)
    if r_val >= r_high:
        if not lighting_shift and score > t_fail:
            return DecisionResult(
                decision="FAIL",
                decision_path="AGENT_CONFIRMED",
                reason=f"Perception score {score:.4f} > t_fail ({t_fail:.4f}) and structural residual {r_val:.4f} >= r_high ({r_high:.4f}) confirm defect",
                suggested_decision=None,
            )
        else:
            return DecisionResult(
                decision="HUMAN",
                decision_path="SUSPECTED_DEFECT",
                reason=f"Structural residual {r_val:.4f} >= r_high ({r_high:.4f}) indicates defect, but score ({score:.4f}) or lighting shift requires human review",
                suggested_decision="FAIL",
            )

    # 6. Mid-band residual branch (r_low < r_val < r_high)
    if r_low < r_val < r_high:
        return DecisionResult(
            decision="HUMAN",
            decision_path="INCONCLUSIVE",
            reason=f"Structural residual {r_val:.4f} lies in indeterminate zone ({r_low:.4f}, {r_high:.4f})",
            suggested_decision=None,
        )

    # 7. Low residual branch (r_val <= r_low across all checked ROIs)
    # Check 3rd ROI rule: when 3 contour-derived ROIs exist and only 2 were checked, force INCONCLUSIVE
    if rois_total >= 3 and rois_checked < rois_total:
        return DecisionResult(
            decision="HUMAN",
            decision_path="INCONCLUSIVE",
            reason=f"Checked {rois_checked} of {rois_total} candidate defect ROIs; remaining unchecked region requires operator review",
            suggested_decision=None,
        )

    if score > t_fail:
        return DecisionResult(
            decision="HUMAN",
            decision_path="EVIDENCE_CONFLICT",
            reason=f"Perception score {score:.4f} > t_fail ({t_fail:.4f}) indicates anomaly, but structural residual {r_val:.4f} <= r_low ({r_low:.4f}) shows normal variation",
            suggested_decision=None,
        )

    if lighting_shift:
        return DecisionResult(
            decision="HUMAN",
            decision_path="LIGHTING_SHIFT_REVIEW",
            reason=f"Structural residual {r_val:.4f} <= r_low ({r_low:.4f}) confirms geometry, but factory illumination covariate shift requires review",
            suggested_decision="PASS",
        )

    return DecisionResult(
        decision="PASS",
        decision_path="AGENT_CLEARED",
        reason=f"Ambiguous perception score {score:.4f} cleared: structural residual {r_val:.4f} <= r_low ({r_low:.4f}) confirms normal assembly",
        suggested_decision=None,
    )


# ---------------------------------------------------------------------------
# Cached Snapshot & Templates Manager
# ---------------------------------------------------------------------------

_SNAPSHOT_CACHE: dict[str, dict[str, Any]] = {}
_TEMPLATES_CACHE: dict[str, dict[str, Any]] = {}


def find_default_snapshot(category: str = "pcb1", root_dir: Path | str = "artifacts") -> Path:
    """Finds the published snapshot directory for category, verifying weight files exist."""
    snaps_dir = Path(root_dir) / "snapshots"
    if snaps_dir.is_dir():
        for d in sorted(snaps_dir.iterdir()):
            if (
                d.is_dir()
                and d.name.startswith(f"{category}-")
                and (d / "model.onnx").is_file()
                and (d / "bank.npz").is_file()
            ):
                return d
    raise FileNotFoundError(f"No published snapshot with weights found for category '{category}' in {snaps_dir}")


def load_snapshot_bundle(snapshot_path: Path | str) -> dict[str, Any]:
    """Loads and caches the immutable snapshot bundle."""
    p = Path(snapshot_path).resolve()
    key = str(p)
    if key in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[key]

    manifest_p = p / "manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Missing manifest.json in snapshot: {p}")
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    onnx_p = p / "model.onnx"
    net = load_backbone(onnx_p)

    bank_p = p / "bank.npz"
    bank = np.load(bank_p)["bank"].astype(np.float32)

    calib_p = p / "calibration.json"
    calib = json.loads(calib_p.read_text(encoding="utf-8"))

    gate_p = p / "quality_gate.json"
    gate = json.loads(gate_p.read_text(encoding="utf-8"))
    assert_live_ready(gate)

    bundle = {
        "path": p,
        "manifest": manifest,
        "net": net,
        "bank": bank,
        "calibration": calib,
        "quality_gate": gate,
        "input_size": manifest.get("preprocessing", {}).get("backbone_input_size", 256),
    }
    _SNAPSHOT_CACHE[key] = bundle
    return bundle


def get_category_templates(category: str = "pcb1") -> dict[str, Any] | None:
    """Loads and caches golden reference templates for a category."""
    if category in _TEMPLATES_CACHE:
        return _TEMPLATES_CACHE[category]

    tmpl_dir = Path("data/templates") / category
    if not tmpl_dir.is_dir():
        return None

    try:
        data = load_templates(tmpl_dir)
        _TEMPLATES_CACHE[category] = data
        return data
    except Exception:
        return None


def get_evidence_thresholds(category: str = "pcb1") -> dict[str, Any]:
    """Loads final evidence thresholds (q_min, r_low, r_high)."""
    p = Path("artifacts/evidence") / f"{category}_256_final.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "thresholds": {
            "q_min": 0.74,
            "r_low": 0.38,
            "r_high": 0.48,
        }
    }


# ---------------------------------------------------------------------------
# Core Engine: inspect_standardized()
# ---------------------------------------------------------------------------

def inspect_standardized(
    image_std: np.ndarray,
    category: str = "pcb1",
    flags: Mapping[str, Any] | None = None,
    alpha_fail: float = 0.05,
    snapshot: Any = None,
) -> dict[str, Any]:
    """
    Run full inspection on an ALREADY standardized BGR uint8 image.
    Never resizes the image a second time.
    """
    validate_standardized_image(image_std)
    t_start = time.time()
    flags = flags or {}

    # 1. Resolve snapshot bundle
    if snapshot is None:
        snap_path = find_default_snapshot(category)
        bundle = load_snapshot_bundle(snap_path)
    elif isinstance(snapshot, (str, Path)):
        bundle = load_snapshot_bundle(snapshot)
    elif isinstance(snapshot, dict) and "net" in snapshot:
        bundle = snapshot
    else:
        raise ValueError(f"Unsupported snapshot specification: {type(snapshot).__name__}")

    net = bundle["net"]
    bank = bundle["bank"]
    calib = bundle["calibration"]
    gate = bundle["quality_gate"]
    input_size = bundle["input_size"]

    trace = []
    step_num = 1
    t0 = time.time()

    # 2. Step 1: Quality Gate
    qg_res = check_quality(image_std, gate)
    qg_duration = (time.time() - t0) * 1000.0

    trace.append({
        "step": step_num,
        "tool": "QUALITY_GATE",
        "reason": qg_res["reason"],
        "status": qg_res["status"],
        "counts_toward_budget": False,
        "duration_ms": round(qg_duration, 1),
        "overlay": None,
    })
    step_num += 1

    if qg_res["status"] == "FAILED":
        dec_res = decide(None, False, None, 1.0, 2.0, {})
        return {
            "decision": dec_res.decision,
            "decision_path": dec_res.decision_path,
            "suggested_decision": None,
            "score": None,
            "quality_gate": qg_res,
            "heatmap": None,
            "patch_map": None,
            "rois": [],
            "alignment": None,
            "measurements": None,
            "trace": trace,
            "steps_used": 0,
            "timing_ms": {
                "quality": round(qg_duration, 1),
                "perception": 0.0,
                "tools": 0.0,
                "total": round((time.time() - t_start) * 1000.0, 1),
            },
            "grounded_report": f"Quality gate rejected frame: {qg_res['reason']}. No inference performed.",
        }

    # 3. Step 2: PatchCore Perception
    t0 = time.time()
    blob = to_blob(image_std, input_size)
    feats, grid = extract_patch_features(net, blob)
    dists = nn_distances(feats, bank)
    patch_map = dists.reshape(grid).astype(np.float32)
    score = float(patch_map.max())
    std_h, std_w = image_std.shape[:2]
    heatmap = make_heatmap(patch_map, (std_w, std_h))
    perc_duration = (time.time() - t0) * 1000.0

    # Resolve conformal calibration thresholds
    t_pass = float(calib["t_pass"]["threshold"])
    alpha_key = f"{alpha_fail:g}"
    if alpha_key in calib.get("t_fail_options", {}):
        t_fail = float(calib["t_fail_options"][alpha_key]["threshold"])
    else:
        t_fail = float(calib.get("t_fail_default", 2.05))

    is_lighting_shift = bool(qg_res.get("lighting_shift_detected", False))

    trace.append({
        "step": step_num,
        "tool": "PATCHCORE_PERCEPTION",
        "reason": f"Anomaly score {score:.3f} computed against coreset memory bank",
        "status": "COMPLETED",
        "counts_toward_budget": False,
        "duration_ms": round(perc_duration, 1),
        "overlay": heatmap,
    })
    step_num += 1

    # 4. Fast Pass Check
    templates_bundle = get_category_templates(category)
    default_ref_bgr = None
    if templates_bundle and templates_bundle.get("items"):
        first_tmpl = templates_bundle["items"][0]
        default_ref_bgr = first_tmpl.get("bgr")
        if default_ref_bgr is None and "gray" in first_tmpl:
            default_ref_bgr = cv2.cvtColor(first_tmpl["gray"], cv2.COLOR_GRAY2BGR)

    if score <= t_pass and not is_lighting_shift:
        dec_res = decide(score, False, None, t_pass, t_fail, {"steps": 0, "budget": 4})
        return {
            "decision": dec_res.decision,
            "decision_path": dec_res.decision_path,
            "suggested_decision": None,
            "score": score,
            "thresholds": {"t_pass": t_pass, "t_fail": t_fail, "alpha_pass": 0.20, "alpha_fail": alpha_fail},
            "quality_gate": qg_res,
            "heatmap": heatmap,
            "patch_map": patch_map,
            "rois": [],
            "alignment": None,
            "measurements": None,
            "reference_aligned": default_ref_bgr,
            "trace": trace,
            "steps_used": 0,
            "timing_ms": {
                "quality": round(qg_duration, 1),
                "perception": round(perc_duration, 1),
                "tools": 0.0,
                "total": round((time.time() - t_start) * 1000.0, 1),
            },
            "grounded_report": f"Score {score:.4f} <= pass threshold {t_pass:.4f} with normal illumination. Fast pass emitted.",
        }

    # 5. Extract ROIs
    patch_thr = float(calib.get("heatmap", {}).get("patch_thr", 1.5))
    rois = find_rois(heatmap, patch_map, patch_thr=patch_thr, max_rois=3)
    primary_roi = rois[0]["box"] if rois else [0, 0, std_w, std_h]

    # 6. Execute Evidence Tools T1-T5
    t_tools_start = time.time()
    steps_used = 0
    evidence_thresholds = get_evidence_thresholds(category)
    th = evidence_thresholds.get("thresholds", {})
    q_min = float(th.get("q_min", 0.65))
    r_low = float(th.get("r_low", 0.38))
    r_high = float(th.get("r_high", 0.48))
    clahe_r_low = float(th.get("clahe_r_low", r_low))
    clahe_r_high = float(th.get("clahe_r_high", r_high))

    alignment_info = None
    measurements_info = None
    evidence_payload: dict[str, Any] = {}
    warped_ref = default_ref_bgr

    test_gray = cv2.cvtColor(image_std, cv2.COLOR_BGR2GRAY)
    if is_lighting_shift:
        t0 = time.time()
        test_gray = normalize_lighting(test_gray)
        t4_duration = (time.time() - t0) * 1000.0
        steps_used += 1
        trace.append({
            "step": step_num,
            "tool": "T4_LIGHT_NORMALIZE",
            "reason": "Lighting shift detected; compute normalized reference comparison",
            "status": "COMPLETED",
            "counts_toward_budget": True,
            "duration_ms": round(t4_duration, 1),
            "overlay": None,
        })
        step_num += 1

    if templates_bundle is not None and templates_bundle.get("items"):
        t0 = time.time()
        feat_mean = feats.mean(axis=0) if feats is not None else None
        order = nearest_order(feat_mean, templates_bundle.get("embeddings"), len(templates_bundle["items"]))
        align_res, tried = align_to_templates(test_gray, templates_bundle["items"], order, q_min=q_min)
        t1_duration = (time.time() - t0) * 1000.0
        steps_used += 1

        trace.append({
            "step": step_num,
            "tool": "T1_ALIGN_REFERENCE",
            "reason": f"Aligned with template {getattr(align_res, 'template_id', 'none')} (quality={getattr(align_res, 'quality', 0):.3f})" if getattr(align_res, 'valid', False) else f"Alignment failed: {getattr(align_res, 'reason', 'failed')}",
            "status": "COMPLETED" if getattr(align_res, 'valid', False) else "FAILED",
            "counts_toward_budget": True,
            "duration_ms": round(t1_duration, 1),
            "overlay": None,
        })
        step_num += 1

        if align_res is not None and align_res.valid and align_res.M is not None:
            alignment_info = {
                "template_id": align_res.template_id,
                "quality": round(float(align_res.quality), 4) if align_res.quality is not None else None,
                "q_min": float(q_min),
                "method": align_res.method,
                "rotation_deg": round(float(align_res.rotation_deg), 2) if align_res.rotation_deg is not None else 0.0,
            }

            ref_item = next(t for t in templates_bundle["items"] if t["id"] == align_res.template_id)
            ref_gray = ref_item["gray"]
            if is_lighting_shift:
                ref_gray = normalize_lighting(ref_gray)

            warped_ref = cv2.warpAffine(
                ref_item["bgr"] if "bgr" in ref_item else cv2.cvtColor(ref_item["gray"], cv2.COLOR_GRAY2BGR),
                align_res.M[:2],
                (std_w, std_h),
                flags=cv2.INTER_LINEAR,
            )

            # T2: ROI 0 residual
            t0 = time.time()
            cmp = prepare_comparison(test_gray, ref_gray, align_res.M, match=True)
            res_dict0 = roi_residual(cmp, primary_roi)
            t2_duration = (time.time() - t0) * 1000.0
            steps_used += 1

            active_res = res_dict0
            active_roi = primary_roi
            rois_checked = 1
            r0 = float(res_dict0.get("residual_p95", 0.0))
            r_low_active = clahe_r_low if is_lighting_shift else r_low
            r_high_active = clahe_r_high if is_lighting_shift else r_high

            trace.append({
                "step": step_num,
                "tool": "T2_ROI_COMPARE",
                "reason": f"Computed structural residual p95={r0:.3f} over ROI",
                "status": "COMPLETED" if res_dict0.get("available") else "FAILED",
                "counts_toward_budget": True,
                "duration_ms": round(t2_duration, 1),
                "overlay": None,
            })
            step_num += 1

            # T5: Next ROI compare (if ROI 0 clean, multiple ROIs, and budget permits)
            if r0 <= r_low_active and len(rois) >= 2 and steps_used < 4 and ((time.time() - t_start) < 2.0):
                t0 = time.time()
                second_roi = rois[1]["box"]
                res_dict1 = roi_residual(cmp, second_roi)
                t5_duration = (time.time() - t0) * 1000.0
                steps_used += 1
                rois_checked = 2
                r1 = float(res_dict1.get("residual_p95", 0.0))
                if r1 > r0:
                    active_res = res_dict1
                    active_roi = second_roi

                trace.append({
                    "step": step_num,
                    "tool": "T5_NEXT_ROI_COMPARE",
                    "reason": f"Compare secondary proposed candidate ROI: residual p95={r1:.3f}",
                    "status": "COMPLETED" if res_dict1.get("available") else "FAILED",
                    "counts_toward_budget": True,
                    "duration_ms": round(t5_duration, 1),
                    "overlay": None,
                })
                step_num += 1

            # T3: Zoom measure
            meas_dict = {"available": False}
            if steps_used < 4 and ((time.time() - t_start) < 2.0):
                t0 = time.time()
                meas_dict = measure(cmp, active_roi, image_area=float(std_w * std_h))
                t3_duration = (time.time() - t0) * 1000.0
                steps_used += 1

                trace.append({
                    "step": step_num,
                    "tool": "T3_ROI_ZOOM_MEASURE",
                    "reason": f"Measured defect: area={meas_dict.get('area_pct', 0):.3f}%, shape={meas_dict.get('shape_class', 'none')}" if meas_dict.get("available") else "No measurable difference contour found",
                    "status": "COMPLETED" if meas_dict.get("available") else "SKIPPED",
                    "counts_toward_budget": True,
                    "duration_ms": round(t3_duration, 1),
                    "overlay": None,
                })
                step_num += 1

            r_p95_val = float(active_res.get("residual_p95", 0.0))
            r_mean_val = float(active_res.get("residual_mean", r_p95_val))
            raw_shape = meas_dict.get("shape_class")
            valid_shape = raw_shape if raw_shape in ("elongated", "compact", "irregular") else None
            measurements_info = {
                "roi_bbox": [int(v) for v in (meas_dict.get("contour_bbox") or active_roi)],
                "roi_source": "threshold" if rois else "peak_fallback",
                "rois_checked": rois_checked,
                "rois_total": len(rois) if rois else 1,
                "residual_mean": round(r_mean_val, 4),
                "r_low": float(r_low_active),
                "r_high": float(r_high_active),
                "area_pct": round(float(meas_dict["area_pct"]), 4) if meas_dict.get("area_pct") is not None else None,
                "area_denominator": "image",
                "elongation": round(float(meas_dict["elongation"]), 2) if meas_dict.get("elongation") is not None else None,
                "shape_class": valid_shape,
            }

            evidence_payload = {
                "alignment_valid": True,
                "residual_p95": r_p95_val,
                "residual_mean": r_mean_val,
                "r_low": r_low,
                "r_high": r_high,
                "clahe_r_low": clahe_r_low,
                "clahe_r_high": clahe_r_high,
                "rois_total": len(rois) if rois else 1,
                "rois_checked": rois_checked,
                "measure": meas_dict,
            }
        else:
            evidence_payload = {
                "alignment_valid": False,
                "reason": getattr(align_res, "reason", "alignment failed"),
            }
    else:
        evidence_payload = {"alignment_valid": False, "reason": "No templates available"}

    tools_duration = (time.time() - t_tools_start) * 1000.0

    # 7. Final Decision via decide()
    budget_state = {"steps": steps_used, "budget": 4}
    dec_res = decide(score, is_lighting_shift, evidence_payload, t_pass, t_fail, budget_state)

    total_duration = (time.time() - t_start) * 1000.0

    report = (
        f"Score {score:.3f} evaluated against t_pass={t_pass:.3f}, t_fail={t_fail:.3f}. "
        f"Decision: {dec_res.decision} ({dec_res.decision_path}). {dec_res.reason}."
    )

    return {
        "decision": dec_res.decision,
        "decision_path": dec_res.decision_path,
        "suggested_decision": dec_res.suggested_decision,
        "score": score,
        "thresholds": {"t_pass": t_pass, "t_fail": t_fail, "alpha_pass": 0.20, "alpha_fail": alpha_fail},
        "quality_gate": qg_res,
        "heatmap": heatmap,
        "patch_map": patch_map,
        "rois": rois,
        "alignment": alignment_info,
        "measurements": measurements_info,
        "reference_aligned": warped_ref,
        "trace": trace,
        "steps_used": steps_used,
        "timing_ms": {
            "quality": round(qg_duration, 1),
            "perception": round(perc_duration, 1),
            "tools": round(tools_duration, 1),
            "total": round(total_duration, 1),
        },
        "grounded_report": report,
    }


# ---------------------------------------------------------------------------
# Backend Integration Hook: run_inspection()
# ---------------------------------------------------------------------------

def run_inspection(
    category: str = "pcb1",
    sample_id: str | None = None,
    image_bytes: bytes | None = None,
    decoded_img: np.ndarray | None = None,
    simulate_blur: bool = False,
    simulate_darken: bool = False,
    simulate_glare: bool = False,
    alpha_fail: float = 0.05,
    path: str | None = None,
    orig_dimensions: tuple[int, int] | None = None,
) -> tuple[Any, dict[str, tuple[bytes, str]]]:
    """
    Standard entrypoint called by backend.executor.InspectionExecutor.
    Returns (InspectionResponse, assets_to_store).
    """
    from backend.schemas import (
        AlignmentInfo,
        AssetsInfo,
        CalibrationInfo,
        CounterfactualItem,
        DecisionEnum,
        DecisionPathEnum,
        ImageMetadata,
        InspectionResponse,
        MeasurementStats,
        QualityGateResult,
        SnapshotInfo,
        ThresholdsInfo,
        TimingInfo,
        ToolEnum,
        TraceStep,
    )

    # 1. Resolve raw image
    if decoded_img is not None:
        raw_bgr = decoded_img
    elif image_bytes is not None:
        raw_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    elif sample_id:
        sample_file = Path("data/samples") / f"{sample_id}.png"
        if sample_file.is_file():
            raw_bgr = cv2.imread(str(sample_file))
        else:
            raise FileNotFoundError(f"Sample not found: {sample_id}")
    else:
        raise ValueError("Either image_bytes, decoded_img, or sample_id must be provided")

    if raw_bgr is None or raw_bgr.size == 0:
        raise ValueError("Failed to load/decode valid image")

    orig_h, orig_w = raw_bgr.shape[:2]
    orig_dims = orig_dimensions or (orig_w, orig_h)

    # 2. Standardize
    std = standardize(raw_bgr)
    img_std = std.image

    # 3. Apply simulated degradations if requested
    applied_degradations = []
    if simulate_blur or simulate_darken or simulate_glare:
        img_std, applied_degradations = apply_degradations(
            img_std, blur=simulate_blur, darken=simulate_darken, glare=simulate_glare
        )

    # 4. Run live inspection
    res = inspect_standardized(
        image_std=img_std,
        category=category,
        flags={"degradations": applied_degradations},
        alpha_fail=alpha_fail,
    )

    insp_id = f"insp_{uuid.uuid4().hex[:8]}"
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 5. Build Assets
    assets_to_store: dict[str, tuple[bytes, str]] = {}

    # Standardized input asset
    success, png_input = cv2.imencode(".png", img_std)
    input_key = f"{insp_id}_input.png"
    assets_to_store[input_key] = (png_input.tobytes(), "image/png")

    # Heatmap asset
    heatmap_key = None
    if res.get("heatmap") is not None:
        norm_heat = (np.clip(res["heatmap"], 0, 3.0) / 3.0 * 255).astype(np.uint8)
        color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
        success, heat_bytes = cv2.imencode(".png", color_heat)
        heatmap_key = f"{insp_id}_heatmap.png"
        assets_to_store[heatmap_key] = (heat_bytes.tobytes(), "image/png")

    # Trace steps conversion
    execution_trace = []
    for st in res.get("trace", []):
        overlay_url = None
        if st.get("overlay") is not None and heatmap_key:
            overlay_url = f"/api/assets/{heatmap_key}"

        tool_val = st.get("tool", "QUALITY_GATE")
        try:
            tool_enum = ToolEnum(tool_val)
        except ValueError:
            tool_enum = ToolEnum.QUALITY_GATE

        execution_trace.append(
            TraceStep(
                step=st["step"],
                tool=tool_enum,
                reason=st["reason"],
                status=st["status"],
                counts_toward_budget=st.get("counts_toward_budget", False),
                duration_ms=st.get("duration_ms", 0.0),
                overlay_url=overlay_url,
            )
        )

    qg_dict = res["quality_gate"]
    quality_gate_obj = QualityGateResult(
        status=qg_dict["status"],
        action=qg_dict.get("action"),
        reason=qg_dict["reason"],
        blur_variance=float(qg_dict.get("blur_variance", 0.0)),
        blur_threshold=float(qg_dict.get("blur_threshold", 0.0)),
        dark_fraction=float(qg_dict.get("dark_fraction", 0.0)),
        bright_fraction=float(qg_dict.get("bright_fraction", 0.0)),
        glare_fraction=float(qg_dict.get("glare_fraction", 0.0)),
        glare_threshold=float(qg_dict.get("glare_threshold", 0.0)),
        l_mean=float(qg_dict.get("l_mean", 0.0)),
        exposure_ok=bool(qg_dict.get("exposure_ok", True)),
        lighting_shift_detected=bool(qg_dict.get("lighting_shift_detected", False)),
    )

    alignment_obj = None
    if res.get("alignment"):
        al = res["alignment"]
        alignment_obj = AlignmentInfo(
            template_id=al["template_id"],
            method=al.get("method", "orb_ecc"),
            quality=al.get("quality"),
            q_min=float(al.get("q_min", 0.65)),
            rotation_deg=float(al.get("rotation_deg", 0.0)),
        )

    measurements_obj = None
    if res.get("measurements"):
        ms = res["measurements"]
        measurements_obj = MeasurementStats(
            roi_bbox=tuple(ms.get("roi_bbox", [0, 0, 0, 0])),
            roi_source=ms.get("roi_source", "peak_fallback"),
            rois_checked=int(ms.get("rois_checked", 1)),
            rois_total=int(ms.get("rois_total", 1)),
            residual_mean=float(ms.get("residual_mean", 0.0)),
            r_low=float(ms.get("r_low", 0.04)),
            r_high=float(ms.get("r_high", 0.12)),
            area_pct=float(ms["area_pct"]) if ms.get("area_pct") is not None else None,
            area_denominator=ms.get("area_denominator", "image"),
            elongation=float(ms["elongation"]) if ms.get("elongation") is not None else None,
            shape_class=ms.get("shape_class"),
        )

    th = res.get("thresholds", {})
    t_pass_val = th.get("t_pass")
    t_fail_val = th.get("t_fail")

    response = InspectionResponse(
        schema_version="1.0",
        inspection_id=insp_id,
        request_id=req_id,
        timestamp=now_iso,
        engine="live",
        category=category,
        snapshot=SnapshotInfo(
            model_version="wide_resnet50_2_256",
            bank_version=f"{category}_wide_resnet50_2_256",
            calibration_version=f"{category}_256",
            policy_version="v8.1_live",
        ),
        image=ImageMetadata(
            original_size=orig_dims,
            standardized_size=std.standardized_size,
            scale_x=std.scale_x,
            scale_y=std.scale_y,
        ),
        decision=DecisionEnum(res["decision"]),
        decision_path=DecisionPathEnum(res["decision_path"]),
        suggested_decision=res.get("suggested_decision"),
        anomaly_score=res.get("score"),
        thresholds=ThresholdsInfo(
            t_pass=t_pass_val,
            t_fail=t_fail_val,
            alpha_pass=0.20,
            alpha_fail=alpha_fail,
            n_calib=300,
        ),
        calibration=CalibrationInfo(
            status="frozen_assumption_dependent" if not qg_dict.get("lighting_shift_detected") else "not_applicable_lighting_shift",
            claim="normal_marginal_upper_tail" if not qg_dict.get("lighting_shift_detected") else None,
            assumptions="Marginal upper-tail guarantee holds under exchangeability with training normals",
        ),
        quality_gate=quality_gate_obj,
        alignment=alignment_obj,
        measurements=measurements_obj,
        assets=AssetsInfo(
            input_standardized=f"/api/assets/{input_key}",
            reference_aligned=None,
            anomaly_heatmap=f"/api/assets/{heatmap_key}" if heatmap_key else None,
            contour_mask=None,
        ),
        execution_trace=execution_trace,
        steps_used=res.get("steps_used", 0),
        step_budget=4,
        timing_ms=TimingInfo(
            quality=res["timing_ms"]["quality"],
            perception=res["timing_ms"]["perception"],
            tools=res["timing_ms"]["tools"],
            total=res["timing_ms"]["total"],
        ),
        grounded_report=res["grounded_report"],
        counterfactuals=[],
    )

    return response, assets_to_store
