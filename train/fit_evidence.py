"""
train/fit_evidence.py
Fit the evidence thresholds on DEVELOPMENT data only (never calibration, never final test).

    python train/fit_evidence.py --root ~/datasets/VisA_pytorch/1cls --category pcb1 \
        --onnx artifacts/onnx/wide_resnet50_2_256.onnx \
        --bank artifacts/bank/pcb1_wide_resnet50_2_256.npz \
        --calibration artifacts/calibration/pcb1_256.json --input 256

For every development image it runs the same path the agent will use:
    score -> heat-map -> ROI1 -> nearest templates (up to 2) -> T1 align -> T2 residual (raw and CLAHE).

Fits
    q_min                alignment-quality cut-off = 1st percentile of development-NORMAL qualities.
                         The held-out alignment failure rate is REPORTED, not promised.
    r_low, r_high        90th / 99th percentile of development-normal ROI1 residuals (raw).
    clahe_r_low/high     the same on CLAHE comparison copies (separate cut-offs, as the spec requires).
Reports how many development anomalies each cut-off captures, so the trade-off is recorded.
Final-test outcomes never trigger retuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import tools as T  # noqa: E402
from core.perception import find_rois, load_backbone, make_heatmap, score_image  # noqa: E402
from core.preprocess import standardize  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


METRICS = ("raw_mean", "raw_p95", "raw_mean_m", "raw_p95_m")      # *_m = photometric matching


def _variants(test_gray, ref_gray, m, roi) -> dict:
    """ROI1 residual under every candidate statistic (raw and CLAHE copies, with/without matching)."""
    out = {}
    pairs = {"raw": (test_gray, ref_gray),
             "clahe": (T.normalize_lighting(test_gray), T.normalize_lighting(ref_gray))}
    for prefix, (tg, rg) in pairs.items():
        for match, suffix in ((False, ""), (True, "_m")):
            r = T.roi_residual(T.prepare_comparison(tg, rg, m, match=match), roi)
            out[f"{prefix}_mean{suffix}"] = r.get("residual_mean") if r["available"] else None
            out[f"{prefix}_p95{suffix}"] = r.get("residual_p95") if r["available"] else None
    return out


def evaluate_image(net, bank, size, templates, patch_thr, img_bgr) -> dict:
    """Run the agent's evidence path for ONE image (no thresholds applied)."""
    std = standardize(img_bgr)
    r = score_image(net, std.image, bank, size)
    heat = make_heatmap(r.patch_map, std.standardized_size)
    rois = find_rois(heat, r.patch_map, patch_thr)
    test_gray = T.to_gray(std.image)
    order = T.nearest_order(r.embedding, templates["embeddings"])
    al, _ = T.align_to_templates(test_gray, templates["items"], order, q_min=None)
    row = {"score": r.score, "roi1": rois[0]["box"], "roi1_source": rois[0]["source"],
           "aligned": bool(al is not None and al.valid), "quality": None, "residual": None,
           "residual_clahe": None, "template_id": None, "reason": None}
    if al is None:
        row["reason"] = "no alignment attempted"
        return row
    row.update(quality=al.quality, template_id=al.template_id, reason=al.reason)
    if not al.valid:
        return row
    ref_gray = next(t["gray"] for t in templates["items"] if t["id"] == al.template_id)
    row["metrics"] = _variants(test_gray, ref_gray, al.M, rois[0]["box"])
    return row


def select_metric(rows: dict, metric: str) -> None:
    """Copy the chosen statistic into row['residual'] / row['residual_clahe'] (CLAHE uses the same statistic)."""
    clahe = metric.replace("raw_", "clahe_", 1)
    for kind in rows.values():
        for r in kind:
            m = r.get("metrics")
            r["residual"] = m.get(metric) if m else None
            r["residual_clahe"] = m.get(clahe) if m else None


def diagnostics(rows: dict, q_min: float) -> dict:
    """Which residual statistic separates development anomalies from normals best? (informational)"""
    out = {}
    for metric in METRICS:
        n = [r["metrics"][metric] for r in rows["normal"]
             if r.get("metrics") and r["quality"] >= q_min and r["metrics"].get(metric) is not None]
        a = [r["metrics"][metric] for r in rows["anomaly"]
             if r.get("metrics") and r["quality"] >= q_min and r["metrics"].get(metric) is not None]
        if len(n) < 10 or len(a) < 3:
            continue
        entry = {"n_normals": len(n), "n_anomalies": len(a),
                 "normal_median": float(np.median(n)), "anomaly_median": float(np.median(a)),
                 "anomalies_above_normal_p90": float(np.mean(np.array(a) > np.percentile(n, 90))),
                 "anomalies_above_normal_p99": float(np.mean(np.array(a) > np.percentile(n, 99)))}
        try:
            from sklearn.metrics import roc_auc_score
            entry["auroc"] = float(roc_auc_score([0] * len(n) + [1] * len(a), n + a))
        except Exception:
            entry["auroc"] = None
        out[metric] = entry
    return out


def _pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fit(rows_normal: list[dict], rows_anom: list[dict], r_low_pct: float = 90.0, r_high_pct: float = 99.0) -> dict:
    qual = [r["quality"] for r in rows_normal if r["aligned"] and r["quality"] is not None]
    if len(qual) < 10:
        raise SystemExit(f"ERROR: only {len(qual)} development normals aligned; cannot fit q_min. "
                         "Check the templates and image standardization.")
    q_min = _pct(qual, 1)
    ok_n = [r for r in rows_normal if r["aligned"] and r["quality"] >= q_min]
    res = [r["residual"] for r in ok_n if r["residual"] is not None]
    res_c = [r["residual_clahe"] for r in ok_n if r["residual_clahe"] is not None]
    if len(res) < 10:
        raise SystemExit("ERROR: too few usable development-normal residuals to fit r_low/r_high.")
    out = {
        "q_min": q_min,
        "r_low": _pct(res, r_low_pct), "r_high": _pct(res, r_high_pct),
        "clahe_r_low": _pct(res_c, r_low_pct), "clahe_r_high": _pct(res_c, r_high_pct),
        "r_low_pct": r_low_pct, "r_high_pct": r_high_pct,
        "n_normals": len(rows_normal),
        "normals_aligned_valid": int(sum(r["aligned"] for r in rows_normal)),
        "alignment_failure_fraction_normals": float(1 - np.mean([r["aligned"] for r in rows_normal])),
        "normals_below_q_min_fraction": float(np.mean([(not r["aligned"]) or r["quality"] < q_min for r in rows_normal])),
        "roi1_fallback_fraction_normals": float(np.mean([r["roi1_source"] == "peak_fallback" for r in rows_normal])),
    }
    if rows_anom:
        a_ok = [r for r in rows_anom if r["aligned"] and r["quality"] >= q_min and r["residual"] is not None]
        out["anomalies"] = {
            "n": len(rows_anom),
            "alignment_failed_or_below_q_min": int(len(rows_anom) - len([r for r in rows_anom if r["aligned"] and r["quality"] >= q_min])),
            "n_with_residual": len(a_ok),
            "residual_above_r_low": float(np.mean([r["residual"] > out["r_low"] for r in a_ok])) if a_ok else None,
            "residual_above_r_high": float(np.mean([r["residual"] > out["r_high"] for r in a_ok])) if a_ok else None,
            "note": "capture rates on development anomalies, informational; the trade-off is recorded, not tuned on final test",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit alignment and residual cut-offs on development data.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--templates", default=None, help="default data/templates/<category>")
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--engine", choices=("auto", "new", "classic"), default="auto")
    ap.add_argument("--metric", choices=METRICS, default="raw_mean",
                    help="residual statistic used for r_low/r_high (see the printed diagnostics table)")
    ap.add_argument("--r-low-pct", type=float, default=90.0, help="percentile of normal residuals for r_low")
    ap.add_argument("--r-high-pct", type=float, default=99.0, help="percentile of normal residuals for r_high")
    ap.add_argument("--limit", type=int, default=None, help="at most N images per class (quick test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    splits_path = Path(args.splits or f"artifacts/splits/{args.category}.json")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    patch_thr = float(cal["heatmap"]["patch_thr"])
    templates = T.load_templates(args.templates or f"data/templates/{args.category}")
    bank = np.load(args.bank)["bank"].astype(np.float32)
    net = load_backbone(args.onnx, args.engine)

    s = splits["splits"]
    seen, normals = set(), []
    for e in s["dev_gate_fit_normals"] + s["dev_eval_normals"]:          # development normals only
        if e["path"] not in seen:
            seen.add(e["path"])
            normals.append(e)
    anomalies = s["dev_anomalies"]
    if args.limit:
        normals, anomalies = normals[: args.limit], anomalies[: args.limit]
    print(f"templates {len(templates['items'])} | dev normals {len(normals)} | dev anomalies {len(anomalies)}")

    t0 = time.time()
    rows = {"normal": [], "anomaly": []}
    for kind, items in (("normal", normals), ("anomaly", anomalies)):
        for i, e in enumerate(items):
            img = cv2.imread(str(Path(args.root) / e["path"]))
            if img is None:
                print(f"ERROR: cannot read {e['path']}", file=sys.stderr)
                return 2
            row = evaluate_image(net, bank, args.input, templates, patch_thr, img)
            row["path"] = e["path"]
            rows[kind].append(row)
            if (i + 1) % 50 == 0:
                print(f"  {kind}: {i + 1}/{len(items)}", flush=True)

    select_metric(rows, args.metric)
    fitted = fit(rows["normal"], rows["anomaly"], args.r_low_pct, args.r_high_pct)
    diag = diagnostics(rows, fitted["q_min"])
    unalignable = [{"path": r["path"], "quality": r["quality"], "reason": r["reason"], "template": r["template_id"],
                    "score": r["score"]} for r in rows["anomaly"] if not (r["aligned"] and r["quality"] >= fitted["q_min"])]
    result = {
        "category": args.category, "input_size": args.input,
        "fitted_on": "development normals (dev_gate_fit + dev_eval) and development anomalies only",
        "thresholds": {k: fitted[k] for k in ("q_min", "r_low", "r_high", "clahe_r_low", "clahe_r_high")},
        "report": {k: v for k, v in fitted.items() if k not in ("q_min", "r_low", "r_high", "clahe_r_low", "clahe_r_high")},
        "percentiles": {"r_low_pct": args.r_low_pct, "r_high_pct": args.r_high_pct},
        "constants": {"orb_features": T.ORB_FEATURES, "ratio": T.RATIO, "ransac_px": T.RANSAC_THRESH,
                      "scale_range": list(T.SCALE_RANGE), "max_rot_deg": T.MAX_ROT_DEG, "min_overlap": T.MIN_OVERLAP,
                      "blur": [T.BLUR_KSIZE, T.BLUR_SIGMA], "clahe": [T.CLAHE_CLIP, T.CLAHE_TILE],
                      "elongated_ratio": T.ELONGATED_RATIO, "compact_ratio": T.COMPACT_RATIO},
        "metric": args.metric,
        "diagnostics": diag,
        "unalignable_anomalies": unalignable,
        "roi1_fallback_fraction_anomalies": float(np.mean([r["roi1_source"] == "peak_fallback" for r in rows["anomaly"]]))
        if rows["anomaly"] else None,
        "templates": [t["id"] for t in templates["items"]],
        "calibration_sha256": sha256_file(Path(args.calibration)),
        "splits_sha256": sha256_file(splits_path),
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out or f"artifacts/evidence/{args.category}_{args.input}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    th, rp = result["thresholds"], result["report"]
    print(f"\nWrote {out}")
    print(f"q_min {th['q_min']:.4f} | r_low {th['r_low']:.4f} r_high {th['r_high']:.4f} | "
          f"CLAHE r_low {th['clahe_r_low']:.4f} r_high {th['clahe_r_high']:.4f}")
    print(f"dev normals: alignment failures {rp['alignment_failure_fraction_normals']:.3f}, "
          f"below q_min {rp['normals_below_q_min_fraction']:.3f}, ROI1 fallbacks {rp['roi1_fallback_fraction_normals']:.3f}")
    print(f"metric used: {args.metric}")
    if diag:
        print("\nResidual statistic diagnostics (aligned dev images; higher AUROC = better separation):")
        print(f"  {'metric':12s} {'AUROC':>6s} {'normal med':>11s} {'anomaly med':>12s} {'anom>p90':>9s} {'anom>p99':>9s}")
        for k, v in diag.items():
            au = f"{v['auroc']:.3f}" if v["auroc"] is not None else "  n/a"
            print(f"  {k:12s} {au:>6s} {v['normal_median']:11.4f} {v['anomaly_median']:12.4f} "
                  f"{v['anomalies_above_normal_p90']:9.2f} {v['anomalies_above_normal_p99']:9.2f}")
    if unalignable:
        print(f"\nDevelopment anomalies that could not be aligned ({len(unalignable)}):")
        for u in unalignable:
            q = f"{u['quality']:.3f}" if u["quality"] is not None else "n/a"
            print(f"  {u['path']}  quality {q}  score {u['score']:.2f}  {u['reason']}")
    if result["roi1_fallback_fraction_anomalies"] is not None:
        print(f"ROI1 is a peak fallback box for {result['roi1_fallback_fraction_anomalies']:.2f} of development anomalies")
    a = rp.get("anomalies")
    if a:
        print(f"dev anomalies: {a['n']} | not alignable {a['alignment_failed_or_below_q_min']} | "
              f"residual > r_low {a['residual_above_r_low']} | > r_high {a['residual_above_r_high']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
