"""
train/fit_calibration.py
Fit the frozen decision thresholds using the SAVED (reloaded) bank.

    python train/fit_calibration.py --root ~/datasets/VisA_pytorch/1cls --category pcb1 \
        --onnx artifacts/onnx/wide_resnet50_2_256.onnx \
        --bank artifacts/bank/pcb1_wide_resnet50_2_256.npz --input 256

What it fits
  * t_pass (alpha_pass) and every allowed t_fail option, by split-conformal calibration on the
    untouched CALIBRATION normals (k = ceil((n+1)(1-alpha)), sorted index k-1; alpha omitted if k > n).
  * patch_thr for the heat-map, on the same upsampled + blurred representation used at runtime,
    from DEVELOPMENT (gate-fit) normals: the --patch-q quantile of each image's heat-map maximum.

It never reads the final test set. Development scores (from eval_dev_auroc) are only used for a
sanity summary, never for fitting.
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

from core.calibration import fit_thresholds  # noqa: E402
from core.perception import load_backbone, make_heatmap, score_image  # noqa: E402
from core.preprocess import standardize  # noqa: E402
from train.prepare_dataset import load_category_config  # noqa: E402

ASSUMPTIONS = (
    "Conformal guarantee: for a new NORMAL image exchangeable with the calibration normals, "
    "P(score > t) <= alpha. The score is a raw max patch distance, not a probability. "
    "It does not hold under lighting shift or other distribution shift, and the true rate on "
    "a finite test set varies around alpha."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_items(net, bank, items, root: Path, size: int, want_heat_max: bool = False, label: str = ""):
    scores, heat_max = [], []
    for i, e in enumerate(items):
        img = cv2.imread(str(root / e["path"]))
        if img is None:
            raise SystemExit(f"ERROR: cannot read {e['path']}")
        std = standardize(img)
        r = score_image(net, std.image, bank, size)
        scores.append(r.score)
        if want_heat_max:
            heat_max.append(float(make_heatmap(r.patch_map, std.standardized_size).max()))
        if (i + 1) % 50 == 0:
            print(f"  {label}: {i + 1}/{len(items)}", flush=True)
    return np.array(scores, dtype=np.float64), np.array(heat_max, dtype=np.float64)


def dev_sanity(dev_scores_path: Path, thresholds: dict) -> dict | None:
    if not dev_scores_path.is_file():
        return None
    rows = json.loads(dev_scores_path.read_text(encoding="utf-8"))["scores"]
    s = np.array([r["score"] for r in rows])
    y = np.array([r["label"] for r in rows])
    tp = thresholds["t_pass"]["threshold"]
    out = {
        "note": "Development data only; informational, never used for fitting.",
        "n_dev_normals": int((y == 0).sum()),
        "n_dev_anomalies": int((y == 1).sum()),
        "fast_pass_fraction_normals": float((s[y == 0] <= tp).mean()),
        "fast_pass_fraction_anomalies": float((s[y == 1] <= tp).mean()),
        "per_alpha_fail": {},
    }
    for a, t in thresholds["t_fail_options"].items():
        thr = t["threshold"]
        out["per_alpha_fail"][a] = {
            "t_fail": thr,
            "dev_normals_above": float((s[y == 0] > thr).mean()),
            "dev_anomalies_above": float((s[y == 1] > thr).mean()),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit conformal thresholds and the heat-map threshold.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--config", default="config/categories.yaml")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--engine", choices=("auto", "new", "classic"), default="auto")
    ap.add_argument("--patch-q", type=float, default=0.90, help="quantile of dev-normal heat-map maxima for patch_thr")
    ap.add_argument("--dev-scores", default=None, help="eval_dev_auroc JSON, for the sanity summary only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cat = load_category_config(Path(args.config), args.category)
    splits_path = Path(args.splits or f"artifacts/splits/{args.category}.json")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    calib = splits["splits"]["calib_normals"]
    gate_fit = splits["splits"]["dev_gate_fit_normals"]
    root = Path(args.root)

    bank = np.load(args.bank)["bank"].astype(np.float32)        # the SAVED bank, reloaded
    net = load_backbone(args.onnx, args.engine)
    print(f"bank {bank.shape} | calibration normals {len(calib)} | dev normals for patch_thr {len(gate_fit)}")

    t0 = time.time()
    cal_scores, _ = _score_items(net, bank, calib, root, args.input, label="calibration")
    thresholds = fit_thresholds(cal_scores, float(cat["alpha_pass"]), [float(a) for a in cat["alpha_fail_options"]])
    _, heat_max = _score_items(net, bank, gate_fit, root, args.input, want_heat_max=True, label="dev heat-maps")
    patch_thr = float(np.quantile(heat_max, args.patch_q))

    default_alpha = f"{float(cat['alpha_fail_default']):g}"
    if default_alpha not in thresholds["t_fail_options"]:
        raise SystemExit(f"ERROR: default alpha_fail {default_alpha} is not available with n={len(cal_scores)}")
    t_pass = thresholds["t_pass"]["threshold"]
    t_fail = thresholds["t_fail_options"][default_alpha]["threshold"]
    if not t_pass < t_fail:
        raise SystemExit(f"ERROR: t_pass ({t_pass}) is not below t_fail ({t_fail}); the uncertain band would be empty.")

    dev_path = Path(args.dev_scores or f"artifacts/bank/dev_auroc_{args.category}_{args.input}.json")
    result = {
        "category": args.category,
        "input_size": args.input,
        "n_calibration": int(len(cal_scores)),
        "alpha_pass": float(cat["alpha_pass"]),
        "alpha_fail_default": float(cat["alpha_fail_default"]),
        "t_pass": thresholds["t_pass"],
        "t_fail_options": thresholds["t_fail_options"],
        "omitted_alpha_fail": thresholds["omitted_alpha_fail"],
        "t_fail_default": t_fail,
        "convention": "fail: score > t_fail; fast pass: score <= t_pass (full precision)",
        "heatmap": {
            "patch_thr": patch_thr,
            "method": f"{args.patch_q} quantile of per-image heat-map maxima on development gate-fit normals",
            "n_images": int(len(heat_max)),
            "representation": "patch map upsampled (bilinear) to the standardized size + Gaussian blur",
        },
        "assumptions": ASSUMPTIONS,
        "sorted_calibration_scores": sorted(float(x) for x in cal_scores),
        "bank_sha256": sha256_file(Path(args.bank)),
        "onnx_sha256": sha256_file(Path(args.onnx)),
        "splits_sha256": sha256_file(splits_path),
        "dev_sanity": None,
        "seconds": round(time.time() - t0, 1),
    }
    result["dev_sanity"] = dev_sanity(dev_path, thresholds)

    out = Path(args.out or f"artifacts/calibration/{args.category}_{args.input}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"t_pass (alpha {result['alpha_pass']}) = {t_pass:.4f}")
    for a, t in thresholds["t_fail_options"].items():
        print(f"t_fail (alpha {a:>5}) = {t['threshold']:.4f}   (k={t['k']} of n={t['n']})")
    if thresholds["omitted_alpha_fail"]:
        print(f"omitted (k > n): {thresholds['omitted_alpha_fail']}")
    print(f"patch_thr (heat-map) = {patch_thr:.4f}")
    ds = result["dev_sanity"]
    if ds:
        print(f"dev sanity: fast-pass normals {ds['fast_pass_fraction_normals']:.2f}, "
              f"anomalies {ds['fast_pass_fraction_anomalies']:.2f}")
        for a, v in ds["per_alpha_fail"].items():
            print(f"  alpha {a:>5}: dev normals above t_fail {v['dev_normals_above']:.3f}, "
                  f"dev anomalies above {v['dev_anomalies_above']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
