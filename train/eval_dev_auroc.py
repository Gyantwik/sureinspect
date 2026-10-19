"""
train/eval_dev_auroc.py
Day-2 detection gate: image AUROC on DEVELOPMENT normals vs DEVELOPMENT anomalies.

    python train/eval_dev_auroc.py --root ~/datasets/VisA_pytorch/1cls --category pcb1 \
        --onnx artifacts/onnx/wide_resnet50_2_256.onnx --bank artifacts/bank/pcb1_wide_resnet50_2_256.npz

Uses ONLY `dev_eval_normals` and `dev_anomalies` from the splits file. It never touches the
calibration set or the final test set. The team target is image AUROC >= 0.85; it is a target,
not an expected benchmark. The score is the max raw patch distance (not a probability).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.perception import load_backbone, score_image  # noqa: E402
from core.preprocess import standardize  # noqa: E402

TARGET_AUROC = 0.85


def auroc_with_ci(labels: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 0):
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Development AUROC for the scorer.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--engine", choices=("auto", "new", "classic"), default="auto")
    ap.add_argument("--limit", type=int, default=None, help="score at most N images per class (quick test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    splits = json.loads(Path(args.splits or f"artifacts/splits/{args.category}.json").read_text(encoding="utf-8"))
    normals = splits["splits"]["dev_eval_normals"]
    anomalies = splits["splits"]["dev_anomalies"]
    if args.limit:
        normals, anomalies = normals[: args.limit], anomalies[: args.limit]

    bank = np.load(args.bank)["bank"].astype(np.float32)
    net = load_backbone(args.onnx, args.engine)
    print(f"bank {bank.shape} ({bank.nbytes / 1e6:.0f} MB) | dev normals {len(normals)} | dev anomalies {len(anomalies)}")

    rows, times = [], []
    for label, items in ((0, normals), (1, anomalies)):
        for e in items:
            img = cv2.imread(str(Path(args.root) / e["path"]))
            if img is None:
                print(f"ERROR: cannot read {e['path']}", file=sys.stderr)
                return 2
            t0 = time.time()
            r = score_image(net, standardize(img).image, bank, args.input)
            times.append(time.time() - t0)
            rows.append({"path": e["path"], "label": label, "score": r.score})
        print(f"  scored {'anomalies' if label else 'normals'}", flush=True)

    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    auc, lo, hi = auroc_with_ci(labels, scores)
    report = {
        "category": args.category,
        "input_size": args.input,
        "n_normals": int((labels == 0).sum()),
        "n_anomalies": int((labels == 1).sum()),
        "image_auroc": auc,
        "auroc_95_bootstrap": [lo, hi],
        "target": TARGET_AUROC,
        "meets_target": bool(auc >= TARGET_AUROC),
        "score_mean_normal": float(scores[labels == 0].mean()),
        "score_mean_anomaly": float(scores[labels == 1].mean()),
        "seconds_per_image_mean": float(np.mean(times)),
        "seconds_per_image_p95": float(np.percentile(times, 95)),
        "bank_ram_mb": round(bank.nbytes / 1e6, 1),
        "scores": rows,
    }
    out = Path(args.out or f"artifacts/bank/dev_auroc_{args.category}_{args.input}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nImage AUROC (development): {auc:.3f}   95% bootstrap interval [{lo:.3f}, {hi:.3f}]")
    print(f"mean score  normals {report['score_mean_normal']:.3f}  anomalies {report['score_mean_anomaly']:.3f}")
    print(f"time/image  mean {report['seconds_per_image_mean']:.2f}s  p95 {report['seconds_per_image_p95']:.2f}s")
    print(f"Report: {out}")
    print("MEETS TEAM TARGET (0.85)" if auc >= TARGET_AUROC
          else "BELOW TARGET: try --input 320 then 384 (re-export the ONNX and rebuild the bank for each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
