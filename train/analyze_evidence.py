"""
train/analyze_evidence.py
Choose the residual statistic and percentiles from the SAVED evidence JSON (no images, instant).

Step 1 - look at the sweep:
    python train/analyze_evidence.py --evidence artifacts/evidence/pcb1_256.json --metric raw_p95
Step 2 - freeze a choice (writes the final thresholds file used by the snapshot):
    python train/analyze_evidence.py --evidence artifacts/evidence/pcb1_256.json --metric raw_p95 \
        --r-low-pct 90 --r-high-pct 95 --write artifacts/evidence/pcb1_256_final.json

Uses development data only. The trade-off you choose is recorded in the output file. Never choose
percentiles by looking at the final test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.fit_evidence import METRICS, fit, select_metric  # noqa: E402

SWEEP = (85, 90, 93, 95, 97, 98, 99)


def sweep(rows: dict, q_min: float, metric: str) -> list[dict]:
    select_metric(rows, metric)
    n = np.array([r["residual"] for r in rows["normal"]
                  if r["aligned"] and r["quality"] >= q_min and r["residual"] is not None])
    a = np.array([r["residual"] for r in rows["anomaly"]
                  if r["aligned"] and r["quality"] >= q_min and r["residual"] is not None])
    out = []
    for p in SWEEP:
        thr = float(np.percentile(n, p))
        out.append({"pct": p, "threshold": thr, "n_normals": int(len(n)), "n_anomalies": int(len(a)),
                    "normals_above": float((n > thr).mean()),
                    "anomalies_above": float((a > thr).mean()) if len(a) else None,
                    "anomalies_above_n": int((a > thr).sum())})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sweep residual percentiles from a saved evidence JSON.")
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--metric", choices=METRICS, default="raw_p95")
    ap.add_argument("--r-low-pct", type=float, default=None)
    ap.add_argument("--r-high-pct", type=float, default=None)
    ap.add_argument("--write", default=None, help="write the final thresholds JSON (needs both percentiles)")
    args = ap.parse_args(argv)

    path = Path(args.evidence)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    if not rows["normal"] or "metrics" not in next((r for r in rows["normal"] if r["aligned"]), {}):
        print("ERROR: this evidence file has no per-metric rows; rerun train/fit_evidence.py with the latest code.",
              file=sys.stderr)
        return 2
    q_min = data["thresholds"]["q_min"]

    for metric in dict.fromkeys([args.metric, args.metric.replace("raw_", "clahe_", 1)]):
        table = sweep(rows, q_min, metric if metric.startswith("raw_") else metric)
        print(f"\nMetric {metric}  (aligned dev images; q_min {q_min:.4f})")
        print(f"  {'pct':>5s} {'threshold':>10s} {'normals above':>14s} {'anomalies above':>16s}")
        for t in table:
            an = f"{t['anomalies_above']:.2f} ({t['anomalies_above_n']}/{t['n_anomalies']})" \
                if t["anomalies_above"] is not None else "n/a"
            print(f"  {t['pct']:5.0f} {t['threshold']:10.4f} {t['normals_above']:14.3f} {an:>16s}")
        if not args.metric.startswith("raw_"):
            break

    if args.write:
        if args.r_low_pct is None or args.r_high_pct is None:
            print("ERROR: --write needs both --r-low-pct and --r-high-pct", file=sys.stderr)
            return 2
        if not args.r_low_pct < args.r_high_pct:
            print("ERROR: r_low percentile must be below r_high percentile", file=sys.stderr)
            return 2
        select_metric(rows, args.metric)
        fitted = fit(rows["normal"], rows["anomaly"], args.r_low_pct, args.r_high_pct)
        final = {
            "category": data["category"], "input_size": data["input_size"], "metric": args.metric,
            "clahe_metric": args.metric.replace("raw_", "clahe_", 1),
            "percentiles": {"r_low_pct": args.r_low_pct, "r_high_pct": args.r_high_pct},
            "thresholds": {k: fitted[k] for k in ("q_min", "r_low", "r_high", "clahe_r_low", "clahe_r_high")},
            "report": {k: v for k, v in fitted.items()
                       if k not in ("q_min", "r_low", "r_high", "clahe_r_low", "clahe_r_high")},
            "chosen_trade_off": "percentiles chosen on development data only; capture rates above are on development anomalies",
            "source_evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "templates": data["templates"], "constants": data["constants"],
            "calibration_sha256": data["calibration_sha256"], "splits_sha256": data["splits_sha256"],
        }
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(final, indent=2), encoding="utf-8")
        t = final["thresholds"]
        a = final["report"].get("anomalies", {})
        print(f"\nWrote {out}\n  metric {args.metric} | q_min {t['q_min']:.4f} | r_low {t['r_low']:.4f} (p{args.r_low_pct:g}) | "
              f"r_high {t['r_high']:.4f} (p{args.r_high_pct:g})")
        print(f"  CLAHE r_low {t['clahe_r_low']:.4f} r_high {t['clahe_r_high']:.4f}")
        print(f"  dev anomalies above r_low {a.get('residual_above_r_low')}, above r_high {a.get('residual_above_r_high')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
