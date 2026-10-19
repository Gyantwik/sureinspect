"""
train/verify_onnx_opencv.py
Prove that OpenCV's cv.dnn gives the same features as PyTorch for the exported backbone.

    python train/verify_onnx_opencv.py --onnx artifacts/onnx/wide_resnet50_2_256.onnx \
        --backbone wide_resnet50_2 --input 256

Two checks, in this order (as the spec requires):
  1. Tensor parity   core.preprocess.to_blob  vs  an independent torch normalisation.
  2. Feature parity  OpenCV DNN outputs       vs  PyTorch outputs, same input tensor.

Target: max absolute feature difference < 1e-3. A graph that loads but gives wrong
features is NOT a pass. Exit code 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# make `core` and `train` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.preprocess import standardize, to_blob  # noqa: E402

TOL = 1e-3
OUT_NAMES = ("layer2_out", "layer3_out")


# ------------------------------------------------------------- OpenCV side

def load_net(onnx_path: str | Path, engine: str = "auto"):
    """Load the ONNX graph with cv.dnn. engine: auto | new | classic."""
    engines = {
        "auto": getattr(cv2.dnn, "ENGINE_AUTO", None),
        "new": getattr(cv2.dnn, "ENGINE_NEW", None),
        "classic": getattr(cv2.dnn, "ENGINE_CLASSIC", None),
    }
    eng = engines.get(engine)
    if engine != "auto" and eng is None:
        raise RuntimeError(f"This OpenCV build has no '{engine}' DNN engine")
    net = cv2.dnn.readNetFromONNX(str(onnx_path), eng) if eng is not None else cv2.dnn.readNetFromONNX(str(onnx_path))
    # default backend/target is CPU; setting a target prints a warning on the new engine
    return net


def opencv_features(net, blob: np.ndarray, out_names=OUT_NAMES) -> list[np.ndarray]:
    net.setInput(blob)
    outs = net.forward(list(out_names))
    if isinstance(outs, np.ndarray):  # single output
        outs = [outs]
    return [np.asarray(o, dtype=np.float32) for o in outs]


# ------------------------------------------------------------ comparison

def compare_features(reference, candidate, tol: float = TOL, names=OUT_NAMES) -> dict:
    """Compare two lists of feature maps. Never passes on a shape mismatch."""
    if len(reference) != len(candidate):
        return {"ok": False, "reason": f"output count {len(reference)} vs {len(candidate)}", "outputs": []}
    rows, ok = [], True
    for name, r, c in zip(names, reference, candidate):
        r, c = np.asarray(r, dtype=np.float32), np.asarray(c, dtype=np.float32)
        if r.shape != c.shape:
            rows.append({"name": name, "ok": False, "reason": f"shape {r.shape} vs {c.shape}"})
            ok = False
            continue
        d = np.abs(r - c)
        row = {
            "name": name,
            "shape": list(r.shape),
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "ref_abs_max": float(np.abs(r).max()),
            "ok": bool(d.max() < tol and np.isfinite(c).all()),
        }
        ok = ok and row["ok"]
        rows.append(row)
    return {"ok": ok, "outputs": rows}


# ---------------------------------------------------------- PyTorch side

def torch_features(backbone: str, blob: np.ndarray, pretrained: bool = True):
    import torch
    from train.export_backbone_onnx import build_backbone

    model, _ = build_backbone(backbone, pretrained)
    with torch.no_grad():
        f2, f3 = model(torch.from_numpy(blob))
    return [f2.numpy(), f3.numpy()]


def tensor_parity(image_std: np.ndarray, size: int) -> dict:
    """to_blob (NumPy) vs an independent torch/torchvision normalisation of the same resize."""
    import torch
    from torchvision.transforms import functional as TF

    from core.preprocess import BLOB_INTERPOLATION

    rgb = cv2.cvtColor(image_std, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=BLOB_INTERPOLATION)
    t = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).contiguous()
    ours = to_blob(image_std, size)
    d = float(np.abs(ours - t.numpy()).max())
    return {"max_abs_diff": d, "ok": d < 1e-5}


# ----------------------------------------------------------------- CLI

def _default_images() -> list[str]:
    root = Path(__file__).resolve().parents[1] / "data" / "samples"
    return sorted(glob.glob(str(root / "*.png")) + glob.glob(str(root / "*.jpg")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenCV DNN vs PyTorch parity check.")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--backbone", default="wide_resnet50_2", choices=("wide_resnet50_2", "resnet18"))
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--images", nargs="*", default=None, help="image files (default: data/samples/*)")
    ap.add_argument("--engine", choices=("auto", "new", "classic"), default="auto")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    images = args.images or _default_images()
    if not images:
        print("ERROR: no images found. Pass --images or check data/samples/.", file=sys.stderr)
        return 2
    try:
        net = load_net(args.onnx, args.engine)
    except Exception as exc:
        print(f"LOAD FAILED (cv.dnn could not read the graph): {exc!r}", file=sys.stderr)
        print("Try --engine classic or --engine new; re-export with --exporter dynamo/legacy; "
              "or use the resnet18 fallback.", file=sys.stderr)
        return 1

    report = {"onnx": str(args.onnx), "engine": args.engine, "tol": args.tol, "images": [], "ok": True}
    for p in images:
        img = cv2.imread(p)
        if img is None:
            print(f"ERROR: cannot read {p}", file=sys.stderr)
            return 2
        std = standardize(img).image
        tp = tensor_parity(std, args.input)
        blob = to_blob(std, args.input)
        try:
            got = opencv_features(net, blob)
        except Exception as exc:
            print(f"FORWARD FAILED on {p}: {exc!r}", file=sys.stderr)
            print("Try --engine classic/new, another exporter, or the resnet18 fallback.", file=sys.stderr)
            return 1
        ref = torch_features(args.backbone, blob)
        cmp = compare_features(ref, got, args.tol)
        ok = tp["ok"] and cmp["ok"]
        report["ok"] = report["ok"] and ok
        report["images"].append({"image": p, "tensor_parity": tp, "feature_parity": cmp})
        worst = max((o.get("max_abs_diff", float("inf")) for o in cmp["outputs"]), default=float("inf"))
        print(f"{Path(p).name:32s} tensor_diff={tp['max_abs_diff']:.2e}  feature_max_diff={worst:.2e}  "
              f"{'OK' if ok else 'FAIL'}")

    out = Path(args.out or f"artifacts/onnx/parity_{args.backbone}_{args.input}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out}")
    print("PARITY PASS" if report["ok"] else "PARITY FAIL (do not continue until fixed; see hints above)")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
