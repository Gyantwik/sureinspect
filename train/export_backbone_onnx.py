"""
train/export_backbone_onnx.py
Export the feature backbone (layer2 + layer3 outputs) to a fixed-shape ONNX graph that
OpenCV's cv.dnn can run.

    python train/export_backbone_onnx.py --backbone wide_resnet50_2 --input 256
    python train/export_backbone_onnx.py --backbone resnet18 --input 256     # fallback

Notes
-----
* Normalisation (ImageNet mean/std) is NOT part of the graph. core.preprocess.to_blob does it
  in NumPy. Never do both.
* One fixed-shape graph per input size. No dynamic axes.
* Weights are the torchvision IMAGENET1K_V1 enum; its name is recorded in the manifest.
* Files go to artifacts/onnx/ (git-ignored). NEVER commit .onnx files.
* Exporter compatibility is not assumed: run train/verify_onnx_opencv.py afterwards.
  If the default exporter fails or the parity check fails, try --exporter dynamo (or legacy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

OUTPUT_NAMES = ("layer2_out", "layer3_out")
BACKBONES = ("wide_resnet50_2", "resnet18")


def build_backbone(name: str, pretrained: bool = True):
    """Return (feature_module, weights_enum_name). Imports torch lazily."""
    import torch
    import torchvision

    tv = torchvision.models
    if name == "wide_resnet50_2":
        weights = tv.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.wide_resnet50_2(weights=weights)
        weights_name = "Wide_ResNet50_2_Weights.IMAGENET1K_V1" if pretrained else "random_init"
    elif name == "resnet18":
        weights = tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.resnet18(weights=weights)
        weights_name = "ResNet18_Weights.IMAGENET1K_V1" if pretrained else "random_init"
    else:
        raise ValueError(f"Unknown backbone {name!r}; choose from {BACKBONES}")

    class Features(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            x = m.layer1(x)
            f2 = m.layer2(x)
            f3 = m.layer3(f2)
            return f2, f3

    return Features(model).eval(), weights_name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(backbone: str, size: int, out_dir: Path, opset: int, exporter: str, pretrained: bool = True) -> dict:
    import onnx
    import torch
    import torchvision

    model, weights_name = build_backbone(backbone, pretrained)
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{backbone}_{size}.onnx"

    with torch.no_grad():
        f2, f3 = model(dummy)
    shapes = {"layer2_out": list(f2.shape), "layer3_out": list(f3.shape)}

    kwargs = dict(
        input_names=["input"],
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
    )
    try:
        if exporter == "dynamo":
            torch.onnx.export(model, (dummy,), str(path), dynamo=True, external_data=False, **kwargs)
        else:
            torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:
        # older torch without the dynamo/external_data arguments: plain legacy export
        torch.onnx.export(model, (dummy,), str(path), **kwargs)

    onnx.checker.check_model(onnx.load(str(path)))
    g = onnx.load(str(path)).graph
    graph_outputs = [o.name for o in g.output]
    if tuple(graph_outputs) != OUTPUT_NAMES:
        # name the outputs explicitly rather than trusting the exporter
        raise RuntimeError(
            f"Exported output names are {graph_outputs}, expected {list(OUTPUT_NAMES)}. "
            "Try the other --exporter."
        )

    manifest = {
        "backbone": backbone,
        "weights": weights_name,
        "input_size": size,
        "input_name": "input",
        "output_names": list(OUTPUT_NAMES),
        "output_shapes": shapes,
        "normalization": "NOT in graph: core.preprocess.to_blob applies ImageNet mean/std in NumPy",
        "opset": opset,
        "exporter": exporter,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnx": onnx.__version__,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    (out_dir / f"{backbone}_{size}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export the backbone to ONNX.")
    ap.add_argument("--backbone", choices=BACKBONES, default="wide_resnet50_2")
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--out-dir", default="artifacts/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--exporter", choices=("legacy", "dynamo"), default="legacy")
    ap.add_argument("--random-init", action="store_true", help="skip pretrained weights (testing only)")
    args = ap.parse_args(argv)
    try:
        m = export(args.backbone, args.input, Path(args.out_dir), args.opset, args.exporter, not args.random_init)
    except Exception as exc:  # report clearly instead of a long traceback
        print(f"EXPORT FAILED: {exc!r}", file=sys.stderr)
        print("Try: --exporter dynamo   (or legacy), then --backbone resnet18 as the fallback.", file=sys.stderr)
        return 1
    print(f"Wrote {Path(args.out_dir) / m['file']}  ({m['bytes'] / 1e6:.1f} MB)")
    print(f"Output shapes: {m['output_shapes']}")
    print("Next: python train/verify_onnx_opencv.py "
          f"--onnx {Path(args.out_dir) / m['file']} --backbone {args.backbone} --input {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
