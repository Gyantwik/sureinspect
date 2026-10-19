"""
train/select_templates.py
Pick up to five defect-free templates from the BANK normals by deterministic embedding diversity.

    python train/select_templates.py --root ~/datasets/VisA_pytorch/1cls --category pcb1 \
        --onnx artifacts/onnx/wide_resnet50_2_256.onnx --input 256

Procedure (no randomness):
  1. embed every bank normal = mean of its backbone patch descriptors (the same global embedding
     the runtime uses to find the nearest templates);
  2. start from the image closest to the mean embedding;
  3. repeatedly add the image farthest (L2) from the already chosen set, until k templates.

Writes, for the category:
    data/templates/<category>/template_00.png ...   original STANDARDIZED images, lossless PNG
    data/templates/<category>/embeddings.npz        the template embeddings
    data/templates/<category>/templates.json        source path, hashes, sizes

Only bank_normals are eligible. Never choose templates from calibration, development or test images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.perception import extract_patch_features, load_backbone  # noqa: E402
from core.preprocess import standardize, to_blob  # noqa: E402

MAX_PNG_MB = 4.5          # the repository CI blocks files over 5 MB


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def farthest_point_selection(emb: np.ndarray, k: int) -> list[int]:
    """Deterministic: start nearest to the mean, then greedily add the farthest-from-set point."""
    n = len(emb)
    k = min(k, n)
    mean = emb.mean(axis=0)
    first = int(np.argmin(((emb - mean) ** 2).sum(axis=1)))
    chosen = [first]
    min_d = ((emb - emb[first]) ** 2).sum(axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))                      # argmax returns the lowest index on ties
        chosen.append(nxt)
        min_d = np.minimum(min_d, ((emb - emb[nxt]) ** 2).sum(axis=1))
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select diverse templates from bank normals.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--engine", choices=("auto", "new", "classic"), default="auto")
    ap.add_argument("--out-dir", default=None, help="default data/templates/<category>")
    args = ap.parse_args(argv)

    splits = json.loads(Path(args.splits or f"artifacts/splits/{args.category}.json").read_text(encoding="utf-8"))
    items = splits["splits"]["bank_normals"]
    root = Path(args.root)
    net = load_backbone(args.onnx, args.engine)

    embeddings, images = [], []
    for i, e in enumerate(items):
        img = cv2.imread(str(root / e["path"]))
        if img is None:
            print(f"ERROR: cannot read {e['path']}", file=sys.stderr)
            return 2
        std = standardize(img)
        feats, _ = extract_patch_features(net, to_blob(std.image, args.input))
        embeddings.append(feats.mean(axis=0))
        images.append(std.image)
        if (i + 1) % 50 == 0:
            print(f"  embedded {i + 1}/{len(items)}", flush=True)
    emb = np.stack(embeddings).astype(np.float32)

    chosen = farthest_point_selection(emb, args.count)
    out = Path(args.out_dir or f"data/templates/{args.category}")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("template_*.png"):
        old.unlink()

    meta = []
    for row, idx in enumerate(chosen):
        name = f"template_{row:02d}.png"
        ok, buf = cv2.imencode(".png", images[idx])
        png = buf.tobytes()
        size_mb = len(png) / 1e6
        if size_mb > MAX_PNG_MB:
            print(f"ERROR: {name} is {size_mb:.1f} MB; the repo CI blocks files over 5 MB. "
                  "Lower image_max_side or commit it another way.", file=sys.stderr)
            return 2
        (out / name).write_bytes(png)
        meta.append({
            "id": f"{args.category}-t{row:02d}",
            "file": name,
            "embedding_row": row,
            "source_path": items[idx]["path"],
            "source_sha256": items[idx]["sha256"],
            "png_sha256": sha256_bytes(png),
            "standardized_size": [int(images[idx].shape[1]), int(images[idx].shape[0])],
            "png_mb": round(size_mb, 2),
        })
    np.savez(out / "embeddings.npz", embeddings=emb[chosen])
    (out / "templates.json").write_text(json.dumps({
        "category": args.category,
        "selection": "deterministic farthest-point on mean patch-descriptor embeddings; start = nearest to the mean",
        "eligible": "bank_normals only",
        "input_size": args.input,
        "onnx": Path(args.onnx).name,
        "templates": meta,
    }, indent=2), encoding="utf-8")

    print(f"Wrote {len(meta)} templates to {out}")
    for m in meta:
        print(f"  {m['id']}  <- {m['source_path']}  ({m['png_mb']} MB, {m['standardized_size']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
