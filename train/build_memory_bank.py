"""
train/build_memory_bank.py
Build the PatchCore memory bank for one category from the BANK normals only.

    python train/build_memory_bank.py --root ~/datasets/VisA_pytorch/1cls --category pcb1 \
        --onnx artifacts/onnx/wide_resnet50_2_256.onnx --input 256

Procedure (recorded in the manifest, fully deterministic for a given seed):
  pass 1  extract descriptors for every bank image, project them to PROJ_DIM with a seeded
          Gaussian random projection (kept in RAM, small);
  select  greedy k-center coreset on the projected descriptors, k = ratio * N, capped at --max-bank;
  pass 2  re-extract descriptors and keep ONLY the selected rows (full precision, fp32);
  save    artifacts/bank/<category>_<backbone>_<size>.npz  (+ .manifest.json). Never commit these.

Only images listed under `bank_normals` in the splits file are used. File hashes are checked
against the splits file so a changed dataset is detected.
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

from core.perception import extract_patch_features, greedy_coreset, load_backbone, random_projection  # noqa: E402
from core.preprocess import standardize, to_blob  # noqa: E402

PROJ_DIM = 128
DEFAULT_RATIO = 0.05
DEFAULT_MAX_BANK = 20000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank_images(splits: dict, root: Path) -> list[tuple[Path, str]]:
    items = splits["splits"]["bank_normals"]
    if not items:
        raise SystemExit("ERROR: bank_normals is empty in the splits file.")
    out = []
    for e in items:
        p = root / e["path"]
        if not p.is_file():
            raise SystemExit(f"ERROR: missing bank image {p}. Check --root.")
        if sha256_file(p) != e["sha256"]:
            raise SystemExit(f"ERROR: hash mismatch for {p}. The dataset changed since the splits were built; "
                             "rebuild the splits with train/prepare_dataset.py.")
        out.append((p, e["sha256"]))
    return out


def _descriptors(net, path: Path, size: int):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"ERROR: cannot read {path}")
    feats, grid = extract_patch_features(net, to_blob(standardize(img).image, size))
    return feats, grid


def build_bank(net, images: list[tuple[Path, str]], size: int, ratio: float, max_bank: int,
               seed: int, proj_dim: int = PROJ_DIM, progress: bool = True) -> dict:
    t0 = time.time()
    # ---- pass 1: projected descriptors
    projections, counts, dim, grid = [], [], None, None
    proj_matrix = None
    for i, (p, _) in enumerate(images):
        feats, g = _descriptors(net, p, size)
        if proj_matrix is None:
            dim, grid = feats.shape[1], g
            proj_matrix = random_projection(dim, proj_dim, seed)
        projections.append(feats @ proj_matrix)
        counts.append(len(feats))
        if progress and (i + 1) % 25 == 0:
            print(f"  pass 1: {i + 1}/{len(images)} images", flush=True)
    proj = np.concatenate(projections, axis=0)
    n_total = len(proj)
    k = int(min(max_bank, max(1, round(ratio * n_total))))

    # ---- coreset selection
    selected = np.sort(greedy_coreset(proj, k, seed, progress=progress))

    # ---- pass 2: keep only selected rows, at full fp32 precision
    offsets = np.concatenate([[0], np.cumsum(counts)])
    rows = []
    for i, (p, _) in enumerate(images):
        lo, hi = offsets[i], offsets[i + 1]
        local = selected[(selected >= lo) & (selected < hi)] - lo
        if len(local):
            feats, _ = _descriptors(net, p, size)
            rows.append(feats[local])
    bank = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
    return {
        "bank": bank,
        "selected_idx": selected,
        "n_total_patches": int(n_total),
        "feature_dim": int(dim),
        "grid": [int(grid[0]), int(grid[1])],
        "seconds": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the PatchCore memory bank.")
    ap.add_argument("--root", required=True, help="dataset root that contains the category folders")
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--splits", default=None, help="splits JSON (default artifacts/splits/<category>.json)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--input", type=int, default=256)
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    ap.add_argument("--max-bank", type=int, default=DEFAULT_MAX_BANK)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--engine", choices=("auto", "new", "classic"), default="auto")
    ap.add_argument("--out-dir", default="artifacts/bank")
    args = ap.parse_args(argv)

    splits_path = Path(args.splits or f"artifacts/splits/{args.category}.json")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    images = load_bank_images(splits, Path(args.root))
    print(f"Bank images: {len(images)}  | ONNX: {args.onnx}")

    net = load_backbone(args.onnx, args.engine)
    r = build_bank(net, images, args.input, args.ratio, args.max_bank, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.category}_{args.backbone}_{args.input}"
    npz = out_dir / f"{stem}.npz"
    np.savez_compressed(npz, bank=r["bank"], selected_idx=r["selected_idx"])

    manifest = {
        "category": args.category,
        "backbone": args.backbone,
        "input_size": args.input,
        "grid": r["grid"],
        "feature_dim": r["feature_dim"],
        "n_bank_images": len(images),
        "n_total_patches": r["n_total_patches"],
        "coreset_ratio": args.ratio,
        "max_bank_cap": args.max_bank,
        "bank_rows": int(len(r["bank"])),
        "bank_ram_mb": round(r["bank"].nbytes / 1e6, 1),
        "projection": {"type": "gaussian", "dim": PROJ_DIM, "seed": args.seed},
        "selection": "greedy k-center on projected descriptors; first centre chosen by seed; selected rows sorted",
        "padding_policy": "3x3 avg pool, zero padding, divisor 9",
        "upsampling": "bilinear, half-pixel centres",
        "dtype": "float32",
        "seed": args.seed,
        "onnx_sha256": sha256_file(Path(args.onnx)),
        "splits_sha256": sha256_file(splits_path),
        "bank_images_sha256": hashlib.sha256("".join(h for _, h in images).encode()).hexdigest(),
        "file": npz.name,
        "file_sha256": sha256_file(npz),
        "build_seconds": round(r["seconds"], 1),
    }
    (out_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {npz}")
    print(f"  patches in total {r['n_total_patches']}  ->  bank rows {manifest['bank_rows']} "
          f"({manifest['bank_ram_mb']} MB fp32), feature dim {r['feature_dim']}, grid {r['grid']}")
    print(f"  build time {manifest['build_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
