"""
train/publish_snapshot.py
Assemble an IMMUTABLE snapshot directory for Person 2:

    artifacts/snapshots/<snapshot-id>/
        model.onnx            exported backbone (copied)
        model.manifest.json   export manifest (explicit weight identity)
        bank.npz              fp32 memory bank (copied)
        bank.manifest.json    bank build manifest
        calibration.json      conformal thresholds + heat-map patch_thr + sorted scores
        quality_gate.json     gate thresholds fitted on development gate-fit normals
        splits.json           split lists with hashes
        manifest.json         versions, preprocessing, score recipe, thresholds, per-file sha256
        SHA256SUMS            standard checksum file (use: sha256sum -c SHA256SUMS)

Readiness
    perception_only : can serve scores/heat-maps only; it CANNOT serve final agent decisions.
    agent_ready     : not available yet (needs templates, alignment and residual cut-offs).

A published snapshot is never overwritten. Nothing here is uploaded; Person 2 uploads and verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.preprocess import BLOB_INTERPOLATION, IMAGENET_MEAN, IMAGENET_STD, MAX_SIDE, standardize  # noqa: E402
from core.quality_gate import fit_gate_thresholds  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_lf(path: Path, text: str) -> None:
    """Write text with plain LF newlines on every OS (sha256sum rejects CRLF checksum files)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _load_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing {what}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fit_gate(root: Path, splits: dict) -> dict:
    items = splits["splits"]["dev_gate_fit_normals"]
    images = []
    for e in items:
        img = cv2.imread(str(root / e["path"]))
        if img is None:
            raise SystemExit(f"ERROR: cannot read {e['path']}")
        images.append(standardize(img).image)
    gate = fit_gate_thresholds(images)
    if gate.get("gate_version") == "stub":
        raise SystemExit("ERROR: stub gate thresholds are not allowed in a published snapshot.")
    return gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publish an immutable perception-only snapshot.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="pcb1")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--readiness", choices=("perception_only", "agent_ready"), default="perception_only")
    ap.add_argument("--out-dir", default="artifacts/snapshots")
    args = ap.parse_args(argv)

    if args.readiness == "agent_ready":
        raise SystemExit("ERROR: agent_ready needs templates, alignment and residual cut-offs, which are not built yet.")

    onnx, bank, cal_path = Path(args.onnx), Path(args.bank), Path(args.calibration)
    splits_path = Path(args.splits or f"artifacts/splits/{args.category}.json")
    onnx_manifest_p = onnx.with_suffix(".manifest.json")
    bank_manifest_p = bank.with_suffix(".manifest.json")
    onnx_m = _load_json(onnx_manifest_p, "ONNX manifest (written by export_backbone_onnx.py)")
    bank_m = _load_json(bank_manifest_p, "bank manifest (written by build_memory_bank.py)")
    cal = _load_json(cal_path, "calibration JSON")
    splits = _load_json(splits_path, "splits JSON")

    # ---- consistency: every artifact must come from the same scorer
    problems = []
    if bank_m["onnx_sha256"] != sha256_file(onnx):
        problems.append("bank was built from a different ONNX file")
    if cal["bank_sha256"] != sha256_file(bank):
        problems.append("calibration was fitted with a different bank")
    if cal["onnx_sha256"] != sha256_file(onnx):
        problems.append("calibration was fitted with a different ONNX file")
    if not (bank_m["splits_sha256"] == cal["splits_sha256"] == sha256_file(splits_path)):
        problems.append("bank, calibration and splits file do not share the same split hashes")
    if not (onnx_m["input_size"] == bank_m["input_size"] == cal["input_size"]):
        problems.append("input size differs between export, bank and calibration")
    if bank_m["category"] != args.category or cal["category"] != args.category:
        problems.append("category mismatch")
    if problems:
        raise SystemExit("ERROR: inconsistent artifacts, rebuild downstream files:\n  - " + "\n  - ".join(problems))

    gate = fit_gate(Path(args.root), splits)

    shas = {"model": sha256_file(onnx), "bank": sha256_file(bank), "calibration": sha256_file(cal_path)}
    snap_hash = hashlib.sha256("".join(shas.values()).encode()).hexdigest()[:12]
    snapshot_id = f"{args.category}-{onnx_m['backbone']}-{onnx_m['input_size']}-{snap_hash}"
    out = Path(args.out_dir) / snapshot_id
    if out.exists():
        raise SystemExit(f"ERROR: snapshot {snapshot_id} already exists. Published snapshots are never overwritten.")
    out.mkdir(parents=True)

    files = {
        "model.onnx": onnx, "model.manifest.json": onnx_manifest_p,
        "bank.npz": bank, "bank.manifest.json": bank_manifest_p,
        "calibration.json": cal_path, "splits.json": splits_path,
    }
    for name, src in files.items():
        shutil.copyfile(src, out / name)
    write_lf(out / "quality_gate.json", json.dumps(gate, indent=2))

    manifest = {
        "snapshot_id": snapshot_id,
        "readiness": args.readiness,
        "category": args.category,
        "versions": {
            "model_version": f"{onnx_m['backbone']}-{onnx_m['input_size']}-{shas['model'][:12]}",
            "bank_version": f"bank-{shas['bank'][:12]}",
            "calibration_version": f"calib-{shas['calibration'][:12]}",
            "policy_version": None,   # set when the agent policy and evidence thresholds are published
        },
        "weights": {"identity": onnx_m["weights"], "backbone": onnx_m["backbone"]},
        "export": {"opset": onnx_m["opset"], "exporter": onnx_m["exporter"], "torch": onnx_m["torch"],
                   "onnx": onnx_m["onnx"]},
        "preprocessing": {
            "standardize": f"keep aspect ratio, longest side <= {MAX_SIDE}, INTER_AREA when shrinking, never upscale",
            "blob": f"BGR->RGB, square resize to {onnx_m['input_size']} (cv2 interpolation {int(BLOB_INTERPOLATION)} = INTER_AREA), "
                    "/255, ImageNet mean/std in NumPy, NCHW float32",
            "mean": [float(x) for x in IMAGENET_MEAN],
            "std": [float(x) for x in IMAGENET_STD],
        },
        "input_size": onnx_m["input_size"],
        "feature_layers": {"names": onnx_m["output_names"], "shapes": onnx_m["output_shapes"],
                           "combined_grid": bank_m["grid"], "combined_channels": bank_m["feature_dim"]},
        "score_recipe": ("upsample layer3 to layer2 grid (bilinear, half-pixel), concat, 3x3 stride-1 average pool "
                         "(zero padding, divisor 9); distance to nearest bank patch; image score = MAX raw patch "
                         "distance (not a probability); heat-map = patch map upsampled to standardized size + "
                         "Gaussian blur (sigma = 4/256 of the longest side)"),
        "bank": {"rows": bank_m["bank_rows"], "dtype": bank_m["dtype"], "ram_mb": bank_m["bank_ram_mb"],
                 "coreset_ratio": bank_m["coreset_ratio"], "max_bank_cap": bank_m["max_bank_cap"],
                 "projection": bank_m["projection"], "selection": bank_m["selection"]},
        "seeds": {"splits": splits["meta"]["seed"], "bank_and_projection": bank_m["seed"]},
        "split_hashes": {"splits_json_sha256": sha256_file(splits_path),
                         "bank_images_sha256": bank_m["bank_images_sha256"]},
        "split_counts": splits["counts"],
        "thresholds": {
            "convention": cal["convention"],
            "alpha_pass": cal["alpha_pass"], "t_pass": cal["t_pass"]["threshold"],
            "alpha_fail_default": cal["alpha_fail_default"], "t_fail_default": cal["t_fail_default"],
            "t_fail_options": {a: v["threshold"] for a, v in cal["t_fail_options"].items()},
            "omitted_alpha_fail": cal["omitted_alpha_fail"],
            "n_calibration": cal["n_calibration"],
            "patch_thr": cal["heatmap"]["patch_thr"],
        },
        "quality_gate": {"gate_version": gate["gate_version"], "n_fit": gate["n_fit"]},
        "tool_policy": None,
        "evaluation_status": "pending",
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "files": {},
    }
    for name in sorted(p.name for p in out.iterdir()):
        manifest["files"][name] = sha256_file(out / name)
    write_lf(out / "manifest.json", json.dumps(manifest, indent=2))

    names = sorted(p.name for p in out.iterdir() if p.name != "SHA256SUMS")
    write_lf(out / "SHA256SUMS", "".join(f"{sha256_file(out / n)}  {n}\n" for n in names))

    print(f"Published {out}")
    print(f"  snapshot id : {snapshot_id}   readiness: {args.readiness}")
    print(f"  t_pass {manifest['thresholds']['t_pass']:.4f}  t_fail(default) {manifest['thresholds']['t_fail_default']:.4f}  "
          f"patch_thr {manifest['thresholds']['patch_thr']:.4f}")
    print("  verify: cd " + str(out) + " && sha256sum -c SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
