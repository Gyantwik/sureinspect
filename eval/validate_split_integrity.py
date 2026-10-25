"""Validate SureInspect split counts, file hashes, masks, and group isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _items(node: Any):
    if isinstance(node, list):
        yield from (item for item in node if isinstance(item, dict) and "path" in item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _items(value)


def validate(manifest: Path, dataset_root: Path, verify_files: bool = True) -> list[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    paths: dict[str, str] = {}
    groups: dict[str, str] = {}
    for split, node in data["splits"].items():
        if split == "excluded":
            continue
        items = list(_items(node))
        expected = data.get("counts", {}).get(split)
        if expected is not None and len(items) != expected:
            errors.append(f"{split}: expected {expected} items, found {len(items)}")
        for item in items:
            rel = item["path"]
            if rel in paths:
                errors.append(f"path appears in {paths[rel]} and {split}: {rel}")
            paths[rel] = split
            group = item.get("group")
            if group and group in groups and groups[group] != split:
                errors.append(f"near-duplicate group {group} crosses {groups[group]} and {split}")
            elif group:
                groups[group] = split
            if verify_files:
                target = dataset_root / rel
                if not target.is_file():
                    errors.append(f"missing image: {rel}")
                elif _sha(target) != item["sha256"]:
                    errors.append(f"image hash mismatch: {rel}")
                if item.get("mask"):
                    mask = dataset_root / item["mask"]
                    if not mask.is_file():
                        errors.append(f"missing mask: {item['mask']}")
                    elif _sha(mask) != item["mask_sha256"]:
                        errors.append(f"mask hash mismatch: {item['mask']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11/splits.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/VisA_pytorch/1cls")
    parser.add_argument("--verify-file-hashes", action="store_true", help="Also require the complete source dataset and verify every image/mask hash")
    args = parser.parse_args()
    errors = validate(args.manifest, args.dataset_root, args.verify_file_hashes)
    if errors:
        print("SPLIT INTEGRITY FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    scope = "counts, unique paths, group isolation, and file hashes" if args.verify_file_hashes else "counts, unique paths, and near-duplicate group isolation"
    print(f"SPLIT INTEGRITY PASSED: {scope} verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
