import os
import sys
import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(".").resolve()
DEFAULT_ZIP_NAME = "sureinspect_ai_evaluation.zip"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv-train",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".turbo",
    "datasets",
    "data/visa",
    "data/VisA_pytorch",
}

EXCLUDE_EXTENSIONS_BASE = {
    ".pt",
    ".pth",
    ".tar",
    ".gz",
    ".zip",
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
}

EXCLUDE_EXACT_FILES = {
    "pcb1_snapshot.zip",
    "test_real_bad_000.jpg",
}


def verify_snapshot_integrity(snapshot_dir: Path) -> bool:
    """Verify SHA256 hashes of all snapshot artifacts before packaging."""
    sums_file = snapshot_dir / "SHA256SUMS"
    if not sums_file.is_file():
        print(f"Warning: No SHA256SUMS file found in {snapshot_dir}")
        return False

    print(f"Verifying snapshot integrity in {snapshot_dir.name}...")
    all_ok = True
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, filename = line.split()
        target = snapshot_dir / filename
        if not target.is_file():
            print(f"  [MISSING] {filename}")
            all_ok = False
            continue
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h != expected_hash:
            print(f"  [CORRUPT] {filename} (hash mismatch)")
            all_ok = False
        else:
            print(f"  [OK] {filename}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Package SureInspect submission archive.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_ZIP_NAME,
        help="Output zip filename (default: sureinspect_ai_evaluation.zip)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Exclude binary model weights (model.onnx, bank.npz) to stay under 15 MB for AI upload.",
    )
    args = parser.parse_args()

    zip_name = args.output
    is_compact = args.compact

    snap_dir = ROOT / "artifacts" / "snapshots" / "pcb1-wide_resnet50_2-256-edc526f4da11"
    if not is_compact:
        if snap_dir.is_dir():
            if not verify_snapshot_integrity(snap_dir):
                print("Error: Snapshot integrity verification failed!")
                sys.exit(1)
        else:
            print(f"Warning: Snapshot directory not found at {snap_dir}")

    exclude_exts = set(EXCLUDE_EXTENSIONS_BASE)
    if is_compact:
        exclude_exts.update({".onnx", ".npz"})

    print(f"\nPackaging SureInspect Archive -> {zip_name} (compact={is_compact})...")

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".venv")]

            rel_root = Path(root).relative_to(ROOT)
            parts = rel_root.parts
            if any(p in EXCLUDE_DIRS for p in parts):
                continue

            for f in files:
                p = Path(root) / f
                rel_file = p.relative_to(ROOT)

                if f in EXCLUDE_EXACT_FILES or f == zip_name:
                    continue
                if p.suffix.lower() in exclude_exts:
                    continue
                if f.endswith(".tmp"):
                    continue

                zf.write(p, arcname=str(rel_file))
                total_files += 1
                total_uncompressed_bytes += p.stat().st_size

    zip_size_bytes = Path(zip_name).stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"SUCCESSFULLY GENERATED: {zip_name}")
    print(f"Mode: {'Compact (weights excluded)' if is_compact else 'Complete Standalone (weights included)'}")
    print(f"Total Files Packaged: {total_files}")
    print(f"Raw Uncompressed Size: {raw_size_mb:.2f} MB")
    print(f"Final Compressed Zip Size: {zip_size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
