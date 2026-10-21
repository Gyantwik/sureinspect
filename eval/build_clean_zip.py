import os
import zipfile
from pathlib import Path

ROOT = Path(".").resolve()
ZIP_NAME = "sureinspect_ai_evaluation.zip"

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
}

EXCLUDE_EXTENSIONS = {
    ".onnx",
    ".npz",
    ".pt",
    ".pth",
    ".tar",
    ".gz",
    ".zip",
    ".pyc",
    ".pyo",
    ".log",
}

EXCLUDE_EXACT_FILES = {
    "pcb1_snapshot.zip",
    ZIP_NAME,
    "test_real_bad_000.jpg",
}

total_files = 0
total_uncompressed_bytes = 0

print(f"Packaging SureInspect AI Evaluation Zip -> {ZIP_NAME}...")

with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for root, dirs, files in os.walk(ROOT):
        # Modify dirs in-place to prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".venv")]
        
        rel_root = Path(root).relative_to(ROOT)
        
        # Check if parent path contains any excluded directory name
        parts = rel_root.parts
        if any(p in EXCLUDE_DIRS for p in parts):
            continue
            
        for f in files:
            p = Path(root) / f
            rel_file = p.relative_to(ROOT)
            
            # Check exclusions
            if f in EXCLUDE_EXACT_FILES:
                continue
            if p.suffix.lower() in EXCLUDE_EXTENSIONS:
                continue
            if f.endswith(".tmp"):
                continue
                
            # Add file to zip
            zf.write(p, arcname=str(rel_file))
            total_files += 1
            total_uncompressed_bytes += p.stat().st_size

zip_size_bytes = Path(ZIP_NAME).stat().st_size
zip_size_mb = zip_size_bytes / (1024 * 1024)
raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

print("\n" + "=" * 60)
print(f"SUCCESSFULLY GENERATED: {ZIP_NAME}")
print(f"Total Files Packaged: {total_files}")
print(f"Raw Uncompressed Size: {raw_size_mb:.2f} MB")
print(f"Final Compressed Zip Size: {zip_size_mb:.2f} MB")
print("=" * 60)
