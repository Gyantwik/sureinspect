"""Download only the official VisA PCB2 one-class test split.

Split metadata comes from Amazon Science's spot-diff repository. Image bytes
come from an unmodified Hugging Face mirror of the official CC BY 4.0 archive;
every downloaded file is hashed in an ignored local acquisition manifest.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "visa" / "pcb2"
SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
MIRROR_BASE = "https://huggingface.co/datasets/imaadd05/visa-anomaly-detection/resolve/main"
OFFICIAL_ARCHIVE = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SureInspect-evaluation/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:  # pragma: no cover
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def official_test_rows(split_csv: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(split_csv.decode("utf-8"))))
    selected = [row for row in rows if row["object"] == "pcb2" and row["split"] == "test"]
    if not selected:
        raise RuntimeError("Official split CSV contained no PCB2 test rows")
    labels = {row["label"] for row in selected}
    if labels != {"normal", "anomaly"}:
        raise RuntimeError(f"Unexpected PCB2 labels: {sorted(labels)}")
    return selected


def destination_for(row: dict[str, str]) -> Path:
    label_dir = "good" if row["label"] == "normal" else "bad"
    return DESTINATION / "test" / label_dir / Path(row["image"]).name


def download_one(row: dict[str, str]) -> dict[str, object]:
    destination = destination_for(row)
    source_path = row["image"]
    url = f"{MIRROR_BASE}/{urllib.parse.quote(source_path, safe='/')}?download=true"
    if destination.is_file() and destination.stat().st_size > 0:
        payload = destination.read_bytes()
    else:
        payload = fetch(url)
        if not payload.startswith(b"\xff\xd8"):
            raise RuntimeError(f"Downloaded payload is not JPEG: {source_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    return {
        "path": destination.relative_to(ROOT).as_posix(),
        "official_path": source_path,
        "label": row["label"],
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def run(workers: int = 8) -> int:
    split_csv = fetch(SPLIT_URL)
    rows = official_test_rows(split_csv)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, row): row for row in rows}
        for completed, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if completed % 25 == 0 or completed == len(rows):
                print(f"Downloaded/verified {completed}/{len(rows)} PCB2 test images", flush=True)

    records.sort(key=lambda record: str(record["path"]))
    label_counts = {
        label: sum(record["label"] == label for record in records)
        for label in ("normal", "anomaly")
    }
    manifest = {
        "schema_version": "sureinspect.visa-pcb2-acquisition.v1",
        "dataset": "Amazon Visual Anomaly (VisA)",
        "category": "pcb2",
        "protocol": "official 1cls test split",
        "license": "CC BY 4.0",
        "official_registry": "https://registry.opendata.aws/visa/",
        "official_archive": OFFICIAL_ARCHIVE,
        "official_split_url": SPLIT_URL,
        "split_sha256": sha256_bytes(split_csv),
        "byte_mirror": "imaadd05/visa-anomaly-detection",
        "counts": label_counts,
        "records": records,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PCB2 ready: {label_counts}; manifest={manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    raise SystemExit(run(workers))
