"""Create visual and machine-readable evidence for every cached defect escape."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval/results/cached_e1_predictions.json"
DATASET = ROOT / "datasets/VisA_pytorch/1cls"
OUT_DIR = ROOT / "docs/failure_cases/escape_evidence"


def main() -> int:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    escapes = [
        (path, pred) for path, pred in cache["predictions"].items()
        if "/bad/" in path.replace("\\", "/") and pred.get("decision") == "PASS"
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []
    for path, pred in sorted(escapes):
        source = DATASET / path
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(f"Missing escape source image: {source}")
        h, w = image.shape[:2]
        scale = min(480 / w, 320 / h)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        tile = np.full((400, 520, 3), 245, dtype=np.uint8)
        x = (520 - resized.shape[1]) // 2
        tile[50:50 + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(tile, path, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        detail = f"{pred['decision_path']} score={pred['score']:.4f} residual={pred.get('residual_mean')}"
        cv2.putText(tile, detail, (12, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 180), 1, cv2.LINE_AA)
        tiles.append(tile)
        out_name = Path(path).stem + "_escape.jpg"
        cv2.imwrite(str(OUT_DIR / out_name), tile, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        rows.append({"image": path, "evidence_file": out_name, **pred})
    if not rows:
        raise SystemExit("No cached defect escapes found; refusing to create empty evidence")
    columns = 2
    rows_n = (len(tiles) + columns - 1) // columns
    while len(tiles) < rows_n * columns:
        tiles.append(np.full_like(tiles[0], 245))
    contact = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    cv2.imwrite(str(OUT_DIR / "all_defect_escapes.jpg"), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    (OUT_DIR / "escape_evidence.json").write_text(json.dumps({"cache_commit": cache["__commit__"], "count": len(rows), "escapes": rows}, indent=2) + "\n", encoding="utf-8")
    report = ["# Defect Escape Visual Evidence", "", f"All `{len(rows)}` cached defect escapes are shown below. These are known failures, not successful detections.", "", "![All recorded defect escapes](escape_evidence/all_defect_escapes.jpg)", "", "| Image | Decision path | Score | Residual | Evidence |", "|---|---|---:|---:|---|"]
    for row in rows:
        report.append(f"| `{row['image']}` | `{row['decision_path']}` | {row['score']:.4f} | {row.get('residual_mean')} | [annotated image](escape_evidence/{row['evidence_file']}) |")
    report.extend(["", "The source dataset is not redistributed in the submission archive. These derived review images are included solely to document failure behavior and must be removed if dataset licensing does not permit derived evaluation evidence.", ""])
    (ROOT / "docs/failure_cases/DEFECT_ESCAPE_EVIDENCE.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Generated visual evidence for {len(rows)} defect escapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
