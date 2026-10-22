"""
eval/download_visa_subset.py
Downloads the exact 493 evaluation images defined in splits.json from the official
VisA mirror and verifies their SHA-256 hashes bit-for-bit.
"""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_HF = "https://huggingface.co/datasets/imaadd05/visa-anomaly-detection/resolve/main"
SPLITS_PATH = Path("artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11/splits.json")
DEST_ROOT = Path("datasets/VisA_pytorch/1cls")

if not SPLITS_PATH.is_file():
    print(f"Error: {SPLITS_PATH} not found!")
    sys.exit(1)

with open(SPLITS_PATH, encoding="utf-8") as f:
    splits_data = json.load(f)

# Collect all evaluation images
eval_splits = ["dev_eval_normals", "dev_anomalies", "final_test_normals", "final_test_anomalies"]
tasks = []
seen_paths = set()

for s_name in eval_splits:
    items = splits_data["splits"].get(s_name, [])
    for it in items:
        p = it["path"]
        sha = it["sha256"]
        if p not in seen_paths:
            seen_paths.add(p)
            tasks.append((p, sha, s_name))

print(f"Total unique evaluation images to download & verify: {len(tasks)}")

def download_and_verify(task):
    rel_path, expected_sha, split_name = task
    dest_file = DEST_ROOT / rel_path
    
    # Check if already exists and hash matches
    if dest_file.is_file():
        h = hashlib.sha256(dest_file.read_bytes()).hexdigest()
        if h == expected_sha:
            return True, rel_path, "cached"
    
    # Map layout to Hugging Face source URL:
    # "pcb1/train/good/xxxx.JPG" -> "pcb1/Data/Images/Normal/xxxx.JPG"
    # "pcb1/test/good/xxxx.JPG"  -> "pcb1/Data/Images/Normal/xxxx.JPG"
    # "pcb1/test/bad/xxx.JPG"    -> "pcb1/Data/Images/Anomaly/xxx.JPG"
    parts = Path(rel_path).parts
    filename = parts[-1]
    
    if "good" in rel_path:
        remote_url = f"{BASE_HF}/pcb1/Data/Images/Normal/{filename}"
    else:
        remote_url = f"{BASE_HF}/pcb1/Data/Images/Anomaly/{filename}"
        
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    req = urllib.request.Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
        
    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}")
        
    with open(dest_file, "wb") as f:
        f.write(content)
        
    return True, rel_path, "downloaded"

t0 = time.time()
completed = 0
errors = []

print("Starting parallel download with 10 worker threads...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_and_verify, t): t for t in tasks}
    for future in concurrent.futures.as_completed(futures):
        task = futures[future]
        try:
            ok, p, status = future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"  [{completed}/{len(tasks)}] verified ({time.time() - t0:.1f}s)")
        except Exception as e:
            errors.append((task[0], str(e)))
            print(f"  Error on {task[0]}: {e}")

elapsed = time.time() - t0
print("\n" + "=" * 60)
print(f"Finished in {elapsed:.1f} seconds.")
print(f"Successfully downloaded and verified: {completed}/{len(tasks)}")
if errors:
    print(f"FAILED: {len(errors)} errors")
    sys.exit(1)
else:
    print("ALL 493 IMAGES VERIFIED WITH 100% SHA256 MATCH!")
print("=" * 60)
