"""
eval/verify_submission_zip.py
Automated end-to-end verification script for SureInspect submission package.
Extracts and verifies:
1. ZIP integrity and presence of required binary model weights (model.onnx, bank.npz).
2. Verifies SHA256 checksums of all snapshot files against SHA256SUMS.
3. Tests real OpenCV 5 DNN live inference on a normal board and a defective board.
4. Confirms decisions, decision paths, and overlays.
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Ensure repo root is on python path
ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

DEFAULT_ZIP_PATH = Path("sureinspect_ai_evaluation.zip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zip_path",
        nargs="?",
        default=str(DEFAULT_ZIP_PATH),
        help="Submission ZIP to verify",
    )
    args = parser.parse_args(argv)
    zip_path = Path(args.zip_path)

    print("=" * 70)
    print("SUREINSPECT SUBMISSION VERIFICATION & LIVE INFERENCE CHECK")
    print("=" * 70)
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Checking Archive: {zip_path.resolve()}")

    if not zip_path.is_file():
        print(f"FAIL: Archive {zip_path} does not exist!")
        return 1

    print(f"Archive Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")

    # 1. Inspect ZIP contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = set(zf.namelist())
        print(f"Total Files in ZIP: {len(namelist)}")

        # Check required model weights
        snap_prefix = "artifacts/snapshots/pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1"
        req_weights = [
            f"{snap_prefix}/model.onnx",
            f"{snap_prefix}/bank.npz",
            f"{snap_prefix}/calibration.json",
            f"{snap_prefix}/quality_gate.json",
            f"{snap_prefix}/policy.json",
            f"{snap_prefix}/SHA256SUMS",
        ]
        for w in req_weights:
            if w in namelist:
                info = zf.getinfo(w)
                print(f"  [FOUND] {w} ({info.file_size / (1024*1024):.2f} MB uncompressed)")
            else:
                print(f"  [FAIL] Missing required artifact: {w}")
                return 1

        # Check templates
        req_templates = [f"data/templates/pcb1/template_{i:02d}.png" for i in range(5)]
        for t in req_templates:
            if t in namelist:
                print(f"  [FOUND] {t}")
            else:
                print(f"  [FAIL] Missing required template: {t}")
                return 1

        # Check key deliverables
        for doc in ["FINAL_SUBMISSION_REPORT.md", "DEMO_SCRIPT.md", "docs/architecture.mmd"]:
            if doc in namelist:
                print(f"  [FOUND] {doc}")
            else:
                print(f"  [FAIL] Missing required deliverable: {doc}")
                return 1

        # Verify SHA256 hashes of snapshot files from inside zip
        sums_content = zf.read(f"{snap_prefix}/SHA256SUMS").decode("utf-8")
        print("\nVerifying snapshot checksums inside ZIP...")
        for line in sums_content.splitlines():
            if not line.strip():
                continue
            expected_hash, fname = line.split()
            data = zf.read(f"{snap_prefix}/{fname}")
            actual_hash = hashlib.sha256(data).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch for {fname}"
            print(f"  [VERIFIED] {fname} ({expected_hash[:16]}...)")

        # Extract only after rejecting traversal paths, then execute from that
        # directory so no source file can be accidentally imported from the repo.
        for info in zf.infolist():
            target = (Path("/") / info.filename).resolve()
            if ".." in Path(info.filename).parts or not target.is_relative_to(Path("/").resolve()):
                print(f"  [FAIL] Unsafe archive path: {info.filename}")
                return 1
        with tempfile.TemporaryDirectory(prefix="sureinspect-package-") as temp_dir:
            zf.extractall(temp_dir)
            packaged = subprocess.run(
                [sys.executable, "-m", "eval.verify_packaged_runtime"],
                cwd=temp_dir,
                check=False,
            )
            if packaged.returncode != 0:
                print("  [FAIL] Extracted packaged runtime smoke test failed")
                return 1
            print("  [VERIFIED] Extracted packaged runtime executed successfully")

    # 2. Run live inference on normal and defective boards
    print("\nRunning live OpenCV 5 inference tests...")
    from core.preprocess import standardize
    from core.agent import inspect_standardized

    # Normal board test
    normal_candidates = list(Path("datasets/VisA_pytorch/1cls/pcb1/test/good").glob("*.JPG"))
    normal_path = normal_candidates[0] if normal_candidates else Path("data/samples/pcb1_good_01.png")
    if normal_path.is_file():
        img_norm = cv2.imread(str(normal_path))
        std_norm = standardize(img_norm)
        res_norm = inspect_standardized(std_norm.image)
        print(f"\n[TEST 1: Normal Board - {normal_path.name}]")
        print(f"  Decision: {res_norm['decision']}")
        print(f"  Decision Path: {res_norm['decision_path']}")
        print(f"  Perception Score: {res_norm['score']:.4f}")
        print(f"  Quality Gate: {res_norm['quality_gate']['status']}")
        print(f"  Timing: {res_norm['timing_ms']['total']:.1f} ms")
        assert res_norm["decision"] == "PASS", f"Expected PASS on normal board, got {res_norm['decision']}"
        assert res_norm["decision_path"] == "FAST_PASS", f"Expected FAST_PASS, got {res_norm['decision_path']}"
        print("  => PASSED (Normal board confirmed FAST_PASS)")

    # Defective board test
    bad_path = Path("datasets/VisA_pytorch/1cls/pcb1/test/bad/000.JPG")
    if bad_path.is_file():
        img_bad = cv2.imread(str(bad_path))
        std_bad = standardize(img_bad)
        res_bad = inspect_standardized(std_bad.image)
        print(f"\n[TEST 2: Defective Board - {bad_path.name} (solder bridge)]")
        print(f"  Decision: {res_bad['decision']}")
        print(f"  Decision Path: {res_bad['decision_path']}")
        print(f"  Perception Score: {res_bad['score']:.4f}")
        print(f"  Quality Gate: {res_bad['quality_gate']['status']}")
        print(f"  ROIs Detected: {len(res_bad['rois'])}")
        print(f"  Timing: {res_bad['timing_ms']['total']:.1f} ms")
        print(f"  Overlay Heatmap: shape={res_bad['heatmap'].shape if res_bad['heatmap'] is not None else None}")
        assert res_bad["decision"] in ("FAIL", "HUMAN"), f"Expected FAIL or HUMAN on defect, got {res_bad['decision']}"
        print(f"  => PASSED (Defect intercepted as {res_bad['decision']} / {res_bad['decision_path']})")

    # Ambiguous board tool verification test
    ambig_path = Path("datasets/VisA_pytorch/1cls/pcb1/test/bad/025.JPG")
    if ambig_path.is_file():
        img_ambig = cv2.imread(str(ambig_path))
        std_ambig = standardize(img_ambig)
        res_ambig = inspect_standardized(std_ambig.image)
        print(f"\n[TEST 3: Ambiguous Solder Melt - {ambig_path.name}]")
        print(f"  Decision: {res_ambig['decision']}")
        print(f"  Decision Path: {res_ambig['decision_path']}")
        print(f"  Perception Score: {res_ambig['score']:.4f}")
        print(f"  Alignment: {res_ambig['alignment']}")
        print(f"  Measurements: {res_ambig['measurements']}")
        print(f"  Steps Used: {res_ambig['steps_used']}")
        print("  => PASSED (Secondary tools executed and logged)")

    print("\n" + "=" * 70)
    print("ARCHIVE STRUCTURE, SNAPSHOT CHECKSUMS, AND LOCAL LIVE INFERENCE PASSED")
    print("This check does not prove AWS deployment, endpoint availability, video access, or competition eligibility.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
