"""Run SureInspect's local judge-facing verification gates.

This command verifies reproducibility and evidence consistency. It does not
claim AWS deployment, endpoint availability, video eligibility, or an
independent holdout evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts" / "snapshots" / "pcb1-wide_resnet50_2-256-edc526f4da11-agent-v1"
MEASURED = ROOT / "eval" / "results" / "measured_e1_results_v2.json"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path = SNAPSHOT) -> CheckResult:
    sums_path = snapshot / "SHA256SUMS"
    if not sums_path.is_file():
        return CheckResult("snapshot checksums", False, f"missing {sums_path}")
    errors: list[str] = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, filename = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed checksum line: {line!r}")
            continue
        artifact = snapshot / filename.strip()
        if not artifact.is_file():
            errors.append(f"missing {filename}")
        elif sha256_file(artifact) != expected:
            errors.append(f"hash mismatch: {filename}")
    return CheckResult(
        "snapshot checksums",
        not errors,
        "; ".join(errors) if errors else f"verified {snapshot.name}",
    )


def run_command(name: str, command: Sequence[str], cwd: Path = ROOT) -> CheckResult:
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    return CheckResult(name, completed.returncode == 0, f"exit {completed.returncode}")


def verify_openapi() -> CheckResult:
    contract = ROOT / "contracts" / "openapi.json"
    if not contract.is_file():
        return CheckResult("OpenAPI drift", False, "tracked contract is missing")
    before = contract.read_bytes()
    completed = subprocess.run(
        [sys.executable, "-m", "backend.export_openapi"], cwd=ROOT, check=False
    )
    after = contract.read_bytes() if contract.is_file() else b""
    if after != before:
        contract.write_bytes(before)
        return CheckResult("OpenAPI drift", False, "export differs from tracked contract")
    return CheckResult(
        "OpenAPI drift", completed.returncode == 0, f"exit {completed.returncode}; no drift"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Also verify this submission ZIP's structure, checksums, and packaged runtime",
    )
    parser.add_argument(
        "--skip-frontend", action="store_true", help="Skip frontend tests, types, and build"
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip the complete Python test suite"
    )
    args = parser.parse_args(argv)

    results: list[CheckResult] = []
    results.append(run_command("Python dependencies", [sys.executable, "-m", "pip", "check"]))
    if not args.skip_tests:
        results.append(run_command("Python tests", [sys.executable, "-m", "pytest", "-q"]))
    results.append(
        run_command(
            "Measured-result validation",
            [sys.executable, "-m", "eval.validate_measured_results", str(MEASURED)],
        )
    )
    results.append(verify_snapshot())
    results.append(
        run_command("Dataset split integrity", [sys.executable, "-m", "eval.validate_split_integrity"])
    )
    results.append(
        run_command("Candidate/result compatibility", [sys.executable, "-m", "eval.verify_candidate_compatibility"])
    )
    results.append(
        run_command("OpenCV 5 runtime proof", [sys.executable, "-m", "eval.verify_opencv5_runtime"])
    )
    results.append(
        run_command("Agentic path matrix", [sys.executable, "-m", "eval.generate_agentic_evidence"])
    )
    results.append(
        run_command(
            "Acquisition robustness",
            [sys.executable, "-m", "eval.run_acquisition_robustness"],
        )
    )
    results.append(verify_openapi())

    if not args.skip_frontend:
        npm = shutil.which("npm")
        if npm is None:
            results.append(CheckResult("frontend", False, "npm is not installed"))
        else:
            frontend = ROOT / "frontend"
            results.extend(
                [
                    run_command("Frontend tests", [npm, "run", "test", "--", "--run"], frontend),
                    run_command("TypeScript", [npm, "run", "types:check"], frontend),
                    run_command("Frontend build", [npm, "run", "build"], frontend),
                ]
            )

    if args.archive is not None:
        archive = args.archive if args.archive.is_absolute() else ROOT / args.archive
        results.append(
            run_command(
                "Submission archive",
                [sys.executable, "-m", "eval.verify_submission_zip", str(archive)],
            )
        )

    print("\n" + "=" * 72)
    for result in results:
        label = "PASS" if result.passed else "FAIL"
        print(f"[{label}] {result.name}: {result.detail}")
    failed = [result for result in results if not result.passed]
    if failed:
        print(f"LOCAL VERIFICATION FAILED: {len(failed)} gate(s) failed")
        return 1
    print("LOCAL REPRODUCIBILITY AND EVIDENCE GATES PASSED")
    print("AWS, endpoint, video, eligibility, and independent-holdout gates remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
