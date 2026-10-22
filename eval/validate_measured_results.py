"""Validate SureInspect measured E1 result provenance and metric arithmetic.

This validator deliberately does not judge whether a split is scientifically
independent. It verifies that the machine-readable result is internally
consistent and surfaces the split's own disclosure label for downstream tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SPLIT_NAMES = ("dev_split", "final_test_split")


def _rate_error(
    errors: list[str],
    path: str,
    metric: dict[str, Any],
    expected_numerator: int,
    expected_denominator: int,
) -> None:
    numerator = metric.get("numerator")
    denominator = metric.get("denominator")
    rate = metric.get("rate")
    if numerator != expected_numerator:
        errors.append(f"{path}.numerator={numerator!r}, expected {expected_numerator}")
    if denominator != expected_denominator:
        errors.append(f"{path}.denominator={denominator!r}, expected {expected_denominator}")
    expected_rate = expected_numerator / expected_denominator if expected_denominator else 0.0
    if not isinstance(rate, (int, float)) or not math.isclose(
        float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append(f"{path}.rate={rate!r}, expected {expected_rate}")


def _validate_binary_baseline(
    errors: list[str], split_name: str, name: str, result: dict[str, Any]
) -> None:
    path = f"{split_name}.{name}"
    counts = result.get("counts")
    if not isinstance(counts, dict):
        errors.append(f"{path}.counts must be an object")
        return
    required = ("normal_pass", "normal_fail", "defective_pass", "defective_fail")
    if any(not isinstance(counts.get(key), int) or counts[key] < 0 for key in required):
        errors.append(f"{path}.counts must contain non-negative integers for {required}")
        return

    n_normal = counts["normal_pass"] + counts["normal_fail"]
    n_defective = counts["defective_pass"] + counts["defective_fail"]
    n_total = n_normal + n_defective
    if result.get("n_normal") != n_normal:
        errors.append(f"{path}.n_normal={result.get('n_normal')!r}, expected {n_normal}")
    if result.get("n_defective") != n_defective:
        errors.append(f"{path}.n_defective={result.get('n_defective')!r}, expected {n_defective}")
    if result.get("n_total") != n_total:
        errors.append(f"{path}.n_total={result.get('n_total')!r}, expected {n_total}")

    for metric_name, numerator, denominator in (
        ("defect_escapes", counts["defective_pass"], n_defective),
        ("defect_recall", counts["defective_fail"], n_defective),
        ("false_fail", counts["normal_fail"], n_normal),
    ):
        metric = result.get(metric_name)
        if not isinstance(metric, dict):
            errors.append(f"{path}.{metric_name} must be an object")
        else:
            _rate_error(errors, f"{path}.{metric_name}", metric, numerator, denominator)


def _validate_agent(
    errors: list[str], split_name: str, split: dict[str, Any]
) -> dict[str, int]:
    path = f"{split_name}.agent"
    result = split.get("agent")
    if not isinstance(result, dict):
        errors.append(f"{path} must be an object")
        return {}
    counts = result.get("counts")
    required = (
        "normal_pass_fast",
        "normal_pass_cleared",
        "normal_fail",
        "normal_human",
        "normal_retake",
        "defective_pass_fast",
        "defective_pass_cleared",
        "defective_fail",
        "defective_human",
        "defective_retake",
    )
    if not isinstance(counts, dict):
        errors.append(f"{path}.counts must be an object")
        return {}
    if any(not isinstance(counts.get(key), int) or counts[key] < 0 for key in required):
        errors.append(f"{path}.counts must contain non-negative integers for {required}")
        return {}

    normal_pass = counts["normal_pass_fast"] + counts["normal_pass_cleared"]
    defect_pass = counts["defective_pass_fast"] + counts["defective_pass_cleared"]
    normal_total = normal_pass + counts["normal_fail"] + counts["normal_human"] + counts["normal_retake"]
    defect_total = defect_pass + counts["defective_fail"] + counts["defective_human"] + counts["defective_retake"]
    total = normal_total + defect_total
    human_total = counts["normal_human"] + counts["defective_human"]
    retake_total = counts["normal_retake"] + counts["defective_retake"]
    coverage = total - human_total - retake_total
    correct_automated = normal_pass + counts["defective_fail"]

    for field, actual, expected in (
        ("n_normal", result.get("n_normal"), normal_total),
        ("n_defective", result.get("n_defective"), defect_total),
        ("n_total", result.get("n_total"), total),
        ("split.n_normals", split.get("n_normals"), normal_total),
        ("split.n_anomalies", split.get("n_anomalies"), defect_total),
    ):
        if actual != expected:
            errors.append(f"{path}.{field}={actual!r}, expected {expected}")

    metric_expectations = (
        ("coverage", coverage, total),
        ("defect_escapes", defect_pass, defect_total),
        ("defect_recall", counts["defective_fail"], defect_total),
        ("false_fail", counts["normal_fail"], normal_total),
        ("review_burden", human_total, total),
        ("retake_rate", retake_total, total),
        ("selective_accuracy", correct_automated, coverage),
    )
    for metric_name, numerator, denominator in metric_expectations:
        metric = result.get(metric_name)
        if not isinstance(metric, dict):
            errors.append(f"{path}.{metric_name} must be an object")
        else:
            _rate_error(errors, f"{path}.{metric_name}", metric, numerator, denominator)

    escapes = result.get("defect_escapes", {})
    if isinstance(escapes, dict):
        if escapes.get("fast_pass_escapes") != counts["defective_pass_fast"]:
            errors.append(
                f"{path}.defect_escapes.fast_pass_escapes does not match counts"
            )
        if escapes.get("agent_cleared_escapes") != counts["defective_pass_cleared"]:
            errors.append(
                f"{path}.defect_escapes.agent_cleared_escapes does not match counts"
            )

    return {
        "normal_total": normal_total,
        "defect_total": defect_total,
        "defect_escapes": defect_pass,
        "normal_human": counts["normal_human"],
        "defect_human": counts["defective_human"],
        "normal_fail": counts["normal_fail"],
        "retakes": retake_total,
        "coverage": coverage,
        "selective_correct": correct_automated,
    }


def validate_payload(
    payload: dict[str, Any], expected_commit: str | None = None
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if payload.get("schema_version") != "sureinspect.measured_e1.v1":
        errors.append("schema_version must be 'sureinspect.measured_e1.v1'")
    commit = payload.get("commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        errors.append("commit must be a 40-character lowercase hexadecimal Git commit")
    if expected_commit is not None and commit != expected_commit:
        errors.append(f"commit={commit!r}, expected current commit {expected_commit!r}")
    if payload.get("benchmark_type") != "measured_real_images":
        errors.append("benchmark_type must be 'measured_real_images'")
    if not isinstance(payload.get("pipeline_version"), str) or not payload["pipeline_version"]:
        errors.append("pipeline_version must be a non-empty string")
    if not isinstance(payload.get("timestamp"), str) or not payload["timestamp"]:
        errors.append("timestamp must be a non-empty string")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for key in (
            "snapshot_manifest_sha256",
            "splits_sha256",
            "model_sha256",
            "bank_sha256",
            "calibration_sha256",
            "quality_gate_sha256",
        ):
            value = provenance.get(key)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                errors.append(f"provenance.{key} must be a SHA-256 digest")
        if provenance.get("prediction_cache_commit") != commit:
            errors.append("provenance.prediction_cache_commit must match commit")
        if provenance.get("prediction_count") != 493:
            errors.append("provenance.prediction_count must equal 493")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds must be an object")
    else:
        for key in ("t_pass", "t_fail"):
            if not isinstance(thresholds.get(key), (int, float)) or not math.isfinite(
                float(thresholds[key])
            ):
                errors.append(f"thresholds.{key} must be finite")

    summary: dict[str, Any] = {
        "benchmark_type": payload.get("benchmark_type"),
        "pipeline_version": payload.get("pipeline_version"),
        "commit": commit,
        "timestamp": payload.get("timestamp"),
        "splits": {},
    }
    for split_name in SPLIT_NAMES:
        split = payload.get(split_name)
        if not isinstance(split, dict):
            errors.append(f"{split_name} must be an object")
            continue
        for baseline_name in ("baseline", "baseline_t_fail", "baseline_t_pass"):
            baseline = split.get(baseline_name)
            if not isinstance(baseline, dict):
                errors.append(f"{split_name}.{baseline_name} must be an object")
            else:
                _validate_binary_baseline(errors, split_name, baseline_name, baseline)
        agent_summary = _validate_agent(errors, split_name, split)
        summary["splits"][split_name] = {
            "label": split.get("label"),
            **agent_summary,
        }

    final_label = payload.get("final_test_split", {}).get("label", "")
    if not isinstance(final_label, str) or "not a clean hold-out" not in final_label:
        errors.append(
            "final_test_split.label must preserve the current previously-seen disclosure"
        )
    return errors, summary


def validate_file(
    path: str | Path, expected_commit: str | None = None
) -> tuple[list[str], dict[str, Any]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"could not read valid JSON from {source}: {exc}"], {}
    if not isinstance(payload, dict):
        return [f"top-level JSON in {source} must be an object"], {}
    errors, summary = validate_payload(payload, expected_commit=expected_commit)
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        repo_root = source.resolve().parents[2]
        snapshot_dir = repo_root / "artifacts" / "snapshots" / str(
            provenance.get("snapshot_id", "")
        )
        artifact_map = {
            "snapshot_manifest_sha256": snapshot_dir / "manifest.json",
            "splits_sha256": snapshot_dir / "splits.json",
            "model_sha256": snapshot_dir / "model.onnx",
            "bank_sha256": snapshot_dir / "bank.npz",
            "calibration_sha256": snapshot_dir / "calibration.json",
            "quality_gate_sha256": snapshot_dir / "quality_gate.json",
        }
        for key, artifact in artifact_map.items():
            if not artifact.is_file():
                errors.append(f"provenance artifact is missing: {artifact}")
                continue
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if provenance.get(key) != actual:
                errors.append(
                    f"provenance.{key} does not match {artifact.relative_to(repo_root)}"
                )
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="eval/results/measured_e1_results_v2.json",
        help="Measured-result JSON to validate",
    )
    parser.add_argument(
        "--expected-commit",
        help="Require the result to have been generated from this exact Git commit",
    )
    args = parser.parse_args(argv)
    errors, summary = validate_file(args.path, expected_commit=args.expected_commit)
    print(json.dumps({"valid": not errors, "errors": errors, "summary": summary}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
