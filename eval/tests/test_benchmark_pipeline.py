"""
eval/tests/test_benchmark_pipeline.py
Unit tests verifying the master benchmark pipeline and report generation (E1-E4).
"""
from pathlib import Path
import pytest

from eval.run_benchmarks import (
    load_calibration_data,
    run_e1_decision_benchmark,
    run_e2_calibration_sensitivity,
    run_e3_corruption_robustness,
    generate_markdown_report,
)


def test_load_calibration_data():
    calib = load_calibration_data()
    assert "category" in calib
    assert "t_pass" in calib
    assert "sorted_calibration_scores" in calib
    assert len(calib["sorted_calibration_scores"]) > 0


def test_load_calibration_data_explicit_missing():
    with pytest.raises(FileNotFoundError):
        load_calibration_data("non_existent_snapshot_path_987", fallback_on_missing=False)


def test_run_e1_decision_benchmark():
    calib = load_calibration_data()
    e1 = run_e1_decision_benchmark(calib)
    assert "baseline" in e1
    assert "agent" in e1
    assert "false_alarm_reduction_pct" in e1
    assert e1["false_alarm_reduction_pct"] >= 0.0
    assert e1["agent"]["counts"]["normal_fail"] == 0


def test_run_e2_calibration_sensitivity():
    calib = load_calibration_data()
    e2 = run_e2_calibration_sensitivity(calib)
    assert "alpha_evaluations" in e2
    assert "alpha_0.05" in e2["alpha_evaluations"]


def test_run_e3_corruption_robustness():
    e3 = run_e3_corruption_robustness()
    assert "clean_baseline" in e3
    assert "individual_corruptions" in e3
    assert "blur" in e3["individual_corruptions"]


def test_generate_markdown_report():
    calib = load_calibration_data()
    e1 = run_e1_decision_benchmark(calib)
    e2 = run_e2_calibration_sensitivity(calib)
    e3 = run_e3_corruption_robustness()
    e4 = {
        "iterations": 1,
        "quality_gate": {"mean_ms": 10.0, "p50_ms": 10.0, "p90_ms": 10.0, "p95_ms": 10.0, "max_ms": 10.0},
        "perception": {"mean_ms": 50.0, "p50_ms": 50.0, "p90_ms": 50.0, "p95_ms": 50.0, "max_ms": 50.0},
        "tools": {"mean_ms": 5.0, "p50_ms": 5.0, "p90_ms": 5.0, "p95_ms": 5.0, "max_ms": 5.0},
        "total_inspection": {"mean_ms": 65.0, "p50_ms": 65.0, "p90_ms": 65.0, "p95_ms": 65.0, "max_ms": 65.0},
    }
    summary = {
        "benchmark_timestamp": "2026-10-21T00:00:00Z",
        "dataset": "Amazon VisA pcb1",
        "snapshot_id": "pcb1",
        "e1_decision_benchmark": e1,
        "e2_calibration_sensitivity": e2,
        "e3_corruption_robustness": e3,
        "e4_latency_profiling": e4,
    }
    md = generate_markdown_report(summary)
    assert "# SureInspect: Empirical Evaluation & Benchmark Report" in md
    assert "Experiment E1" in md
    assert "Experiment E2" in md
    assert "Experiment E3" in md
    assert "Experiment E4" in md
