"""Tests for core/calibration.py, the ROI finder, and the calibration/snapshot scripts."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from core.calibration import conformal_k, conformal_threshold, fit_thresholds
from core.perception import find_rois


# ------------------------------------------------------------ conformal

def test_k_and_index_match_the_formula():
    s = np.arange(300, dtype=float)
    t20 = conformal_threshold(s, 0.20)
    assert (t20["k"], t20["index"], t20["threshold"]) == (241, 240, 240.0)
    assert conformal_threshold(s, 0.05)["k"] == 286
    assert conformal_threshold(s, 0.01)["k"] == 298


def test_alpha_is_omitted_when_k_exceeds_n_not_clipped():
    assert conformal_threshold(np.arange(10.0), 0.01) is None          # k = 11 > 10
    out = fit_thresholds(np.arange(15.0), 0.5, [0.01, 0.3])
    assert out["omitted_alpha_fail"] == [0.01]
    assert list(out["t_fail_options"]) == ["0.3"]


def test_float_noise_does_not_change_k():
    assert conformal_k(99, 0.01) == 99          # 100 * 0.99 is 99.00000000000001 in floats
    assert conformal_k(9, 0.1) == 9


def test_threshold_is_order_statistic_regardless_of_input_order_and_ties():
    rng = np.random.default_rng(0)
    s = rng.normal(size=100)
    a = conformal_threshold(s, 0.1)
    b = conformal_threshold(rng.permutation(s), 0.1)
    assert a == b
    ties = conformal_threshold(np.full(50, 2.5), 0.2)
    assert ties["threshold"] == 2.5
    # fail is strict: a score equal to the threshold is NOT a fail
    assert not (2.5 > ties["threshold"])


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        conformal_threshold([], 0.1)
    with pytest.raises(ValueError):
        conformal_threshold([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        fit_thresholds(np.arange(5.0), 0.01, [0.1])


def test_false_alarm_rate_averaged_over_calibration_draws_matches_alpha():
    """The guarantee is marginal over calibration draws: one draw of n=300 varies (sd ~0.02)."""
    import math

    rng = np.random.default_rng(1)
    n = 300
    for alpha in (0.2, 0.05):
        rates = []
        for _ in range(1500):
            t = conformal_threshold(rng.normal(size=n), alpha)["threshold"]
            rates.append(1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))   # exact N(0,1) tail
        expected = 1.0 - conformal_k(n, alpha) / (n + 1)
        assert abs(float(np.mean(rates)) - expected) < 0.005
        assert float(np.mean(rates)) <= alpha + 0.005


# ------------------------------------------------------------------ ROIs

def _heat(h=480, w=640):
    return np.zeros((h, w), np.float32)


def test_rois_are_ranked_by_peak_and_capped_at_three():
    heat = _heat()
    heat[50:80, 50:80] = 3.0
    heat[200:240, 300:360] = 5.0
    heat[400:420, 500:520] = 4.0
    heat[100:110, 400:410] = 2.5
    rois = find_rois(heat, np.zeros((32, 32), np.float32), patch_thr=1.0, max_rois=3)
    assert [r["peak"] for r in rois] == [5.0, 4.0, 3.0]
    assert all(r["source"] == "threshold" for r in rois)
    assert rois[0]["box"] == [300, 200, 60, 40]


def test_fallback_roi_exists_and_is_clipped_with_independent_axis_strides():
    heat = _heat(480, 640)
    pm = np.zeros((32, 32), np.float32)
    pm[0, 31] = 9.0                                    # peak in the top-right corner cell
    rois = find_rois(heat, pm, patch_thr=100.0)
    assert len(rois) == 1 and rois[0]["source"] == "peak_fallback"
    x, y, w, h = rois[0]["box"]
    assert x + w == 640 and y == 0                      # clipped to the image
    assert w == 640 - int(np.floor(30 * 640 / 32)) and h == int(np.ceil(2 * 480 / 32))


def test_roi_ordering_is_deterministic_for_equal_peaks():
    heat = _heat()
    heat[10:20, 10:20] = 2.0
    heat[300:320, 300:320] = 2.0                        # same peak, larger area wins
    a = find_rois(heat, np.zeros((8, 8), np.float32), 1.0)
    b = find_rois(heat, np.zeros((8, 8), np.float32), 1.0)
    assert a == b and a[0]["box"][2] == 20


# ---------------------------------------------- scripts end to end (tiny ONNX)

onnx = pytest.importorskip("onnx")
from core.tests.test_perception import _img, _tiny_onnx, _write_dataset  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("snap")
    p = d / "wide_resnet50_2_64.onnx"
    _tiny_onnx(p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (d / "wide_resnet50_2_64.manifest.json").write_text(json.dumps({
        "backbone": "wide_resnet50_2", "weights": "random_init", "input_size": 64, "input_name": "input",
        "output_names": ["layer2_out", "layer3_out"],
        "output_shapes": {"layer2_out": [1, 8, 32, 32], "layer3_out": [1, 16, 16, 16]},
        "opset": 17, "exporter": "legacy", "torch": "test", "torchvision": "test", "onnx": onnx.__version__,
        "file": p.name, "bytes": p.stat().st_size, "sha256": sha}), encoding="utf-8")
    return p


def _pipeline(tmp_path, tiny):
    from train import build_memory_bank, fit_calibration, prepare_dataset

    root = tmp_path / "data"
    _write_dataset(root)
    cfg = tmp_path / "categories.yaml"
    cfg.write_text(
        "categories:\n  - id: pcb1\n    bank_images: 10\n    calib_images: 15\n    alpha_pass: 0.5\n"
        "    alpha_fail_default: 0.3\n    alpha_fail_options: [0.01, 0.3]\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    assert prepare_dataset.main(["--root", str(root), "--category", "pcb1", "--config", str(cfg),
                                 "--out", str(splits)]) == 0
    bank_dir = tmp_path / "bank"
    assert build_memory_bank.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                                   "--onnx", str(tiny), "--input", "64", "--max-bank", "200",
                                   "--out-dir", str(bank_dir)]) == 0
    bank = bank_dir / "pcb1_wide_resnet50_2_64.npz"
    cal = tmp_path / "cal.json"
    assert fit_calibration.main(["--root", str(root), "--category", "pcb1", "--config", str(cfg),
                                 "--splits", str(splits), "--onnx", str(tiny), "--bank", str(bank),
                                 "--input", "64", "--out", str(cal)]) == 0
    return root, splits, bank, cal


def test_fit_calibration_writes_expected_fields(tmp_path, tiny):
    _, _, _, cal = _pipeline(tmp_path, tiny)
    c = json.loads(cal.read_text())
    assert c["n_calibration"] == 15 and len(c["sorted_calibration_scores"]) == 15
    assert c["sorted_calibration_scores"] == sorted(c["sorted_calibration_scores"])
    assert c["omitted_alpha_fail"] == [0.01] and "0.3" in c["t_fail_options"]
    assert c["t_pass"]["threshold"] < c["t_fail_default"]
    assert c["heatmap"]["patch_thr"] > 0 and "exchangeable" in c["assumptions"]


def test_publish_snapshot_is_complete_checksummed_and_immutable(tmp_path, tiny):
    from train import publish_snapshot

    root, splits, bank, cal = _pipeline(tmp_path, tiny)
    out = tmp_path / "snaps"
    args = ["--root", str(root), "--category", "pcb1", "--splits", str(splits), "--onnx", str(tiny),
            "--bank", str(bank), "--calibration", str(cal), "--out-dir", str(out)]
    assert publish_snapshot.main(args) == 0
    snap = next(out.iterdir())
    m = json.loads((snap / "manifest.json").read_text())
    assert m["readiness"] == "perception_only" and m["tool_policy"] is None
    assert m["versions"]["policy_version"] is None and m["evaluation_status"] == "pending"
    assert m["quality_gate"]["gate_version"] != "stub"
    for name in ("model.onnx", "bank.npz", "calibration.json", "quality_gate.json", "splits.json", "SHA256SUMS"):
        assert (snap / name).is_file()
    raw = (snap / "SHA256SUMS").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")                      # LF only (Windows safe)
    for line in (snap / "SHA256SUMS").read_text().splitlines():       # sha256sum -c format
        digest, name = line.split("  ")
        assert hashlib.sha256((snap / name).read_bytes()).hexdigest() == digest
    with pytest.raises(SystemExit, match="never overwritten"):         # immutability
        publish_snapshot.main(args)


def test_publish_snapshot_rejects_mismatched_artifacts(tmp_path, tiny):
    from train import publish_snapshot

    root, splits, bank, cal = _pipeline(tmp_path, tiny)
    c = json.loads(cal.read_text())
    c["bank_sha256"] = "0" * 64                                         # calibration from another bank
    bad = tmp_path / "bad_cal.json"
    bad.write_text(json.dumps(c))
    with pytest.raises(SystemExit, match="inconsistent"):
        publish_snapshot.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                               "--onnx", str(tiny), "--bank", str(bank), "--calibration", str(bad),
                               "--out-dir", str(tmp_path / "s2")])


def test_agent_ready_is_refused_until_evidence_exists(tmp_path, tiny):
    from train import publish_snapshot

    with pytest.raises(SystemExit, match="agent_ready"):
        publish_snapshot.main(["--root", "x", "--onnx", "x", "--bank", "x", "--calibration", "x",
                               "--readiness", "agent_ready"])
