"""Tests for train/prepare_dataset.py."""
import json
import shutil

import cv2
import numpy as np
import pytest

from train.prepare_dataset import (
    SplitError,
    build_splits,
    main,
    verify_no_leakage,
    wilson_interval,
)

CAT = "pcb1"


def _write(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8))


@pytest.fixture
def dataset(tmp_path):
    """70 train normals, 12 test normals, 12 test anomalies (with masks)."""
    base = tmp_path / CAT
    for i in range(70):
        _write(base / "train" / "good" / f"t{i:03d}.png", seed=1000 + i)
    for i in range(12):
        _write(base / "test" / "good" / f"g{i:03d}.png", seed=2000 + i)
    for i in range(12):
        _write(base / "test" / "bad" / f"b{i:03d}.png", seed=3000 + i)
        m = base / "ground_truth" / "bad" / f"b{i:03d}.png"
        m.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(m), np.zeros((48, 64), dtype=np.uint8))
    return tmp_path


def _all_paths(result):
    s = result["splits"]
    out = []
    for key in ("bank_normals", "calib_normals", "dev_eval_normals", "dev_anomalies"):
        out += [e["path"] for e in s[key]]
    out += [e["path"] for e in s["demo_pool"]["normals"] + s["demo_pool"]["anomalies"]]
    out += [e["path"] for e in s["final_test"]["normals"] + s["final_test"]["anomalies"]]
    return out


def test_counts_and_no_overlap(dataset):
    r = build_splits(dataset, CAT, bank_images=10, calib_images=15)
    c = r["counts"]
    assert c["bank_normals"] == 10
    assert c["calib_normals"] == 15
    assert c["demo_normals"] == 6
    assert c["dev_eval_normals"] == 70 - 10 - 15 - 6        # 39
    assert c["final_test_normals"] == 12
    assert c["dev_anomalies"] + c["demo_anomalies"] + c["final_test_anomalies"] == 12
    assert c["dev_anomalies"] == 4                          # round(0.30 * 12)
    paths = _all_paths(r)
    assert len(paths) == len(set(paths))                    # nothing in two splits


def test_small_dev_set_shares_gate_and_eval(dataset):
    r = build_splits(dataset, CAT, bank_images=10, calib_images=15)
    assert r["meta"]["dev_normals_shared_between_gate_and_eval"] is True
    assert r["splits"]["dev_gate_fit_normals"] == r["splits"]["dev_eval_normals"]


def test_large_dev_set_separates_gate_and_eval(tmp_path):
    base = tmp_path / CAT
    for i in range(160):
        _write(base / "train" / "good" / f"t{i:03d}.png", seed=i)
    for i in range(5):
        _write(base / "test" / "good" / f"g{i}.png", seed=500 + i)
    for i in range(10):
        _write(base / "test" / "bad" / f"b{i}.png", seed=700 + i)
        m = base / "ground_truth" / "bad" / f"b{i}.png"
        m.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(m), np.zeros((48, 64), dtype=np.uint8))
    r = build_splits(tmp_path, CAT, bank_images=20, calib_images=30)
    assert r["meta"]["dev_normals_shared_between_gate_and_eval"] is False
    gate = {e["path"] for e in r["splits"]["dev_gate_fit_normals"]}
    ev = {e["path"] for e in r["splits"]["dev_eval_normals"]}
    assert gate and ev and not (gate & ev)


def test_deterministic_for_same_seed_and_different_for_other_seed(dataset):
    a = build_splits(dataset, CAT, 10, 15, seed=42)
    b = build_splits(dataset, CAT, 10, 15, seed=42)
    c = build_splits(dataset, CAT, 10, 15, seed=7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["splits"]["bank_normals"] != c["splits"]["bank_normals"]


def test_train_copy_of_a_test_image_is_excluded(dataset):
    src = dataset / CAT / "test" / "good" / "g000.png"
    shutil.copy(src, dataset / CAT / "train" / "good" / "leak.png")
    r = build_splits(dataset, CAT, 10, 15)
    assert r["counts"]["excluded"] == 1
    assert r["splits"]["excluded"][0]["path"].endswith("leak.png")
    assert not any(p.endswith("leak.png") for p in _all_paths(r))


def test_near_duplicates_stay_in_one_split(dataset):
    # tiny edit keeps the perceptual hash but changes the file hash
    a = cv2.imread(str(dataset / CAT / "train" / "good" / "t000.png"))
    a2 = a.copy()
    a2[0, 0] = (a2[0, 0].astype(int) + 1) % 256
    cv2.imwrite(str(dataset / CAT / "train" / "good" / "t000_copy.png"), a2)
    r = build_splits(dataset, CAT, 10, 15)
    s = r["splits"]
    where = {}
    for name in ("bank_normals", "calib_normals", "dev_eval_normals"):
        for e in s[name]:
            where[e["path"].split("/")[-1]] = name
    for e in s["demo_pool"]["normals"]:
        where[e["path"].split("/")[-1]] = "demo"
    assert where["t000.png"] == where["t000_copy.png"]


def test_similar_test_normal_and_anomaly_are_kept_together_in_final(dataset):
    # a test anomaly that is a near-duplicate of a test normal must not cause a false
    # leakage error; both stay in the final test set.
    g = cv2.imread(str(dataset / CAT / "test" / "good" / "g000.png"))
    g2 = g.copy()
    g2[0, 0] = (g2[0, 0].astype(int) + 1) % 256
    cv2.imwrite(str(dataset / CAT / "test" / "bad" / "b_like_good.png"), g2)
    m = dataset / CAT / "ground_truth" / "bad" / "b_like_good.png"
    cv2.imwrite(str(m), np.zeros((48, 64), dtype=np.uint8))
    r = build_splits(dataset, CAT, 10, 15)
    final_anoms = {e["path"].split("/")[-1] for e in r["splits"]["final_test"]["anomalies"]}
    dev_anoms = {e["path"].split("/")[-1] for e in r["splits"]["dev_anomalies"]}
    assert "b_like_good.png" in final_anoms
    assert "b_like_good.png" not in dev_anoms


def test_leakage_check_catches_a_planted_overlap(dataset):
    r = build_splits(dataset, CAT, 10, 15)
    s = r["splits"]
    s["calib_normals"].append(dict(s["bank_normals"][0]))
    with pytest.raises(SplitError, match="Leakage"):
        verify_no_leakage(s, shared_dev_normals=True)


def test_not_enough_training_images_gives_actionable_error(dataset):
    with pytest.raises(SplitError, match="config/categories.yaml"):
        build_splits(dataset, CAT, bank_images=40, calib_images=40)


def test_too_few_anomalies_is_rejected(dataset):
    for i in range(5):
        (dataset / CAT / "test" / "bad" / f"b{i:03d}.png").unlink()
    with pytest.raises(SplitError, match="anomalies"):
        build_splits(dataset, CAT, 10, 15)


def test_missing_mask_is_rejected(dataset):
    (dataset / CAT / "ground_truth" / "bad" / "b000.png").unlink()
    with pytest.raises(SplitError, match="mask"):
        build_splits(dataset, CAT, 10, 15)


def test_missing_folder_message_mentions_prepare_data(tmp_path):
    with pytest.raises(SplitError, match="prepare_data"):
        build_splits(tmp_path, CAT, 10, 15)


def test_wilson_interval_is_sane():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=0.002)
    assert hi == pytest.approx(0.596, abs=0.002)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_cli_writes_json(dataset, tmp_path):
    cfg = tmp_path / "categories.yaml"
    cfg.write_text(
        "categories:\n  - id: pcb1\n    bank_images: 10\n    calib_images: 15\n", encoding="utf-8"
    )
    out = tmp_path / "out" / "splits.json"
    rc = main(["--root", str(dataset), "--category", CAT, "--config", str(cfg), "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["counts"]["bank_normals"] == 10
    assert data["meta"]["seed"] == 42


def test_cli_reports_errors_with_exit_code_2(tmp_path):
    cfg = tmp_path / "categories.yaml"
    cfg.write_text("categories:\n  - id: pcb1\n    bank_images: 10\n    calib_images: 15\n", encoding="utf-8")
    rc = main(["--root", str(tmp_path / "nope"), "--category", CAT, "--config", str(cfg)])
    assert rc == 2
