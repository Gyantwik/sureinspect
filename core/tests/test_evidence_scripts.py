"""End-to-end tests for select_templates.py and fit_evidence.py with a tiny ONNX backbone."""
import json

import cv2
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from core.tests.test_calibration import tiny  # noqa: E402,F401  (fixture)


def _base(seed=0, h=240, w=320):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)
    for _ in range(220):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(2, 10)),
                   int(rng.integers(40, 255)), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _variant(base, seed, defect=False):
    rng = np.random.default_rng(seed)
    h, w = base.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-4, 4)), 1.0)
    m[:, 2] += rng.uniform(-6, 6, 2)
    img = cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    if defect:
        x, y = int(rng.integers(60, 240)), int(rng.integers(40, 180))
        cv2.rectangle(img, (x, y), (x + 22, y + 22), 255, -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _dataset(root):
    cat = root / "pcb1"
    base = _base()
    for sub, n, seed0, defect in (("train/good", 70, 100, False), ("test/good", 12, 300, False), ("test/bad", 12, 500, True)):
        (cat / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(cat / sub / f"{'b' if defect else 'g'}{i:03d}.png"), _variant(base, seed0 + i, defect))
    (cat / "ground_truth" / "bad").mkdir(parents=True, exist_ok=True)
    for i in range(12):
        cv2.imwrite(str(cat / "ground_truth" / "bad" / f"b{i:03d}.png"), np.zeros((240, 320), np.uint8))


@pytest.fixture
def pipeline(tmp_path, tiny):  # noqa: F811
    from train import build_memory_bank, fit_calibration, prepare_dataset

    root = tmp_path / "data"
    _dataset(root)
    cfg = tmp_path / "categories.yaml"
    cfg.write_text("categories:\n  - id: pcb1\n    bank_images: 10\n    calib_images: 15\n    alpha_pass: 0.5\n"
                   "    alpha_fail_default: 0.3\n    alpha_fail_options: [0.01, 0.3]\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    # the synthetic images are all variants of one picture: switch the near-duplicate check off
    assert prepare_dataset.main(["--root", str(root), "--category", "pcb1", "--config", str(cfg),
                                 "--near-dup-hamming", "-1", "--out", str(splits)]) == 0
    bank_dir = tmp_path / "bank"
    assert build_memory_bank.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                                   "--onnx", str(tiny), "--input", "64", "--max-bank", "200",
                                   "--out-dir", str(bank_dir)]) == 0
    bank = bank_dir / "pcb1_wide_resnet50_2_64.npz"
    cal = tmp_path / "cal.json"
    assert fit_calibration.main(["--root", str(root), "--category", "pcb1", "--config", str(cfg),
                                 "--splits", str(splits), "--onnx", str(tiny), "--bank", str(bank),
                                 "--input", "64", "--out", str(cal)]) == 0
    return dict(root=root, splits=splits, bank=bank, cal=cal, tiny=tiny, tmp=tmp_path)


def test_farthest_point_selection_is_deterministic_and_spread():
    from train.select_templates import farthest_point_selection

    pts = np.arange(100, dtype=np.float32)[:, None]
    a, b = farthest_point_selection(pts, 3), farthest_point_selection(pts, 3)
    assert a == b and a[0] in (49, 50) and set(a[1:]) == {0, 99}
    assert len(farthest_point_selection(pts[:2], 5)) == 2


def test_select_templates_writes_standardized_pngs_only_from_bank_normals(pipeline):
    from train import select_templates

    p = pipeline
    out = p["tmp"] / "templates"
    args = ["--root", str(p["root"]), "--category", "pcb1", "--splits", str(p["splits"]), "--onnx", str(p["tiny"]),
            "--input", "64", "--count", "3", "--out-dir", str(out)]
    assert select_templates.main(args) == 0
    meta = json.loads((out / "templates.json").read_text())
    assert len(meta["templates"]) == 3
    bank_paths = {e["path"] for e in json.loads(p["splits"].read_text())["splits"]["bank_normals"]}
    assert all(t["source_path"] in bank_paths for t in meta["templates"])
    assert np.load(out / "embeddings.npz")["embeddings"].shape[0] == 3
    first = [t["source_path"] for t in meta["templates"]]
    assert select_templates.main(args) == 0                              # deterministic, no stale files
    assert [t["source_path"] for t in json.loads((out / "templates.json").read_text())["templates"]] == first
    assert len(list(out.glob("template_*.png"))) == 3


def test_fit_evidence_end_to_end(pipeline):
    from train import fit_evidence, select_templates

    p = pipeline
    out = p["tmp"] / "templates"
    assert select_templates.main(["--root", str(p["root"]), "--category", "pcb1", "--splits", str(p["splits"]),
                                  "--onnx", str(p["tiny"]), "--input", "64", "--count", "3",
                                  "--out-dir", str(out)]) == 0
    result_path = p["tmp"] / "evidence.json"
    rc = fit_evidence.main(["--root", str(p["root"]), "--category", "pcb1", "--splits", str(p["splits"]),
                            "--onnx", str(p["tiny"]), "--bank", str(p["bank"]), "--calibration", str(p["cal"]),
                            "--templates", str(out), "--input", "64", "--out", str(result_path)])
    assert rc == 0
    r = json.loads(result_path.read_text())
    th = r["thresholds"]
    assert set(th) == {"q_min", "r_low", "r_high", "clahe_r_low", "clahe_r_high"}
    assert th["r_low"] <= th["r_high"] and th["clahe_r_low"] <= th["clahe_r_high"]
    assert th["q_min"] > 0.5                                             # textured variants align well
    assert r["report"]["alignment_failure_fraction_normals"] < 0.2
    assert r["report"]["anomalies"]["n"] == 4 and "final" not in r["fitted_on"]
    assert len(r["rows"]["normal"]) == r["report"]["n_normals"]


def test_fit_refuses_when_too_few_normals_align():
    from train.fit_evidence import fit

    rows = [{"aligned": False, "quality": None, "residual": None, "residual_clahe": None,
             "roi1_source": "threshold"}] * 20
    with pytest.raises(SystemExit, match="cannot fit q_min"):
        fit(rows, [])
