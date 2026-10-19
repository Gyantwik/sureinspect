"""Tests for core/perception.py and the bank/eval scripts (tiny hand-built ONNX backbone)."""
import json

import cv2
import numpy as np
import pytest

from core.perception import (
    avg_pool3x3_chw,
    combine_layers,
    greedy_coreset,
    make_heatmap,
    nn_distances,
    random_projection,
    resize_bilinear_chw,
)


# ------------------------------------------------------------ pure numpy

@pytest.mark.parametrize("shape,out", [((3, 16, 16), (32, 32)), ((3, 5, 7), (9, 11)), ((1, 8, 8), (8, 8))])
def test_bilinear_matches_opencv_half_pixel(shape, out):
    x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    ours = resize_bilinear_chw(x, *out)
    ref = np.stack([cv2.resize(c, (out[1], out[0]), interpolation=cv2.INTER_LINEAR) for c in x])
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() < 1e-5


def test_bilinear_handles_many_channels():
    x = np.random.default_rng(1).normal(size=(1024, 16, 16)).astype(np.float32)
    assert resize_bilinear_chw(x, 32, 32).shape == (1024, 32, 32)


def test_avg_pool_is_zero_padded_divide_by_nine():
    x = np.random.default_rng(2).normal(size=(4, 10, 12)).astype(np.float32)
    ours = avg_pool3x3_chw(x)
    ref = np.stack([cv2.boxFilter(c, -1, (3, 3), normalize=True, borderType=cv2.BORDER_CONSTANT) for c in x])
    assert np.abs(ours - ref).max() < 1e-5
    ones = avg_pool3x3_chw(np.ones((1, 5, 5), np.float32))
    assert ones[0, 0, 0] == pytest.approx(4 / 9)      # corner: 4 real cells, 5 zero pads
    assert ones[0, 2, 2] == pytest.approx(1.0)


def test_combine_layers_shapes_are_derived():
    l2 = np.random.default_rng(3).normal(size=(1, 8, 32, 32)).astype(np.float32)
    l3 = np.random.default_rng(4).normal(size=(1, 16, 16, 16)).astype(np.float32)
    f = combine_layers(l2, l3)
    assert f.shape == (32 * 32, 24) and f.dtype == np.float32


def test_nn_distances_match_brute_force_with_small_blocks():
    rng = np.random.default_rng(5)
    q, b = rng.normal(size=(37, 6)).astype(np.float32), rng.normal(size=(53, 6)).astype(np.float32)
    ours = nn_distances(q, b, q_block=8, b_block=10)
    brute = np.sqrt(((q[:, None, :] - b[None, :, :]) ** 2).sum(-1).min(1))
    assert np.allclose(ours, brute, atol=1e-4)
    assert (ours >= 0).all()


def test_identical_rows_give_zero_distance_and_never_nan():
    b = np.random.default_rng(6).normal(size=(20, 1536)).astype(np.float32)
    d = nn_distances(b[:5].copy(), b)
    # float32 squared-norm trick: rounding noise of about 0.02 at 1536 dims is expected and negligible
    assert np.isfinite(d).all() and d.max() < 0.05


def test_greedy_coreset_is_deterministic_unique_and_spread_out():
    pts = np.arange(100, dtype=np.float32)[:, None]            # points on a line
    a = greedy_coreset(pts, 4, seed=1)
    b = greedy_coreset(pts, 4, seed=1)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 4
    assert int(a[1]) in (0, 99)                                # second pick is the farthest point


def test_greedy_coreset_caps_k_at_n():
    pts = np.random.default_rng(7).normal(size=(10, 3)).astype(np.float32)
    assert len(greedy_coreset(pts, 50, seed=0)) == 10


def test_random_projection_is_seeded():
    assert np.array_equal(random_projection(50, 8, 3), random_projection(50, 8, 3))
    assert not np.array_equal(random_projection(50, 8, 3), random_projection(50, 8, 4))


def test_heatmap_size_and_peak_position():
    pm = np.zeros((32, 32), np.float32)
    pm[10, 20] = 5.0
    heat = make_heatmap(pm, (640, 480))                         # (w, h)
    assert heat.shape == (480, 640) and heat.dtype == np.float32
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    assert abs(x - (20.5 * 640 / 32)) < 25 and abs(y - (10.5 * 480 / 32)) < 25


# ------------------------------------------------- with a tiny ONNX backbone

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from core.perception import extract_patch_features, load_backbone, score_image  # noqa: E402
from core.preprocess import standardize, to_blob  # noqa: E402


def _tiny_onnx(path):
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.normal(0, 0.3, (8, 3, 3, 3)).astype(np.float32), "w1")
    b1 = numpy_helper.from_array(rng.normal(0, 0.1, (8,)).astype(np.float32), "b1")
    w2 = numpy_helper.from_array(rng.normal(0, 0.3, (16, 8, 3, 3)).astype(np.float32), "w2")
    b2 = numpy_helper.from_array(rng.normal(0, 0.1, (16,)).astype(np.float32), "b2")
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["layer2_out"]),
        helper.make_node("Conv", ["layer2_out", "w2", "b2"], ["c2"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c2"], ["layer3_out"]),
    ]
    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("layer2_out", TensorProto.FLOAT, [1, 8, 32, 32]),
         helper.make_tensor_value_info("layer3_out", TensorProto.FLOAT, [1, 16, 16, 16])],
        initializer=[w1, b1, w2, b2],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


@pytest.fixture(scope="module")
def tiny_net_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("bb") / "tiny.onnx"
    _tiny_onnx(p)
    return p


def _img(seed):
    return np.random.default_rng(seed).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_extract_patch_features_shapes(tiny_net_path):
    net = load_backbone(tiny_net_path)
    feats, grid = extract_patch_features(net, to_blob(_img(0), 64))
    assert grid == (32, 32) and feats.shape == (1024, 24) and np.isfinite(feats).all()


def test_score_is_near_zero_against_own_bank_and_larger_for_another_image(tiny_net_path):
    net = load_backbone(tiny_net_path)
    a, b = standardize(_img(1)).image, standardize(_img(2)).image
    bank, _ = extract_patch_features(net, to_blob(a, 64))
    own = score_image(net, a, bank, 64)
    other = score_image(net, b, bank, 64)
    assert own.score < 1e-2
    assert other.score > own.score + 0.1
    assert own.patch_map.shape == (32, 32)


# --------------------------------------------------- scripts, end to end

def _write_dataset(root):
    cat = root / "pcb1"
    for i in range(70):
        (cat / "train" / "good").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cat / "train" / "good" / f"t{i:03d}.png"), _img(100 + i))
    for i in range(12):
        (cat / "test" / "good").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cat / "test" / "good" / f"g{i:03d}.png"), _img(300 + i))
    for i in range(12):
        (cat / "test" / "bad").mkdir(parents=True, exist_ok=True)
        (cat / "ground_truth" / "bad").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cat / "test" / "bad" / f"b{i:03d}.png"), _img(500 + i))
        cv2.imwrite(str(cat / "ground_truth" / "bad" / f"b{i:03d}.png"), np.zeros((48, 64), np.uint8))


def test_build_bank_and_eval_scripts_run_end_to_end(tmp_path, tiny_net_path):
    from train import build_memory_bank, prepare_dataset

    root = tmp_path / "data"
    _write_dataset(root)
    cfg = tmp_path / "categories.yaml"
    cfg.write_text("categories:\n  - id: pcb1\n    bank_images: 10\n    calib_images: 15\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    assert prepare_dataset.main(["--root", str(root), "--category", "pcb1", "--config", str(cfg),
                                 "--out", str(splits)]) == 0

    out_dir = tmp_path / "bank"
    rc = build_memory_bank.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                                 "--onnx", str(tiny_net_path), "--input", "64", "--ratio", "0.05",
                                 "--max-bank", "200", "--out-dir", str(out_dir)])
    assert rc == 0
    npz = out_dir / "pcb1_wide_resnet50_2_64.npz"
    manifest = json.loads((out_dir / "pcb1_wide_resnet50_2_64.manifest.json").read_text())
    bank = np.load(npz)["bank"]
    assert bank.dtype == np.float32 and bank.shape == (200, 24)       # 10,240 patches * 5% = 512 -> cap 200
    assert manifest["max_bank_cap"] == 200 and manifest["bank_rows"] == 200
    assert manifest["n_total_patches"] == 10 * 1024 and manifest["grid"] == [32, 32]
    assert len(manifest["file_sha256"]) == 64

    # deterministic: building again gives the identical bank
    out2 = tmp_path / "bank2"
    build_memory_bank.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                            "--onnx", str(tiny_net_path), "--input", "64", "--ratio", "0.05",
                            "--max-bank", "200", "--out-dir", str(out2)])
    assert np.array_equal(bank, np.load(out2 / "pcb1_wide_resnet50_2_64.npz")["bank"])

    pytest.importorskip("sklearn")
    from train import eval_dev_auroc
    rc = eval_dev_auroc.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                              "--onnx", str(tiny_net_path), "--bank", str(npz), "--input", "64",
                              "--limit", "6", "--out", str(tmp_path / "auroc.json")])
    assert rc == 0
    rep = json.loads((tmp_path / "auroc.json").read_text())
    assert 0.0 <= rep["image_auroc"] <= 1.0 and rep["n_normals"] == 6


def test_bank_builder_detects_changed_dataset(tmp_path, tiny_net_path):
    from train import build_memory_bank, prepare_dataset

    root = tmp_path / "data"
    _write_dataset(root)
    cfg = tmp_path / "categories.yaml"
    cfg.write_text("categories:\n  - id: pcb1\n    bank_images: 10\n    calib_images: 15\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    prepare_dataset.main(["--root", str(root), "--category", "pcb1", "--config", str(cfg), "--out", str(splits)])
    first_bank = json.loads(splits.read_text())["splits"]["bank_normals"][0]["path"]
    cv2.imwrite(str(root / first_bank), _img(999))                    # change the file
    with pytest.raises(SystemExit, match="hash mismatch"):
        build_memory_bank.main(["--root", str(root), "--category", "pcb1", "--splits", str(splits),
                                "--onnx", str(tiny_net_path), "--input", "64", "--out-dir", str(tmp_path / "b")])
