"""Tests for train/verify_onnx_opencv.py using a tiny hand-built ONNX graph.

Needs the optional packages `onnx` and `onnxruntime` (skipped when absent, e.g. in CI).
onnxruntime stands in for PyTorch here: it is an independent reference implementation.
"""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from train.verify_onnx_opencv import (  # noqa: E402
    OUT_NAMES,
    compare_features,
    load_net,
    opencv_features,
)


def _tiny_graph(path):
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
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("layer2_out", TensorProto.FLOAT, [1, 8, 32, 32]),
         helper.make_tensor_value_info("layer3_out", TensorProto.FLOAT, [1, 16, 16, 16])],
        initializer=[w1, b1, w2, b2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    p = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    _tiny_graph(p)
    return p


def _blob(seed=1):
    return np.random.default_rng(seed).normal(0, 1, (1, 3, 64, 64)).astype(np.float32)


def test_opencv_dnn_matches_reference_runtime(tiny):
    blob = _blob()
    sess = ort.InferenceSession(str(tiny), providers=["CPUExecutionProvider"])
    ref = sess.run(list(OUT_NAMES), {"input": blob})
    got = opencv_features(load_net(tiny), blob)
    res = compare_features(ref, got, tol=1e-3)
    assert res["ok"], res
    assert [o["shape"] for o in res["outputs"]] == [[1, 8, 32, 32], [1, 16, 16, 16]]


def test_compare_detects_value_differences():
    a = [np.zeros((1, 2, 4, 4), np.float32)] * 2
    b = [np.zeros((1, 2, 4, 4), np.float32), np.full((1, 2, 4, 4), 0.01, np.float32)]
    res = compare_features(a, b, tol=1e-3)
    assert not res["ok"]
    assert res["outputs"][0]["ok"] and not res["outputs"][1]["ok"]


def test_compare_never_passes_on_shape_mismatch_or_nan():
    a = [np.zeros((1, 2, 4, 4), np.float32)] * 2
    assert not compare_features(a, [np.zeros((1, 2, 4, 5), np.float32)] * 2)["ok"]
    assert not compare_features(a, a[:1])["ok"]
    nan = [np.full((1, 2, 4, 4), np.nan, np.float32)] * 2
    assert not compare_features(a, nan)["ok"]


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(Exception):
        load_net(tmp_path / "nope.onnx")
