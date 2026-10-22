from pathlib import Path
import cv2
import yaml


def test_opencv_major():
    major = int(cv2.__version__.split(".")[0])
    assert major >= 4, f"OpenCV major version must be >= 4, got {cv2.__version__}"
    import core.perception as P
    assert hasattr(P, "load_backbone")


def test_enabled_primary_category():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config/categories.yaml").read_text())
    assert any(x["id"] == "pcb1" and x["enabled"] for x in config["categories"])
