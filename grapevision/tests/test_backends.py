"""Wrapper code around the optional GPU backends, run against fakes (see fakes.py)."""

import numpy as np
import pytest
from fakes import install_fake_sahi, install_fake_transformers, install_fake_ultralytics
from PIL import Image

from grapevision.config import load_config
from grapevision.count import build_pipeline
from grapevision.detect import SahiDetector, train_detector, write_data_yaml
from grapevision.quality import AcceptAllQuality, SizeBaselineQuality
from grapevision.rowfilter import DepthRowFilter, KeepAllRowFilter
from grapevision.rowfilter.depth import DepthAnythingEstimator
from grapevision.types import FAR, NEAR, Box


def test_sahi_detector(monkeypatch):
    calls = install_fake_sahi(monkeypatch, [(1, 2, 30, 40, 0.8)])
    detector = SahiDetector("best.pt", tile_size=320, tile_overlap=0.2, conf_floor=0.01, device="cpu")
    detections = detector(Image.new("RGB", (64, 48)))
    assert [(d.box, d.score) for d in detections] == [(Box(1, 2, 30, 40), 0.8)]
    assert calls["init"]["confidence_threshold"] == 0.01
    assert calls["image_shape"] == (48, 64, 3)  # pixels passed, not a path
    assert calls["predict"]["slice_width"] == 320


def test_depth_estimator_resizes_to_input(monkeypatch):
    calls = install_fake_transformers(monkeypatch, np.full((10, 20), 7))
    estimator = DepthAnythingEstimator(model="m", device="cpu")
    depth = estimator(Image.new("RGB", (40, 20)))
    assert depth.shape == (20, 40)
    assert depth.dtype == np.float32
    assert float(depth.mean()) == pytest.approx(7)
    assert calls["pipeline"] == ("depth-estimation", "m", "cpu")


def test_train_detector_passes_args(monkeypatch, tmp_path):
    calls = install_fake_ultralytics(monkeypatch)
    out = train_detector(
        tmp_path / "data.yaml", weights="yolo11n.pt", epochs=3, device="cpu", project=tmp_path, lr0=0.002
    )
    assert out == "trained"
    assert calls["weights"] == "yolo11n.pt"
    assert calls["train"]["epochs"] == 3
    assert calls["train"]["lr0"] == 0.002


def test_write_data_yaml(tmp_path):
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
    text = write_data_yaml(tmp_path).read_text()
    assert "train: train/images" in text
    assert "val: val/images" in text
    assert "test:" not in text
    assert "names: ['bunch']" in text


def _cfg(**overrides):
    return load_config(overrides={"detector": {"weights": "best.pt", "device": "cpu"}, **overrides})


def test_build_pipeline_end_to_end(monkeypatch, tmp_path):
    install_fake_sahi(monkeypatch, [(5, 5, 20, 20, 0.9), (70, 10, 90, 30, 0.9), (5, 60, 15, 70, 0.1)])
    depth = np.full((100, 100), 50)
    depth[:, :50] = 200  # left half is the near row
    install_fake_transformers(monkeypatch, depth)
    pipeline = build_pipeline(_cfg())
    assert isinstance(pipeline.row_filter, DepthRowFilter)
    assert isinstance(pipeline.quality, AcceptAllQuality)

    path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 100)).save(path)
    result = pipeline.run(path)
    assert [d.row for d in result.detections] == [NEAR, FAR]  # the 0.1-score box is below conf
    assert (result.total, result.near, result.countable) == (2, 1, 1)


def test_build_pipeline_alternatives(monkeypatch):
    install_fake_sahi(monkeypatch, [])
    pipeline = build_pipeline(_cfg(row_filter={"method": "none"}, quality={"method": "size"}))
    assert isinstance(pipeline.row_filter, KeepAllRowFilter)
    assert isinstance(pipeline.quality, SizeBaselineQuality)


@pytest.mark.parametrize("section", ["row_filter", "quality"])
def test_build_pipeline_rejects_unknown_method(monkeypatch, section):
    install_fake_sahi(monkeypatch, [])
    with pytest.raises(ValueError, match=section):
        build_pipeline(_cfg(**{"row_filter": {"method": "none"}, section: {"method": "nope"}}))
