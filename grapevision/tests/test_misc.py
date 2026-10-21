"""Config, visualization, crop export and edge cases."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from grapevision.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config, resolve_path
from grapevision.count import ImageResult, aggregate
from grapevision.data import group_kfold, group_split, load_coco, scan_capture
from grapevision.eval import count_metrics
from grapevision.quality import export_quality_crops
from grapevision.rowfilter import DepthRowFilter, box_depth, otsu_split, split_near_far
from grapevision.types import FAR, FULL, NEAR, SPARSE, Box, Detection
from grapevision.viz import draw_result
from grapevision.viz.draw import BACKGROUND, COUNTED, NOT_COUNTED


def test_default_config_loads():
    cfg = load_config()
    assert cfg["split"]["level"] == "section"
    assert cfg["paths"]["captures"] == []  # machine-specific paths live in configs/, not the package
    assert DEFAULT_CONFIG.is_file()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()  # source checkout resolves against the project root


def test_config_layers_file_then_overrides(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("split:\n  seed: 7\ndetector:\n  conf: 0.4\n")
    cfg = load_config(custom, overrides={"detector": {"conf": 0.5}})
    assert cfg["split"]["seed"] == 7
    assert cfg["split"]["level"] == "section"  # untouched keys keep their defaults
    assert cfg["detector"]["conf"] == 0.5
    assert cfg["detector"]["tile_size"] == 640


def test_resolve_path(tmp_path):
    assert resolve_path(tmp_path) == tmp_path
    assert resolve_path("outputs") == PROJECT_ROOT / "outputs"


def test_draw_result_colors_by_stage():
    image = Image.new("RGB", (400, 300))
    dets = [
        Detection(Box(10, 100, 60, 150), 0.9, row=NEAR, quality=FULL),
        Detection(Box(100, 100, 150, 150), 0.9, row=NEAR, quality=SPARSE),
        Detection(Box(200, 100, 250, 150), 0.9, row=FAR),
    ]
    out = draw_result(image, ImageResult("x.jpg", dets))
    assert out.size == image.size
    assert out.getpixel((10, 125)) == COUNTED
    assert out.getpixel((100, 125)) == NOT_COUNTED
    assert out.getpixel((200, 125)) == BACKGROUND
    assert image.getpixel((10, 125)) == (0, 0, 0)  # input left untouched


def test_export_quality_crops(capture, tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "row5/Section1/sequence1.jpg", "width": 80, "height": 60},
            {"id": 2, "file_name": "row5/Section1/sequence2.jpg", "width": 80, "height": 60},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [10, 10, 20, 20], "attributes": {"row": "near", "quality": "full"}},
            {"id": 2, "image_id": 1, "bbox": [40, 10, 20, 20], "attributes": {"row": "near", "quality": "sparse"}},
            {"id": 3, "image_id": 1, "bbox": [60, 10, 10, 10], "attributes": {"row": "far", "quality": "full"}},
            {"id": 4, "image_id": 2, "bbox": [10, 10, 20, 20], "attributes": {"row": "near"}},
        ],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    counts = export_quality_crops(load_coco(tmp_path / "ann.json"), capture, tmp_path / "crops")
    assert counts == {FULL: 1, SPARSE: 1}  # far-row and unjudged bunches are skipped
    assert (tmp_path / "crops/full/row5__Section1__sequence1__0.jpg").exists()


def test_unreadable_image_still_catalogs(tmp_path):
    root = tmp_path / "cap"
    root.mkdir()
    (root / "broken.jpg").write_bytes(b"not a jpeg")
    (record,) = scan_capture(root)
    assert (record.width, record.height, record.orientation) == (None, None, None)


def test_group_key_rejects_unknown_level(capture):
    record = scan_capture(capture)[0]
    with pytest.raises(ValueError, match="unknown group level"):
        record.group_key("vine")


def test_split_argument_errors(capture):
    images = [r for r in scan_capture(capture) if r.media == "image"]
    with pytest.raises(ValueError, match="same length"):
        group_split(images, ratios=(0.5, 0.5))
    with pytest.raises(ValueError, match="need at least"):
        group_kfold(images, "row", k=3)


def test_count_metrics_errors():
    with pytest.raises(ValueError, match="mismatch"):
        count_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="no images"):
        count_metrics([], [])


def test_box_yolo_roundtrip():
    box = Box(10, 20, 50, 80)
    back = Box.from_yolo(*box.to_yolo(100, 100), 100, 100)
    assert (back.x1, back.y1, back.x2, back.y2) == pytest.approx((10, 20, 50, 80))


def test_rowfilter_edge_cases():
    image = Image.new("RGB", (10, 10))
    DepthRowFilter(lambda _: pytest.fail("depth must not run without detections")).apply(image, [])
    edge = np.zeros((10, 10))
    edge[9, 9] = 3
    assert box_depth(edge, Box(20, 20, 30, 30)) == 3  # box past the corner falls back to the edge pixel
    assert math.isnan(box_depth(np.zeros((0, 0)), Box(0, 0, 5, 5)))
    threshold, separability = otsu_split([5.0, 5.0, 5.0])
    assert math.isnan(threshold)
    assert separability == 0.0
    assert split_near_far([200.0, 201.0, float("nan"), 40.0, 41.0]) == [NEAR, NEAR, NEAR, FAR, FAR]


def test_aggregate_rejects_uncataloged_image(capture):
    records = scan_capture(capture)
    with pytest.raises(KeyError, match="not in the catalog"):
        aggregate([ImageResult("elsewhere.jpg", [])], records)
