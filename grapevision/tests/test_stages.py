import numpy as np
import pytest
from PIL import Image

from grapevision.count import CountingPipeline, aggregate
from grapevision.data import scan_capture
from grapevision.quality import SizeBaselineQuality, crop_box
from grapevision.rowfilter import DepthRowFilter, box_depth, netting_score, otsu_split, split_near_far
from grapevision.types import FAR, FULL, NEAR, SPARSE, Box, Detection


def test_otsu_separability():
    _, tight = otsu_split([1.0, 1.1, 0.9, 9.0, 9.1, 8.9])
    _, uniform = otsu_split(np.linspace(0, 1, 200))
    assert tight > 0.99
    assert uniform == pytest.approx(0.75, abs=0.01)


def test_split_near_far():
    assert split_near_far([200, 205, 198, 40, 42]) == [NEAR, NEAR, NEAR, FAR, FAR]
    assert split_near_far([200, 205, 198, 40, 42], higher_is_closer=False) == [FAR, FAR, FAR, NEAR, NEAR]
    # one row seen obliquely: depth spreads evenly, so nothing is declared far
    assert set(split_near_far(list(np.linspace(50, 200, 20)))) == {NEAR}
    assert split_near_far([100]) == [NEAR]


def test_box_depth_uses_center():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[40:60, 40:60] = 10  # center of the box
    assert box_depth(depth, Box(20, 20, 80, 80), inner=0.3) == 10


def test_netting_score():
    rgb = np.zeros((50, 100, 3), dtype=np.uint8)
    rgb[:, :50] = (60, 80, 200)  # blue-tinted left half
    rgb[:, 50:] = (40, 40, 60)  # dark grapes
    assert netting_score(rgb, Box(0, 0, 50, 50)) == 1.0
    assert netting_score(rgb, Box(50, 0, 100, 50)) == 0.0


def test_depth_row_filter_marks_far_boxes():
    image = Image.new("RGB", (100, 100))
    depth = np.full((100, 100), 50, dtype=np.float32)
    depth[:, :50] = 200  # left half is close
    dets = [Detection(Box(5, 5, 20, 20), 0.9), Detection(Box(25, 60, 40, 80), 0.9), Detection(Box(70, 10, 90, 30), 0.9)]
    DepthRowFilter(lambda _: depth).apply(image, dets)
    assert [d.row for d in dets] == [NEAR, NEAR, FAR]
    assert dets[2].features["depth"] == 50


def test_size_baseline_quality():
    image = Image.new("RGB", (100, 100))
    dets = [Detection(Box(0, 0, 50, 50), 0.9), Detection(Box(0, 0, 2, 2), 0.9)]
    SizeBaselineQuality(min_area_frac=0.01).apply(image, dets)
    assert [d.quality for d in dets] == [FULL, SPARSE]


def test_crop_box_pads_and_clips():
    image = Image.new("RGB", (100, 100))
    assert crop_box(image, Box(10, 10, 30, 30), pad=0.5).size == (40, 40)
    assert crop_box(image, Box(90, 90, 100, 100), pad=0.5).size == (15, 15)


def test_pipeline_and_aggregate(capture):
    records = [r for r in scan_capture(capture) if r.media == "image"]

    def fake_detector(image):
        return [
            Detection(Box(0, 0, 30, 30), 0.9),
            Detection(Box(40, 0, 50, 10), 0.6),
            Detection(Box(0, 40, 5, 45), 0.1),
        ]

    def far_right_half(image, detections):
        for d in detections:
            d.row = FAR if d.box.x1 >= 40 else NEAR

    class Stage:
        apply = staticmethod(far_right_half)

    pipeline = CountingPipeline(fake_detector, row_filter=Stage(), conf=0.25)
    results = [pipeline.run(r.path) for r in records]
    assert (results[0].total, results[0].near, results[0].countable) == (2, 1, 1)
    totals = aggregate(results, records, "row")
    assert totals[f"{records[0].capture}/row5"] == {"images": 4, "total": 8, "near": 4, "countable": 4}
