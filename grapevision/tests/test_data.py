import json

import pytest
from PIL import Image

from grapevision.data import (
    export_yolo_split,
    group_kfold,
    group_split,
    label_stem,
    load_coco,
    read_catalog,
    scan_capture,
    write_catalog,
)
from grapevision.data.annotations import include_near
from grapevision.data.catalog import parse_capture_name
from grapevision.types import FAR, FULL, NEAR, SPARSE


def test_parse_capture_name():
    assert parse_capture_name("West_Zone2_09_11_2026") == ("West_Zone2", "2026-09-11")
    assert parse_capture_name("LiveImages") == (None, None)
    assert parse_capture_name("Zone_13_40_2026") == ("Zone", None)


def test_scan_capture_structure(capture):
    records = scan_capture(capture)
    images = [r for r in records if r.media == "image"]
    assert len(images) == 8
    assert [r.media for r in records].count("video") == 1
    first = images[0]
    assert (first.zone, first.date, first.row, first.section, first.sequence) == ("West_Zone2", "2026-09-11", 5, 1, 1)
    assert first.rel_path == "row5/Section1/sequence1.jpg"
    portrait = [r for r in images if r.orientation == "portrait"]
    assert [r.rel_path for r in portrait] == ["row7/Section2/sequence2.jpg"]


def test_catalog_roundtrip(capture, tmp_path):
    records = scan_capture(capture)
    write_catalog(records, tmp_path / "catalog.csv")
    assert read_catalog(tmp_path / "catalog.csv") == records


def test_flat_folder_groups_by_image(tmp_path):
    flat = tmp_path / "LiveImages"
    flat.mkdir()
    for i in range(3):
        Image.new("RGB", (10, 10)).save(flat / f"{i}.jpg")
    records = scan_capture(flat)
    assert all(r.row is None for r in records)
    assert len({r.group_key("section") for r in records}) == 3


@pytest.mark.parametrize("level", ["section", "row"])
def test_group_split_keeps_groups_whole(capture, level):
    images = [r for r in scan_capture(capture) if r.media == "image"]
    splits = group_split(images, level, (0.5, 0.25, 0.25) if level == "section" else (0.5, 0.5, 0.0))
    assert sum(len(v) for v in splits.values()) == len(images)
    seen = {}
    for name, recs in splits.items():
        for r in recs:
            assert seen.setdefault(r.group_key(level), name) == name
    if level == "section":
        assert [len(splits[n]) for n in ("train", "val", "test")] == [4, 2, 2]


def test_group_kfold_each_group_tested_once(capture):
    images = [r for r in scan_capture(capture) if r.media == "image"]
    folds = group_kfold(images, "section", k=4)
    tested = [r.rel_path for _, test in folds for r in test]
    assert sorted(tested) == sorted(r.rel_path for r in images)
    for train, test in folds:
        assert not {r.group_key("section") for r in train} & {r.group_key("section") for r in test}


def test_label_stem_avoids_collisions():
    assert label_stem("row5/Section1/sequence1.jpg") != label_stem("row7/Section1/sequence1.jpg")
    assert label_stem("row5/Section1/sequence1.jpg") == "row5__Section1__sequence1"


def _coco(path, row="near", quality="full"):
    data = {
        "images": [{"id": 1, "file_name": "row5/Section1/sequence1.jpg", "width": 80, "height": 60}],
        "categories": [{"id": 1, "name": "bunch"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "attributes": {"row": row, "quality": quality, "occluded": False},
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 5, 10, 10],
                "attributes": {"row": "far", "quality": ""},
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [40, 30, 30, 20],
                "attributes": {"row": "near", "quality": "Sparse"},
            },
        ],
    }
    path.write_text(json.dumps(data))
    return path


def test_load_coco_attributes(tmp_path):
    (image,) = load_coco(_coco(tmp_path / "ann.json"))
    rows = [b.row for b in image.bunches]
    qualities = [b.quality for b in image.bunches]
    assert rows == [NEAR, FAR, NEAR]
    assert qualities == [FULL, None, SPARSE]
    assert image.count(lambda b: b.countable) == 1


def test_load_coco_rejects_typos(tmp_path):
    with pytest.raises(ValueError, match="quality"):
        load_coco(_coco(tmp_path / "ann.json", quality="ful"))


def test_export_yolo_split(capture, tmp_path):
    labels = load_coco(_coco(tmp_path / "ann.json"))
    n = export_yolo_split(labels, capture, tmp_path / "ds", "train", include=include_near)
    assert n == 1
    lines = (tmp_path / "ds/train/labels/row5__Section1__sequence1.txt").read_text().splitlines()
    assert len(lines) == 2  # the far-row box is excluded
    cls, cx, cy, w, h = (float(v) for v in lines[0].split())
    assert cls == 0
    assert (cx, cy, w, h) == pytest.approx((20 / 80, 20 / 60, 20 / 80, 20 / 60))
    assert (tmp_path / "ds/train/images/row5__Section1__sequence1.jpg").exists()
