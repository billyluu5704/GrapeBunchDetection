"""Annotation import (CVAT COCO export with per-box attributes) and YOLO dataset export.

Label once in CVAT with one `bunch` label carrying two attributes:
    row      near | far
    quality  full | sparse | damaged | occluded
Every stage's training data derives from that single pass: the detector from boxes, the row filter from
`row`, the quality classifier from `quality`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grapevision.types import FAR, FULL, NEAR, QUALITY_VALUES, ROW_VALUES, Box


@dataclass
class BunchLabel:
    box: Box
    row: str = NEAR
    quality: str | None = None  # None = not judged yet

    @property
    def countable(self) -> bool:
        return self.row == NEAR and self.quality == FULL


@dataclass
class ImageLabels:
    file_name: str  # as stored in the export, e.g. "row5/Section1/sequence1.jpg"
    width: int
    height: int
    bunches: list[BunchLabel] = field(default_factory=list)

    def count(self, predicate: Callable[[BunchLabel], bool] | None = None) -> int:
        return sum(1 for b in self.bunches if predicate is None or predicate(b))


def _attribute(attrs: dict[str, Any], name: str, allowed: tuple[str, ...], where: str) -> str | None:
    value = attrs.get(name)
    if value in (None, ""):
        return None
    value = str(value).strip().lower()
    if value not in allowed:
        raise ValueError(f"{where}: attribute {name}={value!r} not in {allowed}")
    return value


def load_coco(path: str | Path, row_attribute: str = "row", quality_attribute: str = "quality") -> list[ImageLabels]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    images = {
        im["id"]: ImageLabels(im["file_name"].replace("\\", "/"), int(im["width"]), int(im["height"]))
        for im in data["images"]
    }
    for ann in data.get("annotations", []):
        image = images[ann["image_id"]]
        attrs = ann.get("attributes") or {}
        where = f"{image.file_name} annotation {ann.get('id')}"
        image.bunches.append(
            BunchLabel(
                box=Box.from_xywh(*ann["bbox"]),
                row=_attribute(attrs, row_attribute, ROW_VALUES, where) or NEAR,
                quality=_attribute(attrs, quality_attribute, QUALITY_VALUES, where),
            )
        )
    return list(images.values())


def label_stem(file_name: str) -> str:
    """Unique flat name for a nested path. Every section has its own sequence1.jpg, so the plain file stem
    collides; YOLO pairs images and labels by name, so a collision silently mixes up labels."""
    return Path(file_name).with_suffix("").as_posix().replace("/", "__")


def include_all(_: BunchLabel) -> bool:
    return True


def include_near(b: BunchLabel) -> bool:
    return b.row != FAR


def export_yolo_split(
    labels: Iterable[ImageLabels],
    image_root: str | Path,
    out_dir: str | Path,
    split: str,
    include: Callable[[BunchLabel], bool] = include_all,
) -> int:
    """Write <out_dir>/<split>/images/<stem>.<ext> and matching single-class YOLO .txt labels.
    Returns the number of images written."""
    image_root, out_dir = Path(image_root), Path(out_dir)
    img_dir, lbl_dir = out_dir / split / "images", out_dir / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for image in labels:
        src = image_root / image.file_name
        stem = label_stem(image.file_name)
        shutil.copy2(src, img_dir / f"{stem}{src.suffix.lower()}")
        lines = []
        for b in image.bunches:
            if include(b):
                cx, cy, w, h = b.box.clip(image.width, image.height).to_yolo(image.width, image.height)
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1
    return n
