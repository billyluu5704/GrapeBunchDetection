"""detect -> row filter -> quality -> count, plus per-row/section totals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from grapevision.data.catalog import MediaRecord
from grapevision.io import load_image
from grapevision.quality import AcceptAllQuality, SizeBaselineQuality
from grapevision.rowfilter import DepthRowFilter, KeepAllRowFilter
from grapevision.types import Detection


class Stage(Protocol):
    def apply(self, image: Image.Image, detections: list[Detection]) -> None: ...


@dataclass
class ImageResult:
    image: str
    detections: list[Detection]

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def near(self) -> int:
        return sum(d.near for d in self.detections)

    @property
    def countable(self) -> int:
        return sum(d.countable for d in self.detections)


class CountingPipeline:
    def __init__(
        self,
        detector: Callable[[Image.Image], list[Detection]],
        row_filter: Stage | None = None,
        quality: Stage | None = None,
        conf: float = 0.25,
    ):
        self.detector = detector
        self.row_filter = row_filter or KeepAllRowFilter()
        self.quality = quality or AcceptAllQuality()
        self.conf = conf

    def run(self, image_path: str | Path) -> ImageResult:
        image = load_image(image_path)
        detections = [d for d in self.detector(image) if d.score >= self.conf]
        self.row_filter.apply(image, detections)
        self.quality.apply(image, [d for d in detections if d.near])
        return ImageResult(str(image_path), detections)


def build_pipeline(cfg: dict[str, Any]) -> CountingPipeline:
    """Assemble the pipeline from a config dict (see grapevision/default_config.yaml)."""
    from grapevision.detect import SahiDetector

    det_cfg = cfg["detector"]
    detector = SahiDetector(
        det_cfg["weights"],
        det_cfg["tile_size"],
        det_cfg["tile_overlap"],
        det_cfg["conf_floor"],
        det_cfg["device"],
    )

    rf_cfg = cfg["row_filter"]
    if rf_cfg["method"] == "depth":
        from grapevision.rowfilter.depth import DepthAnythingEstimator

        row_filter: Stage = DepthRowFilter(
            DepthAnythingEstimator(rf_cfg["depth_model"]),
            rf_cfg["min_separability"],
            rf_cfg["higher_is_closer"],
            rf_cfg["box_inner"],
        )
    elif rf_cfg["method"] == "none":
        row_filter = KeepAllRowFilter()
    else:
        raise ValueError(f"unknown row_filter.method {rf_cfg['method']!r}")

    q_cfg = cfg["quality"]
    if q_cfg["method"] == "accept_all":
        quality: Stage = AcceptAllQuality()
    elif q_cfg["method"] == "size":
        quality = SizeBaselineQuality(q_cfg["min_area_frac"])
    else:
        raise ValueError(f"unknown quality.method {q_cfg['method']!r}")

    return CountingPipeline(detector, row_filter, quality, det_cfg["conf"])


def aggregate(
    results: Iterable[ImageResult], records: Iterable[MediaRecord], level: str = "row"
) -> dict[str, dict[str, int]]:
    """Sum counts per group (row or section). Assumes shots within a group don't overlap; overlapping shots
    would count the same bunch twice."""
    by_path = {str(Path(r.path).resolve()): r for r in records}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"images": 0, "total": 0, "near": 0, "countable": 0})
    for res in results:
        record = by_path.get(str(Path(res.image).resolve()))
        if record is None:
            raise KeyError(f"{res.image} is not in the catalog")
        t = totals[record.group_key(level)]
        t["images"] += 1
        t["total"] += res.total
        t["near"] += res.near
        t["countable"] += res.countable
    return dict(totals)
