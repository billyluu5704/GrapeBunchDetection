"""SAHI tiled inference around an Ultralytics YOLO checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from grapevision.types import Box, Detection


class SahiDetector:
    """Callable: PIL image -> detections above `conf_floor`.

    Keep `conf_floor` low and apply the real threshold afterwards (tuned on val with
    eval.tune_threshold), so one inference pass serves every threshold you want to try.

    `postprocess_type` (NMM merges overlapping tile boxes, NMS | GREEDYNMM | LSNMS alternatives) and its match
    threshold change how touching bunches are merged, so they move the count: tune them on val with the rest.
    """

    def __init__(
        self,
        weights: str,
        tile_size: int = 640,
        tile_overlap: float = 0.25,
        conf_floor: float = 0.05,
        device: str = "cuda:0",
        postprocess_type: str = "NMM",
        postprocess_match_threshold: float = 0.5,
    ):
        from sahi import AutoDetectionModel

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.postprocess_type = postprocess_type
        self.postprocess_match_threshold = postprocess_match_threshold
        self._model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights,
            confidence_threshold=conf_floor,
            device=device,
        )

    def __call__(self, image: Image.Image) -> list[Detection]:
        from sahi.predict import get_sliced_prediction

        # Pass pixels, not a path: the image was already EXIF-rotated by grapevision.io.load_image, so the
        # boxes come back on the same grid that the later stages and the annotations use.
        result = get_sliced_prediction(
            np.asarray(image),
            self._model,
            slice_height=self.tile_size,
            slice_width=self.tile_size,
            overlap_height_ratio=self.tile_overlap,
            overlap_width_ratio=self.tile_overlap,
            postprocess_type=self.postprocess_type,
            postprocess_match_threshold=self.postprocess_match_threshold,
            perform_standard_pred=True,
            verbose=0,
        )
        return [
            Detection(Box(p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy), float(p.score.value))
            for p in result.object_prediction_list
        ]


class FullImageDetector:
    """Callable: PIL image -> detections above `conf_floor`, one forward pass on the whole image (no tiling).

    The same interface as SahiDetector, to compare tiled and untiled inference on the same weights."""

    def __init__(self, weights: str, imgsz: int = 640, conf_floor: float = 0.05, device: str = "cuda:0"):
        from ultralytics import YOLO  # type: ignore[attr-defined]

        self.imgsz = imgsz
        self.conf_floor = conf_floor
        self.device = device
        self._model = YOLO(weights)

    def __call__(self, image: Image.Image) -> list[Detection]:
        results: Any = self._model.predict(
            np.asarray(image), imgsz=self.imgsz, conf=self.conf_floor, device=self.device, verbose=False
        )
        result = results[0]
        return [
            Detection(Box(*map(float, xyxy)), float(score))
            for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True)
        ]
