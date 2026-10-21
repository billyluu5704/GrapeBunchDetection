"""SAHI tiled inference around an Ultralytics YOLO checkpoint."""

from __future__ import annotations

import numpy as np
from PIL import Image

from grapevision.types import Box, Detection


class SahiDetector:
    """Callable: PIL image -> detections above `conf_floor`.

    Keep `conf_floor` low and apply the real threshold afterwards (tuned on val with
    eval.tune_threshold), so one inference pass serves every threshold you want to try.
    """

    def __init__(
        self,
        weights: str,
        tile_size: int = 640,
        tile_overlap: float = 0.25,
        conf_floor: float = 0.05,
        device: str = "cuda:0",
    ):
        from sahi import AutoDetectionModel

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
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
            postprocess_type="NMM",
            postprocess_match_threshold=0.5,
            perform_standard_pred=True,
            verbose=0,
        )
        return [
            Detection(Box(p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy), float(p.score.value))
            for p in result.object_prediction_list
        ]
