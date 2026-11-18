"""Stage 3: judge the quality of each near-row bunch. FULL and OCCLUDED bunches count; SPARSE and DAMAGED
do not (see `COUNTED_QUALITIES`).

Planned model: a small CNN (e.g. EfficientNet/ConvNeXt-tiny) fine-tuned on crops from
`export_quality_crops`, predicting one of the four quality classes so the farm can tune the cutoff without
retraining. The classes here are baselines that it has to beat. Neither can output OCCLUDED, since box
geometry says nothing about a leaf in the way; only the trained model will separate OCCLUDED from SPARSE.
"""

from __future__ import annotations

from PIL import Image

from grapevision.types import FULL, SPARSE, Detection


class AcceptAllQuality:
    """Every bunch counts (the old pipeline's behavior)."""

    def apply(self, image: Image.Image, detections: list[Detection]) -> None:
        for d in detections:
            d.quality = FULL


class SizeBaselineQuality:
    """Bunches whose box covers less than `min_area_frac` of the image are SPARSE, the rest FULL.
    Crude, because apparent size also depends on camera distance, and a small box may really be a partly
    hidden (OCCLUDED, counted) bunch. It's here as the number to beat."""

    def __init__(self, min_area_frac: float = 0.002):
        self.min_area_frac = min_area_frac

    def apply(self, image: Image.Image, detections: list[Detection]) -> None:
        image_area = image.width * image.height
        for d in detections:
            frac = d.box.area / image_area
            d.features["area_frac"] = frac
            d.quality = FULL if frac >= self.min_area_frac else SPARSE
