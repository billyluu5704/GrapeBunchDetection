"""Monocular depth estimation (optional dependency: `pip install -e .[depth]`)."""

from __future__ import annotations

import numpy as np
from PIL import Image


class DepthAnythingEstimator:
    """Relative depth from a single RGB photo via Hugging Face transformers.

    Returns an HxW float array at the input image's size. Depth Anything predicts relative *inverse* depth,
    so by default higher values mean closer. Values are only comparable within one image, which is all the
    near/far split needs. Check the direction on one photo (see the README) before trusting it.
    """

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf", device: int | str | None = None):
        from transformers import pipeline

        self._pipe = pipeline("depth-estimation", model=model, device=device)

    def __call__(self, image: Image.Image) -> np.ndarray:
        depth = self._pipe(image)["depth"]  # PIL image, already resized to the input size
        if depth.size != image.size:
            depth = depth.resize(image.size, Image.Resampling.BILINEAR)
        return np.asarray(depth, dtype=np.float32)
