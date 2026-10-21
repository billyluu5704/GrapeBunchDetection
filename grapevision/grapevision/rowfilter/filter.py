"""Stage 2: decide whether each detected bunch is on the surveyed (near) row or a row behind it.

Baseline here: per-box depth from a monocular depth model, split into two groups per image with Otsu's
method. A far group is declared only when the two groups separate cleanly, since many photos show no
background row at all. Next step once `row` labels exist: train a small near/far classifier on
`Detection.features` (depth, netting, box size), with this rule as the baseline to beat.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image

from grapevision.types import FAR, NEAR, Box, Detection


def _inner_region(arr: np.ndarray, box: Box, inner: float) -> np.ndarray:
    h, w = arr.shape[:2]
    b = box.clip(w, h)
    cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    hw, hh = max(b.width * inner / 2, 0.5), max(b.height * inner / 2, 0.5)
    x1, x2 = int(max(0, cx - hw)), int(min(w, np.ceil(cx + hw)))
    y1, y2 = int(max(0, cy - hh)), int(min(h, np.ceil(cy + hh)))
    return arr[y1:y2, x1:x2]


def box_depth(depth: np.ndarray, box: Box, inner: float = 0.5) -> float:
    """Median depth over the central `inner` fraction of the box. The center is mostly berries; the
    border is mostly leaves and gaps, which can sit at a different depth."""
    region = _inner_region(depth, box, inner)
    return float(np.median(region)) if region.size else float("nan")


def netting_score(rgb: np.ndarray, box: Box, margin: int = 25, inner: float = 0.8) -> float:
    """Fraction of pixels in the box where blue clearly dominates red and green. Bunches seen through
    the blue bird netting on some background rows score high. Only a hint: most rows have no netting."""
    region = _inner_region(rgb, box, inner).astype(np.int16)
    if region.size == 0:
        return 0.0
    r, g, b = region[..., 0], region[..., 1], region[..., 2]
    return float(np.mean((b > r + margin) & (b > g + margin)))


def otsu_split(values: Sequence[float]) -> tuple[float, float]:
    """1-D Otsu over the values. Returns (threshold, separability), where separability is the
    between-class share of total variance: ~1 for two tight clusters, 0.75 for a uniform spread."""
    v = np.sort(np.asarray([x for x in values if np.isfinite(x)], dtype=float))
    if v.size < 2 or np.var(v) == 0:
        return float("nan"), 0.0
    total_var = np.var(v)
    best_between, best_i = -1.0, 1
    for i in range(1, v.size):
        w0 = i / v.size
        between = w0 * (1 - w0) * (v[:i].mean() - v[i:].mean()) ** 2
        if between > best_between:
            best_between, best_i = between, i
    return float((v[best_i - 1] + v[best_i]) / 2), float(best_between / total_var)


def split_near_far(depths: Sequence[float], min_separability: float = 0.85, higher_is_closer: bool = True) -> list[str]:
    """NEAR/FAR per value. Everything is NEAR unless the values form two clearly separated groups."""
    threshold, separability = otsu_split(depths)
    if separability < min_separability:
        return [NEAR] * len(depths)
    out = []
    for d in depths:
        if not np.isfinite(d):
            out.append(NEAR)
        elif higher_is_closer:
            out.append(NEAR if d > threshold else FAR)
        else:
            out.append(NEAR if d < threshold else FAR)
    return out


class KeepAllRowFilter:
    """No filtering: every detection counts as near-row."""

    def apply(self, image: Image.Image, detections: list[Detection]) -> None:
        for d in detections:
            d.row = NEAR


class DepthRowFilter:
    def __init__(
        self,
        estimator: Callable[[Image.Image], np.ndarray],
        min_separability: float = 0.85,
        higher_is_closer: bool = True,
        box_inner: float = 0.5,
    ):
        self.estimator = estimator
        self.min_separability = min_separability
        self.higher_is_closer = higher_is_closer
        self.box_inner = box_inner

    def apply(self, image: Image.Image, detections: list[Detection]) -> None:
        if not detections:
            return
        depth = self.estimator(image)
        rgb = np.asarray(image)
        depths = [box_depth(depth, d.box, self.box_inner) for d in detections]
        for d, z in zip(detections, depths, strict=True):
            d.features["depth"] = z
            d.features["netting"] = netting_score(rgb, d.box)
            d.features["area_frac"] = d.box.area / (image.width * image.height)
        rows = split_near_far(depths, self.min_separability, self.higher_is_closer)
        for d, row in zip(detections, rows, strict=True):
            d.row = row
