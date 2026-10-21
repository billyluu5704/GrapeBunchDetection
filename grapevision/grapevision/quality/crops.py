"""Bunch crops: the training data for the quality classifier, cut from labeled images."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from grapevision.data.annotations import ImageLabels, label_stem
from grapevision.io import load_image
from grapevision.types import FAR, Box


def crop_box(image: Image.Image, box: Box, pad: float = 0.1) -> Image.Image:
    """Crop with `pad` (fraction of box size) of context on each side, clipped to the image."""
    px, py = box.width * pad, box.height * pad
    b = Box(box.x1 - px, box.y1 - py, box.x2 + px, box.y2 + py).clip(image.width, image.height)
    return image.crop((int(b.x1), int(b.y1), round(b.x2), round(b.y2)))


def export_quality_crops(
    labels: Iterable[ImageLabels],
    image_root: str | Path,
    out_dir: str | Path,
    pad: float = 0.1,
) -> Counter[str]:
    """Write <out_dir>/<quality>/<image stem>__<i>.jpg for every near-row bunch with a quality label.
    The folder-per-class layout loads directly with torchvision's ImageFolder. Returns counts per class."""
    image_root, out_dir = Path(image_root), Path(out_dir)
    counts: Counter[str] = Counter()
    for image in labels:
        judged = [(i, b.box, b.quality) for i, b in enumerate(image.bunches) if b.row != FAR and b.quality is not None]
        if not judged:
            continue
        pil = load_image(image_root / image.file_name)
        for i, box, quality in judged:
            dst = out_dir / quality / f"{label_stem(image.file_name)}__{i}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop_box(pil, box, pad).save(dst, quality=95)
            counts[quality] += 1
    return counts
