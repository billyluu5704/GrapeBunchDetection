"""Image loading that every stage shares, so box coordinates always refer to the same pixel grid."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: str | Path) -> Image.Image:
    """RGB image with EXIF orientation applied. Phone photos taken vertically are often stored sideways with
    an orientation tag; skipping this step puts boxes on a rotated grid relative to what annotators saw."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
