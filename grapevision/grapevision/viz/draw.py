"""Draw pipeline results: green = counted, yellow = near row but not countable, gray = background row."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from grapevision.count.pipeline import ImageResult

COUNTED = (40, 220, 60)
NOT_COUNTED = (250, 210, 0)
BACKGROUND = (150, 150, 150)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_result(image: Image.Image, result: ImageResult) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, out.width // 400)
    for d in sorted(result.detections, key=lambda d: (d.near, d.countable)):  # counted boxes drawn last, on top
        color = COUNTED if d.countable else NOT_COUNTED if d.near else BACKGROUND
        draw.rectangle([d.box.x1, d.box.y1, d.box.x2, d.box.y2], outline=color, width=width)

    text = f"counted {result.countable}  |  near row {result.near}  |  detected {result.total}"
    font = _font(max(16, out.width // 45))
    pad = 8
    x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad - y1), text, fill=(255, 255, 255), font=font)
    return out
