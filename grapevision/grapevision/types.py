"""Core value types and label vocabulary shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass, field

# Row position of a bunch relative to the camera: the row being surveyed, or one seen behind it.
NEAR = "near"
FAR = "far"
ROW_VALUES = (NEAR, FAR)

# Bunch quality. Only FULL bunches count toward yield. OCCLUDED means "can't judge" (e.g. half hidden
# behind a leaf) and must not be lumped in with SPARSE, or the quality model learns occlusion = sparse.
FULL = "full"
SPARSE = "sparse"
DAMAGED = "damaged"
OCCLUDED = "occluded"
QUALITY_VALUES = (FULL, SPARSE, DAMAGED, OCCLUDED)


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: float, height: float) -> Box:
        return Box(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

    def to_yolo(self, img_w: float, img_h: float) -> tuple[float, float, float, float]:
        """Normalized (cx, cy, w, h)."""
        return (
            (self.x1 + self.x2) / 2 / img_w,
            (self.y1 + self.y2) / 2 / img_h,
            self.width / img_w,
            self.height / img_h,
        )

    @classmethod
    def from_yolo(cls, cx: float, cy: float, w: float, h: float, img_w: float, img_h: float) -> Box:
        return cls((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Box:
        """COCO-style (top-left x, top-left y, width, height)."""
        return cls(x, y, x + w, y + h)


@dataclass
class Detection:
    """One predicted bunch. Later stages fill in row/quality; `features` carries per-stage diagnostics."""

    box: Box
    score: float
    row: str | None = None
    quality: str | None = None
    fullness: float | None = None
    features: dict[str, float] = field(default_factory=dict)

    @property
    def near(self) -> bool:
        # No row filter applied means every detection is treated as near-row.
        return self.row != FAR

    @property
    def countable(self) -> bool:
        return self.near and self.quality == FULL
