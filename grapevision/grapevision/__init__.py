"""grapevision: grape bunch detection and yield counting.

Pipeline stages, each a subpackage:
    data       catalog of field captures, leakage-free splits, annotation import/export
    detect     YOLO training and SAHI tiled inference (stage 1: find every bunch)
    rowfilter  near-row vs. background-row decision (stage 2)
    quality    "large and full" judgement (stage 3)
    count      runs the stages and aggregates counts per row/section
    eval       count and box metrics, threshold tuning on the val split
    viz        drawing results
"""

from grapevision.types import Box, Detection

__all__ = ["Box", "Detection"]
__version__ = "0.1.0"
