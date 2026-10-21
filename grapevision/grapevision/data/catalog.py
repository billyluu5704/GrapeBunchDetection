"""Index field captures laid out as <capture>/rowN/SectionM/sequenceK.<ext>.

A capture folder is named <zone>_<MM>_<DD>_<YYYY>, e.g. West_Zone2_09_11_2026 -> zone West_Zone2,
date 2026-09-11. Folders without row/section structure (e.g. the older LiveImages/) still catalog,
with row/section left empty.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mov", ".mp4"}
_EXIF_ORIENTATION = 0x0112
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}  # EXIF values meaning width and height swap on display

_CAPTURE_RE = re.compile(r"^(?P<zone>.+?)_(?P<m>\d{1,2})_(?P<d>\d{1,2})_(?P<y>\d{4})$")
_ROW_RE = re.compile(r"^row[ _-]?(\d+)$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^section[ _-]?(\d+)$", re.IGNORECASE)
_SEQUENCE_RE = re.compile(r"^sequence[ _-]?(\d+)$", re.IGNORECASE)


@dataclass
class MediaRecord:
    path: str  # absolute path
    rel_path: str  # path relative to the capture root, with forward slashes
    media: str  # "image" | "video"
    capture: str  # capture folder name
    zone: str | None
    date: str | None  # ISO date
    row: int | None
    section: int | None
    sequence: int | None
    width: int | None  # display size, after EXIF orientation
    height: int | None

    @property
    def orientation(self) -> str | None:
        if self.width is None or self.height is None:
            return None
        return "portrait" if self.height > self.width else "landscape"

    def group_key(self, level: str) -> str:
        """ID of the physical unit this record belongs to. Records sharing a key go to the same split, so
        photos of the same vines can't sit on both sides of a train/test boundary."""
        if level == "image" or self.row is None:
            return f"{self.capture}/{self.rel_path}"
        if level == "row":
            return f"{self.capture}/row{self.row}"
        if level == "section":
            return f"{self.capture}/row{self.row}/section{self.section}"
        raise ValueError(f"unknown group level {level!r}; expected image, section or row")


def parse_capture_name(name: str) -> tuple[str | None, str | None]:
    m = _CAPTURE_RE.match(name)
    if not m:
        return None, None
    try:
        iso = date(int(m["y"]), int(m["m"]), int(m["d"])).isoformat()
    except ValueError:
        return m["zone"], None
    return m["zone"], iso


def _display_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(_EXIF_ORIENTATION) in _ROTATED_ORIENTATIONS:
                w, h = h, w
            return w, h
    except OSError:
        return None, None


def _match_int(pattern: re.Pattern[str], text: str) -> int | None:
    m = pattern.match(text)
    return int(m[1]) if m else None


def scan_capture(root: str | Path) -> list[MediaRecord]:
    root = Path(root).resolve()
    zone, iso_date = parse_capture_name(root.name)
    records = []
    for path in sorted(root.rglob("*")):
        ext = path.suffix.lower()
        if not path.is_file() or ext not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        rel = path.relative_to(root)
        row = section = None
        for part in rel.parts[:-1]:
            row = _match_int(_ROW_RE, part) if row is None else row
            section = _match_int(_SECTION_RE, part) if section is None else section
        is_image = ext in IMAGE_EXTS
        width, height = _display_size(path) if is_image else (None, None)
        records.append(
            MediaRecord(
                path=str(path),
                rel_path=rel.as_posix(),
                media="image" if is_image else "video",
                capture=root.name,
                zone=zone,
                date=iso_date,
                row=row,
                section=section,
                sequence=_match_int(_SEQUENCE_RE, path.stem),
                width=width,
                height=height,
            )
        )
    return records


_INT_FIELDS = {"row", "section", "sequence", "width", "height"}


def write_catalog(records: list[MediaRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(MediaRecord)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)


def read_catalog(path: str | Path) -> list[MediaRecord]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        values: dict[str, Any] = {k: (None if v == "" else int(v) if k in _INT_FIELDS else v) for k, v in row.items()}
        out.append(MediaRecord(**values))
    return out
