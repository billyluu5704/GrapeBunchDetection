from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    """Fake capture: 2 rows x 2 sections x 2 sequences, one portrait shot, plus a video and a stray file."""
    root = tmp_path / "West_Zone2_09_11_2026"
    for row in (5, 7):
        for section in (1, 2):
            folder = root / f"row{row}" / f"Section{section}"
            folder.mkdir(parents=True)
            for seq in (1, 2):
                size = (60, 80) if (row, section, seq) == (7, 2, 2) else (80, 60)
                Image.new("RGB", size, (0, 120, 0)).save(folder / f"sequence{seq}.jpg")
    (root / "video_sample.MOV").write_bytes(b"\x00")
    (root / "notes.txt").write_text("ignored")
    return root
