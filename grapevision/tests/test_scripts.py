"""CLI smoke tests: run the scripts as a user would, in a subprocess, against a fake capture."""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _config(tmp_path: Path, capture: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"paths:\n  captures: ['{capture.as_posix()}']\n  catalog_csv: '{(tmp_path / 'catalog.csv').as_posix()}'\n"
    )
    return cfg


def test_build_catalog_cli(capture, tmp_path):
    proc = _run("build_catalog.py", "--config", str(_config(tmp_path, capture)))
    assert proc.returncode == 0, proc.stderr
    assert "8 images" in proc.stdout
    assert (tmp_path / "catalog.csv").exists()


def test_export_dataset_cli(capture, tmp_path):
    rel_paths = [p.relative_to(capture).as_posix() for p in sorted(capture.rglob("*.jpg"))]
    coco = {
        "images": [{"id": i, "file_name": rel, "width": 80, "height": 60} for i, rel in enumerate(rel_paths)],
        "annotations": [
            {"id": i, "image_id": i, "bbox": [5, 5, 20, 20], "attributes": {"row": "near", "quality": "full"}}
            for i in range(len(rel_paths))
        ],
    }
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    out = tmp_path / "ds"
    proc = _run(
        "export_dataset.py",
        "--coco",
        str(tmp_path / "ann.json"),
        "--capture",
        str(capture),
        "--out",
        str(out),
        "--config",
        str(_config(tmp_path, capture)),
    )
    assert proc.returncode == 0, proc.stderr
    exported = sorted(p.name for p in out.glob("detector/*/images/*.jpg"))
    assert len(exported) == 8
    assert (out / "detector" / "data.yaml").exists()
    assert len(list(out.glob("quality/*/full/*.jpg"))) == 8


def test_export_dataset_cli_reports_missing_files(capture, tmp_path):
    coco = {"images": [{"id": 1, "file_name": "row9/Section1/sequence1.jpg", "width": 1, "height": 1}]}
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    proc = _run(
        "export_dataset.py",
        "--coco",
        str(tmp_path / "ann.json"),
        "--capture",
        str(capture),
        "--out",
        str(tmp_path / "ds"),
    )
    assert proc.returncode != 0
    assert "not found" in proc.stderr
