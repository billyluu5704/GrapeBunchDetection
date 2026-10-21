"""YOLO detector training (single class `bunch`).

On Windows, call `train_detector` from under `if __name__ == "__main__":` (scripts/train_detector.py does).
DataLoader workers start with `spawn` there, which re-imports the calling module. The old notebook forced
`fork` instead, which Windows doesn't support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_data_yaml(
    dataset_dir: str | Path, out_path: str | Path | None = None, names: tuple[str, ...] = ("bunch",)
) -> Path:
    """data.yaml for a directory laid out by data.export_yolo_split: <dir>/{train,val,test}/{images,labels}."""
    dataset_dir = Path(dataset_dir).resolve()
    out_path = Path(out_path) if out_path else dataset_dir / "data.yaml"
    splits = {s: f"{s}/images" for s in ("train", "val", "test") if (dataset_dir / s / "images").is_dir()}
    lines = [f"path: {dataset_dir.as_posix()}"] + [f"{k}: {v}" for k, v in splits.items()]
    lines += [f"nc: {len(names)}", f"names: {list(names)}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def train_detector(
    data_yaml: str | Path,
    weights: str = "yolo11m.pt",
    imgsz: int = 640,
    epochs: int = 150,
    device: int | str = 0,
    workers: int = 4,
    project: str | Path = "runs",
    name: str = "detector",
    seed: int = 42,
    **train_args: Any,
) -> Any:
    """Thin wrapper over Ultralytics' trainer. `train_args` pass straight through (lr0, patience, mosaic...).
    Note: Ultralytics' `copy_paste` needs segmentation masks; with box-only labels it does nothing."""
    from ultralytics import YOLO

    model = YOLO(weights)
    return model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        seed=seed,
        **train_args,
    )
