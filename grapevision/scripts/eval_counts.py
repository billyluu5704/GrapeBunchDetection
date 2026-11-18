"""Held-out count evaluation of a detector, with no row filter or quality stage.

python scripts/eval_counts.py --weights runs/detector/weights/best.pt --dataset-dir ../GrapevineBunchDetection

Runs SAHI inference (or, with --no-sahi, one full-image pass) once per image on the val and test splits,
picks the confidence threshold that minimizes count MAE on val, then reports test count error (with a
bootstrap CI) and box precision/recall at that fixed threshold. Ground truth is the YOLO label files.
Use a labeled split of your *target* view (drone/farm frames): scores on web-style data do not predict
field accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grapevision.config import load_config, resolve_path
from grapevision.detect import FullImageDetector, SahiDetector
from grapevision.eval import bootstrap_ci, count_metrics, detection_metrics, tune_threshold
from grapevision.io import load_image
from grapevision.types import Box, Detection

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            boxes.append(Box((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def infer_split(
    detector: SahiDetector | FullImageDetector, split_dir: Path, limit: int | None
) -> list[tuple[list[Box], list[Detection]]]:
    images = sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:limit]
    out = []
    for i, path in enumerate(images, 1):
        image = load_image(path)
        gt = read_yolo_boxes(split_dir / "labels" / f"{path.stem}.txt", *image.size)
        out.append((gt, detector(image)))
        if i % 100 == 0:
            print(f"  {split_dir.name}: {i}/{len(images)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--no-sahi", action="store_true", help="one full-image pass instead of SAHI tiling")
    parser.add_argument("--limit", type=int, default=None, help="cap images per split (quick check)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    det = cfg["detector"]
    detector: SahiDetector | FullImageDetector
    if args.no_sahi:
        detector = FullImageDetector(args.weights, det["imgsz"], det["conf_floor"], det["device"])
    else:
        detector = SahiDetector(
            args.weights,
            det["tile_size"],
            det["tile_overlap"],
            det["conf_floor"],
            det["device"],
            det.get("postprocess_type", "NMM"),
            det.get("postprocess_match_threshold", 0.5),
        )
    root = Path(args.dataset_dir)
    val = infer_split(detector, root / args.val_split, args.limit)
    test = infer_split(detector, root / args.test_split, args.limit)

    threshold, _ = tune_threshold([len(g) for g, _ in val], [[d.score for d in dets] for _, dets in val])
    kept = [(g, [d for d in dets if d.score >= threshold]) for g, dets in test]
    true_counts = [len(g) for g, _ in kept]
    pred_counts = [len(dets) for _, dets in kept]
    metrics = count_metrics(true_counts, pred_counts)
    lo, hi = bootstrap_ci([abs(p - t) for p, t in zip(pred_counts, true_counts, strict=True)])
    boxes = detection_metrics((g, [d.box for d in dets]) for g, dets in kept)
    report = {
        "inference": "full_image" if args.no_sahi else "sahi",
        "threshold": threshold,
        "count": metrics,
        "mae_ci95": [lo, hi],
        "boxes": boxes,
    }
    out = resolve_path(cfg["paths"]["output_dir"]) / f"eval_counts_{'full' if args.no_sahi else 'sahi'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
