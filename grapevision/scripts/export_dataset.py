"""Turn a CVAT COCO export into a split YOLO detector dataset plus quality-classifier crops.

    python scripts/export_dataset.py --coco labels.json --capture D:/download/West_Zone2_09_11_2026 \
        --out datasets/west_zone2 [--near-only]

Splits follow the config's split.level, so a whole section lands in train, val or test.
"""

from __future__ import annotations

import argparse

from grapevision.config import load_config, resolve_path
from grapevision.data import export_yolo_split, group_split, load_coco, scan_capture
from grapevision.data.annotations import include_all, include_near
from grapevision.detect import write_data_yaml
from grapevision.quality import export_quality_crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--capture", required=True, help="capture root the COCO file_names are relative to")
    parser.add_argument("--out", required=True)
    parser.add_argument("--near-only", action="store_true", help="detector labels only near-row bunches")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    ann_cfg = cfg["annotations"]
    labels = load_coco(args.coco, ann_cfg["row_attribute"], ann_cfg["quality_attribute"])
    by_rel = {lab.file_name: lab for lab in labels}

    records = [r for r in scan_capture(args.capture) if r.rel_path in by_rel]
    missing = set(by_rel) - {r.rel_path for r in records}
    if missing:
        raise SystemExit(f"{len(missing)} labeled files not found under {args.capture}, e.g. {sorted(missing)[:3]}")

    split_cfg = cfg["split"]
    splits = group_split(records, split_cfg["level"], split_cfg["ratios"], seed=split_cfg["seed"])
    out = resolve_path(args.out)
    include = include_near if args.near_only else include_all
    for name, recs in splits.items():
        subset = [by_rel[r.rel_path] for r in recs]
        n = export_yolo_split(subset, args.capture, out / "detector", name, include)
        crops = export_quality_crops(subset, args.capture, out / "quality" / name)
        print(f"{name:5s} {n:3d} images, quality crops {dict(crops)}")
    print(f"data.yaml -> {write_data_yaml(out / 'detector')}")


if __name__ == "__main__":
    main()
