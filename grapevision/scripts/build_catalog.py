"""Index every capture folder in the config and preview the group split.

python scripts/build_catalog.py --config configs/west_zone2.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter

from grapevision.config import load_config, resolve_path
from grapevision.data import group_split, scan_capture, write_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    records = []
    for root in cfg["paths"]["captures"]:
        found = scan_capture(resolve_path(root))
        media = Counter(r.media for r in found)
        orient = Counter(r.orientation for r in found if r.media == "image")
        print(f"{root}: {media['image']} images ({dict(orient)}), {media['video']} videos")
        records.extend(found)

    out = resolve_path(cfg["paths"]["catalog_csv"])
    write_catalog(records, out)
    print(f"catalog -> {out}")

    images = [r for r in records if r.media == "image"]
    level = cfg["split"]["level"]
    splits = group_split(images, level, cfg["split"]["ratios"], seed=cfg["split"]["seed"])
    print(f"\nsplit preview (level={level}):")
    for name, recs in splits.items():
        groups = sorted({r.group_key(level) for r in recs})
        print(f"  {name:5s} {len(recs):3d} images  {groups}")


if __name__ == "__main__":
    main()
