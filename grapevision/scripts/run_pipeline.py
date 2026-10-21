"""Run detect -> row filter -> quality -> count over the cataloged images.

    python scripts/run_pipeline.py --weights runs/detector/weights/best.pt [--level row]

Writes annotated images, a per-image CSV and a per-group CSV under paths.output_dir.
"""

from __future__ import annotations

import argparse
import csv

from grapevision.config import load_config, resolve_path
from grapevision.count import aggregate, build_pipeline
from grapevision.data import read_catalog
from grapevision.data.annotations import label_stem
from grapevision.io import load_image
from grapevision.viz import draw_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--level", default="row", choices=["row", "section", "image"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides={"detector": {"weights": args.weights}})

    records = [r for r in read_catalog(resolve_path(cfg["paths"]["catalog_csv"])) if r.media == "image"]
    pipeline = build_pipeline(cfg)
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    results = []
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["capture", "rel_path", "row", "section", "detected", "near", "countable"])
        for r in records:
            res = pipeline.run(r.path)
            results.append(res)
            writer.writerow([r.capture, r.rel_path, r.row, r.section, res.total, res.near, res.countable])
            draw_result(load_image(r.path), res).save(
                out_dir / "annotated" / f"{r.capture}__{label_stem(r.rel_path)}.jpg"
            )
            print(f"{r.rel_path}: counted {res.countable} / near {res.near} / detected {res.total}")

    totals = aggregate(results, records, args.level)
    with open(out_dir / f"per_{args.level}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([args.level, "images", "detected", "near", "countable"])
        for key, t in sorted(totals.items()):
            writer.writerow([key, t["images"], t["total"], t["near"], t["countable"]])
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
