"""Clean the AI4Agriculture-style online vineyard photos into a YOLO train/val dataset for a stage-1 fine-tune.

python scripts/build_online_dataset.py --src ../vineyard_online_images --out datasets/online_vineyard

- Pairs each .jpg with its YOLO .txt; skips unreadable/truncated images, images without a label file, and classes.txt.
  Images with an empty label file that load fine are kept as background (no bunches visible).
- Writes upright (EXIF-applied, EXIF-stripped) copies with the long side capped at --max-side, so label orientation can
  never disagree with the pixels. Labels are normalised, so they are unchanged by the resize.
- Validates boxes (clamps to the image, drops degenerate ones) and reports what it changed.
- Val is made of contiguous blocks of the timestamp-sorted files, so near-duplicate shots of the same vine do not
  straddle train and val. Only train/val are written; the farm splits are never touched.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageOps

from grapevision.detect import write_data_yaml


def read_labels(path: Path) -> tuple[list[tuple[float, float, float, float]], int, int]:
    boxes, clamped, dropped = [], 0, 0
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:5])
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            clamped += 1
        if cx1 - cx0 <= 1e-4 or cy1 - cy0 <= 1e-4:
            dropped += 1
            continue
        boxes.append(((cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
    return boxes, clamped, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-side", type=int, default=2560)
    parser.add_argument("--val-images", type=int, default=20, help="approximate val size, in blocks of --block")
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)

    kept, skipped = [], {"no label file": [], "unreadable image": []}
    for jpg in sorted(src.glob("*.jpg")):
        txt = jpg.with_suffix(".txt")
        if not txt.exists():
            skipped["no label file"].append(jpg.name)
            continue
        try:
            with Image.open(jpg) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
        except OSError:
            skipped["unreadable image"].append(jpg.name)
            continue
        kept.append((jpg, txt, im))
    for why, names in skipped.items():
        print(f"skipped ({why}): {len(names)} {names}")

    blocks = [list(range(i, min(i + args.block, len(kept)))) for i in range(0, len(kept), args.block)]
    random.Random(args.seed).shuffle(blocks)
    val_idx: set[int] = set()
    for b in blocks:
        if len(val_idx) >= args.val_images:
            break
        val_idx.update(b)

    totals = {"train": [0, 0], "val": [0, 0]}
    clamped = dropped = empty = 0
    for i, (jpg, txt, im) in enumerate(kept):
        split = "val" if i in val_idx else "train"
        boxes, c, d = read_labels(txt)
        clamped, dropped, empty = clamped + c, dropped + d, empty + (not boxes)
        if max(im.size) > args.max_side:
            im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        im.save(out / split / "images" / jpg.name, quality=95)
        (out / split / "labels" / txt.name).write_text(
            "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)
        )
        totals[split][0] += 1
        totals[split][1] += len(boxes)
    print(f"boxes clamped to image: {clamped}, degenerate dropped: {dropped}, background-only images: {empty}")
    for s, (n, b) in totals.items():
        print(f"{s:5s} {n:3d} images, {b} boxes")
    print(f"data.yaml -> {write_data_yaml(out)}")


if __name__ == "__main__":
    main()
