"""Where do a detector's false positives land? Classifies each FP on the held-out split by what it overlaps.

python scripts/fp_diagnostic.py --coco datasets/west_zone2_coco_stripped.json --dataset-dir datasets/west_zone2_visible/detector \
    --model runs/farm_ft/weights/best.pt 0.70 --model runs/farm_ft_visible/weights/best.pt 0.55 --device auto

Ground truth for matching = the YOLO labels in --dataset-dir (full/sparse bunches only). Each unmatched detection
(IoU < 0.5 with every free GT box) is then labelled by its best overlap (IoU >= --loose, default 0.3) with the
CVAT boxes outside the dataset's labels (occluded, damaged or far-row, when the dataset filter dropped them), or another
labelled GT box (duplicate / bad localization). On a dataset that keeps occluded boxes, those are plain GT. Anything
left overlaps no labelled bunch, i.e. background (leaves, netting, unlabeled bunches).
Missed GT boxes (FN) are split into: scored below the threshold, boxed badly, or not found at all.
--overlays draws green = TP, red = background FP, orange = FP on a labelled bunch, cyan = missed bunch. --coco file_names must be
relative to the capture root (strip any leading folder), as for export_dataset.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eval_counts import IMAGE_SUFFIXES, read_yolo_boxes
from grapevision.config import load_config
from grapevision.data import label_stem, load_coco
from grapevision.detect import SahiDetector
from grapevision.gpu import use_best_gpu
from grapevision.io import load_image
from PIL import ImageDraw


def greedy_match(gt, pred, iou_thr=0.5):
    pairs = sorted(((g.iou(p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)), key=lambda t: -t[0])
    used_g, used_p = set(), set()
    for iou, gi, pi in pairs:
        if iou < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
    return used_g, used_p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", nargs=2, action="append", metavar=("WEIGHTS", "THRESHOLD"), required=True)
    parser.add_argument("--loose", type=float, default=0.3)
    parser.add_argument("--overlays", default=None, help="folder to save per-image error overlays")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    det = dict(cfg["detector"])
    if args.device == "auto":
        det["device"] = "cuda:" + use_best_gpu()
    elif args.device:
        det["device"] = args.device

    cvat = {label_stem(lab.file_name): lab for lab in load_coco(args.coco)}
    split = Path(args.dataset_dir) / args.split
    images = sorted(p for p in (split / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    for weights, thr in args.model:
        thr = float(thr)
        detector = SahiDetector(weights, det["tile_size"], det["tile_overlap"], det["conf_floor"], det["device"],
                                det.get("postprocess_type", "NMM"), det.get("postprocess_match_threshold", 0.5))
        kinds, fn_why, fn_quality, tp_quality, tp, total = Counter(), Counter(), Counter(), Counter(), 0, 0
        fn_sizes, tp_sizes, below_scores = [], [], []
        for path in images:
            image = load_image(path)
            gt = read_yolo_boxes(split / "labels" / f"{path.stem}.txt", *image.size)
            dets = detector(image)
            preds = [d.box for d in dets if d.score >= thr]
            matched_g, matched = greedy_match(gt, preds)
            tp += len(matched)
            total += len(preds)
            others = [("far-row" if b.row == "far" else b.quality or "unjudged", b.box) for b in cvat[path.stem].bunches]
            draw = []  # (box, colour)
            for pi, p in enumerate(preds):
                if pi in matched:
                    draw.append((p, "lime"))
                    continue
                best = max(((p.iou(g), "labelled GT (duplicate/localization)") for g in gt), default=(0, ""))
                cand = [(p.iou(b), f"{q} GT") for q, b in others if q in ("occluded", "damaged", "far-row")]
                top = max([best] + cand, key=lambda t: t[0])
                kind = top[1] if top[0] >= args.loose else "background (no labelled bunch)"
                kinds[kind] += 1
                draw.append((p, "red" if kind.startswith("background") else "orange"))
            for gi, g in enumerate(gt):
                quality = max(others, key=lambda qb: g.iou(qb[1]), default=("?", None))[0]
                size = (g.width * g.height) ** 0.5
                if gi in matched_g:
                    tp_quality[quality] += 1
                    tp_sizes.append(size)
                    continue
                fn_quality[quality] += 1
                draw.append((g, "cyan"))
                fn_sizes.append(size)
                below = [d.score for d in dets if d.score < thr and g.iou(d.box) >= 0.5]
                if below:
                    fn_why["found, but score below threshold"] += 1
                    below_scores.append(max(below))
                elif any(g.iou(p) >= 0.1 for p in preds):
                    fn_why["box placed badly (IoU 0.1-0.5)"] += 1
                else:
                    fn_why["missed entirely"] += 1
            if args.overlays:
                out = Path(args.overlays) / Path(weights).parent.parent.name
                out.mkdir(parents=True, exist_ok=True)
                canvas = image.copy()
                pen = ImageDraw.Draw(canvas)
                for b, colour in draw:
                    pen.rectangle([b.x1, b.y1, b.x2, b.y2], outline=colour, width=3)
                canvas.save(out / f"{path.stem}.jpg", quality=90)
        fp, fn = total - tp, len(fn_sizes)
        print(f"\n{weights}  thr={thr}  detections={total}  TP={tp}  FP={fp}  FN={fn}")
        print(" false positives land on:")
        for k, v in kinds.most_common():
            print(f"  {v:3d}  {100 * v / max(fp, 1):4.0f}%  {k}")
        print(" missed bunches (FN), why:")
        for k, v in fn_why.most_common():
            print(f"  {v:3d}  {100 * v / max(fn, 1):4.0f}%  {k}")
        if below_scores:
            print(f"      below-threshold scores: {sorted(round(s, 2) for s in below_scores)}")
        print(" recall by quality tag (found / labelled):")
        for q in sorted(set(tp_quality) | set(fn_quality)):
            n = tp_quality[q] + fn_quality[q]
            print(f"  {q:10s} {tp_quality[q]:3d} / {n:3d}  = {tp_quality[q] / n:.2f}")
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0  # noqa: E731
        print(f" median box size (sqrt area, px): found {med(tp_sizes):.0f}, missed {med(fn_sizes):.0f}")


if __name__ == "__main__":
    main()
