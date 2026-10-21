"""Count and box-level metrics, and confidence-threshold tuning.

Protocol: run inference once per image at a low confidence floor and keep the scores. Pick the threshold
on the *val* split with `tune_threshold`, then report test metrics at that fixed threshold. Tuning and
reporting on the same split (as the old notebook did) gives an optimistic number.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from grapevision.types import Box


def count_metrics(true_counts: Sequence[int], pred_counts: Sequence[int]) -> dict[str, float]:
    true = np.asarray(true_counts, dtype=float)
    pred = np.asarray(pred_counts, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"length mismatch: {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("no images to score")
    err = pred - true
    return {
        "n": int(true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "exact_pct": float(np.mean(err == 0) * 100),
        "within1_pct": float(np.mean(np.abs(err) <= 1) * 100),
        "total_true": float(true.sum()),
        "total_pred": float(pred.sum()),
    }


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI over per-image values, e.g. absolute count errors."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def match_boxes(gt: Sequence[Box], pred: Sequence[Box], iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy highest-IoU-first one-to-one matching. Returns (tp, fp, fn)."""
    pairs = sorted(
        ((g.iou(p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
        key=lambda t: -t[0],
    )
    used_gt, used_pred = set(), set()
    for iou, gi, pi in pairs:
        if iou < iou_thr:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
    tp = len(used_gt)
    return tp, len(pred) - tp, len(gt) - tp


def detection_metrics(images: Iterable[tuple[Sequence[Box], Sequence[Box]]], iou_thr: float = 0.5) -> dict[str, float]:
    """Micro-averaged precision/recall/F1 over (gt_boxes, pred_boxes) pairs, one pair per image."""
    tp = fp = fn = 0
    for gt, pred in images:
        t, f, n = match_boxes(gt, pred, iou_thr)
        tp, fp, fn = tp + t, fp + f, fn + n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def sweep_thresholds(
    true_counts: Sequence[int],
    scores_per_image: Sequence[Sequence[float]],
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    """Count metrics at each threshold, computed offline from stored detection scores."""
    scores = [np.asarray(s, dtype=float) for s in scores_per_image]
    table = []
    for t in thresholds:
        row = count_metrics(true_counts, [int((s >= t).sum()) for s in scores])
        row["threshold"] = float(t)
        table.append(row)
    return table


def tune_threshold(
    true_counts: Sequence[int],
    scores_per_image: Sequence[Sequence[float]],
    thresholds: Sequence[float] | None = None,
    objective: str = "mae",
) -> tuple[float, list[dict[str, float]]]:
    """Threshold minimizing `objective` (a key of count_metrics). Call on the val split only."""
    grid = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()
    table = sweep_thresholds(true_counts, scores_per_image, grid)
    best = min(table, key=lambda r: (r[objective], -r["threshold"]))  # ties -> higher, more conservative threshold
    return best["threshold"], table
