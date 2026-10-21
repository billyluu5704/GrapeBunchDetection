# `grapevision.eval`: metrics and threshold tuning

Everything lives in `metrics.py`. Pure numpy, no model code: it scores predictions you already have.

## The evaluation protocol

1. Run the detector **once** per val and test image at a low `conf_floor` (0.05) and keep every score.
2. `tune_threshold` on the **val** split → the confidence threshold.
3. Report test metrics **at that fixed threshold**. Never tune on test.

The old notebook tuned the threshold on its 9-image test set and reported the best result on that same set.
The reported count error (MAE 0.67) was the best of 16 tries on 9 photos, so it was optimistic. Step 2
done on val is the fix.

---

## Count metrics

### `count_metrics(true_counts, pred_counts) -> dict`

Per image, `err = pred − true`. Returns:

| Key | Formula | Read it as |
|---|---|---|
| `n` | number of images | |
| `mae` | mean \|err\| | average bunches off per photo (headline number) |
| `rmse` | √mean(err²) | like MAE but punishes big misses more |
| `bias` | mean err | **sign matters**: + overcounts, − undercounts. A yield estimate cares about this most, because per-photo errors cancel out over a row but bias doesn't. |
| `exact_pct` | % with err = 0 | |
| `within1_pct` | % with \|err\| ≤ 1 | |
| `total_true`, `total_pred` | sums | compare row totals directly |

Mismatched lengths or empty input → `ValueError`.

### `bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=42) -> (lo, hi)`

Percentile bootstrap: resample the per-image values with replacement `n_boot` times, compute `stat`
each time, and take the 2.5th and 97.5th percentiles. Use it on per-image absolute errors to put an
interval on MAE. With a 5-image test split the interval will be wide, which is an honest reflection of how
little 5 images can tell you.

---

## Box metrics

### `match_boxes(gt, pred, iou_thr=0.5) -> (tp, fp, fn)`

Greedy one-to-one matching:

1. Compute IoU for every (ground truth, prediction) pair. O(G·P), fine for tens of boxes.
2. Sort the pairs by IoU, highest first.
3. Walk down the list. Stop at the first pair below `iou_thr`. Skip pairs where either box is already
   matched; otherwise match them.
4. `tp` = matches, `fp` = unmatched predictions, `fn` = unmatched ground-truth boxes.

Greedy (not Hungarian) matching is the usual choice for detection evaluation and gives the same result
except in rare tie-heavy cases.

### `detection_metrics(images, iou_thr=0.5) -> dict`

`images` is an iterable of `(gt_boxes, pred_boxes)` pairs, one per image. Sums `tp/fp/fn` over all images
first, then computes precision, recall and F1 (**micro-averaged**). This weights images with many bunches
more, which matches "what fraction of all bunches did we find". Any 0/0 → 0.0.

Why both kinds of metric: count error can look fine while boxes are wrong (a miss and a false detection
cancel out). In the old notebook, tuning the threshold brought farm-photo count MAE down from 8.6 to 3.4,
yet box recall stayed around 0.3. Most bunches were still being missed; extra false boxes had pushed the
count up.

---

## Threshold tuning

### `sweep_thresholds(true_counts, scores_per_image, thresholds) -> list[dict]`

`scores_per_image[i]` holds the confidence scores of every detection in image `i` (from the low-floor
run). At threshold `t`, predicted count = `(scores >= t).sum()`. Returns `count_metrics` for each `t`, plus
a `threshold` key. No inference is involved; sweeping 20 thresholds costs microseconds.

### `tune_threshold(true_counts, scores_per_image, thresholds=None, objective="mae") -> (best, table)`

- Default grid: 0.05, 0.10, ..., 0.95.
- Picks the row minimizing `objective` (any `count_metrics` key where lower is better: `mae`, `rmse`). Ties
  go to the **higher** threshold, which keeps fewer low-confidence boxes and is more conservative.
- Returns the full table too: plot `mae` against `threshold` to see whether the minimum is a stable valley
  or a lucky spike.

**Counting `countable` instead of all bunches:** pass `true_counts` = labeled countable bunches per image,
and `scores_per_image` = scores of the detections the row and quality stages kept. The threshold is then
tuned for the number the farm actually uses.
