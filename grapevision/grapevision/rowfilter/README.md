# `grapevision.rowfilter`: stage 2, near row vs. background row

Problem: photographing row 5, the camera also sees bunches on row 6 through gaps in the canopy. They must
not count. This stage labels every detection `near` or `far`.

| File | Contents |
|---|---|
| `filter.py` | Per-box features, the Otsu near/far rule, `DepthRowFilter`, `KeepAllRowFilter` |
| `depth.py` | `DepthAnythingEstimator`: monocular depth model (optional extra `.[depth]`) |

## The approach

```
image ──► depth model ──► depth map (H×W, relative, higher = closer)
                              │
detections ──► median depth inside the center of each box ──► [z1, z2, ..., zn]
                                                                    │
                                    1-D Otsu: best split into two groups + how separable they are
                                                                    │
                     separability ≥ 0.85 ?  ── no ──► everything NEAR (no background row visible)
                              │ yes
                              ▼
                 closer group = NEAR, farther group = FAR
```

Per image, not per dataset. Depth from a single photo is only *relative*: values can be compared within
one image but not across images. The question being asked is "are there two distinct depth layers among
this photo's bunches?", which relative depth answers well. Rows are 2.5–3 m apart, while the camera is
about 1 m from the canopy, so the two layers are usually far apart.

This rule is the **baseline**. Once photos have `row` labels, the plan is a small classifier on
`Detection.features` (depth, netting, size), with this rule as the number to beat.

---

## `filter.py`

### `_inner_region(arr, box, inner)` (internal)

Returns the sub-array covering the **central `inner` fraction** of the box (e.g. `inner=0.5` → the middle
50% of its width and height), after clipping the box to the array. The half-size is at least 0.5 px, so
even a degenerate box yields a 1-pixel region. That means a box lying past the image edge reads the
nearest edge pixel. Only an empty array gives an empty region.

### `box_depth(depth, box, inner=0.5) -> float`

**Median** of the depth map over the box's inner region; `nan` if the region is empty.

- *Center only:* a bunch box's border is mostly leaves and gaps, and gaps show whatever is behind them.
  The center is mostly berries.
- *Median, not mean:* robust to a stray leaf or a gap inside the center.

### `netting_score(rgb, box, margin=25, inner=0.8) -> float`

Fraction of pixels in the central 80% of the box where **blue beats both red and green by more than
`margin`** (0–255 scale). Arithmetic is done in `int16`, because `uint8` would wrap around on subtraction.

Some background rows are covered in blue bird netting, so bunches seen through it are tinted blue. It's
recorded as a feature only. It doesn't decide anything yet, because most rows have no netting and the
older photos have none.

### `otsu_split(values) -> (threshold, separability)`

Exhaustive 1-D Otsu. Non-finite values are dropped and the rest sorted. For every split point `i`:

```
between(i) = w0 · w1 · (mean(v[:i]) − mean(v[i:]))²        w0 = i/n, w1 = 1 − w0
```

The best `i` maximizes `between`. The function returns:

- `threshold` = midpoint between the two values on either side of the best split
- `separability` = `between / var(v)`, the share of total variance explained by the split (Otsu's η)

| Data | separability |
|---|---|
| two tight, well-apart clusters | → 1.0 |
| evenly spread values | 0.75 (for a continuous uniform distribution) |
| < 2 values or all equal | 0.0 (threshold `nan`) |

That 0.75 is why the default cutoff is **0.85, not 0.5**. A single row photographed at an angle has depths
spread continuously from near to far. Otsu would happily split it in half, and the cutoff stops that from
being reported as a background row.

O(n²) in the number of boxes per image. Fine for tens of boxes.

### `split_near_far(depths, min_separability=0.85, higher_is_closer=True) -> list[str]`

Applies `otsu_split`. If separability is below the cutoff, every box is `NEAR`. Otherwise:

- `higher_is_closer=True`: depth `> threshold` → `NEAR`, else `FAR`
- `higher_is_closer=False`: the reverse (for metric-depth sources, e.g. a depth camera in meters)

A `nan` depth is always `NEAR`. The filter only removes a bunch when it has evidence.

### `KeepAllRowFilter.apply(image, detections)`

Sets every `row = NEAR`. Used with `row_filter.method: none`, and as the pipeline default.

### `DepthRowFilter(estimator, min_separability=0.85, higher_is_closer=True, box_inner=0.5)`

`estimator` is any callable `PIL.Image → np.ndarray (H×W)`, normally `DepthAnythingEstimator`. In tests
it's a lambda returning a synthetic map, so this class needs no GPU to test.

`apply(image, detections)`:

1. If there are no detections, it returns immediately, without running the (expensive) depth model.
2. Runs the estimator once per image.
3. For each detection, records `features["depth"]`, `features["netting"]` and `features["area_frac"]`
   (box area / image area).
4. Calls `split_near_far` on the depths and sets `d.row`.

---

## `depth.py`

### `DepthAnythingEstimator(model="depth-anything/Depth-Anything-V2-Small-hf", device=None)`

Builds a Hugging Face `pipeline("depth-estimation")` once (downloads the weights on first use, ~100 MB for
Small). `device=None` lets transformers choose (CPU unless told otherwise). Pass `0` for the first GPU.

`__call__(image) -> np.ndarray` (float32, H×W, the same size as `image`):

- Uses the pipeline's `"depth"` output, a PIL image already resized to the input size and scaled to 0–255.
  8-bit precision is plenty for *ordering* boxes. If the size ever differs, it's resized bilinearly.
- Depth Anything predicts **relative inverse depth** (disparity-like), so **higher = closer**. That's
  `higher_is_closer: true` in the config.

**Before trusting it:** run it on one West_Zone2 photo and look at the map. The near canopy should be
brighter than the netted row behind it. If it's the other way round, set `higher_is_closer: false`.

Swapping models: any other `depth-estimation` checkpoint on the Hub (e.g. the Base/Large Depth Anything V2
variants) works by changing `row_filter.depth_model`. A depth camera would replace this class entirely
with one returning its depth frame.
