# `grapevision.count`: wiring the stages, adding up the counts

Runs detect → row filter → quality on one image, then sums the results per row or section.

Everything lives in `pipeline.py`.

---

## `Stage` (Protocol)

```python
class Stage(Protocol):
    def apply(self, image: Image.Image, detections: list[Detection]) -> None: ...
```

The contract for stages 2 and 3: receive the image and the detections, mutate the detections in place,
return nothing. Any class with this method fits, with no inheritance needed (structural typing). mypy checks
it.

## `ImageResult` (dataclass)

`image` (the path as a string) and `detections` (after every stage). Three derived counts:

| Property | Counts | Meaning |
|---|---|---|
| `total` | all detections above `conf` | every bunch the detector found |
| `near` | `d.near` | on the surveyed row |
| `countable` | `d.countable` (near **and** `quality == full`) | **counts toward yield** |

`total − near` = background-row bunches removed; `near − countable` = near-row bunches judged not full.
Keeping all three makes it obvious which stage changed a number.

## `CountingPipeline(detector, row_filter=None, quality=None, conf=0.25)`

- `detector`: any callable `PIL.Image → list[Detection]` (normally `detect.SahiDetector`; a plain function
  in tests).
- `row_filter` defaults to `KeepAllRowFilter`, `quality` to `AcceptAllQuality`. With both defaults, the
  pipeline counts every detection, which is the old behavior.
- `conf` is the **real** confidence threshold. The detector itself runs at a lower `conf_floor` (see
  `detect/README.md`).

`run(image_path) -> ImageResult`:

```python
image = load_image(image_path)  # EXIF-rotated RGB, loaded once
detections = [d for d in detector(image) if d.score >= conf]  # threshold here, not in the detector
row_filter.apply(image, detections)  # sets .row on all
quality.apply(image, [d for d in detections if d.near])  # judges near-row only
return ImageResult(str(image_path), detections)  # far-row detections are kept, with row=far
```

Far-row detections stay in the result, so they can be drawn in gray and audited.

## `build_pipeline(cfg) -> CountingPipeline`

Builds the pipeline from a config dict (`config.load_config()`):

| Config | Choice → class |
|---|---|
| `detector.*` | `SahiDetector(weights, tile_size, tile_overlap, conf_floor, device)`; `detector.conf` → `CountingPipeline.conf` |
| `row_filter.method` | `depth` → `DepthRowFilter(DepthAnythingEstimator(depth_model), min_separability, higher_is_closer, box_inner)`; `none` → `KeepAllRowFilter` |
| `quality.method` | `accept_all` → `AcceptAllQuality`; `size` → `SizeBaselineQuality(min_area_frac)` |

Unknown method names raise `ValueError`. The GPU-backed classes are imported inside the function, so
importing this module stays lightweight. To add a new stage implementation, add a branch here and a value
to `default_config.yaml`.

Note: the depth model is created without a `device`, so it runs wherever transformers places it (CPU by
default). On a GPU machine, pass `device=0` to `DepthAnythingEstimator` for speed.

## `aggregate(results, records, level="row") -> dict[str, dict[str, int]]`

Matches each `ImageResult` to its catalog `MediaRecord` by resolved absolute path. Results are then summed
under `record.group_key(level)`:

```python
{"West_Zone2_09_11_2026/row5": {"images": 11, "total": 312, "near": 240, "countable": 171}, ...}
```

- An image that isn't in the catalog → `KeyError`. Silently dropping it would undercount a row.
- `level` is `row`, `section` or `image`; it's the same key used for splits (`data/README.md`).
- **It assumes photos within a group don't overlap.** If two consecutive shots both include a bunch, it's
  counted twice. That's fine for the current section photos; continuous video needs tracking (a unique ID
  per bunch across frames) instead of summing.
