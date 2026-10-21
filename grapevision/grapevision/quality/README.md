# `grapevision.quality`: stage 3, "large and pretty full"

The farm only counts bunches that are **large and pretty full**. Small bunches with a few berries, and
large ones with most berries missing or destroyed, are left out of the yield estimate. This stage marks
each **near-row** detection `FULL` (counts) or not.

| File | Contents |
|---|---|
| `crops.py` | Cutting bunch crops out of photos, both to build the training set and at inference |
| `classifier.py` | The quality stages: two baselines now, the CNN later |

## Design: detect first, judge the crop second

Quality is judged on a **crop of one bunch**, not by giving the detector extra classes:

- Labeling crops is fast, and a small CNN learns from a few hundred crops far better than a detector learns
  a subtle class boundary from a few dozen photos.
- "Full" is a judgment call. A model that outputs a **fullness score** (stored in `Detection.fullness`)
  lets the farm move the cutoff without retraining.
- The detector stays a simple one-class model, so improving it and improving quality judgment don't
  interfere with each other.

Only near-row detections reach this stage (`count.CountingPipeline` filters on `d.near`). Far-row bunches
keep `quality = None`.

---

## `crops.py`

### `crop_box(image, box, pad=0.1) -> PIL.Image`

Expands the box by `pad × width` on the left and right and `pad × height` on the top and bottom (10% each
side by default), clips it to the image, and crops. Left/top edges are truncated with `int()` and
right/bottom are rounded, so a crop is never empty for a box with positive size.

The padding gives the classifier a little context: the stem, and whether the outline continues past the
box.

### `export_quality_crops(labels, image_root, out_dir, pad=0.1) -> Counter[str]`

Builds the classifier's training set from labeled images:

- Includes a bunch only if `row != far` **and** `quality` is set. Far-row bunches aren't judged, and
  unjudged bunches have no target.
- Opens each image only once (via `io.load_image`, so EXIF-rotated like everything else), and skips
  images with nothing to export.
- Writes `out_dir/<quality>/<label_stem>__<box index>.jpg` (JPEG quality 95).
- Returns per-class counts, e.g. `{"full": 212, "sparse": 87, "damaged": 14, "occluded": 40}`.

The folder-per-class layout is exactly what `torchvision.datasets.ImageFolder` expects. `scripts/
export_dataset.py` calls this once per split, so the crops inherit the section-level train/val/test split.

---

## `classifier.py`

Every quality stage has the `apply(image, detections) -> None` interface and sets `d.quality` on each
detection it's given.

### `AcceptAllQuality`

Sets `quality = FULL` on everything, which means "count every near-row bunch". It's the pipeline default
and matches what the old notebook counted.

### `SizeBaselineQuality(min_area_frac=0.002)`

`area_frac = box area / image area`, stored in `features["area_frac"]`. `FULL` if `area_frac ≥
min_area_frac`, otherwise `SPARSE`.

Deliberately crude. Apparent size depends on camera distance as much as on the bunch, and it can't see
missing berries at all. It exists so the CNN has a simple number to beat. If the CNN can't beat box size,
it isn't learning fullness.

### Planned: `CnnQuality`

A small pretrained CNN (EfficientNet-B0 or ConvNeXt-Tiny) fine-tuned on the exported crops. Two options
for the output:

- **Classification** over `full / sparse / damaged / occluded`. `fullness = P(full)`, and the cutoff is
  applied to that.
- **Regression** of a fullness score, if labels become graded (e.g. 1–5), which is better suited to a
  fuzzy boundary.

`OCCLUDED` predictions need a policy decision (count or not). Make it a config option, not a hard-coded
rule.
