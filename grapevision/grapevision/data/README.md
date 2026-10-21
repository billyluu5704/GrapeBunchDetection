# `grapevision.data`: catalog, splits, annotations

Everything before the model: finding the photos, deciding which ones are train/val/test, and converting
labels between CVAT and YOLO.

| File | Responsibility |
|---|---|
| `catalog.py` | Walk a capture folder → one `MediaRecord` per photo/video, with row/section/sequence parsed from the path |
| `splits.py` | Train/val/test and k-fold splits that keep whole sections (or rows) together |
| `annotations.py` | Read CVAT's COCO export (boxes + `row`/`quality` attributes); write a YOLO dataset |

---

## `catalog.py`

### Expected layout

```
West_Zone2_09_11_2026/          capture folder:  <zone>_<MM>_<DD>_<YYYY>
├── row5/                       row[ _-]?<n>      (case-insensitive)
│   ├── Section1/               section[ _-]?<n>  (case-insensitive)
│   │   ├── sequence1.jpg       sequence[ _-]?<n> (file stem)
│   │   └── ...
└── video_sample.MOV
```

Any folder works. Missing parts are just `None`: the older flat `LiveImages/` folder catalogs fine, with no
row or section.

### `MediaRecord` (dataclass)

| Field | Example | Notes |
|---|---|---|
| `path` | `D:\download\...\sequence1.jpg` | absolute |
| `rel_path` | `row5/Section1/sequence1.jpg` | relative to the capture root, always `/`. **This is the join key with CVAT's `file_name`.** |
| `media` | `image` / `video` | by extension: `.jpg .jpeg .png` / `.mov .mp4` |
| `capture`, `zone`, `date` | `West_Zone2_09_11_2026`, `West_Zone2`, `2026-09-11` | parsed from the capture folder name |
| `row`, `section`, `sequence` | `5`, `1`, `1` | parsed from folder and file names; `None` if absent |
| `width`, `height` | `1215`, `911` | **display** size: swapped when the EXIF orientation is 5–8 (rotated 90°) |

`orientation` → `"portrait"` if `height > width`, else `"landscape"` (`None` if the size is unknown).

`group_key(level)` → the ID of the physical unit the photo belongs to. `splits.py` keeps records with the
same key together.

| `level` | Key | Used for |
|---|---|---|
| `"image"` | `capture/rel_path` | no grouping (each photo alone) |
| `"section"` | `capture/row5/section1` | default: shots of the same vines stay together |
| `"row"` | `capture/row5` | strictest: train on one row, test on another |

Records without a row (flat folders) always fall back to the image key, since there's nothing to group by.
Unknown levels raise `ValueError`.

### Functions

| Function | Behavior |
|---|---|
| `parse_capture_name(name)` | Regex `^(zone)_(MM)_(DD)_(YYYY)$` (non-greedy zone, so `West_Zone2` keeps its underscore). Returns `(zone, iso_date)`. If the name doesn't match → `(None, None)`; if it matches but isn't a real date (`13_40`) → `(zone, None)`. **Assumes month-first**, which is how these folders are named. |
| `scan_capture(root)` | `rglob("*")` in sorted order (deterministic), skipping non-media files. For each file, scans the parent folder names: the first match for row and the first for section win. The sequence comes from the file stem. Reads image size via `_display_size`; videos get `None`. |
| `_display_size(path)` | Opens the header only (no full decode), swaps w/h when EXIF orientation ∈ {5, 6, 7, 8}. An unreadable file → `(None, None)` rather than an exception, so one bad file doesn't stop a scan. |
| `write_catalog(records, path)` / `read_catalog(path)` | CSV round-trip. Empty cells ↔ `None`; `row/section/sequence/width/height` parse back to `int`. Round-trip is exact (tested). |

---

## `splits.py`

**Why group splits:** photos in the same section show the same vines, often overlapping. A random split by
image puts near-duplicates in both train and test, which inflates test scores. The old notebook's 9-photo
farm test had this problem.

### `_groups(records, level, seed)` (internal)

1. Bucket records by `group_key(level)`.
2. Sort the keys (deterministic regardless of input order), then shuffle with `random.Random(seed)`.
3. **Stable**-sort by group size, largest first. The shuffle only decides the order among equal-size
   groups, so a different seed gives a different but still size-ordered assignment.

### `group_split(records, level="section", ratios=(0.6, 0.2, 0.2), names=("train","val","test"), seed=42)`

Greedy fill, largest group first. Each group goes to the split with the largest **relative deficit**:

```
deficit(split) = (target_images - current_images) / target_images,   target = ratio * total_images
```

- Relative (not absolute) deficit means the small val/test splits get a group early instead of train taking
  everything. With 3+ groups, each split with a nonzero ratio receives at least one.
- Splits with ratio `0` are skipped but still appear in the result as empty lists.
- Ratios are by **image count**, not group count. Actual shares are approximate because groups can't be
  divided: on West_Zone2 the default gives 11 / 6 / 5 images (4 / 2 / 2 sections).
- `len(ratios) != len(names)` → `ValueError`.

### `group_kfold(records, level="section", k=5, seed=42)`

Same group ordering. Each group goes to the currently **smallest** fold (balances image counts). Returns
`k` tuples `(train, test)`, where every group appears in exactly one test fold. Needs at least `k` groups,
otherwise `ValueError`.

Use it while labeled data is small. Averaging a metric over 5 folds is far more stable than one test split
of 2 sections.

---

## `annotations.py`

### Labeling contract (CVAT)

One label `bunch` with two attributes. Each box's attributes appear in the COCO export as
`annotation["attributes"]`:

| Attribute | Allowed values | If missing / empty |
|---|---|---|
| `row` | `near`, `far` | `near` |
| `quality` | `full`, `sparse`, `damaged`, `occluded` | `None` ("not judged") |

Attribute names are configurable (`annotations.row_attribute` / `quality_attribute` in the config).
CVAT's built-in `occluded` checkbox also lands in `attributes`; it's ignored.

### Dataclasses

- `BunchLabel(box, row="near", quality=None)`. `countable` = `row == near and quality == full`.
- `ImageLabels(file_name, width, height, bunches)`. `count(predicate=None)` counts bunches matching a
  predicate, e.g. `img.count(lambda b: b.countable)` = ground-truth countable bunches for that image.

### Functions

| Function | Behavior |
|---|---|
| `load_coco(path, row_attribute="row", quality_attribute="quality")` | Reads COCO JSON. `file_name` backslashes become `/`. `bbox` `[x, y, w, h]` → `Box`. Attribute values are stripped and lowercased (`"Sparse"` → `"sparse"`). **An unknown value raises `ValueError`** naming the file and annotation ID, so a typo in the CVAT setup fails loudly instead of silently becoming a new class. Images with no boxes are kept (they're valid negatives). |
| `label_stem(file_name)` | `row5/Section1/sequence1.jpg` → `row5__Section1__sequence1`. Every section has a `sequence1.jpg`, and YOLO pairs an image with its label by file name only, so flattening the path is what keeps labels from colliding. |
| `include_all(b)` / `include_near(b)` | Predicates choosing which boxes go into detector labels: every bunch (default), or near-row only (`--near-only`). |
| `export_yolo_split(labels, image_root, out_dir, split, include=include_all)` | For each image: copies it to `out_dir/split/images/<stem><ext>` (extension lowercased) and writes `out_dir/split/labels/<stem>.txt` with one `0 cx cy w h` line per included box. Boxes are **clipped to the image** first, because annotation tools can produce boxes slightly past the edge. An image with no included boxes still gets an empty `.txt` (a background example). Returns the image count. |

**Known limitation:** `label_stem` is unique only *within* one capture. Exporting two captures into the
same `--out` folder would collide (both have `row5__Section1__sequence1`). Export each capture to its own
folder, or prefix the stem with the capture name if you merge them.
