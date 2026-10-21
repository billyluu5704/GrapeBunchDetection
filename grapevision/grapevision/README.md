# `grapevision` package

The importable package. Each subfolder is one pipeline stage and has its own README. This page covers the
files at the package root, which every stage depends on, and how the stages fit together.

```
grapevision/
├── __init__.py            re-exports Box, Detection; holds __version__ (single source for the build)
├── types.py               Box, Detection, label vocabulary          ← every module imports this
├── io.py                  load_image()                              ← the only way images get loaded
├── config.py              load_config(), resolve_path()
├── default_config.yaml    every setting with its default
├── py.typed               marks the package as typed (PEP 561) for mypy in downstream code
├── data/      README      catalog, splits, annotation import/export
├── detect/    README      stage 1: YOLO + SAHI
├── rowfilter/ README      stage 2: near row vs. background row
├── quality/   README      stage 3: "large and full"
├── count/     README      wires the stages, aggregates per row/section
├── eval/      README      metrics and threshold tuning
└── viz/       README      annotated images
```

## How the stages fit together

```
                 load_image(path)                         (io.py: EXIF-rotated RGB)
                        │
                        ▼
   detect.SahiDetector(image) ──► list[Detection]         box, score
                        │  keep score >= conf
                        ▼
   rowfilter.*.apply(image, dets)                         sets .row, .features["depth"|"netting"|"area_frac"]
                        │  only d.near continue
                        ▼
   quality.*.apply(image, near_dets)                      sets .quality (and later .fullness)
                        │
                        ▼
   count.ImageResult(total, near, countable) ──► count.aggregate(...) per row / section
```

Stages communicate by **mutating `Detection` objects in place**. Each stage fills in the fields it owns and
reads the ones earlier stages set. Every stage shares one interface, `apply(image, detections) -> None`
(the `Stage` protocol in `count/pipeline.py`). Swapping the depth rule for a learned classifier, or the
size baseline for a CNN, means writing a new class with that one method.

**Import rule:** `ultralytics`, `sahi`, `transformers` and `torch` are only ever imported *inside* functions.
`import grapevision` and every module in it work with just numpy, Pillow and PyYAML. That's what lets CI and
the tests run without a GPU.

---

## `types.py`: shared vocabulary

### Label constants

| Constant | Values | Meaning |
|---|---|---|
| `ROW_VALUES` | `NEAR = "near"`, `FAR = "far"` | Is the bunch on the row being surveyed, or a row seen behind it? |
| `QUALITY_VALUES` | `FULL`, `SPARSE`, `DAMAGED`, `OCCLUDED` | Only `FULL` counts toward yield. `OCCLUDED` = "can't judge" (leaf in the way). It's kept apart from `SPARSE` so a quality model doesn't learn "partly hidden = sparse". |

They're plain strings rather than an `Enum` so they round-trip unchanged through CVAT attributes, CSV and
JSON.

### `Box` (frozen dataclass)

Axis-aligned box in **pixel** coordinates `(x1, y1)` top-left, `(x2, y2)` bottom-right, as floats. Frozen,
so it can't be changed after creation and is hashable.

| Member | What it does |
|---|---|
| `width`, `height` | `max(0, x2 - x1)`: never negative, even for degenerate boxes |
| `area` | `width * height` |
| `iou(other)` | Intersection-over-union, standard formula; `0.0` when the union is empty |
| `clip(w, h)` | Clamps every coordinate into `[0, w] × [0, h]`. A box entirely outside collapses onto the border with zero area. |
| `to_yolo(img_w, img_h)` | → `(cx, cy, w, h)` normalized to 0–1, the YOLO label format |
| `from_yolo(cx, cy, w, h, img_w, img_h)` | Inverse of `to_yolo` (round-trips up to floating-point error) |
| `from_xywh(x, y, w, h)` | From COCO's `[top-left x, top-left y, width, height]` |

### `Detection` (mutable dataclass)

One predicted bunch, filled in progressively by the stages.

| Field | Set by | Meaning |
|---|---|---|
| `box`, `score` | detect | Location and detector confidence (0–1) |
| `row` | rowfilter | `"near"`, `"far"`, or `None` if no row filter ran |
| `quality` | quality | One of `QUALITY_VALUES`, or `None` if not judged (far-row bunches are never judged) |
| `fullness` | quality (planned) | Continuous fullness score from the future CNN; unused by the current baselines |
| `features` | any stage | `dict[str, float]` of diagnostics, e.g. `depth`, `netting`, `area_frac`. The learned near/far classifier will train on these. |

| Property | Rule |
|---|---|
| `near` | `row != FAR`. `None` counts as near, so a pipeline without a row filter keeps everything. |
| `countable` | `near and quality == FULL`: the definition of "counts toward yield" |

---

## `io.py`: image loading

### `load_image(path) -> PIL.Image` (RGB)

Opens the file, applies `ImageOps.exif_transpose`, converts to RGB, closes the file handle.

**Why it exists:** phones often save a portrait photo as landscape pixels plus an EXIF tag saying "rotate
90°". Viewers (and CVAT) honor the tag; raw pixel loaders don't. If one stage applied the rotation and
another didn't, their boxes would land on different grids. Every stage loads through this function, and
the detector receives pixels (not a path), so all coordinates refer to the image as a person sees it.

---

## `config.py`: configuration

Configs are layered: **package defaults → project config file → in-code overrides**, each merged into the
previous one with `_deep_merge`.

| Name | What it is |
|---|---|
| `DEFAULT_CONFIG` | Path to `default_config.yaml` inside the package (ships with the wheel as package data) |
| `PROJECT_ROOT` | The folder holding `pyproject.toml` in a source checkout; the current working directory when installed as a package. Relative config paths resolve against it. |
| `_deep_merge(base, override)` | Recursive dict merge. Nested dicts merge key by key; any other value (lists included) replaces the old one outright. Returns a new dict and never mutates its inputs. |
| `load_config(path=None, overrides=None)` | Defaults, then `path` (if given and not the defaults file itself), then `overrides`. Returns a plain nested `dict`. |
| `resolve_path(p, root=PROJECT_ROOT)` | Absolute paths pass through; relative ones are joined onto `root` and resolved. |

Lists replace rather than merge on purpose. A project config listing `captures: [A]` means "only A", not
"the defaults plus A".

## `default_config.yaml`

Every setting, its default, and a one-line reason. The sections map onto the subpackages:
`paths`, `split` (data), `annotations` (data), `detector` (detect), `row_filter`, `quality`.
Machine-specific values (where the photos are) belong in `../configs/`, not here.
