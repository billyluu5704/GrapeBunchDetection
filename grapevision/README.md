# grapevision

[![grapevision CI](https://github.com/hoangnhatduyle/GrapeBunchDetection/actions/workflows/grapevision-ci.yml/badge.svg)](https://github.com/hoangnhatduyle/GrapeBunchDetection/actions/workflows/grapevision-ci.yml)
![python](https://img.shields.io/badge/python-3.10--3.14-blue) ![types](https://img.shields.io/badge/mypy-strict-blue) ![coverage gate](https://img.shields.io/badge/coverage%20gate-90%25-brightgreen)

Counts grape bunches toward yield on the **surveyed row only**, keeping only bunches that are **large and
pretty full**. It replaces the notebook in `../Deep Learning/` with a package of stages:

```
photo ─► detect ─► rowfilter ─► quality ─► count ─► per-row / per-section totals
         every     near row      countable
         bunch     vs. behind    (full) only
```

## Layout

Every folder has its own `README.md` explaining its functions in detail. Start with
[grapevision/README.md](grapevision/README.md) for how the stages fit together.

```
grapevision/
├── configs/                 project configs (machine-specific paths), layered on the package defaults
├── grapevision/
│   ├── types.py             Box, Detection, label vocabulary (near/far, full/sparse/damaged/occluded)
│   ├── io.py                image loading with EXIF rotation (portrait phone shots)
│   ├── config.py            YAML loading: package defaults -> project config -> overrides
│   ├── default_config.yaml  every setting with its default and an explanation
│   ├── data/                catalog of captures, group splits, CVAT COCO import, YOLO export
│   ├── detect/              YOLO training + SAHI tiled inference           (stage 1)
│   ├── rowfilter/           depth-based near/far split, netting cue        (stage 2)
│   ├── quality/             bunch crops + baseline classifiers             (stage 3)
│   ├── count/               pipeline wiring and aggregation
│   ├── eval/                count + box metrics, threshold tuning on val
│   └── viz/                 annotated output images
├── scripts/                 build_catalog, export_dataset, train_detector, run_pipeline
├── tests/                   CPU-only; GPU backends are faked (tests/fakes.py)
├── CHANGELOG.md
└── CONTRIBUTING.md          dev setup, checks, test rules, release process
```

## Status

| Stage | Done | Next |
|---|---|---|
| data | catalog, section/row splits, k-fold, CVAT import, YOLO + crop export | label West_Zone2 in CVAT |
| detect | SAHI inference, training wrapper | train on the labeled captures |
| rowfilter | per-box depth + Otsu near/far rule; netting and size features recorded | learned near/far classifier on `row` labels |
| quality | crop export; accept-all and box-size baselines | CNN fullness model on the crops |
| eval | count error, P/R/F1, bootstrap CI, threshold tuning on val | per-row yield report |

## Setup

```powershell
cd grapevision
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e .[dev]               # core + tests
# GPU stages: install a CUDA torch build first (pytorch.org), then
pip install -e .[detect,depth]
pytest
```

## Development

Every pull request touching `grapevision/` runs [CI](../.github/workflows/grapevision-ci.yml):

| Job | What it checks |
|---|---|
| Lint & type-check | `ruff check`, `ruff format --check`, `mypy` in strict mode |
| Test | `pytest` on Python 3.10–3.14 (Linux) and 3.14 (Windows); fails under 90% coverage |
| Build | builds the wheel and sdist, installs the wheel in a clean venv, imports it |
| pre-commit | whitespace/YAML/TOML checks, blocks files over 500 KB, ruff |

Pushing a tag `grapevision-vX.Y.Z` runs the [release workflow](../.github/workflows/grapevision-release.yml):
it tests, builds and publishes a GitHub Release using the matching `CHANGELOG.md` section. Dependabot opens
weekly update PRs for Python packages and GitHub Actions. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
local commands and the release steps.

## Workflow

1. **Catalog the captures.** List capture folders under `paths.captures` in a project config (see
   `configs/west_zone2.yaml`), then run `python scripts/build_catalog.py --config configs/west_zone2.yaml`.
   It prints image counts and a preview of the split by section. Pass the same `--config` to the other
   scripts.
2. **Label in CVAT.** Upload each capture as a zip of the whole folder, so file names keep their
   `row5/Section1/...` paths: every section has its own `sequence1.jpg`. Create one label, `bunch`, with two
   attributes:
   - `row`: select, values `near`, `far` (default `near`)
   - `quality`: select, values `full`, `sparse`, `damaged`, `occluded`

   Box every visible bunch, background row included, and mark it `far`. Use `occluded` when a leaf
   hides enough of the bunch that you can't judge fullness. Export as **COCO 1.0**.
3. **Export training data.** Run
   `python scripts/export_dataset.py --coco <export>.json --capture <capture folder> --out datasets/<name>`.
   It writes the YOLO detector dataset (split by section) and the quality crops.
4. **Train the detector.** Run `python scripts/train_detector.py --data datasets/<name>/detector/data.yaml`.
5. **Pick the confidence threshold on val, never on test.** Run the detector over the val split, keep the
   scores, and call `grapevision.eval.tune_threshold`. Put the result in `detector.conf`.
6. **Count.** Run `python scripts/run_pipeline.py --weights runs/detector/weights/best.pt --level row`.
   It writes annotated images (green = counted, yellow = near row but not full, gray = background row)
   and per-image and per-row CSVs to `outputs/`.

## Things to check

- **Depth direction.** Depth Anything outputs relative inverse depth (higher = closer), so the default is
  `higher_is_closer: true`. Check it on one photo: the near canopy should come out brighter than the row
  behind it.
- **The near/far rule declares a far group only when depths form two clearly separate clusters**
  (`min_separability`, 0.85 by default). A single row shot at an angle has spread-out depths
  (separability around 0.75) and is left as all near. Tune this on labeled photos.
- **Aggregation assumes shots in a section don't overlap.** If they do, the same bunch is counted twice.
  Video with tracking is the planned fix for whole-row counts.
- **Ultralytics `copy_paste` needs segmentation masks.** With box-only labels it does nothing.
