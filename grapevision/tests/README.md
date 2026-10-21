# `tests/`

```powershell
pytest                    # all tests (~3 s)
pytest --cov              # plus coverage; fails under 90%
pytest tests/test_data.py -k split -v      # one file / matching tests
pytest -m "not gpu"       # what CI runs
```

## Rules

- **CPU only, no downloads, no real photos.** Images are made with Pillow inside `tmp_path`, pytest's
  per-test temporary folder, which is deleted afterwards.
- **GPU libraries are faked, not installed** (`fakes.py`). These tests check *our* code: the arguments we
  pass and how we convert results. They don't check whether YOLO is accurate.
- **Warnings are errors** (`filterwarnings = ["error"]` in `pyproject.toml`), so a deprecation shows up in
  CI before it becomes a break.
- Tests that truly need a GPU or real weights get `@pytest.mark.gpu`. CI deselects them.

## Shared helpers

### `conftest.py`: the `capture` fixture

Builds a fake capture shaped like the real one:

```
<tmp>/West_Zone2_09_11_2026/
├── row5/Section1/sequence1.jpg, sequence2.jpg      80×60 green images
├── row5/Section2/...
├── row7/Section1/...
├── row7/Section2/sequence2.jpg                     60×80, the one portrait shot
├── video_sample.MOV                                must be cataloged as video
└── notes.txt                                       must be ignored
```

That's 8 images in 4 sections across 2 rows, enough to test grouping at both levels.

### `fakes.py`

| Installer | Replaces | Fake behavior | Records |
|---|---|---|---|
| `install_fake_sahi(monkeypatch, boxes)` | `sahi`, `sahi.predict` | `get_sliced_prediction` returns the given `(x1, y1, x2, y2, score)` boxes | init kwargs, image shape, predict kwargs |
| `install_fake_transformers(monkeypatch, depth)` | `transformers` | the depth pipeline returns `depth` as an 8-bit image **at its own size**, which forces the resize path | `(task, model, device)` |
| `install_fake_ultralytics(monkeypatch)` | `ultralytics` | `YOLO(w).train(**kw)` returns `"trained"` | weights, train kwargs |

They work by inserting stub modules into `sys.modules` with `monkeypatch.setitem`, so the package's lazy
`from sahi import ...` picks up the stub. pytest restores `sys.modules` after each test.

## What each file covers

| File | Covers |
|---|---|
| `test_data.py` | capture-name parsing; scanning (structure, orientation, video, ignored files); catalog CSV round-trip; flat folders; group splits at section and row level keep groups whole and hit the expected sizes; k-fold tests each group exactly once; `label_stem` collisions; COCO attribute parsing (defaults, case, typos rejected); YOLO export (far rows excluded, normalized coordinates) |
| `test_stages.py` | Otsu separability (tight clusters ≈ 1, uniform ≈ 0.75); near/far split, including the reversed depth direction and a single oblique row; center-weighted box depth; netting score; `DepthRowFilter` with a synthetic depth map; size baseline; crop padding and clipping; the full pipeline with a fake detector + `aggregate` |
| `test_metrics.py` | count metrics; one-to-one box matching; precision/recall; threshold tuning picks the minimum MAE; bootstrap CI |
| `test_backends.py` | `SahiDetector` (passes pixels, not a path); depth estimator resizing; `train_detector` argument pass-through; `write_data_yaml`; `build_pipeline` for every config option, end to end, and rejecting unknown methods |
| `test_misc.py` | config layering and path resolution; `draw_result` colors and not modifying the input; quality-crop export (far and unjudged bunches skipped); unreadable images; argument errors; edge cases (empty arrays, NaN depths, boxes outside the image); uncataloged images in `aggregate` |
| `test_scripts.py` | `build_catalog.py` and `export_dataset.py` run in a subprocess, including the missing-file error |

## Adding a test

Fixing a bug: first write a test that fails because of it, then fix the code. New stage implementations
(e.g. the quality CNN) should get the same treatment as `DepthRowFilter`: inject the model as a callable,
so the stage logic is tested with a stand-in and the real model is only exercised in `gpu`-marked tests.
