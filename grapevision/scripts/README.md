# `scripts/`: command-line entry points

Thin CLIs over the package. Each one parses arguments, loads the config, calls package functions, and
prints a summary. No pipeline logic lives here; if a script grows logic, move it into the package and test
it there.

Run from the `grapevision/` folder with the venv active. Every script takes `--config` (a project config
layered on the package defaults; see `../configs/README.md`).

| Order | Script | Needs a GPU stack | Input → output |
|---|---|---|---|
| 1 | `build_catalog.py` | no | capture folders → `outputs/catalog.csv` + split preview |
| 2 | *(label in CVAT, export COCO 1.0)* | | |
| 3 | `export_dataset.py` | no | COCO JSON + photos → YOLO dataset + quality crops |
| 4 | `train_detector.py` | yes | YOLO dataset → `runs/<name>/weights/best.pt` |
| 5 | `run_pipeline.py` | yes | catalog + weights → annotated images + count CSVs |

---

## `build_catalog.py --config configs/west_zone2.yaml`

1. For each folder in `paths.captures`: `data.scan_capture`, then prints image and video counts and the
   landscape/portrait split.
2. Writes all records to `paths.catalog_csv` with `data.write_catalog`.
3. Runs `data.group_split` on the images with the config's `split` settings and prints which groups land
   in train/val/test. It only *previews* the split; nothing is copied. `export_dataset.py` recomputes the
   same split, which is deterministic for a given seed and set of photos.

## `export_dataset.py --coco <export.json> --capture <capture folder> --out <dir> [--near-only] [--config ...]`

1. `data.load_coco` reads the labels, then matches them to photos by `rel_path` = COCO `file_name`.
2. If any labeled file isn't found under `--capture`, it **exits with an error** listing examples. The
   usual cause is uploading files to CVAT one by one instead of as a zip, which loses the
   `row5/Section1/` part of the name.
3. Splits only the *labeled* photos with `data.group_split` (by `split.level`).
4. Per split:
   - `data.export_yolo_split` → `<out>/detector/<split>/{images,labels}/`. With `--near-only`, far-row boxes
     are left out.
   - `quality.export_quality_crops` → `<out>/quality/<split>/<class>/*.jpg`.
5. `detect.write_data_yaml` → `<out>/detector/data.yaml`.

Relative `--out` paths resolve against the project root.

## `train_detector.py --data <data.yaml> [--epochs 150] [--name detector] [--workers 4] [--config ...]`

Calls `detect.train_detector` with `detector.weights`, `detector.imgsz` and `detector.device` from the
config (`cuda:0` becomes Ultralytics' `0`), and `split.seed`. Output goes to `runs/<name>/`. The
`if __name__ == "__main__":` guard is required on Windows (see `detect/README.md`). Lower `--workers` if
RAM runs short.

## `run_pipeline.py --weights <best.pt> [--level row|section|image] [--config ...]`

1. Reads `outputs/catalog.csv` (run `build_catalog.py` first) and keeps the images.
2. `count.build_pipeline(cfg)` with `detector.weights` overridden by `--weights`.
3. Per image: `pipeline.run`, then appends a row to `outputs/per_image.csv`
   (`capture, rel_path, row, section, detected, near, countable`). Saves
   `viz.draw_result` to `outputs/annotated/<capture>__<label_stem>.jpg` and prints the three counts.
4. `count.aggregate` at `--level` → `outputs/per_<level>.csv`.

It runs on *all* cataloged images, including those used for training. Those counts show what the model
does, but they don't measure accuracy. Accuracy comes from `eval/` on the test split.

## Testing

`tests/test_scripts.py` runs `build_catalog.py` and `export_dataset.py` in a subprocess against a fake
capture, as a user would, including the missing-file error. The two GPU scripts are covered through the
package functions they call (`tests/test_backends.py`).
