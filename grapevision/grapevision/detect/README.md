# `grapevision.detect`: stage 1, find every bunch

Wraps Ultralytics YOLO (training) and SAHI (tiled inference). The detector's only job is **"where are the
bunches?"**: one class, `bunch`, near row and background row alike. Deciding which bunches count is
left to stages 2 and 3, so the detector never has to learn "counts toward yield", a harder target that
depends on context.

Needs the optional extra: `pip install -e ".[detect]"`, after a CUDA build of torch. Both libraries are
imported inside the functions, so `import grapevision.detect` works without them.

| File | Contents |
|---|---|
| `infer.py` | `SahiDetector`: image → `list[Detection]` |
| `train.py` | `write_data_yaml`, `train_detector` |

---

## `infer.py`

### `SahiDetector(weights, tile_size=640, tile_overlap=0.25, conf_floor=0.05, device="cuda:0")`

`__init__` loads the checkpoint once through `sahi.AutoDetectionModel.from_pretrained(model_type="ultralytics")`.
SAHI then drops every prediction scoring below `conf_floor`.

`__call__(image: PIL.Image) -> list[Detection]` runs `sahi.predict.get_sliced_prediction`:

```
image ──► split into tile_size × tile_size slices, overlapping by tile_overlap (25%)
      ├─► YOLO on every slice                    (small bunches seen at native resolution)
      ├─► YOLO on the whole image too            (perform_standard_pred=True: large bunches that
      │                                            a single tile would cut in half)
      └─► merge: NMM (non-maximum *merging*), match threshold 0.5
             duplicates of one bunch from overlapping tiles are merged into one box,
             not just suppressed, so a bunch cut by a tile seam becomes one full box again
──► Detection(Box(minx, miny, maxx, maxy), score) for each merged prediction
```

Design points:

- **It takes pixels, not a path.** The pipeline has already EXIF-rotated the image (`io.load_image`); if
  SAHI re-read the file it could see the unrotated version, and boxes on portrait photos would come back
  on a different grid from the labels.
- **`conf_floor` is deliberately low (0.05).** The real threshold (`detector.conf`) is applied later, in
  `count.CountingPipeline`. Running once at a low floor and keeping all scores lets
  `eval.tune_threshold` try every threshold offline, with no extra inference.
- **The tile size follows the image size.** West_Zone2 photos are ~1215×911. 640 px tiles with 25% overlap
  make a ~2×2 grid, plus the full-image pass. The old notebook's 1024 tiles were sized for 2048 px photos.
  If you get full-resolution originals (4032×3024), try 1024.
- The merge uses SAHI's default match metric (IOS, intersection over the smaller box) at 0.5: two boxes
  merge when half of the smaller one lies inside the larger. Tune `postprocess_match_threshold` if adjacent
  bunches get merged or one bunch comes out as two boxes.

---

## `train.py`

### `write_data_yaml(dataset_dir, out_path=None, names=("bunch",)) -> Path`

Writes the Ultralytics dataset file for a folder produced by `data.export_yolo_split`:

```yaml
path: D:/.../datasets/west_zone2/detector     # absolute, forward slashes
train: train/images
val: val/images
test: test/images                              # only splits whose images/ folder exists are listed
nc: 1
names: ['bunch']
```

Ultralytics finds each label by swapping `/images/` for `/labels/` in the image path and `.jpg` for
`.txt`. That's why `export_yolo_split` writes the two side by side with identical stems.

### `train_detector(data_yaml, weights="yolo11m.pt", imgsz=640, epochs=150, device=0, workers=4, project="runs", name="detector", seed=42, **train_args)`

A thin pass-through to `YOLO(weights).train(...)`. It returns Ultralytics' results object. Outputs land in
`<project>/<name>/`: `weights/best.pt`, `weights/last.pt`, `results.csv` and plots. `best.pt` is the epoch
with the best validation *fitness* (a weighted mix of mAP50 and mAP50-95), a box-accuracy measure that
isn't the same as count error. Compare checkpoints on count MAE with `eval/` too.

- `weights` can be a pretrained name (`yolo11m.pt`, downloaded automatically on first use) or your own
  checkpoint, to fine-tune from it.
- Any Ultralytics training argument passes through `**train_args`: `lr0`, `patience`, `batch`, `mosaic`,
  `hsv_h`, `close_mosaic`, ...
- **Windows:** DataLoader workers start with `spawn`, which re-imports the calling module. Call this from
  under `if __name__ == "__main__":`, as `scripts/train_detector.py` does. The old notebook forced `fork`
  instead, which doesn't exist on Windows.
- **`copy_paste` does nothing for us.** Ultralytics' copy-paste augmentation needs segmentation masks; our
  labels are boxes only.
- **6 GB GPUs** (e.g. a laptop RTX 3060): pass `batch=-1` so Ultralytics picks the largest batch that fits;
  if `yolo11m` still runs out of memory, use `yolo11s.pt`.

### Choosing what the detector learns

`scripts/export_dataset.py` writes labels for **all** bunches by default, so the detector learns
"bunch" and the row filter separates rows afterwards. `--near-only` drops far-row boxes, which teaches
the detector to ignore the background row itself. Then far-row bunches become *negatives*, which is
harder to learn from small data but needs no stage 2. Training both and comparing on val tells you which
works better.
