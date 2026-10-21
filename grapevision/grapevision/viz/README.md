# `grapevision.viz`: annotated images

Everything lives in `draw.py`. It turns an `ImageResult` into a picture, so every stage's decision can be
checked by eye.

## Colors

| Constant | RGB | Box means |
|---|---|---|
| `COUNTED` | green `(40, 220, 60)` | near row **and** full: counts toward yield |
| `NOT_COUNTED` | yellow `(250, 210, 0)` | near row, but judged not full |
| `BACKGROUND` | gray `(150, 150, 150)` | background row, removed by the row filter |

Each color answers "which stage removed this?": gray → stage 2, yellow → stage 3.

## `draw_result(image, result) -> PIL.Image`

1. Draws on a **copy**; the input image is never modified.
2. Line width scales with the image: `max(2, width // 400)` px, so boxes stay visible on 4000-px photos.
3. Boxes are drawn in the order gray → yellow → green (sorted by `(near, countable)`). Where boxes overlap,
   the counted one is on top.
4. A black banner in the top-left corner reads
   `counted N  |  near row M  |  detected K` (the three `ImageResult` counts). The font size scales with
   width (`width // 45`, at least 16 px).

## `_font(size)` (internal)

`ImageFont.load_default(size=...)` gives a scalable font on Pillow ≥ 10.1. On older Pillow, that call raises
`TypeError`, and it falls back to the fixed-size bitmap default rather than crashing.

Used by `scripts/run_pipeline.py`, which saves one annotated image per photo to `outputs/annotated/`.
