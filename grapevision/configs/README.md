 # `configs/`: project configs

A project config holds the settings that are specific to a capture or a machine, mainly **where the
photos are**. It's layered on top of the package defaults, so it only lists what differs from them.

```
grapevision/default_config.yaml     package defaults: every setting, with comments (ships with the package)
          ▲  deep-merged with
configs/<name>.yaml                 this folder: only the overrides
          ▲  deep-merged with
in-code overrides                   e.g. run_pipeline.py's --weights
```

Merge rules (`config._deep_merge`): nested sections merge key by key; **lists and scalars replace** the
default outright. So `captures: [A]` means "only A".

Pass a config to any script with `--config configs/<name>.yaml`.

## Files

| File | Purpose |
|---|---|
| `west_zone2.yaml` | The 2026-09-11 West Zone 2 capture (rows 5 and 7) at `D:/download/West_Zone2_09_11_2026` |

## Adding one

Copy `west_zone2.yaml`, change `paths.captures`, and add any setting you want to override, using the
same section and key names as `default_config.yaml`. For example, a config for full-resolution photos:

```yaml
paths:
  captures:
    - D:/data/West_Zone2_09_11_2026_fullres
detector:
  imgsz: 1024
  tile_size: 1024
```

Relative paths resolve against the `grapevision/` folder. Absolute paths are machine-specific; that's
expected here, and it's why they live in this folder rather than in the package defaults.
