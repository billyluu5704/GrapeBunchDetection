# Changelog

All notable changes to grapevision. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). The release workflow publishes the section
matching the tag, so every release needs a section here.

## [Unreleased]

## [0.1.0] - 2026-10-07

### Added
- Package layout with one module per pipeline stage: `data`, `detect`, `rowfilter`, `quality`, `count`,
  `eval`, `viz`.
- Capture catalog for `<capture>/rowN/SectionM/sequenceK` folders, with EXIF-aware orientation.
- Train/val/test splits and k-fold by section or row, so shots of the same vines can't leak across splits.
- CVAT COCO import with per-box `row` (near/far) and `quality` (full/sparse/damaged/occluded) attributes;
  YOLO export with collision-free file names; quality-classifier crop export.
- Near/far row filter: per-box monocular depth (Depth Anything V2) split with Otsu, plus netting and size
  features.
- Quality baselines (accept-all, box size) for the planned fullness classifier.
- Count metrics, box precision/recall/F1, bootstrap CIs, and confidence-threshold tuning on the val split
  from a single inference pass.
- CLI scripts: `build_catalog`, `export_dataset`, `train_detector`, `run_pipeline`.
- CI (ruff, strict mypy, pytest on Python 3.10–3.14 and Windows, 90% coverage gate, wheel build and install
  check, pre-commit), tag-driven GitHub Releases, Dependabot.
