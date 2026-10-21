## What and why

<!-- What does this change, and what problem does it solve? Link the issue if there is one. -->

## Type

- [ ] Bug fix
- [ ] New feature / pipeline stage
- [ ] Model or threshold change (affects counts)
- [ ] Refactor / tooling / docs

## Model impact

<!-- Required if this changes what the pipeline counts. Report on the VAL split; the test split is only
     for the final number. Delete this section otherwise. -->

| Metric (val) | Before | After |
|---|---|---|
| Count MAE | | |
| Precision / Recall | | |
| Countable bunches per row | | |

## Checklist

- [ ] Tests added or updated; `pytest` passes locally
- [ ] `ruff check`, `ruff format --check` and `mypy` pass (or `pre-commit run --all-files`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No photos, videos, datasets or model weights committed
