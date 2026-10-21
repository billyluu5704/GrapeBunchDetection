# Contributing to grapevision

## Setup

```powershell
cd grapevision
python -m venv .venv; .venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install                                    # run from the repo root; hooks run on every commit
```

GPU work (training, real inference, depth model) also needs `pip install -e ".[detect,depth]"`, after
installing a CUDA build of torch from pytorch.org.

## Checks: the same ones CI runs

```powershell
ruff check .              # lint (add --fix to auto-fix)
ruff format .             # format
mypy                      # strict type check of the package
pytest --cov              # tests + coverage; fails below 90%
pre-commit run --all-files
```

CI runs these on every pull request that touches `grapevision/`. Tests run on Python 3.10–3.14 on Linux
and 3.14 on Windows. A PR should merge only when every check is green.

## Tests

- Tests must run on a CPU-only machine with no downloads. Code that wraps `ultralytics`, `sahi` or
  `transformers` is tested against the fakes in `tests/fakes.py`, which check our wiring, not the libraries.
- Tests that truly need a GPU or real weights get `@pytest.mark.gpu`; CI deselects them with `-m "not gpu"`.
- Build test images in `tmp_path` (see the `capture` fixture in `tests/conftest.py`); never commit photos.
- A bug fix comes with a test that fails without the fix.

## Model changes

Anything that changes what gets counted (weights, thresholds, row filter, quality rules) is a model
change. Report before/after metrics on the **val** split in the PR (the template has a table). Leave the
test split alone until you report a final number; tuning against it makes that number meaningless.

## Branches, commits, releases

- Branch from `main` (`feat/quality-classifier`, `fix/exif-rotation`); open a PR; squash-merge.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
  `test:`, `docs:`, `refactor:`, `ci:`, `chore:`.
- Add a line under `[Unreleased]` in `CHANGELOG.md` for user-visible changes.
- To release: bump `__version__` in `grapevision/__init__.py`, rename `[Unreleased]` to the new version and
  date in `CHANGELOG.md`, merge, then tag `main`:
  `git tag grapevision-v0.2.0 && git push origin grapevision-v0.2.0`.
  The release workflow tests, builds and publishes a GitHub Release with the wheel attached.

## Never commit

Photos, videos, datasets, `outputs/`, `runs/`, or model weights (`*.pt`). `.gitignore` covers the usual
paths, and the `check-added-large-files` hook blocks anything over 500 KB. Share data through the team
drive; share released weights as GitHub Release assets.
