"""YAML config loading: package defaults (default_config.yaml), then a project config, then overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = _PACKAGE_DIR / "default_config.yaml"
# In a source checkout, relative config paths resolve against the folder holding pyproject.toml; when
# installed as a package there is no such folder, so they resolve against the working directory.
PROJECT_ROOT = _PACKAGE_DIR.parent if (_PACKAGE_DIR.parent / "pyproject.toml").is_file() else Path.cwd()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the default config, layer `path` on top of it, then `overrides`."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def resolve_path(p: str | Path, root: Path = PROJECT_ROOT) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()
