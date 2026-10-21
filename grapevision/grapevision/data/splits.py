"""Train/val/test splits that keep whole groups (sections or rows) together.

Splitting by individual image lets near-identical shots of the same vines land in both train and test,
which inflates test scores. Grouping by section or row closes that leak.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from grapevision.data.catalog import MediaRecord


def _groups(records: Sequence[MediaRecord], level: str, seed: int) -> list[list[MediaRecord]]:
    by_key: dict[str, list[MediaRecord]] = defaultdict(list)
    for r in records:
        by_key[r.group_key(level)].append(r)
    keys = sorted(by_key)
    random.Random(seed).shuffle(keys)
    # Largest groups first so the greedy fill below lands close to the target ratios; the shuffle
    # breaks ties between equal-size groups differently per seed.
    keys.sort(key=lambda k: -len(by_key[k]))
    return [by_key[k] for k in keys]


def group_split(
    records: Sequence[MediaRecord],
    level: str = "section",
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    names: Sequence[str] = ("train", "val", "test"),
    seed: int = 42,
) -> dict[str, list[MediaRecord]]:
    """Assign each group to the split furthest below its target share of images."""
    if len(ratios) != len(names):
        raise ValueError("ratios and names must have the same length")
    total = len(records)
    targets = {n: r * total for n, r in zip(names, ratios, strict=True) if r > 0}
    splits: dict[str, list[MediaRecord]] = {n: [] for n in names}
    for group in _groups(records, level, seed):
        name = max(targets, key=lambda n: (targets[n] - len(splits[n])) / targets[n])
        splits[name].extend(group)
    return splits


def group_kfold(
    records: Sequence[MediaRecord],
    level: str = "section",
    k: int = 5,
    seed: int = 42,
) -> list[tuple[list[MediaRecord], list[MediaRecord]]]:
    """K folds of (train, test), each group in exactly one test fold. Useful while labeled data is small:
    averaging over folds gives a steadier estimate than one tiny test split."""
    groups = _groups(records, level, seed)
    if len(groups) < k:
        raise ValueError(f"only {len(groups)} groups at level {level!r}; need at least k={k}")
    folds: list[list[MediaRecord]] = [[] for _ in range(k)]
    for group in groups:
        min(folds, key=len).extend(group)
    return [([r for j, f in enumerate(folds) if j != i for r in f], folds[i]) for i in range(k)]
