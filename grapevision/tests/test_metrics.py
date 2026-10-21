import pytest

from grapevision.eval import bootstrap_ci, count_metrics, detection_metrics, match_boxes, tune_threshold
from grapevision.types import Box


def test_count_metrics():
    m = count_metrics([3, 5, 2], [3, 7, 1])
    assert m["mae"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(1 / 3)
    assert m["exact_pct"] == pytest.approx(100 / 3)
    assert m["within1_pct"] == pytest.approx(200 / 3)


def test_match_boxes_one_to_one():
    gt = [Box(0, 0, 10, 10), Box(20, 20, 30, 30)]
    pred = [Box(0, 0, 10, 10), Box(1, 1, 10, 10), Box(50, 50, 60, 60)]
    assert match_boxes(gt, pred) == (1, 2, 1)
    assert match_boxes([], pred) == (0, 3, 0)
    assert match_boxes(gt, []) == (0, 0, 2)


def test_detection_metrics():
    m = detection_metrics([([Box(0, 0, 10, 10)], [Box(0, 0, 10, 10), Box(30, 30, 40, 40)])])
    assert (m["precision"], m["recall"]) == (0.5, 1.0)


def test_tune_threshold_picks_best_mae():
    true = [2, 1]
    scores = [[0.9, 0.8, 0.3], [0.7, 0.2]]
    best, table = tune_threshold(true, scores, thresholds=[0.1, 0.5, 0.85])
    assert best == 0.5
    assert {r["threshold"]: r["mae"] for r in table} == {0.1: 1.0, 0.5: 0.0, 0.85: 1.0}


def test_bootstrap_ci_contains_mean():
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo <= 3 <= hi
