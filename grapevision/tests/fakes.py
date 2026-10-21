"""Stand-ins for the optional GPU backends (sahi, transformers, ultralytics), so CI can exercise the wrapper
code on CPU without installing them. Each installer returns a dict that records how the fake was called."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image


def install_fake_sahi(monkeypatch: pytest.MonkeyPatch, boxes: list[tuple[float, float, float, float, float]]) -> dict:
    calls: dict[str, Any] = {}

    class AutoDetectionModel:
        @staticmethod
        def from_pretrained(**kwargs: Any) -> str:
            calls["init"] = kwargs
            return "fake-model"

    def get_sliced_prediction(image: np.ndarray, model: str, **kwargs: Any) -> SimpleNamespace:
        calls["image_shape"] = image.shape
        calls["predict"] = kwargs
        return SimpleNamespace(
            object_prediction_list=[
                SimpleNamespace(
                    bbox=SimpleNamespace(minx=x1, miny=y1, maxx=x2, maxy=y2), score=SimpleNamespace(value=s)
                )
                for x1, y1, x2, y2, s in boxes
            ]
        )

    sahi = types.ModuleType("sahi")
    predict = types.ModuleType("sahi.predict")
    sahi.AutoDetectionModel = AutoDetectionModel  # type: ignore[attr-defined]
    predict.get_sliced_prediction = get_sliced_prediction  # type: ignore[attr-defined]
    sahi.predict = predict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sahi", sahi)
    monkeypatch.setitem(sys.modules, "sahi.predict", predict)
    return calls


def install_fake_transformers(monkeypatch: pytest.MonkeyPatch, depth: np.ndarray) -> dict:
    """The fake depth pipeline returns `depth` as an 8-bit image at its own size, so callers must resize."""
    calls: dict[str, Any] = {}

    def pipeline(task: str, model: str, device: Any = None) -> Any:
        calls["pipeline"] = (task, model, device)
        return lambda image: {"depth": Image.fromarray(depth.astype(np.uint8))}

    transformers = types.ModuleType("transformers")
    transformers.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict[str, Any] = {}

    class YOLO:
        def __init__(self, weights: str):
            calls["weights"] = weights

        def train(self, **kwargs: Any) -> str:
            calls["train"] = kwargs
            return "trained"

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = YOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    return calls
