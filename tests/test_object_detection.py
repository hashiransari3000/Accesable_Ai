"""Tests for the object detection module.

Pure post-processing is tested directly; model-dependent paths are
exercised with a stub injected into the module cache. A real-inference
smoke test runs only when the cached weights are present locally.
"""

import io

import pytest
from PIL import Image

from modules import object_detection as od
from modules.object_detection import (
    DEFAULT_MODEL_PATH,
    Detection,
    analyze_image,
    clamp_bbox,
    filter_by_confidence,
    format_label,
)


def _d(label, conf, bbox=(0, 0, 10, 10)):
    return Detection(label=label, confidence=conf, bbox=bbox)


# ---- filter_by_confidence (D1) ---------------------------------------------

def test_threshold_filters_detections():
    dets = [_d("a", 0.9), _d("b", 0.41), _d("c", 0.39), _d("d", 0.40)]
    kept = filter_by_confidence(dets, 0.40)
    assert [d.label for d in kept] == ["a", "b", "d"]


def test_results_sorted_best_first():
    kept = filter_by_confidence([_d("low", 0.5), _d("high", 0.8)], 0.1)
    assert kept[0].label == "high"


# ---- clamp_bbox (D2) ---------------------------------------------------------

def test_bbox_inside_bounds_unchanged():
    assert clamp_bbox((5, 5, 20, 20), 100, 100) == (5, 5, 20, 20)


def test_bbox_clamped_to_bounds():
    assert clamp_bbox((-10, -10, 200, 200), 100, 80) == (0, 0, 100, 80)


def test_fully_outside_bbox_removed():
    assert clamp_bbox((150, 150, 200, 200), 100, 100) is None
    assert clamp_bbox((50, 50, 50, 50), 100, 100) is None  # degenerate


# ---- format_label (D3) --------------------------------------------------------

def test_label_formatting():
    assert format_label("chair", 0.823) == "chair (82%)"


# ---- error paths (D4) via stubbed model ---------------------------------------

class _StubBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [xyxy]
        self.cls = [cls]
        self.conf = [conf]


class _StubRaw:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _StubModel:
    def predict(self, img, **kwargs):
        return [
            _StubRaw(
                boxes=[_StubBox((10, 10, 50, 50), 0, 0.91), _StubBox((0, 0, 5, 5), 1, 0.55)],
                names={0: "chair", 1: "table"},
            )
        ]


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def stub_model(monkeypatch):
    monkeypatch.setitem(od._MODEL_CACHE, DEFAULT_MODEL_PATH, _StubModel())
    yield


def test_analyze_full_pipeline_with_stub(stub_model):
    result = analyze_image(_png_bytes(), 0.40)
    assert result.error is None
    assert [d.label for d in result.detections] == ["chair", "table"]
    assert result.annotated_png is not None
    assert result.image_size == (100, 80)
    assert result.detections[0].bbox == (10, 10, 50, 50)


def test_model_load_failure_is_actionable(monkeypatch):
    def boom(path):
        raise RuntimeError("FileNotFoundError: weights missing")
    monkeypatch.setattr(od, "_load_model", boom)
    result = analyze_image(_png_bytes())
    assert result.error is not None
    assert "Could not load the detection model" in result.error


def test_corrupted_image_reports_error(monkeypatch):
    monkeypatch.setitem(od._MODEL_CACHE, DEFAULT_MODEL_PATH, _StubModel())
    result = analyze_image(b"definitely not an image")
    assert result.error is not None
    assert "could not be decoded" in result.error


# ---- model caching (D5) --------------------------------------------------------

def test_second_load_uses_cache(monkeypatch):
    import sys
    import types

    calls = {"n": 0}

    class FakeYOLO:
        def __init__(self, path):
            calls["n"] += 1

    # Stub the heavy 'ultralytics' import, then exercise the REAL loader,
    # so this test verifies the cache logic itself (D5).
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    od._load_model("stub-cache-key")
    od._load_model("stub-cache-key")
    assert calls["n"] == 1


def test_is_model_loaded_reflects_cache():
    assert od.is_model_loaded("definitely-not-a-key") is False


# ---- real-inference smoke test (runs only with cached weights) ------------------

@pytest.mark.skipif(
    not __import__("os").path.exists(DEFAULT_MODEL_PATH),
    reason="yolo11n.pt not downloaded yet (one-time download on first use)",
)
def test_real_model_smoke():
    result = analyze_image(_png_bytes(), 0.40)
    assert result.error is None
    assert result.inference_ms >= 0
    # A plain white square may or may not yield detections; the contract is
    # a clean run with valid structure, not specific objects.
    for d in result.detections:
        assert 0.40 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 80
