"""Tests for relative proximity bands.

The depth model itself is not exercised here (heavy download); the pure
band-mapping logic and failure paths are.
"""

from modules import depth_estimation as depth
from modules.object_detection import Detection, DetectionResult


def test_band_boundaries():
    assert depth.proximity_band(1.0) == "very close"
    assert depth.proximity_band(0.70) == "very close"
    assert depth.proximity_band(0.69) == "close"
    assert depth.proximity_band(0.45) == "close"
    assert depth.proximity_band(0.44) == "mid-distance"
    assert depth.proximity_band(0.22) == "mid-distance"
    assert depth.proximity_band(0.21) == "far"
    assert depth.proximity_band(0.0) == "far"


def test_annotate_no_detections_is_noop_and_does_not_load_model():
    result = DetectionResult()
    assert depth.annotate_result(result, b"not-an-image") is None
    assert depth.is_loaded() is False


def test_annotate_bad_image_returns_friendly_error():
    result = DetectionResult(
        detections=[Detection("chair", 0.8, (0, 0, 10, 10))]
    )
    error = depth.annotate_result(result, b"not-an-image")
    assert error and "proximity" in error.lower()
    assert depth.is_loaded() is False


def test_bands_are_relative_wording_only():
    # Guard against metric/safety language creeping into band labels.
    labels = {label for _, label in depth.BAND_THRESHOLDS}
    assert all("meter" not in label and "safe" not in label for label in labels)
