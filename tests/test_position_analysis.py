"""Tests for relative position analysis (TEST_PLAN P1-P7)."""

import pytest

from modules.object_detection import Detection
from modules.position_analysis import (
    describe_position,
    horizontal_region,
    vertical_region,
)

W, H = 300, 150


def _det(bbox):
    return Detection(label="chair", confidence=0.9, bbox=bbox)


# ---- Horizontal regions (P1-P4) ---------------------------------------------

def test_box_centered_is_center():
    assert horizontal_region((100, 0, 200, 100), W) == "center"


def test_box_in_left_third_is_left():
    assert horizontal_region((0, 0, 80, 100), W) == "left"


def test_box_in_right_third_is_right():
    assert horizontal_region((220, 0, 300, 100), W) == "right"


def test_box_straddling_left_boundary():
    # Centre at 100 == exactly the first third boundary -> center side
    assert horizontal_region((60, 0, 140, 100), W) == "center"
    # Centre just left of the boundary -> left
    assert horizontal_region((40, 0, 120, 100), W) == "left"


# ---- Vertical regions (P5) ---------------------------------------------------

def test_vertical_regions():
    assert vertical_region((0, 0, 100, 30), H) == "upper"
    assert vertical_region((0, 60, 100, 90), H) == "middle"
    assert vertical_region((0, 120, 100, 150), H) == "lower"


def test_vertical_option_in_phrase():
    text = describe_position(_det((0, 0, 80, 40)), W, H, include_vertical=True)
    assert "upper" in text and "left" in text


# ---- Degenerate inputs (P6) ---------------------------------------------------

def test_degenerate_dimensions_default_safely():
    assert horizontal_region((0, 0, 10, 10), 0) == "center"
    assert vertical_region((0, 0, 10, 10), 0) == "middle"


# ---- Cautious phrasing (P7) ----------------------------------------------------

def test_phrasing_is_cautious_and_image_scoped():
    text = describe_position(_det((0, 0, 80, 100)), W, H)
    assert text == "appears on the left side of the image"
    # never a distance/collision claim
    for banned in ("meters", "metres", "away", "close to you", "safe"):
        assert banned not in text
