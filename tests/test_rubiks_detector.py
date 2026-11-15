"""Tests for the heuristic 3x3 colour-grid Rubik's cube check."""

import cv2
import numpy as np

from modules import rubiks_detector as rd

RED = (0, 0, 220)
ORANGE = (0, 140, 250)
YELLOW = (0, 220, 240)
GREEN = (0, 180, 0)
BLUE = (200, 80, 30)
WHITE = (245, 245, 245)


def _cube_image(colors, size=420, margin=60, rotate=0):
    image = np.full((size, size, 3), 255, np.uint8)
    cell = (size - 2 * margin) // 3
    for row in range(3):
        for col in range(3):
            x0 = margin + col * cell
            y0 = margin + row * cell
            cv2.rectangle(
                image,
                (x0 + 3, y0 + 3),
                (x0 + cell - 3, y0 + cell - 3),
                colors[row * 3 + col],
                -1,
            )
    if rotate:
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rotate, 1.0)
        image = cv2.warpAffine(
            image, matrix, (size, size), borderValue=(255, 255, 255)
        )
    return cv2.imencode(".png", image)[1].tobytes()


def _scrambled():
    return _cube_image([RED, ORANGE, YELLOW, GREEN, BLUE, WHITE, YELLOW, GREEN, ORANGE])


def test_scrambled_face_is_detected():
    detection, error = rd.detect_rubiks_cube(_scrambled())
    assert error is None
    assert detection is not None
    assert detection.label == "rubik's cube"
    assert detection.source == "color-grid"
    assert 0.55 <= detection.confidence <= 0.99
    x1, y1, x2, y2 = detection.bbox
    assert x2 - x1 > 150 and y2 - y1 > 150


def test_rotated_face_is_detected():
    detection, error = rd.detect_rubiks_cube(
        _cube_image(
            [RED, ORANGE, YELLOW, GREEN, BLUE, WHITE, YELLOW, GREEN, ORANGE],
            rotate=20,
        )
    )
    assert error is None
    assert detection is not None


def test_solved_single_colour_face_is_not_claimed():
    detection, error = rd.detect_rubiks_cube(_cube_image([YELLOW] * 9))
    assert detection is None
    assert error is None


def test_blank_image_is_not_claimed():
    blank = np.full((420, 420, 3), 255, np.uint8)
    data = cv2.imencode(".png", blank)[1].tobytes()
    detection, error = rd.detect_rubiks_cube(data)
    assert detection is None
    assert error is None


def test_junk_bytes_return_error():
    detection, error = rd.detect_rubiks_cube(b"not-an-image")
    assert detection is None
    assert error and "could not be read" in error


def test_classify_hsv_reference_colours():
    assert rd.classify_hsv(0, 220, 200) == "red"
    assert rd.classify_hsv(178, 220, 200) == "red"  # red wraps at 180
    assert rd.classify_hsv(15, 220, 220) == "orange"
    assert rd.classify_hsv(30, 220, 220) == "yellow"
    assert rd.classify_hsv(60, 200, 180) == "green"
    assert rd.classify_hsv(115, 200, 180) == "blue"
    assert rd.classify_hsv(0, 20, 240) == "white"
    assert rd.classify_hsv(90, 200, 20) is None  # too dark
    assert rd.classify_hsv(0, 25, 100) is None  # grey mid-tone
    assert rd.classify_hsv(90, 200, 200) is None  # off-hue cyan/teal
