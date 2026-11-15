"""Classical 3x3 colour-grid checker for Rubik's cube faces.

This is NOT a trained model and NOT 100% accurate. It looks for the
visual signature of a scrambled cube face: a quadrilateral region made
of a 3x3 grid of saturated stickers (red, orange, yellow, green, blue,
white). It is fast, offline, and deliberately conservative.

What it handles well:
  - a clear, mostly face-on view of a cube face with several colours
  - moderate rotation and mild perspective (four-corner warp)

Known misses (documented in SAFETY_AND_LIMITATIONS):
  - solved or nearly solved faces dominated by one colour (rejected on
    purpose to avoid matching plain coloured objects)
  - heavy glare, motion blur, very small cubes, occluded faces
  - cartoon drawings and other colourful 3x3 patterns (false positives
    are still possible)

The returned confidence is a heuristic pattern score, not the YOLO
model confidence and never a probability of safety.
"""

from __future__ import annotations

import numpy as np

from modules.object_detection import Detection

MAX_DIM = 700
WARP_SIZE = 300
MIN_QUAD_AREA_FRAC = 0.01
MIN_DISTINCT_COLORS = 3
MIN_TILES_CLASSIFIED = 7
MIN_SCORE = 0.55

# OpenCV HSV reference hues (H in 0..179).
_COLOR_HUES = {
    "red": 0,        # wraps to 179 as well
    "orange": 15,
    "yellow": 30,
    "green": 60,
    "blue": 115,
}
_HUE_TOLERANCE = 9
_RED_WRAP_TOLERANCE = 7


def _hue_distance(hue: float, center: float) -> float:
    diff = abs(hue - center)
    return min(diff, 180 - diff)


def classify_hsv(h: float, s: float, v: float) -> str | None:
    """Name a sticker colour from one HSV sample, or None if not a sticker."""
    if v < 55:
        return None  # too dark (shadows, grid gaps)
    if s < 65:
        return "white" if v > 150 else None
    best_label = None
    best_distance = 1e9
    for label, center in _COLOR_HUES.items():
        distance = _hue_distance(h, center)
        if distance < best_distance:
            best_distance = distance
            best_label = label
    tolerance = _RED_WRAP_TOLERANCE if best_label == "red" else _HUE_TOLERANCE
    if best_distance <= tolerance and s >= 80:
        return best_label
    return None


def _classify_tile(tile_bgr: np.ndarray) -> str | None:
    """Majority colour of a tile's centre area (ignores sticker edges)."""
    import cv2

    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    y0, y1 = int(h * 0.30), int(h * 0.70)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    samples = hsv[y0:y1, x0:x1].reshape(-1, 3)
    votes: dict[str, int] = {}
    for hue, sat, val in samples[::4]:
        label = classify_hsv(float(hue), float(sat), float(val))
        if label:
            votes[label] = votes.get(label, 0) + 1
    if not votes:
        return None
    label, count = max(votes.items(), key=lambda item: item[1])
    if count < max(1, (len(samples) // 4) * 0.40):
        return None
    return label


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    diffs = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def _warp_quad(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    import cv2

    dst = np.array(
        [[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1],
         [0, WARP_SIZE - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, matrix, (WARP_SIZE, WARP_SIZE))


def _analyse_warp(warped: np.ndarray) -> tuple[float, int, int]:
    """Return (score, tiles_classified, distinct_colours) for a warped face."""
    cell = WARP_SIZE // 3
    labels: list[str | None] = []
    for row in range(3):
        for col in range(3):
            y0, x0 = row * cell, col * cell
            labels.append(_classify_tile(warped[y0:y0 + cell, x0:x0 + cell]))
    classified = sum(1 for label in labels if label)
    distinct = len({label for label in labels if label})
    score = 0.65 * (classified / 9) + 0.35 * min(1.0, distinct / 5)
    return score, classified, distinct


def detect_rubiks_cube(
    image_bytes: bytes, score_threshold: float = MIN_SCORE
) -> tuple[Detection | None, str | None]:
    """Find the best 3x3 cube-face candidate; (None, None) when not found."""
    import cv2

    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return None, "The image could not be read for the Rubik's cube check."

    original_height, original_width = bgr.shape[:2]
    scale = MAX_DIM / max(bgr.shape[:2])
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = bgr.shape[:2]
    inverse = 1.0 / scale if scale < 1.0 else 1.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 85, 50), (179, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_QUAD_AREA_FRAC * width * height

    best: tuple[float, tuple[int, int, int, int], int, int] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.03 * perimeter, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(contour)
            approx = cv2.boxPoints(rect).reshape(-1, 1, 2)
        approx = approx.astype(np.int32)
        if not cv2.isContourConvex(approx):
            continue
        warped = _warp_quad(bgr, _order_quad(approx))
        score, classified, distinct = _analyse_warp(warped)
        if distinct < MIN_DISTINCT_COLORS or classified < MIN_TILES_CLASSIFIED:
            continue
        if score < score_threshold:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        box = (
            max(0, int(x * inverse)),
            max(0, int(y * inverse)),
            min(original_width, int((x + w) * inverse)),
            min(original_height, int((y + h) * inverse)),
        )
        if best is None or score > best[0]:
            best = (score, box, classified, distinct)

    if best is None:
        return None, None

    score, box, _, _ = best
    detection = Detection(
        label="rubik's cube",
        confidence=round(min(score, 0.99), 3),
        bbox=box,
        source="color-grid",
    )
    return detection, None


__all__ = [
    "MIN_SCORE",
    "classify_hsv",
    "detect_rubiks_cube",
]
